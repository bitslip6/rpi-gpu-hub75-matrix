#!/usr/bin/env python3
"""Pi 5 isolation environment: read-only audit; explicit boot/runtime transactions.

No GPIO output, scheduler/governor/RT-budget changes, service installation,
kernel selection, driver reset, shutdown or reboot. Uses rio_isolate's checked
runtime ownership/restore protocol. CLI paths target the actual host only;
function arguments are injectable for pin-free tests, not a fake-root CLI.
"""
import argparse
import collections
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import tempfile
import time

import rio_isolate as iso

BOOT = Path('/boot/firmware')
BASE = Path('/var/lib/rpihub75')
MANAGED = {'isolcpus', 'irqaffinity', 'nohz_full', 'rcu_nocbs'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def private_parent(path):
    """Create only root/caller-owned private state hierarchy, never through links."""
    path = Path(path).absolute()
    if '..' in path.parts:
        raise ValueError('state paths must not contain parent traversal')
    # Check ancestors BEFORE creating anything beneath them. Sticky root-owned
    # /tmp is safe for a private caller-owned child; ordinary writable dirs are not.
    for part in [*reversed(path.parents), path]:
        try:
            info = part.lstat()
        except FileNotFoundError:
            part.mkdir(mode=0o700)
            sync_directory(part.parent)
            info = part.lstat()
        sticky_root = info.st_uid == 0 and bool(info.st_mode & stat.S_ISVTX)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0, os.geteuid()) or \
                (info.st_mode & 0o022 and not sticky_root):
            raise ValueError(f'untrusted state ancestor: {part}')
    if path.stat().st_mode & 0o077:
        raise ValueError(f'state parent must be private: {path}')


def read_regular(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o022 or \
                info.st_uid not in (0, os.geteuid()):
            raise ValueError(f'unsafe regular file: {path}')
        return stream.read()


def atomic_bytes(path, raw):
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, name = tempfile.mkstemp(prefix='.rio-cmdline-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            if stat.S_IMODE(os.fstat(stream.fileno()).st_mode) != mode:
                os.fchmod(stream.fileno(), mode)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def boot_line(raw, flags, cpu, housekeeping):
    text = raw.decode('ascii')
    if len(text.splitlines()) != 1 or any(c in text for c in ['"', "'", '\\', '\x00']):
        raise ValueError('require one unquoted ASCII command line; review complex syntax manually')
    tokens = text.split()
    if not tokens or not any(t.startswith('root=') for t in tokens):
        raise ValueError('missing root argument')
    if flags['CPU_ISOLATION'] != 'y' or flags.get('GENERIC_IRQ_MIGRATION') != 'y':
        raise ValueError('CPU/IRQ isolation capability is not established for this kernel')
    if any(flags[k] not in ('y', 'n') for k in ['NO_HZ_FULL', 'RCU_NOCB_CPU']):
        raise ValueError('unknown full-dynticks/RCU capability; refuse to guess')
    keep = [t for t in tokens if t.split('=', 1)[0] not in MANAGED]
    keep += [f'isolcpus=domain,managed_irq,{cpu}',
             'irqaffinity=' + iso.format_list(housekeeping)]
    if flags['NO_HZ_FULL'] == 'y':
        keep.append(f'nohz_full={cpu}')
    if flags['RCU_NOCB_CPU'] == 'y':
        keep.append(f'rcu_nocbs={cpu}')
    return (' '.join(keep) + '\n').encode('ascii')


def inspect_boot(cpu, housekeeping, boot=BOOT, config_path=None, release=None):
    release = platform.release() if release is None else release
    config_path = Path('/boot/config-' + release) if config_path is None else config_path
    config = read_regular(boot/'config.txt')
    active, kernel, arm64 = True, None, True
    for original in config.decode().splitlines():
        line = original.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith('['):
            if line not in ['[all]', '[pi5]', '[cm4]', '[cm5]']:
                raise ValueError('unsupported firmware filter; review active configuration manually')
            active = line in ['[all]', '[pi5]']
        elif re.match(r'(?:include\s|cmdline\s*=|os_prefix\s*=|tryboot|boot_partition\s*=)', line):
            raise ValueError('indirect firmware/command-line selection requires manual review')
        elif active and re.match(r'kernel\s*=', line):
            kernel = line.split('=', 1)[1].strip()
        elif active and re.match(r'arm_64bit\s*=', line):
            arm64 = line.split('=', 1)[1].strip() == '1'
    # Pi 5's normal 64-bit automatic image. The banner check below still MUST
    # match the running release; do not guess a fallback kernel or switch images.
    selection = 'explicit' if kernel else 'pi5-default-verified-by-release'
    if not arm64:
        raise ValueError('32-bit firmware selection is not supported')
    kernel = kernel or 'kernel_2712.img'
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', kernel) or kernel in ('.', '..'):
        raise ValueError('simple Pi 5 kernel selection is required')
    image = read_regular(boot/kernel)
    unpacked = gzip.decompress(image) if image.startswith(b'\x1f\x8b') else image
    if b'Linux version ' + release.encode() + b' ' not in unpacked:
        raise ValueError('selected image does not identify the running kernel; refuse config guess')
    kernel_config = read_regular(config_path)
    flags = iso.flags_from_config(kernel_config.decode())
    # Linux omits this unselectable symbol entirely when both of its enabling
    # choices are off (TREE_RCU && (RCU_EXPERT || NO_HZ_FULL)). Do not treat an
    # arbitrary missing key as disabled; require these explicit dependency facts.
    rcu_dependency_disabled = (flags['RCU_NOCB_CPU'] == 'unknown' and
                               flags['NO_HZ_FULL'] == 'n' and bool(re.search(
                                   r'^# CONFIG_RCU_EXPERT is not set$', kernel_config.decode(), re.M)))
    if rcu_dependency_disabled:
        flags['RCU_NOCB_CPU'] = 'n'
    flags['GENERIC_IRQ_MIGRATION'] = ('y' if re.search(
        r'^CONFIG_GENERIC_IRQ_MIGRATION=y$', kernel_config.decode(), re.M) else 'unknown')
    before = read_regular(boot/'cmdline.txt')
    after = boot_line(before, flags, cpu, housekeeping)
    return dict(schema=1, kind='rio-boot', boot_directory=str(boot), kernel=kernel,
                kernel_selection=selection, kernel_release=release, kernel_image_sha256=sha(image),
                config_path=str(config_path), kernel_config_sha256=sha(kernel_config),
                config_text=config.decode(), config_sha256=sha(config),
                cmdline_before=before.decode(), cmdline_after=after.decode(),
                before_sha256=sha(before), after_sha256=sha(after), flags=flags,
                rcu_nocb_disabled_by_dependencies=rcu_dependency_disabled,
                cpu=cpu, housekeeping=sorted(housekeeping), changed=before != after)


def stage_boot(plan, state, boot=BOOT):
    if not plan['changed']:
        return {'phase': 'unchanged', 'plan': plan}
    private_parent(state.parent)
    # Durable original contents are saved BEFORE the firmware partition changes.
    journal = iso.Journal(state, dict(plan, phase='pending', created_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    try:
        if sha(read_regular(boot/'config.txt')) != plan['config_sha256'] or \
                sha(read_regular(boot/plan['kernel'])) != plan['kernel_image_sha256'] or \
                sha(read_regular(Path(plan['config_path']))) != plan['kernel_config_sha256'] or \
                sha(read_regular(boot/'cmdline.txt')) != plan['before_sha256']:
            raise ValueError('boot configuration changed since review')
        for name, text in [('cmdline.txt.before', plan['cmdline_before']),
                           ('config.txt.before', plan['config_text']),
                           ('cmdline.txt.after', plan['cmdline_after'])]:
            fd = os.open(state/name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(text.encode())
                stream.flush()
                os.fsync(stream.fileno())
        sync_directory(state)
        atomic_bytes(boot/'cmdline.txt', plan['cmdline_after'].encode())
        if sha(read_regular(boot/'cmdline.txt')) != plan['after_sha256']:
            raise ValueError('boot command-line readback mismatch')
        journal.data['phase'] = 'staged'
        journal.save()
    except BaseException:
        # Best effort only; durable pending journal also supports explicit recovery.
        if sha(read_regular(boot/'cmdline.txt')) == plan['after_sha256']:
            atomic_bytes(boot/'cmdline.txt', plan['cmdline_before'].encode())
            journal.data['phase'] = 'rolled_back'
            journal.save()
        raise
    return journal.data


def restore_boot(state, boot=BOOT):
    journal = iso.Journal(state)
    d = journal.data
    if d.get('kind') != 'rio-boot' or d.get('schema') != 1 or d['boot_directory'] != str(boot):
        raise ValueError('not a matching boot backup')
    if sha(d['cmdline_before'].encode()) != d['before_sha256'] or \
            sha(d['cmdline_after'].encode()) != d['after_sha256']:
        raise ValueError('boot backup content/hash mismatch')
    if sha(read_regular(boot/'config.txt')) != d['config_sha256'] or \
            sha(read_regular(boot/d['kernel'])) != d['kernel_image_sha256']:
        raise ValueError('firmware config/image changed; refusing another boot policy')
    current = sha(read_regular(boot/'cmdline.txt'))
    if current == d['before_sha256']:
        pass
    elif current == d['after_sha256']:
        atomic_bytes(boot/'cmdline.txt', d['cmdline_before'].encode())
    else:
        raise ValueError('command line changed by another actor; refusing restore')
    if sha(read_regular(boot/'cmdline.txt')) != d['before_sha256']:
        raise ValueError('boot restore readback mismatch')
    d['phase'] = 'restored'
    journal.save()
    return d


def rp1_health():
    """Read PCI config only, never GPIO MMIO; PM bookkeeping is not an access test."""
    result = {}
    try:
        with Path('/sys/bus/pci/devices/0002:01:00.0/config').open('rb') as stream:
            raw = stream.read(64)
        result['config_hex'] = raw.hex()
        result['pci_ok'] = (len(raw) == 64 and int.from_bytes(raw[:2], 'little') == 0x1de4 and
                            int.from_bytes(raw[2:4], 'little') == 1 and raw[14] == 0)
        p = subprocess.run(['vcgencmd', 'get_throttled'], capture_output=True, text=True, timeout=10)
        result['throttled_raw'] = p.stdout.strip()
        match = re.fullmatch(r'throttled=0x([0-9a-fA-F]+)', p.stdout.strip())
        flags = int(match[1], 16) if match and p.returncode == 0 else None
        result['throttled_flags'] = flags
        result['ok'] = result['pci_ok'] and flags is not None and not (flags & 0xf)
    except (OSError, subprocess.SubprocessError) as exc:
        result.update(ok=False, error=str(exc))
    return result


def gpio_users(proc=Path('/proc')):
    users, errors = [], []
    for directory in proc.glob('[0-9]*'):
        try:
            name = (directory/'comm').read_text().strip()
            argv = (directory/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if name == 'example' or name.startswith(('rio_scope', 'pio_')) or any(
                    Path(a).name in {'rio_baseline.py', 'pio_baseline.py'} for a in argv if a):
                users.append({'pid': directory.name, 'name': name, 'reason': 'recognized workload'})
            maps = (directory/'maps').read_text()
            mapped = any(s in maps for s in
                         ['/dev/gpiomem', '/dev/mem', '/dev/pio', '/dev/gpiochip'])
            held = []
            for fd in (directory/'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    if target == '/dev/mem' or target.startswith(('/dev/gpiomem', '/dev/pio', '/dev/gpiochip')):
                        held.append(target)
                except FileNotFoundError:
                    pass
            if mapped or held:
                users.append({'pid': directory.name, 'name': name, 'mapped': mapped, 'held': held})
        except (FileNotFoundError, ProcessLookupError):
            pass
        except OSError as exc:
            errors.append({'pid': directory.name, 'error': str(exc)})
    return {'users': users, 'errors': errors}


def verify(host, journal, original=False):
    iso.check_journal(host, journal)
    for entry in journal.data['entries']:
        iso.check_identity(host, entry)
        wanted = set(entry['old' if original else 'new'])
        if iso.read_current(host, entry) != wanted:
            raise ValueError('journaled setting drift; restore/review rather than stacking apply')
        if not original and entry['status'] != 'applied':
            raise ValueError('journal is partial; review/restore before applying again')
        if not original and 'effective_path' in entry:
            effective = iso.cpu_list(host.read(entry['effective_path']), allow_empty=True)
            if not effective or not effective <= wanted:
                raise ValueError('effective IRQ affinity drift')


def runtime_apply(host, cpu, hk, state, health=rp1_health, users=gpio_users, sleep=time.sleep):
    check = users()
    if check['users'] or check['errors']:
        raise ValueError('GPIO/PIO workload or unavailable process inspection; stop/review first')
    before = health()
    if not before['ok']:
        raise ValueError(f'RP1/power preflight failed; no apply: {before}')
    if state.exists():
        journal = iso.Journal(state)
        if journal.data['phase'] != 'applied':
            raise ValueError('state is not active; preserve it and select a NEW --state directory')
        if journal.data['cpu'] != cpu or set(journal.data['housekeeping']) != hk:
            raise ValueError('state belongs to another CPU policy')
        verify(host, journal)
        current = iso.audit(host, cpu, hk)
        if current['operations'] or current['irqbalance_tids']:
            raise ValueError('new movable work/irqbalance discovered; review before another apply')
        return dict(phase='already_applied', state=str(state), health=before)
    report = iso.audit(host, cpu, hk)
    if report['irqbalance_tids']:
        raise ValueError('irqbalance is active; explicitly configure its exclusion first')
    if not report['operations']:
        return dict(phase='already_confined_without_owned_journal', state=None,
                    health=before, note='No settings changed; no original values invented')
    report.update(environment_health=before, gpio_user_check=check)
    journal = None
    try:
        journal, problems = iso.apply(host, report, state)
        journal.data['environment_after'] = health()
        journal.save()
        if problems or not journal.data['environment_after']['ok']:
            raise ValueError('partial apply or post-apply RP1/power failure')
        sleep(1)
        journal.data['environment_settled'] = health()
        journal.save()
        if not journal.data['environment_settled']['ok']:
            raise ValueError('settled RP1/power failure')
        verify(host, journal)
        current = iso.audit(host, cpu, hk)
        if current['operations'] or current['irqbalance_tids']:
            raise ValueError('new movable work/irqbalance appeared during apply; rolling back')
        return dict(phase='applied', state=str(state), entries=len(journal.data['entries']),
                    statuses=dict(collections.Counter(e['status'] for e in journal.data['entries'])),
                    health=journal.data['environment_settled'])
    except BaseException as exc:
        if journal is not None:
            problems = iso.restore(host, journal)  # same-boot; never force conflicts
            if problems:
                raise ValueError(f'rollback incomplete; inspect {state}/journal.json: {problems}') from exc
        raise


def runtime_restore(host, state, users=gpio_users):
    check = users()
    if check['users'] or check['errors']:
        raise ValueError('stop GPIO/PIO workloads and resolve process inspection before restore')
    journal = iso.Journal(state)
    problems = iso.restore(host, journal)
    if problems:
        raise ValueError('partial restore; inspect journal, never force conflicts')
    verify(host, journal, original=True)
    return dict(phase='restored', state=str(state), entries=len(journal.data['entries']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='audit', choices=[
        'audit', 'boot-plan', 'boot-apply', 'boot-restore', 'apply', 'restore'])
    parser.add_argument('--cpu', type=int, default=3)
    parser.add_argument('--housekeeping', type=iso.cpu_list, default={0, 1, 2})
    parser.add_argument('--state', type=Path, help='NEW state for apply; matching journal for restore')
    parser.add_argument('--yes', action='store_true')
    args = parser.parse_args(argv)
    host = iso.Host()
    online = host.online()
    if args.cpu not in online or args.housekeeping != online - {args.cpu}:
        parser.error('profile requires all other online CPUs as housekeeping')
    mutating = args.action not in ('audit', 'boot-plan')
    if mutating and (not args.yes or os.geteuid() != 0):
        parser.error('mutations require root and --yes; stop GPIO and retain recovery access')
    if args.action == 'boot-restore' and args.state is None:
        parser.error('boot-restore requires the exact --state backup directory')
    if args.action in ('boot-plan', 'boot-apply'):
        if 'Raspberry Pi 5' not in Path('/proc/device-tree/model').read_text():
            raise ValueError('boot edits restricted to Pi 5')
        plan = inspect_boot(args.cpu, args.housekeeping)
        if args.action == 'boot-plan':
            result = plan
        else:
            with host.lock():
                check = gpio_users()
                if check['users'] or check['errors']:
                    raise ValueError('stop GPIO/PIO and resolve process inspection first')
                state = args.state or BASE/'boot'/dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
                result = dict(state=str(state), **stage_boot(plan, state))
    elif args.action == 'boot-restore':
        with host.lock():
            result = restore_boot(args.state)
    elif args.action == 'audit':
        result = dict(isolation=iso.audit(host, args.cpu, args.housekeeping), rp1_health=rp1_health())
    else:
        if args.state is None:
            # UUID is kernel supplied, but validate before using it as a path component.
            boot_id = host.boot_id()
            if not re.fullmatch(r'[0-9a-f-]{36}', boot_id):
                raise ValueError('invalid boot identity')
            state = BASE/'runtime'/f'{boot_id}-cpu{args.cpu}'
        else:
            state = args.state.absolute()
        with host.lock():
            if args.action == 'apply':
                private_parent(state.parent)
                result = runtime_apply(host, args.cpu, args.housekeeping, state)
            else:
                result = runtime_restore(host, state)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    import signal
    signal.signal(signal.SIGTERM, iso.interrupted)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted; inspect saved journals. No SIGKILL/power-loss cleanup guarantee.', file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
        print(f'Environment operation refused/failed: {exc}. Inspect journals; no forced recovery.', file=sys.stderr)
        sys.exit(1)
