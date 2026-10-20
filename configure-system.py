#!/usr/bin/python3 -I
"""Persist the Pi 5 CPU3 display-isolation profile; --restore undoes owned changes.

Run with sudo, stop GPIO/PIO applications first, and keep recovery access.
Use --dry-run to inspect. No GPIO output, reboot, kernel installation, governor,
scheduler, RT-budget, thermal or electrical-policy changes are made.
"""
import argparse
import base64
import datetime as dt
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import uuid

UNIT = 'rpihub75-isolation.service'
IRQBALANCE = 'irqbalance.service'
KIND = 'rpihub75-system-configuration'
PHASES = {'installing', 'installed', 'restoring', 'restore_incomplete', 'restored'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def regular(path, private=False):
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        s = os.fstat(stream.fileno())
        if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1 or s.st_uid != os.geteuid() or \
                s.st_mode & (0o077 if private else 0o022):
            raise ValueError(f'unsafe owned file: {path}')
        return stream.read()


def directory(path, private=False):
    s = path.lstat()
    if not stat.S_ISDIR(s.st_mode) or s.st_uid not in (0, os.geteuid()) or s.st_mode & 0o022:
        # Root-owned sticky /tmp is acceptable above private test/state directories.
        if not (stat.S_ISDIR(s.st_mode) and s.st_uid == 0 and s.st_mode & stat.S_ISVTX):
            raise ValueError(f'untrusted directory: {path}')
    if private and (s.st_uid != os.geteuid() or s.st_mode & 0o077):
        raise ValueError(f'private owned directory required: {path}')


class Paths:
    """Alternate roots are function-level test injection ONLY, never a CLI option."""
    def __init__(self, root=Path('/')):
        self.root = root
        self.base = root/'var/lib/rpihub75'
        self.state = self.base/'configure-system'
        self.boot = root/'boot/firmware'
        self.libexec = root/'usr/libexec'
        self.package = self.libexec/'rpihub75-system'
        self.units = root/'etc/systemd/system'
        self.wants = self.units/'multi-user.target.wants'
        self.files = {
            'runner': self.package/'configure-system.py',
            'environment': self.package/'rio_environment.py',
            'isolate': self.package/'rio_isolate.py',
            'unit': self.units/UNIT,
            'enable': self.wants/UNIT,
            'irq_mask': self.units/IRQBALANCE,
        }
        self.dirs = {'libexec': self.libexec, 'package': self.package, 'wants': self.wants}

    def check_parents(self):
        candidates = [self.units, self.libexec.parent, self.base.parent]
        candidates += [p for p in (self.libexec, self.wants) if os.path.lexists(p)]
        for path in candidates:
            for parent in [*reversed(path.parents), path]:
                directory(parent)


def load_state(paths):
    if not os.path.lexists(paths.state):
        return None
    for parent in [*reversed(paths.state.parents), paths.state]:
        directory(parent, private=parent == paths.state)
    data = json.loads(regular(paths.state/'journal.json', private=True))
    if data.get('schema') != 1 or data.get('kind') != KIND or data.get('phase') not in PHASES:
        raise ValueError('unknown system configuration journal')
    seen = set()
    for f in data['files']:
        key = f['key']
        if key not in paths.files or key in seen or f['status'] not in {
                'planned', 'pending', 'installed', 'not_owned', 'removed'}:
            raise ValueError('invalid asset journal')
        seen.add(key)
        if f['kind'] != ('link' if key in ('enable', 'irq_mask') else 'file'):
            raise ValueError('invalid asset type')
        if f['kind'] == 'file':
            raw = base64.b64decode(f['data'], validate=True)
            if digest(raw) != f['sha256'] or f['mode'] not in (0o644, 0o755):
                raise ValueError('asset backup/hash mismatch')
        elif f['target'] != ('/dev/null' if key == 'irq_mask' else str(paths.files['unit'])):
            raise ValueError('invalid managed link')
    for key, record in data['directories'].items():
        if key not in paths.dirs or record['status'] not in {
                'pending', 'created', 'not_owned', 'removed', 'preserved'}:
            raise ValueError('invalid directory journal')
    for boot_id, relative in data['runtime'].items():
        uuid.UUID(boot_id)
        parts = Path(relative).parts
        if len(parts) != 2 or parts[0] != 'runtime' or \
                not parts[1].startswith(boot_id + '-'):
            raise ValueError('invalid runtime journal path')
        uuid.UUID(parts[1][len(boot_id)+1:])
    if type(data['boot_owned']) is not bool or type(data['irq_was_active']) is not bool:
        raise ValueError('invalid configuration ownership')
    return data


def load_helpers(paths, installed=False, data=None):
    """Boot imports only verified root-owned copies, never cwd/PYTHONPATH code."""
    here = Path(__file__).resolve().parent
    sources = ({'rio_isolate': paths.files['isolate'], 'rio_environment': paths.files['environment']}
               if installed else
               {'rio_isolate': here/'tests/rio_isolate.py', 'rio_environment': here/'tests/rio_environment.py'})
    if installed:
        if data is None:
            raise ValueError('installed helper manifest is missing')
        expected = {f['key']: f for f in data['files']}
        for key in ('runner', 'environment', 'isolate'):
            if key not in expected or digest(regular(paths.files[key])) != expected[key]['sha256']:
                raise ValueError('installed helper differs from its root-owned manifest')
    modules = {}
    for name, path in sources.items():
        # Explicit checkout execution trusts that checkout. Installed boot copies
        # have already passed ownership/hash checks. Avoid local .pyc artifacts.
        raw = regular(path) if installed else path.read_bytes()
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        exec(compile(raw, str(path), 'exec'), module.__dict__)
        modules[name] = module
    return modules['rio_environment'], modules['rio_isolate']


def source_payload(paths, installed=False):
    here = Path(__file__).resolve()
    source = ({'runner': paths.files['runner'], 'environment': paths.files['environment'],
               'isolate': paths.files['isolate']} if installed else
              {'runner': here, 'environment': here.parent/'tests/rio_environment.py',
               'isolate': here.parent/'tests/rio_isolate.py'})
    return {key: path.read_bytes() for key, path in source.items()}


def service_text(paths):
    return f'''[Unit]
Description=Pi 5 HUB75 CPU3 isolation (no GPIO output)
After=local-fs.target systemd-udev-trigger.service
Before=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStartPre=/usr/bin/udevadm settle --timeout=20
ExecStart=/usr/bin/python3 -I -B {paths.files['runner']} --apply-runtime
CPUAffinity=0 1 2
UMask=0077
NoNewPrivileges=yes
Environment=PATH=/usr/sbin:/usr/bin:/sbin:/bin
TimeoutStartSec=90
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
'''.encode()


class System:
    def __init__(self, env, iso, paths):
        self.env, self.iso, self.paths = env, iso, paths
        self.host = iso.Host()

    def run(self, *args):
        p = subprocess.run(['/usr/bin/systemctl', *args], capture_output=True, text=True, timeout=60)
        if p.returncode:
            raise ValueError(f'systemctl {args} failed: {p.stderr.strip() or p.stdout.strip()}')
        return p.stdout

    def unit(self, name):
        p = subprocess.run(['/usr/bin/systemctl', 'show', name,
                            '--property=LoadState,ActiveState,MainPID,UnitFileState,FragmentPath'],
                           capture_output=True, text=True, timeout=30)
        fields = dict(s.split('=', 1) for s in p.stdout.splitlines() if '=' in s)
        if (p.returncode and fields.get('LoadState') != 'not-found') or \
                not {'LoadState', 'ActiveState', 'MainPID'} <= fields.keys():
            raise ValueError(f'cannot inspect {name}: {p.stderr}')
        if fields['ActiveState'] not in ('active', 'inactive', 'failed'):
            raise ValueError(f'{name} is transitioning; retry after it settles')
        return fields

    def preflight(self):
        if 'Raspberry Pi 5' not in Path('/proc/device-tree/model').read_text() or \
                self.host.online() != {0, 1, 2, 3}:
            raise ValueError('requires a Pi 5 with CPUs0–3 online')
        if not Path('/run/systemd/system').is_dir():
            raise ValueError('systemd must be the running service manager')
        for p in ['/usr/bin/python3', '/usr/bin/udevadm', '/usr/bin/systemctl']:
            if not os.access(p, os.X_OK):
                raise ValueError(f'required executable missing: {p}')

    def idle(self):
        check = self.env.gpio_users()
        if check['users'] or check['errors']:
            raise ValueError('stop GPIO/PIO apps and collectors; process inspection must be clear')

    def plan(self):
        return self.env.inspect_boot(3, {0, 1, 2})

    def health(self):
        result = self.env.rp1_health()
        if not result['ok']:
            raise ValueError(f'RP1/power preflight failed: {result}')
        return result

    def apply_runtime(self, state):
        return self.env.runtime_apply(self.host, 3, {0, 1, 2}, state)

    def restore_runtime(self, state):
        return self.env.runtime_restore(self.host, state)


class Configuration:
    def __init__(self, paths, system, env, iso):
        self.p, self.s, self.env, self.iso = paths, system, env, iso

    def saved(self):
        return load_state(self.p)

    def journal(self):
        data = self.saved()
        if data is None:
            raise ValueError('no configure-system journal exists')
        return self.iso.Journal(self.p.state)

    def plan(self, payload):
        if set(payload) != {'runner', 'environment', 'isolate'}:
            raise ValueError('exactly the installer and two runtime helpers are required')
        for name, raw in payload.items():
            compile(raw, name, 'exec')
        self.s.preflight()
        self.p.check_parents()
        self.s.idle()
        health = self.s.health()
        boot = self.s.plan()
        irq = self.s.unit(IRQBALANCE)
        profile = self.s.unit(UNIT)
        if profile['LoadState'] != 'not-found' or os.path.lexists(self.p.package):
            raise ValueError('unowned profile service/package exists; refusing replacement')
        for key in ('runner', 'environment', 'isolate', 'unit', 'enable'):
            if os.path.lexists(self.p.files[key]):
                raise ValueError(f'unowned install path exists: {self.p.files[key]}')
        mask = self.p.files['irq_mask']
        masked = mask.is_symlink() and os.readlink(mask) == '/dev/null'
        if os.path.lexists(mask) and not masked:
            raise ValueError('local irqbalance unit/override exists; review it manually')
        if masked and irq['ActiveState'] == 'active':
            raise ValueError('irqbalance is both masked and active; review inconsistent state')
        report = self.iso.audit(self.s.host)
        for tid in report['irqbalance_tids']:
            if irq['ActiveState'] != 'active' or self.s.host.task(tid)['tgid'] != int(irq['MainPID']):
                raise ValueError('unmanaged irqbalance process; no arbitrary process termination')
        return dict(boot=boot, health=health, irqbalance=irq, irq_mask_preexisting=masked,
                    payload=payload, service=service_text(self.p))

    def asset_record(self, key, raw=None, target=None):
        d = dict(key=key, status='planned', kind='link' if target is not None else 'file')
        if target is not None:
            d['target'] = target
        else:
            d.update(data=base64.b64encode(raw).decode(), sha256=digest(raw),
                     mode=0o755 if key == 'runner' else 0o644)
        return d

    def check_asset(self, f):
        path = self.p.files[f['key']]
        if not os.path.lexists(path):
            return False
        st = path.lstat()
        if f.get('identity') and [st.st_dev, st.st_ino] != f['identity']:
            raise ValueError(f'owned path identity changed: {path}')
        if f['kind'] == 'link':
            if not stat.S_ISLNK(st.st_mode) or st.st_uid != os.geteuid() or os.readlink(path) != f['target']:
                raise ValueError(f'owned link changed: {path}')
        elif digest(regular(path)) != f['sha256'] or stat.S_IMODE(st.st_mode) != f['mode']:
            raise ValueError(f'owned file changed: {path}')
        return True

    def put_asset(self, journal, f):
        f['status'] = 'pending'
        journal.save()
        path = self.p.files[f['key']]
        try:
            if f['kind'] == 'link':
                os.symlink(f['target'], path)
            else:
                fd, name = tempfile.mkstemp(prefix='.rpihub75-', dir=path.parent)
                try:
                    with os.fdopen(fd, 'wb') as stream:
                        os.fchmod(stream.fileno(), f['mode'])
                        stream.write(base64.b64decode(f['data'], validate=True))
                        stream.flush()
                        os.fsync(stream.fileno())
                    # Exclusive publication: never overwrite a concurrently created path.
                    os.link(name, path, follow_symlinks=False)
                finally:
                    os.unlink(name)
        except FileExistsError:
            f['status'] = 'not_owned'
            journal.save()
            raise
        sync_dir(path.parent)
        st = path.lstat()
        f.update(status='installed', identity=[st.st_dev, st.st_ino])
        journal.save()

    def remove_asset(self, journal, key):
        for f in journal.data['files']:
            if f['key'] != key or f['status'] in ('planned', 'not_owned', 'removed'):
                continue
            if self.check_asset(f):
                self.p.files[key].unlink()
                sync_dir(self.p.files[key].parent)
            f['status'] = 'removed'
            journal.save()

    def runtime(self, journal):
        boot_id = self.s.host.boot_id()
        uuid.UUID(boot_id)
        relative = journal.data['runtime'].get(boot_id)
        if relative is None:
            relative = f'runtime/{boot_id}-{uuid.uuid4()}'
            journal.data['runtime'][boot_id] = relative
            journal.save()  # Record ownership path BEFORE the runtime transaction.
        state = self.p.state/relative
        self.env.private_parent(state.parent)
        if state.exists():
            old = self.iso.Journal(state)
            if old.data['phase'] == 'restored':
                # Preserve failed/rolled-back attempts and create a fresh journal.
                journal.data['past_runtime'].append(relative)
                relative = f'runtime/{boot_id}-{uuid.uuid4()}'
                journal.data['runtime'][boot_id] = relative
                journal.save()
                state = self.p.state/relative
        result = self.s.apply_runtime(state)
        journal.data['last_runtime'] = result
        journal.save()
        return result

    def configure(self, payload):
        previous = self.saved()
        if previous and previous['phase'] == 'installed':
            journal = self.journal()
            for f in journal.data['files']:
                if f['status'] == 'installed' and not self.check_asset(f):
                    raise ValueError('installed asset is missing; restore before repairing')
                if f['key'] in payload and digest(payload[f['key']]) != f['sha256']:
                    raise ValueError('installer/helper version changed; restore before reinstalling')
            current = self.s.plan()
            if current['changed']:
                raise ValueError('boot policy drifted; restore/review before configuring again')
            self.s.idle()
            result = self.runtime(journal)
            return dict(phase='already_configured', runtime=result, reboot_required=False)
        if previous and previous['phase'] != 'restored':
            raise ValueError('incomplete prior transaction; use --restore before another configure')
        plan = self.plan(payload)
        self.env.private_parent(self.p.base)
        if previous:
            archive = self.p.base/f'configure-system-restored-{uuid.uuid4()}'
            os.rename(self.p.state, archive)
            sync_dir(self.p.base)
        files = [self.asset_record(k, raw=v) for k, v in payload.items()]
        files += [self.asset_record('unit', raw=plan['service']),
                  self.asset_record('enable', target=str(self.p.files['unit']))]
        if not plan['irq_mask_preexisting']:
            files.append(self.asset_record('irq_mask', target='/dev/null'))
        data = dict(schema=1, kind=KIND, phase='installing',
                    created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    boot_owned=plan['boot']['changed'], irq_was_active=plan['irqbalance']['ActiveState']=='active',
                    irq_before=plan['irqbalance'], irq_stop_attempted=False,
                    files=files, directories={}, runtime={}, past_runtime=[])
        journal = self.iso.Journal(self.p.state, data)
        try:
            for key, path in self.p.dirs.items():
                if not path.exists():
                    journal.data['directories'][key] = {'status':'pending'}
                    journal.save()
                    try:
                        path.mkdir(mode=0o755)
                    except FileExistsError:
                        journal.data['directories'][key]['status'] = 'not_owned'
                        journal.save()
                        raise
                    sync_dir(path.parent)
                    st = path.lstat()
                    journal.data['directories'][key].update(status='created', identity=[st.st_dev, st.st_ino])
                    journal.save()
                directory(path)
            if journal.data['boot_owned']:
                self.env.stage_boot(plan['boot'], self.p.state/'boot', self.p.boot)
            # Mask before stopping, so dependencies cannot immediately restart it.
            for f in files:
                if f['key'] == 'irq_mask':
                    self.put_asset(journal, f)
            self.s.run('daemon-reload')
            if journal.data['irq_was_active']:
                journal.data['irq_stop_attempted'] = True
                journal.save()
                self.s.run('stop', IRQBALANCE)
                if self.s.unit(IRQBALANCE)['ActiveState'] == 'active':
                    raise ValueError('irqbalance did not stop')
            for f in files:
                if f['key'] not in ('irq_mask', 'enable'):
                    self.put_asset(journal, f)
            runtime = self.runtime(journal)
            for f in files:
                if f['key'] == 'enable':
                    self.put_asset(journal, f)
            self.s.run('daemon-reload')
            journal.data['phase'] = 'installed'
            journal.save()
            return dict(phase='configured', runtime=runtime, reboot_required=plan['boot']['changed'],
                        service=UNIT, state=str(self.p.state),
                        note='Runtime applied now; enabled oneshot runs on subsequent boots. No reboot performed.')
        except BaseException as exc:
            try:
                self.restore()
            except BaseException as cleanup:
                raise ValueError(f'configure failed ({exc}); rollback incomplete ({cleanup}); inspect {self.p.state}') from exc
            raise

    def restore(self):
        data = self.saved()
        if data is None or data['phase'] == 'restored':
            return dict(phase='already_restored', note='No owned changes to revert')
        self.s.idle()
        journal = self.journal()
        # Validate all owned assets before disabling/removing anything.
        if self.p.package.exists():
            directory(self.p.package)
            expected = {self.p.files[k].name for k in ('runner', 'environment', 'isolate')}
            if any(child.name not in expected for child in self.p.package.iterdir()):
                raise ValueError('unmanaged files in installed package; review before restore')
        for f in journal.data['files']:
            if f['status'] in ('pending', 'installed'):
                self.check_asset(f)
        journal.data['phase'] = 'restoring'
        journal.save()
        problems = []
        def attempt(label, fn):
            try:
                fn()
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                problems.append(f'{label}: {exc}')
        # No ExecStop changes affinities; stopping an in-progress runtime hook is
        # serialized by the same isolation lock used by the CLI and boot hook.
        attempt('disable profile', lambda: self.remove_asset(journal, 'enable'))
        if any(f['key']=='unit' and f['status'] in ('pending','installed') for f in journal.data['files']):
            attempt('stop profile', lambda: self.s.run('stop', UNIT))
        attempt('remove profile unit', lambda: self.remove_asset(journal, 'unit'))
        attempt('reload units', lambda: self.s.run('daemon-reload'))
        boot_id = self.s.host.boot_id()
        relative = journal.data['runtime'].get(boot_id)
        if relative and (self.p.state/relative).exists():
            attempt('restore current-boot runtime', lambda: self.s.restore_runtime(self.p.state/relative))
        # Older boot journals are evidence only; NEVER replay their identities.
        if journal.data['boot_owned'] and (self.p.state/'boot').exists():
            attempt('restore boot', lambda: self.env.restore_boot(self.p.state/'boot', self.p.boot))
        if not problems:
            attempt('unmask irqbalance', lambda: self.remove_asset(journal, 'irq_mask'))
            attempt('reload units', lambda: self.s.run('daemon-reload'))
            if not problems and journal.data['irq_stop_attempted'] and journal.data['irq_was_active']:
                attempt('restart prior irqbalance', lambda: self.s.run('start', IRQBALANCE))
        if not problems:
            for key in ('environment', 'isolate', 'runner'):
                attempt('remove '+key, lambda key=key: self.remove_asset(journal, key))
            for key in reversed(list(journal.data['directories'])):
                path = self.p.dirs[key]
                record = journal.data['directories'][key]
                if record['status'] not in ('removed', 'not_owned', 'preserved'):
                    def remove_dir(path=path, key=key, record=record):
                        if os.path.lexists(path):
                            directory(path)
                            st = path.lstat()
                            if record.get('identity') and record['identity'] != [st.st_dev, st.st_ino]:
                                raise ValueError(f'owned directory identity changed: {path}')
                            try:
                                path.rmdir()  # Never recursively delete unknown files.
                            except OSError as exc:
                                if key != 'package' and exc.errno == errno.ENOTEMPTY:
                                    record['status'] = 'preserved'  # Now shared by other software.
                                    journal.save()
                                    return
                                raise
                            sync_dir(path.parent)
                        record['status'] = 'removed'
                        journal.save()
                    attempt('remove directory '+key, remove_dir)
        journal.data['phase'] = 'restore_incomplete' if problems else 'restored'
        journal.data['restore_problems'] = problems
        journal.save()
        if problems:
            raise ValueError('; '.join(problems))
        return dict(phase='restored', reboot_required=journal.data['boot_owned'],
                    note='Backups retained. Preexisting/unowned settings were not reverted; no reboot performed.')


def interrupted(signum, frame):
    del signum, frame
    raise KeyboardInterrupt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--restore', action='store_true', help='revert this script\'s owned changes; retain backups')
    group.add_argument('--dry-run', action='store_true', help='inspect proposed changes without writing')
    group.add_argument('--status', action='store_true', help='show saved installation state; no changes')
    group.add_argument('--apply-runtime', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--yes', action='store_true', help='noninteractive acknowledgement of boot/service/global affinity changes')
    args = p.parse_args(argv)
    if os.geteuid() != 0:
        p.error('run with sudo (including --dry-run/--status for private state and process inspection)')
    mutating = not (args.dry_run or args.status)
    if mutating and not args.apply_runtime and not args.yes:
        if not sys.stdin.isatty():
            p.error('use --yes noninteractively; stop GPIO/PIO and retain recovery access first')
        print(('This restores owned boot/service/runtime changes.\n' if args.restore else
               'This changes Pi 5 boot CPU isolation, masks irqbalance.service, and installs boot-time setup.\n') +
              'Restore reverses only changes owned by this script. No GPIO, RT-budget/governor change or reboot.\n'
              'Stop GPIO/PIO workloads and keep recovery access.')
        if input('Type YES to continue: ') != 'YES':
            return 1
    # Firmware queries must not resolve through user-writable /usr/local paths.
    # This affects only this process/its children, not the user's shell or system.
    os.environ['PATH'] = '/usr/sbin:/usr/bin:/sbin:/bin'
    paths = Paths()
    data = load_state(paths)
    if args.status:
        result = ({'phase':'not_configured'} if data is None else dict(
            phase=data['phase'], state=str(paths.state), boot_changes_owned=data['boot_owned'],
            assets=[{k:v for k,v in f.items() if k != 'data'} for f in data['files']],
            runtime_journals=data['runtime'], last_runtime=data.get('last_runtime'),
            restore_problems=data.get('restore_problems', []),
            note='Saved installation state, not continuous live-affinity verification'))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    installed = Path(__file__).resolve() == paths.files['runner']
    if args.apply_runtime and not installed:
        raise ValueError('boot hook must execute the root-owned installed copy')
    env, iso = load_helpers(paths, installed, data)
    system = System(env, iso, paths)
    config = Configuration(paths, system, env, iso)
    if args.dry_run:
        if data and data['phase'] == 'installed':
            result = dict(phase='already_configured', state=str(paths.state), note='Use --status to inspect backups')
        else:
            plan = config.plan(source_payload(paths, installed))
            result = dict(boot_before=plan['boot']['cmdline_before'], boot_after=plan['boot']['cmdline_after'],
                          mask_irqbalance=not plan['irq_mask_preexisting'], irqbalance=plan['irqbalance'],
                          install_paths={k:str(v) for k,v in paths.files.items()}, health=plan['health'],
                          no_changes=True, note='No governor, RT-budget, scheduler, GPIO, kernel-selection or reboot changes')
    else:
        with system.host.lock():
            if args.restore:
                result = config.restore()
            elif args.apply_runtime:
                journal = config.journal()
                if journal.data['phase'] != 'installed':
                    raise ValueError('configuration is not installed; refusing boot-time apply')
                result = config.runtime(journal)
            else:
                result = config.configure(source_payload(paths, installed))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted; inspect /var/lib/rpihub75/configure-system and run --restore if needed. '
              'No SIGKILL/power-loss cleanup guarantee.', file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, TypeError, SyntaxError, subprocess.SubprocessError) as exc:
        print(f'System configuration refused/failed: {exc}. No forced takeover; inspect saved state.', file=sys.stderr)
        sys.exit(1)
