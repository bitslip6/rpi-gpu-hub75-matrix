#!/usr/bin/env python3
"""Read-only CPU isolation audit by default; explicit, journaled apply/restore.

Never drives GPIO, edits boot/cgroup policy, stops services, or changes scheduling,
RT budgets, thermal protection or kernel modules. Linux, Python standard library.
"""
import argparse
import contextlib
import datetime as dt
import fcntl
import gzip
import json
import os
from pathlib import Path
import platform
import re
import signal
import stat
import sys
import tempfile
import time
import zlib

SCHEMA = 1
CONFIG_KEYS = ('NO_HZ_FULL', 'NO_HZ_IDLE', 'RCU_NOCB_CPU', 'PREEMPT_RT',
               'IRQ_FORCED_THREADING', 'CPU_ISOLATION', 'HZ')
# Deliberately not every kernel thread. Never manipulate workqueue workers,
# migration, cpuhp or ksoftirqd. IRQ-thread fallback has separate eligibility checks.
REVIEWED_THREADS = re.compile(
    r'(?:rcu_preempt|rcu_sched|rcu_tasks.*|rcu_exp.*|watchdogd|khungtaskd|'
    r'kauditd|oom_reaper|kswapd\d+|kcompactd\d+|jbd2/.+|hwrng|kdevtmpfs|'
    r'brcmf_wdog/.+|cec-.+|card\d+-crtc\d+|spi\d+|krfcommd|pr/tty.+)\Z')
CONTROL_PATHS = (
    r'/proc/irq/default_smp_affinity', r'/proc/irq/[0-9]+/smp_affinity_list',
    r'/sys/devices/virtual/workqueue/cpumask', r'/proc/sys/kernel/watchdog_cpumask',
    r'/sys/class/net/(?!\.{1,2}/)[A-Za-z0-9_.:-]+/queues/(?:rx|tx)-[0-9]+/(?:rps|xps)_cpus',
)


def cpu_list(text, allow_empty=False):
    text = text.strip()
    if not text and allow_empty:
        return set()
    result = set()
    for part in text.split(','):
        if not re.fullmatch(r'\d+(?:-\d+)?', part):
            raise ValueError(f'invalid CPU list: {text!r}')
        ends = [int(x) for x in part.split('-')]
        if ends[-1] < ends[0] or ends[-1] > 4095:
            raise ValueError('invalid CPU range')
        result.update(range(ends[0], ends[-1] + 1))
    return result


def cpu_mask(text):
    text = text.strip()
    if not re.fullmatch(r'[0-9a-fA-F]{1,8}(?:,[0-9a-fA-F]{8})*', text):
        raise ValueError(f'invalid hexadecimal CPU mask: {text!r}')
    value = int(text.replace(',', ''), 16)
    if value.bit_length() > 4096:
        raise ValueError('CPU mask too large')
    return {i for i in range(value.bit_length()) if value & (1 << i)}


def format_list(cpus):
    # Expanded lists are accepted by the kernel and easy to compare semantically.
    return ','.join(str(i) for i in sorted(cpus))


def format_mask(cpus):
    value = sum(1 << i for i in cpus)
    groups = []
    while value > 0xffffffff:
        groups.append(f'{value & 0xffffffff:08x}')
        value >>= 32
    return ','.join([f'{value:x}', *reversed(groups)])


def valid_control(path):
    return any(re.fullmatch(pattern, path) for pattern in CONTROL_PATHS)


def flags_from_config(text):
    result = {}
    for key in CONFIG_KEYS:
        found = re.search(rf'^CONFIG_{key}=(\S+)$', text, re.M)
        result[key] = found.group(1) if found else (
            'n' if re.search(rf'^# CONFIG_{key} is not set$', text, re.M) else 'unknown')
    return result


class Host:
    """Filesystem root is injectable in tests only; CLI always uses the live host."""
    def __init__(self, root=Path('/')):
        self.root = Path(root)

    def path(self, path):
        return self.root / path.lstrip('/')

    def read(self, path):
        return self.path(path).read_text().strip()

    def optional(self, path):
        try:
            return {'text': self.read(path)}
        except OSError as exc:
            return {'error': str(exc)}

    def online(self):
        return cpu_list(self.read('/sys/devices/system/cpu/online'))

    def boot_id(self):
        return self.read('/proc/sys/kernel/random/boot_id')

    def config(self):
        errors = {}
        for path in ['/proc/config.gz', '/boot/config-' + platform.release()]:
            try:
                data = self.path(path).read_bytes()
                text = gzip.decompress(data).decode() if path.endswith('.gz') else data.decode()
                return {'source': path, 'flags': flags_from_config(text)}
            except (OSError, ValueError, EOFError, zlib.error) as exc:
                errors[path] = str(exc)
        return {'flags': flags_from_config(''), 'errors': errors}

    def irq_identity(self, irq):
        # Description and proc inode guard against a removed/reused IRQ number.
        identity = {'inode': self.path(f'/proc/irq/{irq}').stat().st_ino}
        for key in ['actions', 'chip_name', 'hwirq', 'type']:
            identity[key] = self.read(f'/sys/kernel/irq/{irq}/{key}')
        return identity

    def network_identity(self, path):
        iface = path.split('/')[4]
        return {'ifindex': int(self.read(f'/sys/class/net/{iface}/ifindex')),
                'queue_inode': self.path(path).parent.stat().st_ino}

    def tids(self):
        return sorted({int(p.name) for p in self.path('/proc').glob('[0-9]*/task/[0-9]*')})

    def affinity(self, tid):
        return set(os.sched_getaffinity(tid))

    def task(self, tid):
        fields = dict(line.split(':', 1) for line in self.read(f'/proc/{tid}/status').splitlines()
                      if ':' in line)
        tail = self.read(f'/proc/{tid}/stat').rsplit(') ', 1)[1].split()
        flags = int(tail[6])  # stat field 9; comm can contain spaces/parentheses.
        return {'tid': tid, 'tgid': int(fields['Tgid']), 'name': fields['Name'].strip(),
                'starttime': int(tail[19]), 'kernel': bool(flags & 0x00200000),
                'no_setaffinity': bool(flags & 0x04000000),
                'allowed': sorted(self.affinity(tid))}

    def set_affinity(self, tid, cpus):
        os.sched_setaffinity(tid, cpus)

    def writable(self, path):
        return bool(self.path(path).stat().st_mode & 0o222)

    def write(self, path, value):
        if not valid_control(path):
            raise ValueError(f'control path not allowlisted: {path}')
        # No following a substituted final-component symlink during privileged writes.
        fd = os.open(self.path(path), os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(fd, 'w') as stream:
            stream.write(value + '\n')

    @contextlib.contextmanager
    def lock(self):
        path = self.path('/run/lock/rio-isolate.lock')
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise ValueError('isolation lock must be a private regular file owned by this user')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        finally:
            os.close(fd)


def irq_thread_binding(host, name):
    """Movable thread work of a non-movable cascaded IRQ, not its primary IRQ."""
    match = re.fullmatch(r'irq/(\d+)-.+', name)
    if not match:
        return None
    irq = int(match.group(1))
    if host.writable(f'/proc/irq/{irq}/smp_affinity_list'):
        return None  # Let the ordinary IRQ affinity operation manage its thread.
    debug = host.read(f'/sys/kernel/debug/irq/irqs/{irq}')
    if 'handler:' not in debug or any(flag in debug for flag in
                        ['IRQD_PER_CPU', 'IRQD_AFFINITY_MANAGED', '_IRQ_NO_BALANCING']):
        return None
    identity = host.irq_identity(irq)
    if not identity['actions']:
        return None
    return {'thread_irq': irq, 'irq_identity': identity}


def task_identity(task):
    return {key: task[key] for key in ['tid', 'tgid', 'name', 'starttime', 'kernel', 'no_setaffinity']}


def audit(host, cpu=3, housekeeping=None):
    online = host.online()
    hk = online - {cpu} if housekeeping is None else set(housekeeping)
    if cpu not in online or not hk or cpu in hk or not hk <= online:
        raise ValueError('output CPU must be online; housekeeping must be nonempty, online and disjoint')
    report = {'schema': SCHEMA, 'kernel': platform.release(), 'boot_id': host.boot_id(),
              'cpu': cpu, 'online': sorted(online), 'housekeeping': sorted(hk),
              'config': host.config(), 'controls': [], 'irqs': [], 'threads': [],
              'operations': [], 'warnings': [], 'irqbalance_tids': [], 'read_only': True}
    for name, path in {
        'cmdline': '/proc/cmdline', 'isolated': '/sys/devices/system/cpu/isolated',
        'nohz_full': '/sys/devices/system/cpu/nohz_full',
        'rt_runtime_us': '/proc/sys/kernel/sched_rt_runtime_us',
        'rt_period_us': '/proc/sys/kernel/sched_rt_period_us',
        'workqueue_requested': '/sys/devices/virtual/workqueue/cpumask_requested',
        'workqueue_isolated': '/sys/devices/virtual/workqueue/cpumask_isolated',
        'cgroup_controllers': '/sys/fs/cgroup/cgroup.controllers',
        'interrupts': '/proc/interrupts', 'softirqs': '/proc/softirqs',
    }.items():
        report[name] = host.optional(path)
    try:
        listed = cpu in cpu_list(report['nohz_full']['text'], allow_empty=True)
    except (KeyError, ValueError):
        listed = None
    report['full_dynticks_cpu_listed'] = listed
    if report['config']['flags']['NO_HZ_FULL'] != 'y':
        report['warnings'].append('Full dynticks unavailable/unverified: cannot move the busy-core local timer tick with IRQ affinity.')
    elif listed is not True:
        report['warnings'].append('NO_HZ_FULL compiled in, but output CPU is not listed or runtime mask is unavailable; tick suppression is not established.')
    else:
        report['warnings'].append('Output CPU is listed for full dynticks; measure actual timer deltas under load, not just mask membership.')
    if report['config']['flags']['RCU_NOCB_CPU'] != 'y':
        report['warnings'].append('RCU callback offload unavailable/unverified; moving GP threads is not callback offload.')
    else:
        report['warnings'].append('RCU callback offload is compiled in; this audit does not establish runtime callback offload.')
    report['warnings'].append('FIFO is NOT applied: a free-running emitter may hit the RT budget. Hard IRQs and MMIO stalls remain possible.')

    def control(path, encoding, identity=None, effective=None, source=None):
        record = {'path': path, 'encoding': encoding}
        try:
            old = (cpu_mask if encoding == 'mask' else cpu_list)(host.read(source or path))
            record['current'] = sorted(old)
            new = old & hk
            if effective:
                record['effective'] = host.optional(effective)
            if not host.writable(path):
                record['skip'] = 'read-only/per-CPU/managed control'
            elif not old:
                record['skip'] = 'disabled mask; leave disabled'
            elif not new:
                record['skip'] = 'no existing housekeeping overlap; do not broaden affinity'
            elif new == old:
                record['skip'] = 'already confined to housekeeping'
            else:
                op = {'kind': 'control', 'path': path, 'encoding': encoding,
                      'old': sorted(old), 'new': sorted(new)}
                if identity is not None:
                    op['irq'] = int(path.split('/')[3])
                    op['identity'] = identity
                if path.startswith('/sys/class/net/'):
                    op['net_identity'] = host.network_identity(path)
                if effective:
                    op['effective_path'] = effective
                if source:
                    op['read_path'] = source
                report['operations'].append(op)
        except (OSError, ValueError) as exc:
            record['skip'] = str(exc)
        report['controls'].append(record)

    control('/proc/irq/default_smp_affinity', 'mask')
    for directory in sorted(host.path('/proc/irq').glob('[0-9]*'), key=lambda p: int(p.name)):
        irq = int(directory.name)
        path = f'/proc/irq/{irq}/smp_affinity_list'
        record = {'irq': irq, 'debug': host.optional(f'/sys/kernel/debug/irq/irqs/{irq}')}
        try:
            identity = host.irq_identity(irq)
            record['identity'] = identity
            debug = record['debug'].get('text', '')
            if any(flag in debug for flag in ['IRQD_PER_CPU', 'IRQD_AFFINITY_MANAGED', '_IRQ_NO_BALANCING']):
                record['skip'] = 'per-CPU/managed/non-balancing IRQ'
            elif not identity['actions']:
                record['skip'] = 'no registered handler; future allocations use the default IRQ mask'
            else:
                control(path, 'list', identity, f'/proc/irq/{irq}/effective_affinity_list')
        except (OSError, ValueError) as exc:
            record['skip'] = str(exc)
        report['irqs'].append(record)
    requested = '/sys/devices/virtual/workqueue/cpumask_requested'
    control('/sys/devices/virtual/workqueue/cpumask', 'mask',
            source=requested if host.path(requested).exists() else None)
    control('/proc/sys/kernel/watchdog_cpumask', 'list')
    for pattern in ['*/queues/rx-*/rps_cpus', '*/queues/tx-*/xps_cpus']:
        for path in sorted(host.path('/sys/class/net').glob(pattern)):
            relative = '/' + str(path.relative_to(host.root))
            if valid_control(relative):
                control(relative, 'mask')
    for tid in host.tids():
        try:
            task = host.task(tid)
            allowed = set(task['allowed'])
            if not task['kernel'] and task['name'] == 'irqbalance':
                report['irqbalance_tids'].append(tid)
            if cpu not in allowed:
                continue
            record = dict(task)
            new = allowed & hk
            if not task['kernel']:
                record['skip'] = 'userspace is never automatically moved'
            elif task['no_setaffinity'] or not new:
                record['skip'] = 'kernel-required/per-CPU/no-setaffinity task'
            else:
                binding = irq_thread_binding(host, task['name'])
                if not REVIEWED_THREADS.fullmatch(task['name']) and binding is None:
                    record['skip'] = 'not a reviewed movable thread; writable IRQ threads follow IRQ affinity'
                else:
                    report['operations'].append(dict(kind='task', tid=tid,
                                                      identity=task_identity(task),
                                                      old=sorted(allowed), new=sorted(new),
                                                      **(binding or {})))
            report['threads'].append(record)
        except (OSError, ValueError, KeyError, IndexError) as exc:
            report['threads'].append({'tid': tid, 'skip': str(exc)})
    if report['irqbalance_tids']:
        report['warnings'].append('irqbalance is running and may undo IRQ placement; configure its exclusion separately (not stopped automatically).')
    return report


def read_current(host, entry):
    if entry['kind'] == 'task':
        return host.affinity(entry['tid'])
    return (cpu_mask if entry['encoding'] == 'mask' else cpu_list)(
        host.read(entry.get('read_path', entry['path'])))


def check_identity(host, entry):
    if entry['kind'] == 'task':
        if task_identity(host.task(entry['tid'])) != entry['identity']:
            raise ValueError('task identity changed; possible TID reuse')
        if 'thread_irq' in entry:
            binding = irq_thread_binding(host, entry['identity']['name'])
            if binding != {key: entry[key] for key in ['thread_irq', 'irq_identity']}:
                raise ValueError('IRQ-thread binding/eligibility changed')
    elif 'irq' in entry:
        if host.irq_identity(entry['irq']) != entry['identity']:
            raise ValueError('IRQ identity changed; possible IRQ reuse')
    elif 'net_identity' in entry:
        if host.network_identity(entry['path']) != entry['net_identity']:
            raise ValueError('network interface/queue identity changed')


def set_value(host, entry, value):
    if entry['kind'] == 'task':
        host.set_affinity(entry['tid'], value)
    else:
        text = (format_mask if entry['encoding'] == 'mask' else format_list)(value)
        host.write(entry['path'], text)
    if read_current(host, entry) != value:
        raise ValueError('requested setting readback mismatch')


def validate_entry(entry):
    if entry['kind'] == 'task':
        if not isinstance(entry['tid'], int) or entry['tid'] <= 0 or \
                not entry['identity']['kernel'] or entry['identity']['no_setaffinity']:
            raise ValueError('invalid task journal entry')
        name = entry['identity']['name']
        if 'thread_irq' in entry:
            if type(entry['thread_irq']) is not int or entry['thread_irq'] <= 0 or \
                    not re.fullmatch(rf"irq/{entry['thread_irq']}-.*", name) or 'irq_identity' not in entry:
                raise ValueError('invalid IRQ-thread binding')
        elif not REVIEWED_THREADS.fullmatch(name):
            raise ValueError('unreviewed task journal entry')
    elif entry['kind'] == 'control':
        path = entry['path']
        if not valid_control(path) or entry['encoding'] not in ('list', 'mask'):
            raise ValueError('invalid control journal entry')
        if entry.get('read_path', path) != path and not (
                path == '/sys/devices/virtual/workqueue/cpumask' and
                entry['read_path'] == '/sys/devices/virtual/workqueue/cpumask_requested'):
            raise ValueError('invalid readback path')
        is_irq = bool(re.fullmatch(r'/proc/irq/[0-9]+/smp_affinity_list', path))
        if is_irq != ('irq' in entry):
            raise ValueError('IRQ operations require an identity')
        if 'irq' in entry and (path != f"/proc/irq/{entry['irq']}/smp_affinity_list" or
                               entry.get('effective_path') != f"/proc/irq/{entry['irq']}/effective_affinity_list"):
            raise ValueError('invalid IRQ journal entry')
        if 'effective_path' in entry and 'irq' not in entry:
            raise ValueError('unexpected effective affinity path')
        if path.startswith('/sys/class/net/') != ('net_identity' in entry):
            raise ValueError('network operations require interface/queue identity')
    else:
        raise ValueError('invalid journal operation kind')
    for key in ('old', 'new'):
        if not isinstance(entry[key], list) or not entry[key] or any(
                type(i) is not int or not 0 <= i <= 4095 for i in entry[key]):
            raise ValueError('invalid journal CPU set')
    if not set(entry['new']) < set(entry['old']):
        raise ValueError('journal operation must only narrow affinity')


class Journal:
    def __init__(self, directory, data=None):
        self.directory = Path(directory)
        if data is not None:
            self.directory.mkdir(mode=0o700)  # Exclusive, never overwrite a state directory.
            self.data = data
            self.save()
            # The new directory link must also be durable before any system write.
            parent_fd = os.open(self.directory.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        else:
            info = self.directory.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise ValueError('state directory must be private, owned by this user, and not a symlink')
            fd = os.open(self.directory / 'journal.json',
                         os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
            with os.fdopen(fd) as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
                    raise ValueError('journal must be a private regular file owned by this user')
                self.data = json.load(stream)

    def save(self):
        fd, name = tempfile.mkstemp(prefix='.journal-', dir=self.directory)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(self.data, stream, indent=2, sort_keys=True)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.directory / 'journal.json')
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(name):
                os.unlink(name)


def check_journal(host, journal):
    data = journal.data
    if data['schema'] != SCHEMA or data['boot_id'] != host.boot_id() or \
            set(data['online']) != host.online():
        raise ValueError('schema, boot identity or online CPUs changed; refusing stale restoration')
    hk = set(data['housekeeping'])
    if not hk or data['cpu'] not in host.online() or data['cpu'] in hk or not hk <= host.online():
        raise ValueError('invalid housekeeping policy in journal')
    for entry in data['entries']:
        validate_entry(entry)
        if entry['status'] not in {'pending', 'applied', 'refused', 'rolled_back',
                                   'restored', 'already_original', 'restore_failed'}:
            raise ValueError('invalid journal operation status')
        if not set(entry['new']) <= hk:
            raise ValueError('journal target outside housekeeping')


def restore(host, journal):
    check_journal(host, journal)  # Validate the WHOLE journal before any write.
    problems = []
    for entry in reversed(journal.data['entries']):
        if entry['status'] not in ('pending', 'applied', 'restore_failed'):
            continue
        try:
            check_identity(host, entry)
            current = read_current(host, entry)
            if current == set(entry['old']):
                entry['status'] = 'already_original'
            elif current != set(entry['new']):
                raise ValueError('setting changed by another actor; refusing to overwrite it')
            else:
                set_value(host, entry, set(entry['old']))
                entry['status'] = 'restored'
        except (OSError, ValueError) as exc:
            entry['status'] = 'restore_failed'
            entry['restore_error'] = str(exc)
            problems.append(str(exc))
        journal.save()
    try:
        journal.data['after_restore'] = audit(host, journal.data['cpu'], journal.data['housekeeping'])
    except (OSError, ValueError) as exc:
        journal.data['after_restore'] = {'error': str(exc)}
        problems.append(f'restore audit failed: {exc}')
    journal.data['phase'] = 'restore_incomplete' if problems else 'restored'
    journal.save()
    return problems


def apply(host, report, state):
    if report['boot_id'] != host.boot_id() or set(report['online']) != host.online():
        raise ValueError('host changed since audit')
    data = {key: report[key] for key in ('schema', 'boot_id', 'online', 'cpu', 'housekeeping')}
    data.update(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                phase='applying', before=report, entries=[])
    journal = Journal(state, data)
    problems = []
    try:
        for operation in report['operations']:
            entry = dict(operation, status='pending')
            validate_entry(entry)
            try:
                check_identity(host, entry)
                if read_current(host, entry) != set(entry['old']):
                    raise ValueError('concurrent setting change')
            except (OSError, ValueError) as exc:
                problems.append(f'skipped {operation}: {exc}')
                continue
            # Write-ahead: durably record old/new state BEFORE the system call.
            journal.data['entries'].append(entry)
            journal.save()
            try:
                set_value(host, entry, set(entry['new']))
                if 'effective_path' in entry:
                    for attempt in range(6):
                        effective = cpu_list(host.read(entry['effective_path']), allow_empty=True)
                        if effective and effective <= set(entry['new']):
                            break
                        if attempt < 5:
                            time.sleep(.02)
                    else:
                        raise ValueError('effective IRQ affinity does not match restriction')
                    entry['effective_after'] = sorted(effective)
                entry['status'] = 'applied'
            except (OSError, ValueError) as exc:
                problems.append(str(exc))
                entry['error'] = str(exc)
                # A refused/ineffective setting is rolled back locally. Others can proceed.
                check_identity(host, entry)
                current = read_current(host, entry)
                if current == set(entry['new']):
                    set_value(host, entry, set(entry['old']))
                    entry['status'] = 'rolled_back'
                elif current == set(entry['old']):
                    entry['status'] = 'refused'
                else:
                    raise ValueError('unexpected partial change; manual recovery required') from exc
            journal.save()
        journal.data['phase'] = 'partial' if problems else 'applied'
        journal.data['problems'] = problems
        journal.data['after'] = audit(host, report['cpu'], report['housekeeping'])
        journal.save()
    except BaseException:
        # Ordinary errors/signals attempt rollback. SIGKILL/power failure require
        # explicit restore from the durable pending records; no such guarantee.
        restore(host, journal)
        raise
    return journal, problems


def summary(report):
    print(f"CPU{report['cpu']} output; housekeeping {format_list(report['housekeeping'])}; kernel {report['kernel']}")
    print('Kernel features: ' + json.dumps(report['config']['flags'], sort_keys=True))
    controls = sum(op['kind'] == 'control' for op in report['operations'])
    print(f'Planned restrictions: {controls} controls, {len(report["operations"])-controls} reviewed kernel threads')
    for warning in report['warnings']:
        print('WARNING: ' + warning)
    for op in report['operations']:
        target = op.get('path', f"TID {op.get('tid')} {op.get('identity', {}).get('name', '')}")
        print(f"  {target}: {format_list(op['old'])} -> {format_list(op['new'])}")
    print('No GPIO, boot edits, service stops, cpuset changes, scheduling or RT-budget changes.')


def interrupted(signum, frame):
    del signum, frame
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=['audit', 'apply', 'restore'], default='audit')
    parser.add_argument('--cpu', type=int, default=3)
    parser.add_argument('--housekeeping', type=cpu_list, help='default: other online CPUs')
    parser.add_argument('--state', type=Path, help='new private directory for apply; same directory for restore')
    parser.add_argument('--yes', action='store_true', help='acknowledge global affinity changes and recovery readiness')
    parser.add_argument('--json', action='store_true', help='print full audit/journal JSON')
    parser.add_argument('--report', type=Path, help='save JSON to a NEW file, never overwrite')
    args = parser.parse_args(argv)
    if args.action != 'audit' and (not args.yes or args.state is None):
        parser.error('apply/restore require --state and --yes; stop GPIO tests and retain recovery access first')
    if args.action != 'audit' and os.geteuid() != 0:
        parser.error('apply/restore require root; audit does not')
    # Reserve an exclusive report destination before any global setting changes.
    with (args.report.open('x') if args.report else contextlib.nullcontext()) as stream:
        return execute(Host(), args, stream)


def execute(host, args, stream=None):
    if args.action == 'audit':
        result = audit(host, args.cpu, args.housekeeping)
        rc = 0
        if not args.json:
            summary(result)
    else:
        with host.lock():
            if args.action == 'apply':
                report = audit(host, args.cpu, args.housekeeping)
                active = [t for t in report['threads'] if not t.get('kernel', True) and
                          re.match(r'(?:rio_scope|pio_|example$)', t.get('name', ''))]
                if active:
                    raise ValueError('a known GPIO benchmark/application is present; stop it before applying')
                journal, problems = apply(host, report, args.state)
            else:
                journal = Journal(args.state)
                problems = restore(host, journal)
            result = journal.data
            rc = 2 if problems else 0
        if not args.json:
            print(f"{result['phase']}: state saved in {args.state}/journal.json")
            for entry in result['entries']:
                print(entry['status'], entry.get('path', f"TID {entry.get('tid')}"),
                      entry.get('restore_error', entry.get('error', '')))
            print('Partial/refused/conflicting controls remain explicit; inspect the journal and audit again.')
            snapshot = result.get('after' if args.action == 'apply' else 'after_restore', {})
            for warning in snapshot.get('warnings', []):
                print('WARNING: ' + warning)
    if stream is not None:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    return rc


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted; check the journal and restore before another experiment.', file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'Isolation operation failed: {exc}. No forced takeover; inspect saved state.', file=sys.stderr)
        sys.exit(1)
