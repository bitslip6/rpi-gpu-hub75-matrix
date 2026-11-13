"""All mutations use a temporary filesystem and fake task affinities, never the host."""
import contextlib
import copy
import errno
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('rio_isolate', ROOT / 'tests/rio_isolate.py')
iso = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iso)


class FakeHost(iso.Host):
    def __init__(self, root):
        super().__init__(root)
        self.tasks = {}
        self.writes = []
        self.fail = set()
        self.ineffective = set()
        self.hook = None

    def put(self, path, value, mode=0o644):
        p = self.path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value + '\n')
        p.chmod(mode)

    def tids(self):
        return sorted(self.tasks)

    def affinity(self, tid):
        if tid not in self.tasks:
            raise ProcessLookupError(tid)
        return set(self.tasks[tid]['allowed'])

    def task(self, tid):
        if tid not in self.tasks:
            raise ProcessLookupError(tid)
        return copy.deepcopy(self.tasks[tid])

    def set_affinity(self, tid, value):
        if self.hook:
            self.hook({'kind': 'task', 'tid': tid})
        self.writes.append(('task', tid, set(value)))
        self.tasks[tid]['allowed'] = sorted(value)

    def write(self, path, value):
        assert iso.valid_control(path), path
        if self.hook:
            self.hook({'kind': 'control', 'path': path})
        self.writes.append(('control', path, value))
        if path in self.fail:
            raise OSError(errno.EPERM, 'simulated driver refusal')
        self.path(path).write_text(value + '\n')
        if path.endswith('/smp_affinity_list'):
            if path not in self.ineffective:
                effective = path.replace('/smp_affinity_list', '/effective_affinity_list')
                self.path(effective).write_text(value + '\n')
        if path == '/sys/devices/virtual/workqueue/cpumask':
            self.put(path + '_requested', value)


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.host = FakeHost(self.root/'host')
        h = self.host
        h.put('/sys/devices/system/cpu/online', '0-3')
        h.put('/proc/sys/kernel/random/boot_id', 'test-boot-1')
        h.put('/proc/cmdline', 'isolcpus=3 nohz_full=3 rcu_nocbs=3')
        h.put('/sys/devices/system/cpu/isolated', '3')
        h.put('/proc/irq/default_smp_affinity', 'f')
        h.put('/sys/devices/virtual/workqueue/cpumask', '7')
        h.put('/sys/devices/virtual/workqueue/cpumask_requested', '7')
        h.put('/proc/sys/kernel/watchdog_cpumask', '0-3')
        h.put('/proc/sys/kernel/sched_rt_runtime_us', '950000')
        h.put('/proc/sys/kernel/sched_rt_period_us', '1000000')
        h.put('/sys/class/net/eth0/ifindex', '5')
        h.put('/sys/class/net/eth0/queues/rx-0/rps_cpus', '0')
        h.put('/sys/class/net/eth0/queues/tx-0/xps_cpus', 'f')
        config = h.path('/proc/config.gz')
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_bytes(gzip.compress(b'CONFIG_PREEMPT_RT=y\n# CONFIG_NO_HZ_FULL is not set\nCONFIG_HZ=250\n'))
        self.irq(13, 'IRQD_PER_CPU', writable=False)
        self.irq(40, 'IRQD_AFFINITY_MANAGED')
        self.irq(141)
        self.irq(142, active=False)
        self.task(11, 'rcu_preempt')
        self.task(12, 'ksoftirqd/3', allowed=[3])
        self.task(14, 'rcu_tasks_kthread', no_setaffinity=True)
        self.task(15, 'ordinary-user', kernel=False)
        self.task(16, 'kworker/R-test')
        self.task(17, 'irq/141-test')
        self.state = self.root/'state'

    def irq(self, number, debug='', writable=True, active=True):
        h = self.host
        h.put(f'/proc/irq/{number}/smp_affinity_list', '0-3', 0o644 if writable else 0o444)
        h.put(f'/proc/irq/{number}/effective_affinity_list', '0-3')
        h.put(f'/sys/kernel/debug/irq/irqs/{number}', debug)
        for key, value in {'actions': 'test' if active else '', 'chip_name': 'test-chip',
                           'hwirq': str(number), 'type': 'level'}.items():
            h.put(f'/sys/kernel/irq/{number}/{key}', value)

    def task(self, tid, name, allowed=None, kernel=True, no_setaffinity=False):
        self.host.tasks[tid] = {'tid': tid, 'tgid': tid, 'name': name, 'starttime': tid*100,
                                'kernel': kernel, 'no_setaffinity': no_setaffinity,
                                'allowed': [0, 1, 2, 3] if allowed is None else allowed}

    def test_masks_and_lists(self):
        for cpus in [{0, 1, 2}, {0, 31, 32, 64, 99}, set()]:
            self.assertEqual(iso.cpu_mask(iso.format_mask(cpus)), cpus)
            self.assertEqual(iso.cpu_list(iso.format_list(cpus), allow_empty=True), cpus)
        for bad in ['-1', '3-1', '0,,2', '4096', '0-999999999']:
            with self.assertRaises(ValueError):
                iso.cpu_list(bad)
        for bad in ['0x7', 'fffffffff', '7,1', '../x', '']:
            with self.assertRaises(ValueError):
                iso.cpu_mask(bad)

    def test_audit_is_read_only_and_does_not_infer_nohz_from_rt(self):
        report = iso.audit(self.host)
        self.assertEqual(self.host.writes, [])
        self.assertEqual(report['config']['flags']['PREEMPT_RT'], 'y')
        self.assertEqual(report['config']['flags']['NO_HZ_FULL'], 'n')
        self.assertTrue(any('local timer' in text for text in report['warnings']))
        paths = {op.get('path') for op in report['operations']}
        self.assertIn('/proc/irq/default_smp_affinity', paths)
        self.assertIn('/proc/irq/141/smp_affinity_list', paths)
        self.assertNotIn('/proc/irq/13/smp_affinity_list', paths)
        self.assertNotIn('/proc/irq/40/smp_affinity_list', paths)
        self.assertNotIn('/proc/irq/142/smp_affinity_list', paths)
        self.assertNotIn('/sys/devices/virtual/workqueue/cpumask', paths)
        self.assertNotIn('/sys/class/net/eth0/queues/rx-0/rps_cpus', paths)
        self.assertEqual([op['tid'] for op in report['operations'] if op['kind']=='task'], [11])

    def test_nohz_requires_runtime_membership_not_only_config(self):
        self.host.path('/proc/config.gz').write_bytes(gzip.compress(b'CONFIG_NO_HZ_FULL=y\n'))
        for mask, listed in [('0-2', False), ('3', True), ('(null)', None)]:
            with self.subTest(mask=mask):
                self.host.put('/sys/devices/system/cpu/nohz_full', mask)
                report = iso.audit(self.host)
                self.assertIs(report['full_dynticks_cpu_listed'], listed)
                if listed is not True:
                    self.assertTrue(any('tick suppression is not established' in w
                                        for w in report['warnings']))
        self.assertEqual(self.host.writes, [])

    def test_invalid_housekeeping(self):
        for cpu, hk in [(3, set()), (3, {3}), (3, {4}), (4, {0})]:
            with self.assertRaises(ValueError):
                iso.audit(self.host, cpu, hk)
        self.assertEqual(self.host.writes, [])

    def test_apply_write_ahead_and_restore_round_trip(self):
        report = iso.audit(self.host)
        def hook(op):
            data = json.loads((self.state/'journal.json').read_text())
            key = 'path' if op['kind']=='control' else 'tid'
            self.assertTrue(any(e.get(key)==op[key] and e['status'] in ['pending','applied']
                                for e in data['entries']))
        self.host.hook = hook
        journal, problems = iso.apply(self.host, report, self.state)
        self.assertEqual(problems, [])
        self.assertEqual(journal.data['phase'], 'applied')
        self.assertEqual(self.host.read('/proc/irq/default_smp_affinity'), '7')
        self.assertEqual(self.host.affinity(11), {0, 1, 2})
        self.assertEqual(self.host.affinity(15), {0, 1, 2, 3})
        self.assertEqual(self.host.read('/sys/class/net/eth0/queues/rx-0/rps_cpus'), '0')
        self.assertEqual(self.state.stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.state/'journal.json').stat().st_mode & 0o777, 0o600)
        self.assertEqual(iso.restore(self.host, iso.Journal(self.state)), [])
        self.assertEqual(self.host.read('/proc/irq/default_smp_affinity'), 'f')
        self.assertEqual(self.host.affinity(11), {0, 1, 2, 3})
        count = len(self.host.writes)
        iso.restore(self.host, iso.Journal(self.state))
        self.assertEqual(len(self.host.writes), count)

    def test_readonly_cascaded_irq_thread_fallback(self):
        self.irq(173, 'handler: handle_level_irq', writable=False)
        self.task(18, 'irq/173-hdmi')
        report = iso.audit(self.host)
        op = next(e for e in report['operations'] if e.get('tid') == 18)
        self.assertEqual(op['thread_irq'], 173)
        self.assertFalse(any(e.get('path') == '/proc/irq/173/smp_affinity_list'
                             for e in report['operations']))
        journal, problems = iso.apply(self.host, report, self.state)
        self.assertEqual(problems, [])
        self.assertEqual(self.host.affinity(18), {0, 1, 2})
        self.assertEqual(iso.restore(self.host, journal), [])
        self.assertEqual(self.host.affinity(18), {0, 1, 2, 3})

    def test_irq_thread_fallback_excludes_managed_and_unknown(self):
        self.irq(173, 'handler: handle_level_irq\nIRQD_AFFINITY_MANAGED', writable=False)
        self.irq(174, '', writable=False)
        self.task(18, 'irq/173-hdmi')
        self.task(19, 'irq/174-unknown')
        report = iso.audit(self.host)
        self.assertFalse(any(e.get('tid') in [18, 19] for e in report['operations']))

    def test_irq_thread_fallback_restoration_checks_binding(self):
        self.irq(173, 'handler: handle_level_irq', writable=False)
        self.task(18, 'irq/173-hdmi')
        journal, _ = iso.apply(self.host, iso.audit(self.host), self.state)
        self.host.put('/sys/kernel/irq/173/actions', 'replacement')
        count = len(self.host.writes)
        self.assertTrue(iso.restore(self.host, journal))
        self.assertFalse(any(e[1] == 18 for e in self.host.writes[count:]))

    def test_global_lock_excludes_other_instances(self):
        self.host.path('/run/lock').mkdir(parents=True)
        with self.host.lock():
            with self.assertRaises(BlockingIOError):
                with self.host.lock():
                    self.fail('second lock unexpectedly acquired')

    def test_workqueue_requested_value_and_network_masks(self):
        self.host.put('/sys/devices/virtual/workqueue/cpumask_requested', 'f')
        report = iso.audit(self.host)
        op = next(op for op in report['operations'] if op.get('path')=='/sys/devices/virtual/workqueue/cpumask')
        self.assertEqual(op['old'], [0,1,2,3])
        journal, _ = iso.apply(self.host, report, self.state)
        self.assertEqual(self.host.read('/sys/devices/virtual/workqueue/cpumask_requested'), '7')
        iso.restore(self.host, journal)
        self.assertEqual(self.host.read('/sys/devices/virtual/workqueue/cpumask_requested'), 'f')

    def test_driver_refusal_is_partial_not_success(self):
        self.host.fail.add('/proc/irq/141/smp_affinity_list')
        journal, problems = iso.apply(self.host, iso.audit(self.host), self.state)
        self.assertTrue(problems)
        self.assertEqual(journal.data['phase'], 'partial')
        self.assertTrue(any(e['status']=='refused' for e in journal.data['entries']))
        self.assertEqual(self.host.read('/proc/irq/default_smp_affinity'), '7')
        self.assertEqual(iso.restore(self.host, journal), [])

    def test_ineffective_irq_is_rolled_back(self):
        path = '/proc/irq/141/smp_affinity_list'
        self.host.ineffective.add(path)
        with mock.patch.object(iso.time, 'sleep'):
            journal, problems = iso.apply(self.host, iso.audit(self.host), self.state)
        self.assertTrue(problems)
        self.assertEqual(self.host.read(path), '0,1,2,3')
        self.assertEqual(iso.cpu_list(self.host.read(path)), {0,1,2,3})
        entry = next(e for e in journal.data['entries'] if e.get('path')==path)
        self.assertEqual(entry['status'], 'rolled_back')
        self.assertEqual(iso.restore(self.host, journal), [])

    def test_network_identity_change_blocks_stale_restore(self):
        journal, _ = iso.apply(self.host, iso.audit(self.host), self.state)
        self.host.put('/sys/class/net/eth0/ifindex', '8')
        count = len(self.host.writes)
        self.assertTrue(iso.restore(self.host, journal))
        self.assertFalse(any(str(e[1]).endswith('xps_cpus') for e in self.host.writes[count:]))

    def test_irqbalance_warning_even_when_it_is_off_core(self):
        self.task(20, 'irqbalance', allowed=[0, 1, 2], kernel=False)
        report = iso.audit(self.host)
        self.assertEqual(report['irqbalance_tids'], [20])
        self.assertTrue(any('irqbalance is running' in text for text in report['warnings']))
        self.assertEqual(self.host.writes, [])

    def test_no_broadening_of_core_only_affinity(self):
        self.host.put('/proc/irq/141/smp_affinity_list', '3')
        self.host.put('/sys/class/net/eth0/queues/tx-0/xps_cpus', '8')
        report = iso.audit(self.host)
        self.assertFalse(any(op.get('path')=='/proc/irq/141/smp_affinity_list' for op in report['operations']))
        self.assertFalse(any(op.get('path','').endswith('xps_cpus') for op in report['operations']))

    def test_boot_and_online_changes_reject_restore_before_writes(self):
        for path, value in [('/proc/sys/kernel/random/boot_id','other-boot'),
                            ('/sys/devices/system/cpu/online','0-2')]:
            with self.subTest(path=path):
                state = self.root/('s'+str(len(self.host.writes)))
                journal,_=iso.apply(self.host,iso.audit(self.host),state)
                original = self.host.read(path)
                self.host.put(path,value)
                count=len(self.host.writes)
                with self.assertRaises(ValueError):iso.restore(self.host,journal)
                self.assertEqual(len(self.host.writes),count)
                self.host.put(path,original)
                iso.restore(self.host,journal)

    def test_conflicting_setting_not_overwritten(self):
        journal,_=iso.apply(self.host,iso.audit(self.host),self.state)
        self.host.put('/proc/irq/default_smp_affinity','3')  # Another actor chose CPUs0-1.
        self.assertTrue(iso.restore(self.host,journal))
        self.assertEqual(self.host.read('/proc/irq/default_smp_affinity'),'3')
        self.assertEqual(journal.data['phase'],'restore_incomplete')

    def test_tid_reuse_and_irq_identity_reuse(self):
        journal,_=iso.apply(self.host,iso.audit(self.host),self.state)
        self.host.tasks[11]['starttime'] += 1
        self.host.put('/sys/kernel/irq/141/actions','different-device')
        count=len(self.host.writes)
        self.assertTrue(iso.restore(self.host,journal))
        new=self.host.writes[count:]
        self.assertFalse(any(e[1] in [11,'/proc/irq/141/smp_affinity_list'] for e in new))

    def test_concurrent_change_and_task_exit_before_apply(self):
        report=iso.audit(self.host)
        self.host.put('/proc/irq/default_smp_affinity','3')
        del self.host.tasks[11]
        journal,problems=iso.apply(self.host,report,self.state)
        self.assertTrue(problems)
        self.assertEqual(journal.data['phase'],'partial')
        self.assertEqual(self.host.read('/proc/irq/default_smp_affinity'),'3')
        self.assertFalse(any(e[0]=='task' for e in self.host.writes))

    def test_interrupt_attempts_rollback(self):
        def hook(op):
            if op.get('tid')==11:
                self.host.hook=None
                raise KeyboardInterrupt
        self.host.hook=hook
        with self.assertRaises(KeyboardInterrupt):iso.apply(self.host,iso.audit(self.host),self.state)
        self.assertEqual(self.host.read('/proc/irq/default_smp_affinity'),'f')
        self.assertEqual(iso.Journal(self.state).data['phase'],'restored')

    def test_pending_record_recovers_after_simulated_crash(self):
        report=iso.audit(self.host)
        op=report['operations'][0]
        data={key:report[key] for key in ['schema','boot_id','online','cpu','housekeeping']}
        data.update(phase='applying',entries=[dict(op,status='pending')])
        iso.Journal(self.state,data)
        iso.set_value(self.host,op,set(op['new']))  # Crash after write, before marking applied.
        self.assertEqual(iso.restore(self.host,iso.Journal(self.state)),[])
        self.assertEqual(self.host.read(op['path']),'f')

    def test_state_security_and_no_overwrite(self):
        journal,_=iso.apply(self.host,iso.audit(self.host),self.state)
        count=len(self.host.writes)
        with self.assertRaises(FileExistsError):iso.apply(self.host,iso.audit(self.host),self.state)
        self.assertEqual(len(self.host.writes),count)
        self.state.chmod(0o777)
        with self.assertRaises(ValueError):iso.Journal(self.state)
        self.state.chmod(0o700)
        saved=json.loads((self.state/'journal.json').read_text())
        saved['entries'].append({'kind':'control','path':'/etc/passwd','encoding':'list',
                                  'old':[0,1,2,3],'new':[0,1,2],'status':'applied'})
        journal.data=saved
        with self.assertRaises(ValueError):iso.restore(self.host,journal)
        self.assertEqual(len(self.host.writes),count)

    def test_symlink_and_corrupt_status_are_rejected(self):
        journal, _ = iso.apply(self.host, iso.audit(self.host), self.state)
        alias = self.root/'alias'
        alias.symlink_to(self.state, target_is_directory=True)
        with self.assertRaises(ValueError):
            iso.Journal(alias)
        count = len(self.host.writes)
        journal.data['entries'][-1]['status'] = 'invented'
        with self.assertRaises(ValueError):
            iso.restore(self.host, journal)
        self.assertEqual(len(self.host.writes), count)

    def test_active_benchmark_refused_before_state_creation(self):
        self.task(20, 'rio_scope.execu', kernel=False)
        with mock.patch.object(iso, 'Host', return_value=self.host), \
                mock.patch.object(iso.os, 'geteuid', return_value=0), \
                mock.patch.object(self.host, 'lock', return_value=contextlib.nullcontext()):
            with self.assertRaisesRegex(ValueError, 'stop it'):
                iso.main(['apply', '--state', str(self.state), '--yes'])
        self.assertFalse(self.state.exists())
        self.assertEqual(self.host.writes, [])

    def test_audit_reports_are_exclusive(self):
        report = self.root/'audit.json'
        with mock.patch.object(iso, 'Host', return_value=self.host), mock.patch('builtins.print'):
            self.assertEqual(iso.main(['--report', str(report)]), 0)
            with self.assertRaises(FileExistsError):
                iso.main(['--report', str(report)])
            with mock.patch.object(iso.os, 'geteuid', return_value=0):
                with self.assertRaises(FileExistsError):
                    iso.main(['apply', '--state', str(self.state), '--yes', '--report', str(report)])
        self.assertFalse(self.state.exists())
        self.assertTrue(json.loads(report.read_text())['read_only'])
        self.assertEqual(self.host.writes, [])

    def test_stat_parser_handles_comm_parentheses(self):
        tid=11
        self.host.put(f'/proc/{tid}/status','Name:\trcu_preempt\nTgid:\t11')
        tail=['0']*22
        tail[0]='S';tail[6]=str(0x00200000);tail[19]='12345'
        self.host.put(f'/proc/{tid}/stat',f'{tid} (name with ) parentheses) '+' '.join(tail))
        parsed=iso.Host.task(self.host,tid)
        self.assertEqual(parsed['starttime'],12345)
        self.assertTrue(parsed['kernel'])
        self.assertFalse(parsed['no_setaffinity'])

    def test_cli_refuses_mutation_without_acknowledgement(self):
        for action in ['apply','restore']:
            proc=subprocess.run([sys.executable,str(ROOT/'tests/rio_isolate.py'),action],
                                text=True,capture_output=True,timeout=10)
            self.assertEqual(proc.returncode,2)
            self.assertIn('--state and --yes',proc.stderr)
        self.assertFalse(self.state.exists())

    def test_default_cli_audit_has_no_mutation_calls(self):
        with mock.patch.object(iso,'Host',return_value=self.host), \
                mock.patch.object(self.host,'write',side_effect=AssertionError('write in audit')), \
                mock.patch.object(self.host,'set_affinity',side_effect=AssertionError('affinity in audit')), \
                mock.patch('builtins.print'):
            self.assertEqual(iso.main([]),0)
            self.assertEqual(iso.main(['audit','--json']),0)


if __name__ == '__main__':
    unittest.main()
