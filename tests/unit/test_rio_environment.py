"""Pin-free boot transactions and fake-host runtime guards. Never mutate host policy."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tests'))
import rio_environment as env
import test_rio_isolate as fixtures

FLAGS = dict(CPU_ISOLATION='y', GENERIC_IRQ_MIGRATION='y', NO_HZ_FULL='n', RCU_NOCB_CPU='n')
ORIGINAL = ('console=serial0,115200 root=PARTUUID=keep rootwait isolcpus=3 '
            'nohz_full=3 rcu_nocbs=3 rp1_pio.heavy_tx_burst4_sm=0\n')
CONFIG = ('CONFIG_CPU_ISOLATION=y\nCONFIG_GENERIC_IRQ_MIGRATION=y\n'
          '# CONFIG_NO_HZ_FULL is not set\n# CONFIG_RCU_NOCB_CPU is not set\n')


class BootTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.boot = self.root/'boot'
        self.boot.mkdir()
        (self.boot/'config.txt').write_text('[all]\ndtparam=audio=on\n[pi5]\nkernel=test.img\n[all]\n')
        (self.boot/'test.img').write_bytes(b'header Linux version test-release (builder)\0')
        (self.boot/'cmdline.txt').write_text(ORIGINAL)
        self.config = self.root/'kernel.config'
        self.config.write_text(CONFIG)
        for path in [*self.boot.iterdir(), self.config]:
            path.chmod(0o644)
        self.state = self.root/'private'/'transaction'

    def plan(self):
        return env.inspect_boot(3, {0, 1, 2}, self.boot, self.config, 'test-release')

    def test_only_managed_tokens_change(self):
        after = env.boot_line(ORIGINAL.encode(), FLAGS, 3, {0, 1, 2}).decode().split()
        before_keep = [s for s in ORIGINAL.split() if s.split('=')[0] not in env.MANAGED]
        self.assertEqual(after[:-2], before_keep)
        self.assertEqual(after[-2:], ['isolcpus=domain,managed_irq,3', 'irqaffinity=0,1,2'])

    def test_retains_capability_supported_nohz_and_rcu(self):
        flags = dict(FLAGS, NO_HZ_FULL='y', RCU_NOCB_CPU='y')
        out = env.boot_line((ORIGINAL.strip()+' irqaffinity=f nohz_full=2\n').encode(), flags, 3, {0, 1, 2})
        self.assertEqual(out.count(b'nohz_full='), 1)
        self.assertIn(b'nohz_full=3 rcu_nocbs=3', out)
        self.assertNotIn(b'irqaffinity=f', out)

    def test_complex_or_incomplete_inputs_rejected(self):
        for text in ['', 'root=x\nother=y\n', 'root="a b"\n', 'root=a\\b\n', 'root=x\0', 'console=foo\n']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                env.boot_line(text.encode(), FLAGS, 3, {0, 1, 2})
        for key in FLAGS:
            with self.subTest(key=key), self.assertRaises(ValueError):
                env.boot_line(ORIGINAL.encode(), dict(FLAGS, **{key:'unknown'}), 3, {0, 1, 2})

    def test_image_mismatch_or_indirection_rejected(self):
        (self.boot/'test.img').write_bytes(b'Linux version other-release (builder)')
        with self.assertRaises(ValueError):
            self.plan()
        (self.boot/'test.img').write_bytes(b'Linux version test-release (builder)')
        for line in ['include extra.txt', 'cmdline=other.txt', 'os_prefix=other/', '[tryboot]']:
            (self.boot/'config.txt').write_text('kernel=test.img\n'+line+'\n')
            with self.subTest(line=line), self.assertRaises(ValueError):
                self.plan()

    def test_stock_pi5_automatic_image_requires_matching_release(self):
        (self.boot/'config.txt').write_text('[all]\narm_64bit=1\n')
        (self.boot/'kernel_2712.img').write_bytes((self.boot/'test.img').read_bytes())
        (self.boot/'kernel_2712.img').chmod(0o644)
        plan = self.plan()
        self.assertEqual(plan['kernel'], 'kernel_2712.img')
        self.assertEqual(plan['kernel_selection'], 'pi5-default-verified-by-release')
        (self.boot/'kernel_2712.img').write_bytes(b'Linux version other-release (builder)')
        with self.assertRaises(ValueError):
            self.plan()

    def test_32bit_firmware_selection_refused(self):
        (self.boot/'config.txt').write_text('arm_64bit=0\nkernel=test.img\n')
        with self.assertRaises(ValueError):
            self.plan()

    def test_rcu_omitted_symbol_requires_explicit_disabled_dependencies(self):
        self.config.write_text(CONFIG.replace('# CONFIG_RCU_NOCB_CPU is not set\n', '') +
                               '# CONFIG_RCU_EXPERT is not set\n')
        self.assertTrue(self.plan()['rcu_nocb_disabled_by_dependencies'])
        self.config.write_text(CONFIG.replace('# CONFIG_RCU_NOCB_CPU is not set\n', ''))
        with self.assertRaises(ValueError):
            self.plan()

    def test_symlink_fifo_and_hardlink_refused(self):
        target = self.boot/'cmdline.txt'
        target.unlink()
        target.symlink_to(self.config)
        with self.assertRaises(OSError):
            self.plan()
        target.unlink()
        os.mkfifo(target)
        with self.assertRaises(ValueError):
            self.plan()
        target.unlink()
        os.link(self.config, target)
        with self.assertRaises(ValueError):
            self.plan()

    def test_stage_and_cross_boot_restore_exact_contents_and_mode(self):
        plan = self.plan()
        original_config = (self.boot/'config.txt').read_bytes()
        mode = (self.boot/'cmdline.txt').stat().st_mode
        result = env.stage_boot(plan, self.state, self.boot)
        self.assertEqual(result['phase'], 'staged')
        self.assertEqual((self.boot/'cmdline.txt').read_text(), plan['cmdline_after'])
        self.assertEqual((self.state/'cmdline.txt.before').read_text(), ORIGINAL)
        self.assertEqual((self.state/'config.txt.before').read_bytes(), original_config)
        self.assertEqual((self.state/'journal.json').stat().st_mode & 0o777, 0o600)
        self.assertEqual((self.boot/'cmdline.txt').stat().st_mode, mode)
        with mock.patch.object(env.platform, 'release', return_value='another-boot'):
            self.assertEqual(env.restore_boot(self.state, self.boot)['phase'], 'restored')
        self.assertEqual((self.boot/'cmdline.txt').read_text(), ORIGINAL)
        self.assertEqual((self.boot/'config.txt').read_bytes(), original_config)
        self.assertEqual(env.restore_boot(self.state, self.boot)['phase'], 'restored')

    def test_stage_drift_and_state_reuse_refused(self):
        plan = self.plan()
        (self.boot/'cmdline.txt').write_text(ORIGINAL+'unexpected\n')
        with self.assertRaises(ValueError):
            env.stage_boot(plan, self.state, self.boot)
        self.assertTrue((self.boot/'cmdline.txt').read_text().endswith('unexpected\n'))
        with self.assertRaises(FileExistsError):
            env.stage_boot(plan, self.state, self.boot)

    def test_post_replace_error_rolls_back_from_durable_backup(self):
        plan = self.plan()
        original = env.atomic_bytes
        def fail_once(path, raw):
            self.assertTrue((self.state/'cmdline.txt.before').exists())
            self.assertEqual(json.loads((self.state/'journal.json').read_text())['phase'], 'pending')
            original(path, raw)
            if raw == plan['cmdline_after'].encode():
                raise OSError('simulated post-replace failure')
        with mock.patch.object(env, 'atomic_bytes', side_effect=fail_once), self.assertRaises(OSError):
            env.stage_boot(plan, self.state, self.boot)
        self.assertEqual((self.boot/'cmdline.txt').read_text(), ORIGINAL)
        self.assertEqual(json.loads((self.state/'journal.json').read_text())['phase'], 'rolled_back')

    def test_restore_conflicts_and_backup_corruption_refused(self):
        env.stage_boot(self.plan(), self.state, self.boot)
        (self.boot/'cmdline.txt').write_text('root=someone-else\n')
        with self.assertRaises(ValueError):
            env.restore_boot(self.state, self.boot)
        self.assertEqual((self.boot/'cmdline.txt').read_text(), 'root=someone-else\n')
        j = env.iso.Journal(self.state)
        j.data['cmdline_before'] += 'tampered'
        j.save()
        with self.assertRaises(ValueError):
            env.restore_boot(self.state, self.boot)

    def test_private_parent_checks_before_creating_under_symlink(self):
        link = self.root/'link'
        link.symlink_to(self.boot, target_is_directory=True)
        with self.assertRaises(ValueError):
            env.private_parent(link/'new')
        self.assertFalse((self.boot/'new').exists())

    def test_already_staged_plan_is_noop_without_state_creation(self):
        env.stage_boot(self.plan(), self.state, self.boot)
        second = self.root/'unused'/'state'
        self.assertEqual(env.stage_boot(self.plan(), second, self.boot)['phase'], 'unchanged')
        self.assertFalse(second.parent.exists())


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.IsolationTests('test_masks_and_lists')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.host, self.state = self.fixture.host, self.fixture.state
        self.health = mock.Mock(return_value={'ok': True})
        self.users = mock.Mock(return_value={'users': [], 'errors': []})

    def apply(self):
        return env.runtime_apply(self.host, 3, {0, 1, 2}, self.state,
                                 self.health, self.users, sleep=lambda _: None)

    def test_checked_apply_idempotence_restore_and_no_reuse(self):
        self.assertEqual(self.apply()['phase'], 'applied')
        writes = len(self.host.writes)
        self.assertEqual(self.apply()['phase'], 'already_applied')
        self.assertEqual(len(self.host.writes), writes)
        result = env.runtime_restore(self.host, self.state, self.users)
        self.assertEqual(result['phase'], 'restored')
        self.assertEqual(self.host.read('/proc/irq/default_smp_affinity'), 'f')
        with self.assertRaises(ValueError):
            self.apply()

    def test_busy_scan_errors_or_bad_health_do_not_create_state(self):
        for users, health in [({'users': [123], 'errors': []}, True),
                              ({'users': [], 'errors': ['unreadable']}, True),
                              ({'users': [], 'errors': []}, False)]:
            self.users.return_value = users
            self.health.return_value = {'ok': health}
            with self.subTest(users=users, health=health), self.assertRaises(ValueError):
                self.apply()
            self.assertFalse(self.state.exists())
            self.assertEqual(self.host.writes, [])

    def test_partial_apply_is_rolled_back_not_left_partial(self):
        self.host.fail.add('/proc/irq/141/smp_affinity_list')
        with self.assertRaises(ValueError):
            self.apply()
        self.assertEqual(self.host.read('/proc/irq/default_smp_affinity'), 'f')
        self.assertEqual(env.iso.Journal(self.state).data['phase'], 'restored')

    def test_post_apply_and_settled_health_failure_or_interrupt_roll_back(self):
        for failure in [1, 2]:
            state = self.state.with_name(f'failure-{failure}')
            health = mock.Mock(side_effect=[{'ok': i != failure} for i in range(3)])
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                env.runtime_apply(self.host, 3, {0, 1, 2}, state, health, self.users, sleep=lambda _: None)
            self.assertEqual(self.host.read('/proc/irq/default_smp_affinity'), 'f')
        def stop(_):
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            env.runtime_apply(self.host, 3, {0, 1, 2}, self.state, self.health, self.users, sleep=stop)
        self.assertEqual(self.host.read('/proc/irq/default_smp_affinity'), 'f')

    def test_drift_effective_affinity_and_stale_boot_fail_without_new_writes(self):
        self.apply()
        writes = len(self.host.writes)
        self.host.put('/proc/irq/141/effective_affinity_list', '3')
        with self.assertRaises(ValueError):
            self.apply()
        self.assertEqual(len(self.host.writes), writes)
        self.host.put('/proc/sys/kernel/random/boot_id', 'another-boot')
        with self.assertRaises(ValueError):
            env.runtime_restore(self.host, self.state, self.users)
        self.assertEqual(len(self.host.writes), writes)

    def test_busy_restore_refused_without_new_writes(self):
        self.apply()
        writes = len(self.host.writes)
        self.users.return_value['users'] = [123]
        with self.assertRaises(ValueError):
            env.runtime_restore(self.host, self.state, self.users)
        self.assertEqual(len(self.host.writes), writes)

    def test_irqbalance_requires_explicit_configuration(self):
        self.fixture.task(51, 'irqbalance', kernel=False)
        with self.assertRaises(ValueError):
            self.apply()
        self.assertFalse(self.state.exists())
        self.assertEqual(self.host.writes, [])

    def test_irqbalance_after_apply_is_not_accepted_by_repeated_apply(self):
        self.apply()
        writes = len(self.host.writes)
        self.fixture.task(51, 'irqbalance', kernel=False)
        with self.assertRaises(ValueError):
            self.apply()
        self.assertEqual(len(self.host.writes), writes)

    def test_new_movable_work_during_settle_rolls_back_owned_changes(self):
        def new_thread(_):
            self.fixture.task(52, 'rcu_tasks_new')
        with self.assertRaises(ValueError):
            env.runtime_apply(self.host, 3, {0, 1, 2}, self.state,
                              self.health, self.users, sleep=new_thread)
        self.assertEqual(self.host.read('/proc/irq/default_smp_affinity'), 'f')
        self.assertEqual(self.host.affinity(52), {0, 1, 2, 3})

    def test_unowned_already_confined_state_does_not_invent_originals(self):
        env.iso.apply(self.host, env.iso.audit(self.host), self.state)
        next_state = self.state.with_name('unowned')
        result = env.runtime_apply(self.host, 3, {0, 1, 2}, next_state, self.health, self.users)
        self.assertEqual(result['phase'], 'already_confined_without_owned_journal')
        self.assertFalse(next_state.exists())

    def test_process_scan_catches_offcore_controller_and_device_handles(self):
        proc = self.fixture.root/'proc-scan'
        p = proc/'123'
        p.mkdir(parents=True)
        (p/'comm').write_text('python3\n')
        (p/'cmdline').write_bytes(b'python3\0/tests/rio_baseline.py\0')
        (p/'maps').write_text('mapping /dev/gpiomem0\n')
        (p/'fd').mkdir()
        (p/'fd'/'4').symlink_to('/dev/gpiochip0')
        result = env.gpio_users(proc)
        self.assertEqual(len(result['users']), 2)
        self.assertEqual(result['errors'], [])

    def test_health_rejects_bad_config_active_flags_and_missing_telemetry(self):
        valid = bytearray(64)
        valid[:4] = bytes.fromhex('e41d0100')
        for raw, stdout, code, ok in [
                (valid, 'throttled=0x0', 0, True),
                (valid, 'throttled=0x50000', 0, True),  # history is not active
                (valid, 'throttled=0x1', 0, False),
                (valid, '', 1, False), (valid, 'garbage', 0, False),
                (b'\\xff'*64, 'throttled=0x0', 0, False),
                (valid[:16], 'throttled=0x0', 0, False)]:
            process = subprocess.CompletedProcess([], code, stdout=stdout, stderr='')
            with self.subTest(stdout=stdout, size=len(raw)), \
                    mock.patch.object(Path, 'open', return_value=io.BytesIO(raw)), \
                    mock.patch.object(env.subprocess, 'run', return_value=process):
                self.assertIs(env.rp1_health()['ok'], ok)
        with mock.patch.object(Path, 'open', side_effect=OSError('unavailable')):
            self.assertFalse(env.rp1_health()['ok'])

    def test_cli_mutations_without_ack_are_rejected(self):
        for action in ['apply', 'restore', 'boot-apply', 'boot-restore']:
            p = subprocess.run([sys.executable, str(ROOT/'tests/rio_environment.py'), action],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 2)
            self.assertIn('root and --yes', p.stderr)


if __name__ == '__main__':
    unittest.main()
