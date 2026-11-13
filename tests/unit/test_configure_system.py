"""Installer tests: temporary files, fake systemd and fake affinities; no host setup."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tests'))
import rio_environment as env
import rio_isolate as iso
import test_rio_isolate as fixtures

spec = importlib.util.spec_from_file_location('configure_system', ROOT/'configure-system.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
BOOT1 = '11111111-1111-4111-8111-111111111111'
BOOT2 = '22222222-2222-4222-8222-222222222222'
ORIGINAL = 'console=serial0 root=PARTUUID=keep rootwait isolcpus=3 nohz_full=3 rcu_nocbs=3\n'
CONFIG = ('CONFIG_CPU_ISOLATION=y\nCONFIG_GENERIC_IRQ_MIGRATION=y\n'
          '# CONFIG_NO_HZ_FULL is not set\n# CONFIG_RCU_NOCB_CPU is not set\n')


class FakeSystem:
    def __init__(self, fixture, paths):
        self.fixture, self.host, self.paths = fixture, fixture.host, paths
        self.irq_active = False
        self.irq_exists = False
        self.busy = False
        self.good_health = True
        self.calls = []
        self.restores = []
        self.fail_once = set()

    def preflight(self):
        pass

    def idle(self):
        if self.busy:
            raise ValueError('fake GPIO workload active')

    def health(self):
        if not self.good_health:
            raise ValueError('fake bad PCI/power')
        return {'ok':True}

    def plan(self):
        return env.inspect_boot(3, {0,1,2}, self.paths.boot,
                                self.paths.root/'boot/kernel.config', 'test-release')

    def unit(self, name):
        if name == app.UNIT:
            loaded = self.paths.files['unit'].exists()
            return dict(LoadState='loaded' if loaded else 'not-found', ActiveState='inactive', MainPID='0')
        masked = self.paths.files['irq_mask'].is_symlink()
        return dict(LoadState='masked' if masked else ('loaded' if self.irq_exists else 'not-found'),
                    ActiveState='active' if self.irq_active else 'inactive', MainPID='51' if self.irq_active else '0')

    def run(self, *args):
        self.calls.append(args)
        if args in self.fail_once:
            self.fail_once.remove(args)
            raise ValueError('fake systemctl refusal')
        if args == ('stop', app.IRQBALANCE):
            self.irq_active = False
            self.host.tasks.pop(51, None)
        if args == ('start', app.IRQBALANCE):
            self.irq_active = True
            self.fixture.task(51, 'irqbalance', kernel=False)
        return ''

    def apply_runtime(self, state):
        return env.runtime_apply(self.host, 3, {0,1,2}, state,
                                 health=self.health, users=lambda:{'users':[], 'errors':[]}, sleep=lambda _:None)

    def restore_runtime(self, state):
        self.restores.append(str(state))
        return env.runtime_restore(self.host, state, users=lambda:{'users':[], 'errors':[]})


class ConfigureTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.IsolationTests('test_masks_and_lists')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.host = self.fixture.host
        self.host.put('/proc/sys/kernel/random/boot_id', BOOT1)
        self.p = app.Paths(self.host.root)
        for p in [self.p.units, self.p.libexec.parent, self.p.base.parent, self.p.boot]:
            p.mkdir(parents=True, exist_ok=True)
        for p in [self.host.root, *self.host.root.rglob('*')]:
            if p.is_dir():
                p.chmod(0o755)
        (self.p.boot/'config.txt').write_text('[all]\n[pi5]\nkernel=test.img\n[all]\n')
        (self.p.boot/'cmdline.txt').write_text(ORIGINAL)
        (self.p.boot/'test.img').write_bytes(b'Linux version test-release (builder)\0')
        (self.p.root/'boot/kernel.config').write_text(CONFIG)
        for p in [*self.p.boot.iterdir(), self.p.root/'boot/kernel.config']:
            p.chmod(0o644)
        self.system = FakeSystem(self.fixture, self.p)
        self.manager = app.Configuration(self.p, self.system, env, iso)
        self.payload = app.source_payload(self.p)
        self.original_tasks = copy.deepcopy(self.host.tasks)
        self.original_controls = {str(p):p.read_bytes() for p in self.host.root.glob('proc/irq/*/smp_affinity_list')}
        self.original_controls[str(self.host.path('/proc/irq/default_smp_affinity'))] = b'f\n'

    def install(self):
        return self.manager.configure(self.payload)

    def test_round_trip_persists_files_and_restores_all_owned_changes(self):
        result = self.install()
        self.assertEqual(result['phase'], 'configured')
        self.assertTrue(result['reboot_required'])
        self.assertIn('isolcpus=domain,managed_irq,3', (self.p.boot/'cmdline.txt').read_text())
        self.assertEqual(os.readlink(self.p.files['enable']), str(self.p.files['unit']))
        self.assertEqual(os.readlink(self.p.files['irq_mask']), '/dev/null')
        for key in self.payload:
            self.assertEqual(self.p.files[key].read_bytes(), self.payload[key])
            self.assertFalse(self.p.files[key].stat().st_mode & 0o022)
        self.assertEqual(self.host.read('/proc/irq/default_smp_affinity'), '7')
        self.assertEqual(self.manager.restore()['phase'], 'restored')
        self.assertEqual((self.p.boot/'cmdline.txt').read_text(), ORIGINAL)
        self.assertEqual(self.host.read('/proc/irq/default_smp_affinity'), 'f')
        self.assertTrue((self.p.state/'journal.json').exists())
        self.assertFalse(any(os.path.lexists(p) for p in self.p.files.values()))
        self.assertEqual(self.manager.restore()['phase'], 'already_restored')

    def test_repeat_configure_does_not_replace_originals(self):
        self.install()
        data = app.load_state(self.p)
        before = (self.p.state/'boot/journal.json').read_bytes()
        count = len(self.host.writes)
        self.assertEqual(self.install()['phase'], 'already_configured')
        self.assertEqual(len(self.host.writes), count)
        self.assertEqual((self.p.state/'boot/journal.json').read_bytes(), before)
        self.assertEqual(data['runtime'], app.load_state(self.p)['runtime'])

    def test_restore_reconfigure_archives_old_backups(self):
        self.install()
        self.manager.restore()
        old = (self.p.state/'boot/journal.json').read_bytes()
        self.assertEqual(self.install()['phase'], 'configured')
        archives = list(self.p.base.glob('configure-system-restored-*'))
        self.assertEqual(len(archives), 1)
        self.assertEqual((archives[0]/'boot/journal.json').read_bytes(), old)

    def test_runtime_new_boot_gets_new_journal_and_restore_never_replays_old(self):
        self.install()
        j = self.manager.journal()
        old_path = self.p.state/j.data['runtime'][BOOT1]
        old_raw = (old_path/'journal.json').read_bytes()
        self.host.tasks = copy.deepcopy(self.original_tasks)
        for name, raw in self.original_controls.items():
            p = Path(name);p.chmod(0o644);p.write_bytes(raw)
        self.host.put('/proc/sys/kernel/random/boot_id', BOOT2)
        self.manager.runtime(self.manager.journal())
        self.manager.restore()
        self.assertEqual(len(self.system.restores), 1)
        self.assertIn(BOOT2, self.system.restores[0])
        self.assertEqual((old_path/'journal.json').read_bytes(), old_raw)

    def test_live_owned_restore_conflict_is_partial_and_not_forced(self):
        self.install()
        self.host.put('/proc/irq/default_smp_affinity', '3')
        with self.assertRaises(ValueError):
            self.manager.restore()
        self.assertEqual(self.host.read('/proc/irq/default_smp_affinity'), '3')
        self.assertEqual(app.load_state(self.p)['phase'], 'restore_incomplete')
        self.assertTrue(self.p.files['runner'].exists())
        self.assertFalse(self.p.files['enable'].exists())

    def test_unowned_configuration_is_preserved_on_restore(self):
        (self.p.boot/'cmdline.txt').write_text(self.system.plan()['cmdline_after'])
        external = self.p.root/'external-runtime'
        iso.apply(self.host, iso.audit(self.host), external)
        self.assertFalse(self.install()['reboot_required'])
        self.manager.restore()
        self.assertIn('managed_irq', (self.p.boot/'cmdline.txt').read_text())
        self.assertEqual(self.host.read('/proc/irq/default_smp_affinity'), '7')
        self.assertEqual(iso.Journal(external).data['phase'], 'applied')

    def test_irqbalance_service_stop_mask_and_prior_active_state_restore(self):
        self.system.irq_exists = self.system.irq_active = True
        self.fixture.task(51, 'irqbalance', kernel=False)
        self.install()
        self.assertFalse(self.system.irq_active)
        self.assertTrue(self.p.files['irq_mask'].is_symlink())
        self.manager.restore()
        self.assertTrue(self.system.irq_active)
        self.assertFalse(os.path.lexists(self.p.files['irq_mask']))

    def test_failed_irqbalance_stop_restores_prior_service_and_boot(self):
        self.system.irq_exists = self.system.irq_active = True
        self.fixture.task(51, 'irqbalance', kernel=False)
        self.system.fail_once.add(('stop', app.IRQBALANCE))
        with self.assertRaises(ValueError):
            self.install()
        self.assertTrue(self.system.irq_active)
        self.assertFalse(os.path.lexists(self.p.files['irq_mask']))
        self.assertEqual((self.p.boot/'cmdline.txt').read_text(), ORIGINAL)
        self.assertEqual(app.load_state(self.p)['phase'], 'restored')

    def test_shared_new_parent_is_kept_if_other_software_uses_it(self):
        self.install()
        other = self.p.libexec/'another-package'
        other.mkdir()
        (other/'keep').write_text('unrelated')
        self.assertEqual(self.manager.restore()['phase'], 'restored')
        self.assertEqual((other/'keep').read_text(), 'unrelated')
        self.assertEqual(app.load_state(self.p)['directories']['libexec']['status'], 'preserved')

    def test_preexisting_irqbalance_mask_is_not_removed(self):
        self.p.files['irq_mask'].symlink_to('/dev/null')
        self.install()
        self.manager.restore()
        self.assertEqual(os.readlink(self.p.files['irq_mask']), '/dev/null')

    def test_unmanaged_irqbalance_or_local_override_refused_before_state(self):
        self.fixture.task(51, 'irqbalance', kernel=False)
        with self.assertRaises(ValueError):
            self.install()
        self.assertFalse(self.p.state.exists())
        self.host.tasks.pop(51)
        self.p.files['irq_mask'].write_text('local override')
        with self.assertRaises(ValueError):
            self.install()
        self.assertFalse(self.p.state.exists())

    def test_existing_package_or_unit_never_overwritten(self):
        self.p.package.mkdir(parents=True)
        (self.p.package/'unrelated').write_text('keep')
        with self.assertRaises(ValueError):
            self.install()
        self.assertEqual((self.p.package/'unrelated').read_text(), 'keep')
        self.assertFalse(self.p.state.exists())

    def test_busy_bad_health_and_boot_drift_are_refused(self):
        self.system.busy = True
        with self.assertRaises(ValueError):
            self.install()
        self.system.busy = False
        self.system.good_health = False
        with self.assertRaises(ValueError):
            self.install()
        self.assertFalse(self.p.state.exists())
        self.system.good_health = True
        self.install()
        self.system.busy = True
        with self.assertRaises(ValueError):
            self.manager.restore()
        self.assertTrue(self.p.files['enable'].is_symlink())

    def test_install_runtime_failure_rolls_back_boot_files_and_mask(self):
        with mock.patch.object(self.system, 'apply_runtime', side_effect=ValueError('refused')):
            with self.assertRaises(ValueError):
                self.install()
        self.assertEqual((self.p.boot/'cmdline.txt').read_text(), ORIGINAL)
        self.assertFalse(any(os.path.lexists(p) for p in self.p.files.values()))
        self.assertEqual(app.load_state(self.p)['phase'], 'restored')

    def test_install_systemd_failure_attempts_rollback(self):
        self.system.fail_once.add(('daemon-reload',))
        with self.assertRaises(ValueError):
            self.install()
        self.assertEqual((self.p.boot/'cmdline.txt').read_text(), ORIGINAL)
        self.assertFalse(self.p.package.exists())
        self.assertEqual(app.load_state(self.p)['phase'], 'restored')

    def test_signal_during_install_attempts_rollback(self):
        with mock.patch.object(self.system, 'apply_runtime', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.install()
        self.assertEqual((self.p.boot/'cmdline.txt').read_text(), ORIGINAL)
        self.assertFalse(self.p.files['enable'].exists())
        self.assertEqual(app.load_state(self.p)['phase'], 'restored')

    def test_foreign_asset_drift_refuses_restore_before_any_mutation(self):
        self.install()
        self.p.files['unit'].write_text('foreign administrator edit')
        calls = len(self.system.calls)
        with self.assertRaises(ValueError):
            self.manager.restore()
        self.assertEqual(len(self.system.calls), calls)
        self.assertTrue(self.p.files['enable'].is_symlink())
        self.assertEqual(app.load_state(self.p)['phase'], 'installed')

    def test_owned_link_drift_or_path_identity_replacement_not_removed(self):
        self.install()
        self.p.files['enable'].unlink()
        self.p.files['enable'].symlink_to('/unrelated')
        with self.assertRaises(ValueError):
            self.manager.restore()
        self.assertEqual(os.readlink(self.p.files['enable']), '/unrelated')

    def test_partial_boot_restore_keeps_recovery_tools_and_can_retry(self):
        self.install()
        after = (self.p.boot/'cmdline.txt').read_text()
        (self.p.boot/'cmdline.txt').write_text(after.strip()+' administrator=change\n')
        with self.assertRaises(ValueError):
            self.manager.restore()
        self.assertTrue(self.p.files['runner'].exists())
        self.assertEqual(app.load_state(self.p)['phase'], 'restore_incomplete')
        (self.p.boot/'cmdline.txt').write_text(after)
        self.assertEqual(self.manager.restore()['phase'], 'restored')

    def test_untrusted_install_ancestors_refused_before_writes(self):
        self.p.libexec.parent.chmod(0o777)
        with self.assertRaises(ValueError):
            self.install()
        self.assertFalse(self.p.state.exists())
        self.p.libexec.parent.chmod(0o755)
        elsewhere = self.p.root/'elsewhere'
        elsewhere.mkdir()
        self.p.wants.symlink_to(elsewhere, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.install()
        self.assertFalse(self.p.state.exists())
        self.assertEqual(list(elsewhere.iterdir()), [])

    def test_dry_plan_has_no_writes_or_state_creation(self):
        before = (self.p.boot/'cmdline.txt').read_bytes()
        self.manager.plan(self.payload)
        self.assertEqual(self.host.writes, [])
        self.assertEqual(self.system.calls, [])
        self.assertFalse(self.p.state.exists())
        self.assertEqual((self.p.boot/'cmdline.txt').read_bytes(), before)

    def test_installed_loader_hash_guard_and_no_project_dependency(self):
        self.install()
        loaded_env, loaded_iso = app.load_helpers(self.p, installed=True, data=app.load_state(self.p))
        self.assertTrue(callable(loaded_env.runtime_apply))
        self.assertTrue(callable(loaded_iso.restore))
        self.p.files['isolate'].write_bytes(b'raise RuntimeError("must not execute")\n')
        with self.assertRaises(ValueError):
            app.load_helpers(self.p, installed=True, data=app.load_state(self.p))

    def test_service_has_isolated_imports_no_project_paths_and_no_execstop(self):
        text = app.service_text(self.p).decode()
        self.assertIn('/usr/bin/python3 -I -B ', text)
        self.assertIn('CPUAffinity=0 1 2', text)
        self.assertIn('Environment=PATH=/usr/sbin:/usr/bin:/sbin:/bin', text)
        self.assertNotIn('/usr/local', text)
        self.assertIn('Before=multi-user.target', text)
        self.assertNotIn(str(ROOT), text)
        self.assertNotIn('ExecStop=', text)
        self.assertNotIn('SCHED_FIFO', text)

    def test_unknown_file_in_package_is_preserved_not_recursively_deleted(self):
        self.install()
        unknown = self.p.package/'administrator-file'
        unknown.write_text('keep')
        with self.assertRaises(ValueError):
            self.manager.restore()
        self.assertEqual(unknown.read_text(), 'keep')
        self.assertEqual(app.load_state(self.p)['phase'], 'installed')
        self.assertTrue(self.p.files['runner'].exists())

    def test_exclusive_publication_race_preserves_foreign_file(self):
        original = app.os.link
        def race(source, target, **kwargs):
            if target == self.p.files['unit']:
                target.write_text('foreign unit')
            return original(source, target, **kwargs)
        with mock.patch.object(app.os, 'link', side_effect=race), self.assertRaises(FileExistsError):
            self.install()
        self.assertEqual(self.p.files['unit'].read_text(), 'foreign unit')
        self.assertEqual(app.load_state(self.p)['phase'], 'restored')
        self.assertEqual((self.p.boot/'cmdline.txt').read_text(), ORIGINAL)

    def test_payload_syntax_and_private_manifest_permissions(self):
        payload = dict(self.payload, environment=b'invalid python !')
        with self.assertRaises(SyntaxError):
            self.manager.configure(payload)
        self.assertFalse(self.p.state.exists())
        self.install()
        self.p.state.chmod(0o755)
        with self.assertRaises(ValueError):
            self.manager.restore()
        self.assertTrue(self.p.files['enable'].is_symlink())

    def test_manifest_path_tampering_rejected_before_changes(self):
        self.install()
        j = self.manager.journal()
        j.data['runtime'][BOOT1] = '../../unrelated'
        j.save()
        calls = len(self.system.calls)
        with self.assertRaises(ValueError):
            self.manager.restore()
        self.assertEqual(len(self.system.calls), calls)

    def test_help_and_noninteractive_ack_guard(self):
        p = subprocess.run([sys.executable, str(ROOT/'configure-system.py'), '--help'], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0)
        self.assertIn('--restore', p.stdout)
        with mock.patch.object(app.os, 'geteuid', return_value=0), \
                mock.patch.object(app.sys.stdin, 'isatty', return_value=False), \
                mock.patch.object(app.sys, 'stderr', io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                app.main([])
        self.assertEqual(raised.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
