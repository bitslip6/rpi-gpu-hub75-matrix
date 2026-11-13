import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
EXE = Path(os.environ.get('RIO_SCOPE_EXE', ROOT / 'build/release/rio_scope'))
spec = importlib.util.spec_from_file_location('rio_baseline', ROOT / 'tests/rio_baseline.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class ScopeTests(unittest.TestCase):
    def test_default_is_ram(self):
        proc = subprocess.run([str(EXE), '--cycles', '3'], text=True, capture_output=True, timeout=10)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        metrics = json.loads(proc.stdout)
        self.assertEqual(metrics['backend'], 'ram-dry-run')
        self.assertIsNone(metrics['blank_readback_ok'])
        self.assertEqual(metrics['workload_cycles'], 3)
        self.assertEqual(metrics['full_cycles'], 0)
        self.assertEqual(metrics['plane_sweeps'], 0)
        self.assertFalse(metrics['scope_verified'])
        self.assertIsNone(metrics['deadline_misses'])
        self.assertIsNone(metrics['shutdown_latency_ns'])
        self.assertFalse(metrics['pad_output_enable_requested'])
        self.assertEqual(metrics['pad_output_enable_applied_mask'], 0)

    def test_modes_paths_and_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            for mode in baseline.MODES:
                for store in ('xor', 'out'):
                    with self.subTest(mode=mode, store=store):
                        samples = Path(temp) / f'{mode}-{store}.csv'
                        argv = [str(EXE), '--mode', mode, '--store', store, '--cycles', '3',
                                '--width', '17', '--rows', '4', '--planes', '3',
                                '--sample-every', '1', '--samples', str(samples)]
                        proc = subprocess.run(argv, text=True, capture_output=True, timeout=10)
                        self.assertEqual(proc.returncode, 0, proc.stderr)
                        data = json.loads(proc.stdout)
                        self.assertEqual(data['backend'], 'ram-dry-run')
                        self.assertEqual(data['issued_pixel_edges'], 3 * 17 * 4 * 3)
                        self.assertEqual(data['repeats'], 2)
                        self.assertEqual(data['prepared_generations'], 1)
                        self.assertEqual(data['full_cycles'], 3 if mode == 'protocol' else 0)
                        self.assertEqual(data['plane_sweeps'], 9 if mode == 'protocol' else 0)
                        self.assertEqual(data['sample_count'], 3)
                        self.assertEqual(data['emit_total_ns'] + data['service_total_ns'], data['elapsed_ns'])
                        with samples.open() as file:
                            rows = list(csv.DictReader(file))
                        self.assertEqual([int(row['workload_cycle']) for row in rows], [1, 2, 3])
                        self.assertTrue(all(int(row['emit_including_blank_ns']) > 0 for row in rows))

    def test_invalid_options_never_reach_output(self):
        for args in (['--output'], ['--output', '--panels-disconnected', '--store', 'out'],
                     ['--mode', 'bad'], ['--rows', '3', '--mode', 'protocol'],
                     ['--seconds', '-1'], ['--seconds', 'nan'], ['--cpu', '1024'],
                     ['--width', '0'], ['--sample-every', '0'], ['unexpected'],
                     ['--enable-pad-output'], ['--output', '--enable-pad-output']):
            with self.subTest(args=args):
                proc = subprocess.run([str(EXE), *args], text=True, capture_output=True, timeout=10)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(proc.stdout, '')
                self.assertIn('Invalid options', proc.stderr)
                self.assertNotIn('GPIO OUTPUT:', proc.stderr)

    def test_existing_samples_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'existing.csv'
            path.write_text('preserve me\n')
            proc = subprocess.run([str(EXE), '--cycles', '1', '--samples', str(path)],
                                  text=True, capture_output=True, timeout=10)
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(path.read_text(), 'preserve me\n')
            self.assertNotIn('RAM DRY RUN', proc.stderr)  # Fails before emission.

    def test_cooperative_signal_stop(self):
        proc = subprocess.Popen([str(EXE), '--seconds', '30'], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            # Banner is emitted only after sigwait is ready, before emission starts.
            banner = proc.stderr.readline()
            self.assertIn('RAM DRY RUN', banner)
            proc.send_signal(signal.SIGTERM)
            out, err = proc.communicate(timeout=10)
            self.assertEqual(proc.returncode, 130, err)
            data = json.loads(out)
            self.assertTrue(data['interrupted'])
            self.assertGreaterEqual(data['shutdown_latency_ns'], 0)
            self.assertEqual(data['backend'], 'ram-dry-run')
        finally:
            if proc.poll() is None:
                proc.kill()  # Test subprocess is explicitly RAM-only, never GPIO.
                proc.communicate()

    def test_interrupt_deltas_and_missing_data(self):
        before = {'text': ' CPU0 CPU3\n 17: 3 4 RP1 foo\n IPI0: 10 20 Rescheduling\n Err: 0\n'}
        after = {'text': ' CPU0 CPU3\n 17: 4 6 RP1 foo\n IPI0: 12 25 Rescheduling\n Err: 0\n'}
        result = baseline.interrupt_delta(before, after, 3)
        self.assertEqual(result['cpu_total_delta'], 7)
        self.assertEqual(result['rows']['IPI0']['delta']['CPU3'], 5)
        self.assertEqual(result['changed_rows'], [])
        reset = {'text': after['text'].replace('12 25', '0 1')}
        self.assertIn('IPI0', baseline.interrupt_delta(before, reset, 3)['changed_rows'])
        self.assertIn('error', baseline.interrupt_delta({'error': 'denied'}, after, 3))
        self.assertIn('error', baseline.interrupt_delta({'text': ''}, after, 3))
        self.assertIn('error', baseline.interrupt_delta(before, after, 7))

    def test_kernel_config_does_not_infer_tickless_from_rt(self):
        result = baseline.config_flags('CONFIG_PREEMPT_RT=y\n# CONFIG_NO_HZ_FULL is not set\n')
        self.assertEqual(result['PREEMPT_RT'], 'y')
        self.assertEqual(result['NO_HZ_FULL'], 'n')
        self.assertEqual(result['RCU_NOCB_CPU'], 'unknown')

    def test_noninteractive_hardware_collector_refused(self):
        proc = subprocess.run([sys.executable, str(ROOT / 'tests/rio_baseline.py'), '--output'],
                              input='', text=True, capture_output=True, timeout=10)
        self.assertEqual(proc.returncode, 2)
        self.assertIn('interactive terminal', proc.stderr)

    def test_pad_opt_in_requires_hardware_collection(self):
        proc = subprocess.run([sys.executable, str(ROOT / 'tests/rio_baseline.py'),
                               '--enable-pad-output'], text=True, capture_output=True, timeout=10)
        self.assertEqual(proc.returncode, 2)
        self.assertIn('--enable-pad-output requires --output', proc.stderr)

    def test_fifo_requires_explicit_budget_acknowledgement(self):
        proc = subprocess.run([sys.executable, str(ROOT / 'tests/rio_baseline.py'),
                               '--scheduler', 'fifo'], text=True, capture_output=True, timeout=10)
        self.assertEqual(proc.returncode, 2)
        self.assertIn('--ack-rt-throttling', proc.stderr)
        self.assertIn('No RT limits will be changed', proc.stderr)

    def test_failed_run_reports_stderr_not_json_error(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / 'stdout.json').write_text('')
            diagnostic = ('GPIO17 ctrl=0x0000009f pad=0x00000096\n'
                          'GPIO17 pad output-disable is set; requires --enable-pad-output\n')
            (run / 'stderr.log').write_text(diagnostic)
            metrics, error = baseline.read_run_summary(run, 1, None, 'clock')
            self.assertEqual(metrics, {})
            self.assertIn('benchmark exit 1', error)
            self.assertIn('GPIO17 pad output-disable', error)
            self.assertIn('--enable-pad-output', error)
            self.assertNotIn('Expecting value', error)
            _, error = baseline.read_run_summary(run, 0, None, 'clock')
            self.assertEqual(error, 'missing JSON summary')

    def test_collector_failure_requests_child_stop(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        with tempfile.TemporaryDirectory() as temp, \
                mock.patch.object(baseline.subprocess, 'Popen', return_value=proc), \
                mock.patch.object(baseline, 'telemetry', side_effect=OSError('telemetry failed')):
            with self.assertRaises(OSError):
                baseline.run_child(['not-executed'], Path(temp), 0)
        proc.send_signal.assert_called_once_with(signal.SIGTERM)
        proc.wait.assert_called_once_with(timeout=10)

    def test_invalid_summary_stops_collector(self):
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) < 2:
            self.skipTest('collector requires two available CPUs')
        with tempfile.TemporaryDirectory() as temp:
            fake = Path(temp) / 'fake-benchmark'
            fake.write_text('#!/usr/bin/env python3\nprint("[]")\n')
            fake.chmod(0o755)
            dest = Path(temp) / 'capture'
            argv = [sys.executable, str(ROOT / 'tests/rio_baseline.py'), '--skip-build',
                    '--exe', str(fake), '--cycles', '1', '--seconds', '1', '--repeats', '1',
                    '--modes', 'alternating', 'sequence', '--cpu', str(cpus[-1]),
                    '--controller-cpu', str(cpus[0]), '--results', str(dest)]
            proc = subprocess.run(argv, text=True, capture_output=True, timeout=45)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertFalse((dest / 'sequence-xor-r1').exists())
            result = json.loads((dest / 'alternating-xor-r1/result.json').read_text())
            self.assertTrue(result['error'])
            with (dest / 'summary.csv').open() as file:
                self.assertIn('INVALID', list(csv.DictReader(file))[0]['status'])

    def test_collector_ram_artifacts(self):
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) < 2:
            self.skipTest('collector requires two available CPUs')
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / 'capture'
            argv = [sys.executable, str(ROOT / 'tests/rio_baseline.py'), '--skip-build',
                    '--exe', str(EXE), '--cycles', '3', '--seconds', '1', '--repeats', '1',
                    '--modes', 'protocol', '--sample-every', '1', '--cpu', str(cpus[-1]),
                    '--controller-cpu', str(cpus[0]), '--results', str(dest)]
            proc = subprocess.run(argv, text=True, capture_output=True, timeout=45)
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            manifest = json.loads((dest / 'manifest.json').read_text())
            self.assertEqual(manifest['executable_sha256'], hashlib.sha256(EXE.read_bytes()).hexdigest())
            self.assertTrue((dest / 'sources/tests/rio_scope.c').exists())
            self.assertTrue((dest / 'sources/tests/rio_scope_lock.h').exists())
            self.assertTrue((dest / 'sources/tests/rio_isolate.py').exists())
            self.assertTrue((dest / 'sources/tests/README-rio-isolate.md').exists())
            self.assertTrue((dest / 'sources/tests/rio_environment.py').exists())
            self.assertTrue((dest / 'sources/tests/README-rio-environment.md').exists())
            self.assertTrue((dest / 'sources/configure-system.py').exists())
            self.assertTrue((dest / 'sources/tests/README-configure-system.md').exists())
            self.assertTrue((dest / 'provenance.json').exists())
            run = dest / 'protocol-xor-r1'
            self.assertTrue((run / 'cycles.csv').exists())
            self.assertTrue((run / 'interrupt-deltas.json').exists())
            self.assertTrue((run / 'telemetry.jsonl').exists())
            for phase in ['before', 'after']:
                isolation = json.loads((run / phase / 'isolation.json').read_text())
                self.assertEqual(isolation['exit'], 0)
                self.assertTrue(isolation['audit']['read_only'])
                self.assertEqual(isolation['audit']['cpu'], cpus[-1])
                self.assertNotIn(cpus[-1], isolation['audit']['housekeeping'])
            result = json.loads((run / 'result.json').read_text())
            self.assertEqual(result['error'], '')
            self.assertEqual(result['metrics']['full_cycles'], 3)
            args = SimpleNamespace(**manifest)
            metrics = result['metrics']
            baseline.validate_metrics(metrics, args, 'protocol')
            for key, value in [('issued_pixel_edges', 1), ('software_issued_mword_s', -1),
                               ('full_cycles', 9), ('cpu_start', -1), ('sample_count', 0),
                               ('prepared_generations', 2), ('interrupted', True),
                               ('pad_output_enable_applied_mask', 1),
                               ('pad_output_enable_requested', True)]:
                with self.subTest(corrupted_metric=key), self.assertRaises(ValueError):
                    baseline.validate_metrics(dict(metrics, **{key: value}), args, 'protocol')
            # Plane sweeps must not masquerade as scans in a non-protocol mode.
            with self.assertRaises(ValueError):
                baseline.validate_metrics(dict(metrics, mode='alternating'), args, 'alternating')
            # Argument construction only: NEVER execute this hardware command in tests.
            args.output = args.enable_pad_output = True
            forwarded = baseline.benchmark_argv(args, 'clock', run / 'not-created.csv')
            self.assertIn('--enable-pad-output', forwarded)
            self.assertIn('--output', forwarded)
            self.assertIn('--panels-disconnected', forwarded)
            with (dest / 'summary.csv').open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertIn('RAM ONLY', rows[0]['status'])
            self.assertEqual(rows[0]['scope_long_term_mhz'], '')
            # Existing captures must never be silently overwritten.
            again = subprocess.run(argv, text=True, capture_output=True, timeout=15)
            self.assertNotEqual(again.returncode, 0)


if __name__ == '__main__':
    unittest.main()
