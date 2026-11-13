#!/usr/bin/env python3
"""Sequential RIO baseline collector. Default RAM only; no global system tuning."""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
MODES = ('clock', 'alternating', 'sequence', 'protocol')
CONFIG_KEYS = ('PREEMPT_RT', 'NO_HZ_FULL', 'NO_HZ_IDLE', 'RCU_NOCB_CPU',
               'IRQ_FORCED_THREADING', 'CPU_ISOLATION', 'HZ', 'FTRACE', 'IRQSOFF_TRACER')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')


def read_text(path):
    try:
        return {'text': Path(path).read_text()}
    except OSError as exc:
        return {'error': str(exc)}


def command(argv):
    try:
        result = subprocess.run(argv, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=15)
        return {'exit': result.returncode, 'text': result.stdout}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'exit': -1, 'error': str(exc)}


def parse_interrupts(text):
    """Parse per-CPU /proc/interrupts or /proc/softirqs; ignore non-per-CPU totals."""
    lines = text.splitlines()
    if not lines:
        raise ValueError('empty interrupt table')
    cpus = lines[0].split()
    if not cpus or any(not re.fullmatch(r'CPU\d+', cpu) for cpu in cpus):
        raise ValueError('missing CPU header')
    result = {}
    for line in lines[1:]:
        if ':' not in line:
            continue
        key, rest = line.split(':', 1)
        fields = rest.split()
        if len(fields) < len(cpus) or not all(v.isdigit() for v in fields[:len(cpus)]):
            continue
        result[key.strip()] = {'counts': dict(zip(cpus, map(int, fields[:len(cpus)]))),
                               'description': ' '.join(fields[len(cpus):])}
    return result


def interrupt_delta(before, after, cpu):
    if 'text' not in before or 'text' not in after:
        return {'error': 'interrupt snapshots unavailable', 'before': before, 'after': after}
    try:
        old, new = parse_interrupts(before['text']), parse_interrupts(after['text'])
    except ValueError as exc:
        return {'error': str(exc)}
    target = f'CPU{cpu}'
    rows, changed = {}, []
    for key in sorted(old.keys() | new.keys()):
        if key not in old or key not in new:
            changed.append(key)
            continue
        a, b = old[key], new[key]
        if a['description'] != b['description'] or a['counts'].keys() != b['counts'].keys():
            changed.append(key)
            continue
        counts = {name: b['counts'][name] - count for name, count in a['counts'].items()}
        if any(value < 0 for value in counts.values()):
            changed.append(key)
            continue
        rows[key] = {'delta': counts, 'description': b['description']}
    if not any(target in row['delta'] for row in rows.values()):
        return {'error': f'{target} not found in comparable interrupt rows', 'changed_rows': changed}
    return {'cpu': target, 'cpu_total_delta': sum(row['delta'].get(target, 0) for row in rows.values()),
            'changed_rows': changed, 'rows': rows,
            'note': 'Includes IRQ/IPI counts, not interrupt durations or proof of isolation.'}


def config_flags(text):
    flags = {}
    for key in CONFIG_KEYS:
        found = re.search(rf'^CONFIG_{key}=(\S+)$', text, re.M)
        flags[key] = found.group(1) if found else (
            'n' if re.search(rf'^# CONFIG_{key} is not set$', text, re.M) else 'unknown')
    return flags


def save_kernel_config(dest):
    errors = {}
    for source in (Path('/proc/config.gz'), Path('/boot') / ('config-' + platform.release())):
        try:
            raw = source.read_bytes()
            text = gzip.decompress(raw).decode() if source.suffix == '.gz' else raw.decode()
            (dest / 'kernel-config.raw').write_bytes(raw)
            (dest / 'kernel-config.txt').write_text(text)
            return {'source': str(source), 'flags': config_flags(text), 'earlier_errors': errors}
        except (OSError, UnicodeError, gzip.BadGzipFile, EOFError) as exc:
            errors[str(source)] = str(exc)
    return {'error': 'exact running kernel config unavailable', 'attempts': errors}


def system_snapshot(dest, cpu):
    dest.mkdir()
    sources = {
        'interrupts': '/proc/interrupts', 'softirqs': '/proc/softirqs', 'stat': '/proc/stat',
        'cmdline': '/proc/cmdline', 'rt-runtime-us': '/proc/sys/kernel/sched_rt_runtime_us',
        'rt-period-us': '/proc/sys/kernel/sched_rt_period_us',
        'isolated': '/sys/devices/system/cpu/isolated',
        'nohz-full': '/sys/devices/system/cpu/nohz_full',
        'irq-default-affinity': '/proc/irq/default_smp_affinity',
        'unbound-workqueue-cpumask': '/sys/devices/virtual/workqueue/cpumask',
        'watchdog-cpumask': '/proc/sys/kernel/watchdog_cpumask',
        'governor': f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor',
        'frequency': f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq',
        'temperature': '/sys/class/thermal/thermal_zone0/temp',
    }
    results = {name: read_text(path) for name, path in sources.items()}
    for name, result in results.items():
        write_json(dest / (name + '.json'), result)
    write_json(dest / 'throttled.json', command(['vcgencmd', 'get_throttled']))
    write_json(dest / 'arm-clock.json', command(['vcgencmd', 'measure_clock', 'arm']))
    write_json(dest / 'dmesg.json', command(['dmesg', '--color=never']))
    write_json(dest / 'pci.json', command(['lspci', '-D', '-vv']))
    isolation = command([sys.executable, str(ROOT / 'tests/rio_isolate.py'),
                         'audit', '--cpu', str(cpu), '--json'])
    if isolation.get('exit') == 0:
        try:
            isolation = {'exit': 0, 'audit': json.loads(isolation['text'])}
        except (ValueError, KeyError) as exc:
            isolation = dict(isolation, error=f'invalid isolation audit: {exc}')
    write_json(dest / 'isolation.json', isolation)
    write_json(dest / 'thread-priorities.json',
               command(['ps', '-eLo', 'pid,tid,psr,cls,rtprio,pri,ni,comm', '--sort=-rtprio']))
    return results


def telemetry(pid, cpu):
    return {'monotonic_ns': time.monotonic_ns(),
            'leader_stat': read_text(f'/proc/{pid}/stat'),
            'leader_status': read_text(f'/proc/{pid}/status'),
            'frequency_khz': read_text(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq'),
            'temperature_millidegrees': read_text('/sys/class/thermal/thermal_zone0/temp'),
            'throttled': command(['vcgencmd', 'get_throttled']),
            'arm_clock_hz': command(['vcgencmd', 'measure_clock', 'arm'])}


def validate_metrics(metrics, args, mode):
    """Never accept plausible-looking rates without matching workload/counter labels."""
    if not isinstance(metrics, dict):
        raise ValueError('summary must be a JSON object')
    expected = {'schema': 1, 'backend': 'rio-mmio' if args.output else 'ram-dry-run',
                'mode': mode, 'store': args.store, 'scheduler': args.scheduler,
                'priority': args.priority if args.scheduler == 'fifo' else 0,
                'width': args.width, 'scanned_rows': args.rows, 'planes': args.planes,
                'delay_nops': args.delay_nops, 'cpu_requested': args.cpu,
                'controller_cpu': args.controller_cpu, 'emitter_affinity': [args.cpu],
                'cpu_start': args.cpu, 'cpu_end': args.cpu, 'mlocked': args.mlock,
                'scope_verified': False, 'rgb_timing_verified': False,
                'renderer_content_frames': 0, 'interrupted': False, 'underruns': 0,
                'pad_output_enable_requested': args.enable_pad_output}
    for name, value in expected.items():
        if metrics.get(name) != value:
            raise ValueError(f'{name}: expected {value!r}, received {metrics.get(name)!r}')
    integer_keys = ('workload_cycles', 'elapsed_ns', 'emit_total_ns', 'service_total_ns',
                    'issued_pixel_edges', 'words_per_cycle', 'full_cycles', 'plane_sweeps',
                    'prepared_generations', 'repeats', 'minor_faults', 'major_faults',
                    'sample_every', 'sample_count', 'samples_dropped', 'emit_min_ns',
                    'emit_max_observed_ns', 'sample_emit_p50_ns', 'sample_emit_p99_ns',
                    'service_boundaries', 'emitter_user_ns', 'emitter_system_ns',
                    'voluntary_switches', 'involuntary_switches',
                    'pad_output_enable_applied_mask', 'owned_mask')
    for name in integer_keys:
        if type(metrics.get(name)) is not int or metrics[name] < 0:
            raise ValueError(f'invalid integer metric {name}')
    cycles = metrics['workload_cycles']
    words = args.width * args.rows * args.planes
    if not cycles or not metrics['elapsed_ns'] or metrics['words_per_cycle'] != words:
        raise ValueError('missing cycles/time or wrong pixel count')
    if metrics['issued_pixel_edges'] != cycles * words:
        raise ValueError('issued edge count mismatch')
    if metrics['full_cycles'] != (cycles if mode == 'protocol' else 0):
        raise ValueError('non-protocol work mislabeled as complete scans')
    if metrics['plane_sweeps'] != (cycles * args.planes if mode == 'protocol' else 0):
        raise ValueError('plane sweep count mismatch')
    if metrics['prepared_generations'] != 1 or metrics['repeats'] != cycles - 1:
        raise ValueError('static schedule generation/repeat mismatch')
    if metrics['emit_total_ns'] + metrics['service_total_ns'] != metrics['elapsed_ns']:
        raise ValueError('emission/service accounting mismatch')
    rate = metrics.get('software_issued_mword_s')
    if type(rate) not in (int, float) or not math.isfinite(rate) or \
            abs(rate - cycles * words * 1000 / metrics['elapsed_ns']) > 0.000002:
        raise ValueError('software rate does not match issued edges / measured wall time')
    if metrics.get('service_boundaries') != cycles:
        raise ValueError('boundary count mismatch')
    expected_samples = (cycles - 1) // args.sample_every + 1
    if metrics['sample_every'] != args.sample_every or \
            metrics['sample_count'] != min(expected_samples, 65536) or \
            metrics['samples_dropped'] != max(0, expected_samples - 65536):
        raise ValueError('sample count/stride mismatch')
    if not metrics['emit_min_ns'] <= metrics['sample_emit_p50_ns'] <= \
            metrics['sample_emit_p99_ns'] <= metrics['emit_max_observed_ns']:
        raise ValueError('emission duration quantiles outside observed bounds')
    if args.cycles is None and metrics['elapsed_ns'] < args.seconds * 10**9:
        raise ValueError('timed run ended early')
    if args.cycles is not None and cycles > args.cycles:
        raise ValueError('cycle cap exceeded')
    applied = metrics['pad_output_enable_applied_mask']
    if applied & ~metrics['owned_mask'] or (applied and not args.enable_pad_output):
        raise ValueError('unacknowledged or unowned pad output-enable change')
    if not args.output and applied:
        raise ValueError('RAM run reported pad changes')
    if args.output and metrics.get('blank_readback_ok') is not True:
        raise ValueError('blanking or restoration readback failed')


def read_run_summary(run, rc, args, mode):
    """Keep the process failure/diagnostic primary, even if stdout has no JSON."""
    metrics, error = {}, ''
    try:
        stdout = (run / 'stdout.json').read_text()
        if not stdout.strip():
            raise ValueError('missing JSON summary')
        metrics = json.loads(stdout, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
        if not isinstance(metrics, dict):
            raise ValueError('summary must be a JSON object')
        if not rc:
            validate_metrics(metrics, args, mode)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        error = str(exc)
        if not isinstance(metrics, dict):
            metrics = {}
    if rc:
        try:
            diagnostic = (run / 'stderr.log').read_text(errors='replace').strip()
        except OSError:
            diagnostic = ''
        error = f'benchmark exit {rc}'
        if diagnostic:
            error += ': ' + diagnostic[-2048:]
    return metrics, error


def benchmark_argv(args, mode, samples):
    argv = [str(args.exe), '--mode', mode, '--store', args.store, '--seconds', str(args.seconds),
            '--width', str(args.width), '--rows', str(args.rows), '--planes', str(args.planes),
            '--delay-nops', str(args.delay_nops), '--cpu', str(args.cpu),
            '--controller-cpu', str(args.controller_cpu), '--scheduler', args.scheduler,
            '--priority', str(args.priority), '--sample-every', str(args.sample_every),
            '--samples', str(samples)]
    if args.cycles:
        argv += ['--cycles', str(args.cycles)]
    if args.mlock:
        argv += ['--mlock']
    if args.output:
        argv += ['--output', '--panels-disconnected']
        if args.exclusive_bank:
            argv += ['--exclusive-bank']
        if args.enable_pad_output:
            argv += ['--enable-pad-output']
    return argv


def run_child(argv, run, cpu):
    with (run / 'stdout.json').open('w') as out, (run / 'stderr.log').open('w') as err, \
            (run / 'telemetry.jsonl').open('w') as log:
        proc = subprocess.Popen(argv, cwd=ROOT, stdout=out, stderr=err, start_new_session=True)
        try:
            while proc.poll() is None:
                log.write(json.dumps(telemetry(proc.pid, cpu)) + '\n')
                log.flush()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            # Also stop on disk/telemetry failures; do not orphan a live emitter.
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f'PID {proc.pid} did not stop: manual recovery required. NOT sending SIGKILL.',
                      file=sys.stderr)
            raise
        return proc.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', action='store_true', help='interactive scope only; panels disconnected')
    parser.add_argument('--exclusive-bank', action='store_true', help='required with hardware --store out')
    parser.add_argument('--enable-pad-output', action='store_true',
                        help='explicitly clear owned pad OD bits only; restore them on cleanup')
    parser.add_argument('--seconds', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cycles', type=int, help='optional smoke-test cap, not a timed baseline')
    parser.add_argument('--modes', choices=MODES, nargs='+', default=list(MODES))
    parser.add_argument('--store', choices=('xor', 'out'), default='xor')
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--rows', type=int, default=32)
    parser.add_argument('--planes', type=int, default=8)
    parser.add_argument('--delay-nops', type=int, default=8)
    parser.add_argument('--sample-every', type=int, default=64)
    parser.add_argument('--cpu', type=int, default=3)
    parser.add_argument('--controller-cpu', type=int, default=0)
    parser.add_argument('--scheduler', choices=('other', 'fifo'), default='other')
    parser.add_argument('--priority', type=int, default=80)
    parser.add_argument('--mlock', action='store_true')
    parser.add_argument('--ack-rt-throttling', action='store_true',
                        help='explicitly acknowledge a free-running FIFO diagnostic and current RT limits')
    parser.add_argument('--results', type=Path, help='new directory only')
    parser.add_argument('--exe', type=Path, default=ROOT / 'build/release/rio_scope')
    parser.add_argument('--skip-build', action='store_true', help='use/hash existing executable')
    parser.add_argument('--notes', default='', help='load, scope/probe setup, and experiment purpose')
    args = parser.parse_args(argv)
    limits = ((args.seconds, 1, 3600), (args.repeats, 1, 100), (args.width, 1, 4096),
              (args.rows, 1, 32), (args.planes, 1, 64), (args.delay_nops, 0, 1024),
              (args.sample_every, 1, 10**9), (args.cpu, 0, 1023),
              (args.controller_cpu, 0, 1023), (args.priority, 1, 90))
    if any(not low <= value <= high for value, low, high in limits) or \
            (args.cycles is not None and not 1 <= args.cycles <= 10**9):
        parser.error('numeric option outside benchmark limits')
    if len(args.modes) != len(set(args.modes)):
        parser.error('duplicate modes would overwrite run names')
    if 'protocol' in args.modes and args.rows & (args.rows - 1):
        parser.error('protocol rows must be a power of two')
    if args.cpu == args.controller_cpu:
        parser.error('collector/signal CPU must differ from emitter CPU')
    if args.enable_pad_output and not args.output:
        parser.error('--enable-pad-output requires --output')
    if args.scheduler == 'fifo' and not args.ack_rt_throttling:
        runtime = read_text('/proc/sys/kernel/sched_rt_runtime_us')
        period = read_text('/proc/sys/kernel/sched_rt_period_us')
        parser.error('free-running FIFO can be RT-throttled or starve dependencies; '
                     f'current runtime={runtime}, period={period}. '
                     'No RT limits will be changed. Review the budget and use '
                     '--ack-rt-throttling only for an intentional diagnostic')
    if args.output:
        if args.enable_pad_output:
            print('Explicit pad-output enable: owned OD bits only; drive/slew/pulls unchanged. '
                  'Scope CLK/OE first.', flush=True)
        if args.store == 'out' and not args.exclusive_bank:
            parser.error('hardware Out needs --exclusive-bank: GPIO0/1 and all bank users must be quiescent')
        if not sys.stdin.isatty():
            parser.error('--output requires an interactive terminal')
        if input('Panels disconnected, fixed HZELLER pinout, no GPIO/PIO users, recovery ready? Type YES: ') != 'YES':
            return 1
    args.exe = args.exe.resolve()
    if not args.skip_build and args.exe != ROOT / 'build/release/rio_scope':
        parser.error('custom --exe requires --skip-build')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    dest = (args.results or ROOT / 'tests/results' / (stamp + '-rio')).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    original_affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {args.controller_cpu})
    provenance = {'uname': command(['uname', '-a']), 'compiler': command(['cc', '--version']),
                  'git-head': command(['git', 'rev-parse', 'HEAD']),
                  'git-status': command(['git', 'status', '--short']),
                  'git-diff': command(['git', 'diff']),
                  'firmware': command(['vcgencmd', 'version']),
                  'kernel-config': save_kernel_config(dest)}
    write_json(dest / 'provenance.json', provenance)
    if not args.skip_build:
        build = command(['make', '-B', 'BUILD=release', 'V=1', 'rio-scope'])
        write_json(dest / 'build.json', build)
        if build['exit']:
            return 1
    sources = ['Makefile', 'tests/rio_scope.c', 'tests/rio_scope_io.h', 'tests/rio_scope_lock.h',
               'tests/rio_baseline.py', 'tests/rio_isolate.py', 'tests/README-rio-isolate.md',
               'tests/rio_environment.py', 'tests/README-rio-environment.md',
               'configure-system.py', 'tests/README-configure-system.md',
               'tests/README-rio-scope.md', 'tests/RIO_USERSPACE_PLAN.md',
               'include/rio_scan.h', 'src/rio_scan.c']
    hashes = {}
    for relative in sources:
        data = (ROOT / relative).read_bytes()
        target = dest / 'sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[relative] = hashlib.sha256(data).hexdigest()
    exe_bytes = args.exe.read_bytes()
    captured_exe = dest / 'rio_scope.executable'
    captured_exe.write_bytes(exe_bytes)
    captured_exe.chmod(0o755)
    write_json(dest / 'manifest.json', dict(vars(args), timestamp_utc=stamp,
               original_collector_affinity=original_affinity,
               actual_collector_affinity=sorted(os.sched_getaffinity(0)),
               executable_sha256=hashlib.sha256(exe_bytes).hexdigest(), sources_sha256=hashes,
               status='scope/RGB/IRQ isolation unverified; RAM is not a RIO baseline',
               build_skipped=args.skip_build, telemetry_interval_seconds=1))
    # Execute the captured bytes, not a build artifact that could be replaced
    # during a long soak. Manifest retains the original executable path/hash.
    args.exe = captured_exe
    print(f'Results: {dest}', flush=True)
    fields = ['run', 'mode', 'backend', 'store', 'exit', 'status', 'software_issued_mword_s',
              'workload_cycles', 'full_cycles', 'plane_sweeps', 'repeats', 'minor_faults',
              'major_faults', 'emit_max_observed_ns', 'sample_emit_p99_ns', 'cpu_irq_ipi_delta',
              'scope_long_term_mhz', 'scope_max_gap_ns', 'scope_capture', 'notes']
    with (dest / 'summary.csv').open('w', newline='') as summary:
        writer = csv.DictWriter(summary, fieldnames=fields)
        writer.writeheader()
        summary.flush()
        for repeat in range(1, args.repeats + 1):
            for mode in args.modes:
                name = f'{mode}-{args.store}-r{repeat}'
                run = dest / name
                run.mkdir()
                if args.output:
                    input(f'Prepare scope for {name}; press Enter to drive pins (Ctrl-C stops): ')
                before = system_snapshot(run / 'before', args.cpu)
                child_argv = benchmark_argv(args, mode, run / 'cycles.csv')
                write_json(run / 'command.json', child_argv)
                started = time.monotonic_ns()
                started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
                print(f'Running {name}', flush=True)
                rc = run_child(child_argv, run, args.cpu)
                wall_ns = time.monotonic_ns() - started
                after = system_snapshot(run / 'after', args.cpu)
                irq = interrupt_delta(before['interrupts'], after['interrupts'], args.cpu)
                softirq = interrupt_delta(before['softirqs'], after['softirqs'], args.cpu)
                write_json(run / 'interrupt-deltas.json', {'irq_ipi': irq, 'softirq': softirq})
                metrics, error = read_run_summary(run, rc, args, mode)
                notes = args.notes
                if args.output and not error:
                    notes += '; ' + input('Scope readings/capture filenames (blank if unavailable): ')
                row = {key: metrics.get(key, '') for key in fields}
                row.update(run=name, mode=mode, store=args.store, exit=rc,
                           status='INVALID: ' + error if error else (
                               'scope/data/isolation unverified' if args.output else 'RAM ONLY, not RIO throughput'),
                           cpu_irq_ipi_delta=irq.get('cpu_total_delta', ''), notes=notes)
                writer.writerow(row)
                summary.flush()
                write_json(run / 'result.json', {'exit': rc, 'wall_ns': wall_ns, 'error': error,
                                                'timestamp_utc': started_utc,
                                                'started_monotonic_ns': started,
                                                'metrics': metrics, 'notes': notes})
                if error:
                    print(f'Stopping on invalid run: {error}. Inspect {run}', file=sys.stderr)
                    return 1
    print('Complete. Add raw scope captures; software counts do not verify physical timing or RGB.')
    return 0


def stop_collector(signum, frame):
    del signum, frame
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop_collector)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nInterrupted: partial run is INVALID; suite stopped.', file=sys.stderr)
        sys.exit(130)
    except OSError as exc:
        print(f'Collector failed (no further runs): {exc}', file=sys.stderr)
        sys.exit(1)
