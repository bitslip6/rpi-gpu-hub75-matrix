#!/usr/bin/env python3
"""Sequential baseline capture; never changes kernel, DMACTRL, or boot files."""
import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
ERRORS = re.compile(r"non-idle|failed to stop|DMA bounce interrupted|underflow|overrun", re.I)


def capture(argv):
    proc = subprocess.run(argv, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', action='store_true', help='manual GPIO scope runs; prompts before every run')
    parser.add_argument('--seconds', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--chunks', type=int, nargs='+', default=[256], choices=[64, 256, 1024])
    parser.add_argument('--results', type=Path, help='new directory only')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 3600 or args.repeats < 1:
        parser.error('seconds must be 1..3600 and repeats must be positive')
    if args.output:
        if not sys.stdin.isatty():
            parser.error('--output requires an interactive terminal')
        if input('Only scope connected, panels disconnected, no other GPIO/PIO application? Type YES: ') != 'YES':
            return 1
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dest = args.results or ROOT / 'tests/results' / (stamp + ('-gpio' if args.output else '-no-pins'))
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    commands = {
        'uname': ['uname', '-a'], 'module': ['modinfo', 'rp1_pio'],
        'git-status': ['git', 'status', '--short'], 'git-head': ['git', 'rev-parse', 'HEAD'],
        'git-diff': ['git', 'diff'], 'compiler': ['cc', '--version'],
        'firmware': ['vcgencmd', 'version'],
    }
    for name, command in commands.items():
        try:
            rc, text = capture(command)
        except OSError as exc:
            rc, text = -1, str(exc)
        (dest / (name + '.txt')).write_text(f'exit={rc}\n{text}')
    for name, path in {
        'clk_sys.txt': '/sys/kernel/debug/clk/clk_sys/clk_rate',
        'config.gz': '/proc/config.gz',
        'firmware-git-hash.txt': '/boot/firmware/.firmware_revision',
        'device-tree.dtb': '/sys/firmware/fdt',
    }.items():
        try:
            (dest / name).write_bytes(Path(path).read_bytes())
        except OSError as exc:
            (dest / (name + '.unavailable.txt')).write_text(str(exc))
    rc, before = capture(['dmesg', '--color=never'])
    (dest / 'dmesg-start.txt').write_text(before)
    if rc:
        print(f'Cannot read dmesg; stopping. See {dest}', file=sys.stderr)
        return 1
    rc, text = capture(['make', 'BUILD=release', 'pio-hub75-scope'])
    (dest / 'build.txt').write_text(text)
    if rc:
        return rc
    exe = ROOT / 'build/release/pio_hub75_scope'
    source = ROOT / 'tests/pio_hub75_scope.c'
    (dest / source.name).write_bytes(source.read_bytes())
    metadata = vars(args).copy()
    metadata.update(timestamp_utc=stamp, executable=str(exe),
                    executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    burst_verification='UNVERIFIED: requires matching kernel/controller source',
                    data_correctness='UNVERIFIED: GPIO17-only scope cannot check RGB')
    (dest / 'manifest.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    # Preserve the actual executable as well as its hash, not an installed copy.
    (dest / exe.name).write_bytes(exe.read_bytes())
    fields = ['run', 'timestamp_utc', 'mode', 'pins', 'divider', 'instructions', 'chunk_kib',
              'seconds_requested', 'wall_seconds_including_drain', 'cpu_user_seconds',
              'cpu_system_seconds', 'submission_mword_s', 'dmactrl', 'exit', 'status',
              'scope_long_term_mhz', 'scope_min_interval_ns', 'scope_p99_interval_ns',
              'scope_max_gap_ns', 'scope_pulse_width_ns', 'scope_capture', 'notes']
    print(f'Results: {dest}', flush=True)
    with (dest / 'summary.csv').open('w', newline='') as summary:
        writer = csv.DictWriter(summary, fieldnames=fields)
        writer.writeheader()
        summary.flush()
        for chunk in args.chunks:
            for repeat in range(1, args.repeats + 1):
                for light, divider in [(False, 8), (False, 4), (True, 8), (True, 4)]:
                    mode = 'light' if light else 'heavy-default'
                    name = f'{chunk}KiB-{mode}-div{divider}-r{repeat}'
                    argv = [str(exe), str(args.seconds), str(divider), str(chunk)]
                    if light:
                        argv.append('--light-dma')
                    if args.output:
                        argv.append('--output')
                        input(f'Prepare CH1 GPIO17 for {name}; press Enter to run (Ctrl-C stops suite): ')
                    (dest / (name + '.command.json')).write_text(json.dumps(argv) + '\n')
                    print(f'Running {name}', flush=True)
                    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                    started = time.monotonic()
                    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
                    # No forced timeout/SIGKILL: abruptly stopping DMA can leave the channel stalled.
                    with (dest / (name + '.log')).open('w') as log:
                        proc = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    elapsed = time.monotonic() - started
                    after_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                    text = (dest / (name + '.log')).read_text()
                    print(text, end='', flush=True)
                    rc, after = capture(['dmesg', '--color=never'])
                    (dest / (name + '.dmesg.txt')).write_text(after)
                    # Detect ring-buffer truncation rather than silently missing new errors.
                    complete = after.startswith(before)
                    delta = after[len(before):] if complete else after
                    (dest / (name + '.dmesg-new.txt')).write_text(delta)
                    bad = bool(rc or not complete or ERRORS.search(delta))
                    rate = re.search(r'Total: ([0-9.]+) Mword/s', text)
                    ctrl = re.search(r'TX DMACTRL: (0x[0-9a-fA-F]+)', text)
                    valid = proc.returncode == 0 and not bad and bool(rate) and bool(ctrl)
                    row = dict(run=name, timestamp_utc=timestamp, mode=mode,
                               pins='output' if args.output else 'none', divider=divider,
                               instructions=3, chunk_kib=chunk, seconds_requested=args.seconds,
                               wall_seconds_including_drain=f'{elapsed:.6f}',
                               cpu_user_seconds=f'{after_usage.ru_utime - usage.ru_utime:.6f}',
                               cpu_system_seconds=f'{after_usage.ru_stime - usage.ru_stime:.6f}',
                               submission_mword_s=rate.group(1) if rate else '',
                               dmactrl=ctrl.group(1) if ctrl else '', exit=proc.returncode,
                               status='submission-only; scope/data unverified' if valid else 'INVALID',
                               notes='burst unverified; RGB correctness unverified')
                    if args.output and valid:
                        row['notes'] += '; ' + input('Scope readings/capture filenames/notes (or leave blank): ')
                    writer.writerow(row)
                    summary.flush()
                    if not valid:
                        print('Stopping: failed run, missing metrics, kernel error, or incomplete kernel log. '
                              'Inspect logs; do not reuse these timings as baseline.', file=sys.stderr)
                        return 1
                    before = after
    print(f'Completed. Submission throughput only; add manual scope readings to {dest / "summary.csv"}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nInterrupted: suite stopped; inspect dmesg before another run. Partial run is invalid.', file=sys.stderr)
        sys.exit(130)
