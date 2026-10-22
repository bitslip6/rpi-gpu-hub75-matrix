"""Matched direct-C/Python workloads. Offscreen only; never activates GPIO.

python python/benchmarks/build_reference.py
python python/benchmarks/benchmark.py
The reference links the installed package's exact CPU/GPU native library.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import tempfile
import time

import hub75gpu as h


def workload(args):
    batch = h.DrawList().clear()
    raw = None
    if args.workload == "upload":
        raw = bytes(v for i in range(args.width * args.height)
                    for v in ((i*17) % 256, (i*31) % 256, (i*47) % 256, 255))
    elif args.workload == "scene":
        scene = h.Scene3D()
        mesh = h.Mesh.cube()
        mesh.set_color((255, 120, 30))
        scene.add(mesh, angular_velocity=(.3, .7, 0) if args.retained else (0, 0, 0))
        scene.directional_light(shadows=args.shadows)
        batch.render(scene, h.Camera(aspect=args.width / args.height))
    elif args.workload == "text":
        batch.text(h.Text(h.Font(args.font, size=16), "NATIVE HUB75", position=(2, 2), speed=20))
    elif args.workload == "lines":
        for i in range(args.lines):
            batch.line(i % args.width, i*3 % args.height, i*7 % args.width, i*11 % args.height,
                       (i*17 % 256, i*31 % 256, i*47 % 256))
    return batch, raw


def python_run(args, *, unbatched=False):
    batch, raw = workload(args)
    config = h.DisplayConfig(width=args.width, height=args.height, fps=args.fps,
                            queue_depth=args.queue_depth, bit_depth=args.bit_depth, offscreen=True)
    frames = min(50, args.frames) if unbatched else args.frames
    warmup = min(10, args.warmup) if unbatched else args.warmup
    samples = []
    with h.Display(config) as display:
        start = time.perf_counter_ns()
        cpu_start = time.process_time_ns()
        before = display.stats()
        if args.retained or args.workload in ("shader", "video"):
            if args.workload == "shader":
                playback = display.play_shader(args.file)
            elif args.workload == "video":
                playback = display.play_video(args.file, loop=True)
            else:
                playback = display.play(batch)
            playback.ready(10)
            while display.stats()["consumed"] < warmup and playback.playing:
                time.sleep(.001)
            before = display.stats()
            start = time.perf_counter_ns()
            cpu_start = time.process_time_ns()
            while display.stats()["consumed"] < before["consumed"] + frames and playback.playing:
                time.sleep(.001)
            playback.stop(5)
        else:
            for i in range(frames + warmup):
                if i == warmup:
                    display.flush(5)
                    before = display.stats()
                    start = time.perf_counter_ns()
                    cpu_start = time.process_time_ns()
                begin = time.perf_counter_ns()
                with display.frame(1) as frame:
                    if raw is not None:
                        frame.upload(raw)
                    elif unbatched:
                        frame.clear()
                        for n in range(args.lines):
                            frame.line(n % args.width, n*3 % args.height,
                                       n*7 % args.width, n*11 % args.height,
                                       (n*17 % 256, n*31 % 256, n*47 % 256))
                    else:
                        frame.draw(batch, time=i/args.fps)
                if i >= warmup:
                    samples.append(time.perf_counter_ns() - begin)
        display.flush(5)
        elapsed = time.perf_counter_ns() - start
        cpu_elapsed = time.process_time_ns() - cpu_start
        after = display.stats()
        image = display.snapshot()
        shutdown_start = time.perf_counter_ns()
        display.close(5)
        shutdown_ns = time.perf_counter_ns() - shutdown_start
    samples.sort()
    return {"frames": after["consumed"]-before["consumed"], "elapsed_ns": elapsed,
            "cpu_ns": cpu_elapsed, "shutdown_ns": shutdown_ns, "latency_ns": after["latency_ns"],
            "mean_latency_ns": (after["latency_total_ns"]-before["latency_total_ns"])/max(1,after["consumed"]-before["consumed"]),
            **{f"p{q}_ns": samples[(len(samples)-1)*q//100] if samples else 0 for q in (50, 95, 99)},
            **{key: after[key]-before[key] for key in ("render_ns", "queue_wait_ns", "map_ns", "copy_ns", "scan_ns")},
            "sha256": hashlib.sha256(image).hexdigest()}


def native_run(args):
    command = shlex.split(args.native) + ["--offscreen", "--width", str(args.width), "--height", str(args.height),
        "--fps", str(args.fps), "--queue-depth", str(args.queue_depth), "--bit-depth", str(args.bit_depth),
        "--frames", str(args.frames), "--warmup", str(args.warmup), "--lines", str(args.lines)]
    if args.retained:
        command.append("--retained")
    if args.shadows:
        command.append("--shadows")
    if args.workload in ("upload", "scene"):
        command.append("--" + args.workload)
    elif args.workload == "text":
        command += ["--text", str(Path(args.font).resolve())]
    elif args.workload in ("shader", "video"):
        command += ["--" + args.workload, str(Path(args.file).resolve())]
        if args.workload == "video":
            command.append("--loop")
    with tempfile.TemporaryDirectory() as td:
        dump = Path(td) / "frame.rgba"
        output = subprocess.check_output(command + ["--dump", str(dump)], text=True, timeout=300)
        result = json.loads(output.strip().splitlines()[-1])
        result["sha256"] = hashlib.sha256(dump.read_bytes()).hexdigest()
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", default="build/python-reference/example_runtime", help="binary plus optional CLI prefix (quote it)")
    p.add_argument("--workload", choices=("lines", "upload", "scene", "text", "shader", "video"), default="lines")
    p.add_argument("--retained", action="store_true", help="native timed producer instead of manual transactions")
    p.add_argument("--shadows", action="store_true", help="enable native shadow mapping for scene workload")
    p.add_argument("--font")
    p.add_argument("--file")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--fps", type=int, default=120)
    p.add_argument("--bit-depth", type=int, default=64)
    p.add_argument("--queue-depth", type=int, default=4)
    p.add_argument("--lines", type=int, default=2048)
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--compare-unbatched", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.frames <= 0 or args.repeats <= 0 or args.warmup < 0:
        p.error("frames/repeats must be positive, warmup nonnegative")
    if args.shadows and args.workload != "scene":
        p.error("--shadows requires --workload scene")
    if args.workload == "text" and not args.font:
        p.error("text workload requires --font")
    if args.workload in ("shader", "video") and not args.file:
        p.error("media workload requires --file")
    if args.retained and args.workload == "upload":
        p.error("upload workload is manual only")
    pairs = []
    for _ in range(args.repeats):
        native = native_run(args)
        python = python_run(args)
        parity = native["sha256"] == python["sha256"]
        if not args.retained and args.workload not in ("shader", "video") and not parity:
            raise AssertionError(f"native/Python image mismatch: {native['sha256']} != {python['sha256']}")
        c_fps = native["frames"] * 1e9 / native["elapsed_ns"]
        py_fps = python["frames"] * 1e9 / python["elapsed_ns"]
        pairs.append({"native": native, "python": python, "image_parity": parity,
                      "native_fps": c_fps, "python_fps": py_fps,
                      "native_mean_cpu_cores": native["cpu_ns"] / native["elapsed_ns"],
                      "python_mean_cpu_cores": python["cpu_ns"] / python["elapsed_ns"],
                      "throughput_regression_percent": (1 - py_fps/c_fps)*100 if c_fps else None})
    result = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "capabilities": h.capabilities(), "runs": pairs,
              "median_throughput_regression_percent": statistics.median(r["throughput_regression_percent"] for r in pairs)}
    if args.compare_unbatched and args.workload == "lines" and not args.retained:
        unbatched = python_run(args, unbatched=True)
        fast_fps = statistics.median(r["python_fps"] for r in pairs)
        result["unbatched_python"] = unbatched
        result["batch_speedup"] = fast_fps / (unbatched["frames"] * 1e9 / unbatched["elapsed_ns"])
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
