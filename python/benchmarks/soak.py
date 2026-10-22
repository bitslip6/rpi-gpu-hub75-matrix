#!/usr/bin/env python3
"""Bounded-resource lifecycle/content soak. Offscreen only; never opens GPIO."""
from __future__ import annotations

import argparse
from array import array
from collections import deque
import gc
import json
import math
import os
from pathlib import Path
import platform
import struct
import tempfile
import time
import zlib

import hub75gpu as h


def fixture(directory: Path) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    (directory / "A.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\0" + b"\xff" * 8) * 8)) + chunk(b"IEND", b""))
    (directory / "metrics.csv").write_text(
        "char,advance_width,bearing_x,bearing_y,width,height,path\nA,8,0,0,8,8,A.png\n")


def resources() -> dict[str, int]:
    return {"rss_bytes": int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE"),
            "fds": len(list(Path("/proc/self/fd").iterdir())),
            "threads": len(list(Path("/proc/self/task").iterdir()))}


def cycle(index: int, font: h.Font | None, shader: str | None, video: str | None) -> dict:
    with h.Display(h.DisplayConfig(width=64, height=64, offscreen=True, queue_depth=2)) as d:
        with h.Image(8, 8, bytes((32, 64, 128, 255)) * 64) as image, h.Mesh.cube() as mesh, \
                h.Camera() as camera, h.Scene3D() as scene, h.DrawList() as batch:
            instance = scene.add(mesh, angular_velocity=(.3, .5, .1))
            scene.directional_light(shadows=index % 4 == 2)
            batch.clear().render(scene, camera).blit(image, x=56, y=56)
            text = h.Text(font, "AAAA", position=(2, 2)) if font else None
            if text:
                text.speed = 15
                batch.text(text)
            # Exercise native ownership after closure of the Python mesh/image wrappers.
            mesh.close()
            image.close()
            if index % 3 == 0:
                for i in range(4):
                    with d.frame() as frame:
                        if i & 1:
                            frame.upload(bytes((index % 256, 7, 11, 255)) * (64 * 64))
                        else:
                            frame.draw(batch, time=i / 120)
            else:
                if shader and index % 3 == 1:
                    playback = d.play_shader(shader, overlay=batch)
                elif video and index % 3 == 2:
                    playback = d.play_video(video, loop=True, overlay=batch)
                else:
                    playback = d.play(batch)
                playback.ready(5)
                for i in range(3):
                    instance.transform.rotation = (i * .01, i * .02, 0)
                    scene.set_transforms(array("f", [0, 0, 0, 0, i * .01, 0, 1, 1, 1]))
                    if text:
                        text.weight = 1 + i * .05
                    time.sleep(.02)  # Native rendering/pacing continues while Python is idle.
                playback.pause()
                playback.resume()
                playback.stop(5)
            d.flush(5)
            result = d.stats()
            if text:
                text.close()
            instance.transform.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--rss-growth-mib", type=float, default=64)
    parser.add_argument("--shader", help="optional local shader; requires GPU build")
    parser.add_argument("--video", help="optional local video; requires GPU build")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not math.isfinite(args.rss_growth_mib) or args.seconds <= 0 or args.rss_growth_mib < 0:
        parser.error("duration must be positive and RSS allowance nonnegative")
    for mode in ("shader", "video"):
        if getattr(args, mode) and not h.capabilities()[mode]:
            parser.error(f"{mode} requires an appropriate GPU installation")
    samples: deque[dict] = deque(maxlen=4096)
    started = time.monotonic()
    baseline = None
    peaks = {"rss_bytes": 0, "fds": 0, "threads": 0}
    totals = {"submitted": 0, "consumed": 0, "dropped": 0}
    index = 0
    with tempfile.TemporaryDirectory(prefix="hub75-soak-") as td:
        directory = Path(td)
        fixture(directory)
        font = h.Font(directory, size=8) if h.capabilities()["text"] else None
        try:
            while index < 4 or time.monotonic() - started < args.seconds:
                stats = cycle(index, font, args.shader, args.video)
                gc.collect()
                current = resources()
                if index == 3:
                    baseline = current.copy()
                if baseline:
                    for key in peaks:
                        peaks[key] = max(peaks[key], current[key])
                    if current["fds"] > baseline["fds"] + 2 or current["threads"] > baseline["threads"] + 2:
                        raise AssertionError(f"descriptor/thread growth: {baseline} -> {current}")
                    if current["rss_bytes"] > baseline["rss_bytes"] + args.rss_growth_mib * 1024**2:
                        raise AssertionError(f"RSS growth beyond allowance: {baseline} -> {current}")
                    elapsed = time.monotonic() - started
                    if not samples or elapsed - samples[-1]["seconds"] >= 1:
                        samples.append({"seconds": elapsed, "cycles": index + 1, **current})
                for key in totals:
                    totals[key] += stats[key]
                index += 1
        finally:
            if font:
                font.close()
    report = {"platform": platform.platform(), "python": platform.python_version(),
              "capabilities": h.capabilities(), "offscreen": True,
              "seconds": time.monotonic() - started, "cycles": index,
              "baseline_after_four_cycles": baseline, "peaks_after_warmup": peaks,
              "final": resources(), "totals": totals, "samples_tail": list(samples)}
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
