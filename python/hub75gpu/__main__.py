"""Safe-by-default demo: python -m hub75gpu --help."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import hub75gpu as h


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("cpu", "retained", "scene", "text", "shader", "video"), default="retained")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--ports", type=int, default=1)
    p.add_argument("--chains", type=int, default=1)
    p.add_argument("--panel-width", type=int, default=64)
    p.add_argument("--panel-height", type=int, default=64)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--seconds", type=float, default=2)
    p.add_argument("--file", type=Path, help="external shader or local video")
    p.add_argument("--font", type=Path, help="external SDF font directory")
    p.add_argument("--output", type=Path, help="save final raw RGBA8 snapshot")
    p.add_argument("--gpio", action="store_true", help="activate physical output (default: offscreen)")
    p.add_argument("--acknowledge-wiring", action="store_true")
    args = p.parse_args()
    if args.gpio and not args.acknowledge_wiring:
        p.error("--gpio requires --acknowledge-wiring; check panel voltage/pins/topology first")
    if args.mode in ("shader", "video") and args.file is None:
        p.error("--shader/--video modes require --file")
    if args.mode == "text" and args.font is None:
        p.error("text mode requires --font")
    if not 0 < args.seconds <= 86400:
        p.error("--seconds must be in (0, 86400]")
    config = h.DisplayConfig(width=args.width, height=args.height, ports=args.ports,
                            chains=args.chains, panel_width=args.panel_width,
                            panel_height=args.panel_height, fps=args.fps,
                            brightness=64, offscreen=not args.gpio)
    batch = h.DrawList().clear()
    batch.gradient(((0, 0), (1, 0), (1, 1), (0, 1)), (0, 0, 60), (80, 0, 0))
    batch.lines(((0, 0, args.width-1, args.height-1), (0, args.height-1, args.width-1, 0)),
                (255, 100, 20), antialias=True)
    if args.mode == "scene":
        scene = h.Scene3D()
        mesh = h.Mesh.cube()
        mesh.set_color((255, 110, 40))
        scene.add(mesh, angular_velocity=(.3, .8, .2))
        scene.directional_light()
        batch = h.DrawList().clear().render(scene, h.Camera(aspect=args.width/args.height))
    elif args.mode == "text":
        assert args.font is not None
        text = h.Text(h.Font(args.font, size=16), "NATIVE HUB75", position=(args.width, 2), speed=25)
        text.sine = (2, 2, 1)
        batch = h.DrawList().clear().text(text)
    print(h.capabilities())
    try:
        with h.Display(config) as display:
            if args.mode == "cpu":
                end = time.monotonic() + args.seconds
                while time.monotonic() < end:
                    with display.frame(timeout=1) as frame:
                        frame.draw(batch)
                    time.sleep(1 / args.fps)
            else:
                if args.mode == "shader":
                    assert args.file is not None
                    playback = display.play_shader(args.file)
                elif args.mode == "video":
                    assert args.file is not None
                    playback = display.play_video(args.file, loop=True)
                else:
                    playback = display.play(batch)
                playback.ready(10)
                time.sleep(args.seconds)  # no Python frame callbacks
                playback.stop(5)
            if args.output:
                args.output.write_bytes(display.snapshot())
            print(display.stats())
    except KeyboardInterrupt:
        pass  # context-manager shutdown has already joined native workers


if __name__ == "__main__":
    main()
