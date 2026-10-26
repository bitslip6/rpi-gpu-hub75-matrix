"""Safe-by-default demo: python -m hub75gpu --help."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import hub75gpu as h


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("cpu", "retained", "scene", "text", "shader", "video"),
                   help="content source (default: retained)")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--ports", type=int, default=1)
    p.add_argument("--chains", type=int, default=1)
    p.add_argument("--panel-width", type=int, default=64)
    p.add_argument("--panel-height", type=int, default=64)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--output-mode", choices=("equal-slot", "weighted-bcm"), default="equal-slot")
    p.add_argument("--pwm-bits", type=int, default=11)
    p.add_argument("--pwm-lsb-ns", type=int, default=130)
    p.add_argument("--color-curve", choices=("cie", "gamma"), default="cie")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--brightness", type=int, default=64, help="overall brightness, 0–254 (default: 64)")
    p.add_argument("--seconds", type=float, default=2)
    p.add_argument("--file", type=Path, help="external shader or local video")
    p.add_argument("--video", type=Path, metavar="FILE", help="shortcut for --mode video --file FILE; loops for --seconds")
    p.add_argument("--preserve-aspect", action="store_true", help="fit video with black bars instead of stretching")
    p.add_argument("--font", type=Path, help="external SDF font directory")
    p.add_argument("--output", type=Path, help="save final raw RGBA8 snapshot")
    p.add_argument("--gpio", action="store_true", help="activate physical output (default: offscreen)")
    p.add_argument("--gpio-cpu", type=int, default=-1,
                   help="pin only the GPIO scanner thread to this CPU; -1 leaves affinity unchanged (default)")
    p.add_argument("--acknowledge-wiring", action="store_true")
    args = p.parse_args()
    if args.video is not None:
        if args.mode not in (None, "video"):
            p.error("--video cannot be combined with a different --mode")
        if args.file is not None and args.file != args.video:
            p.error("--video and --file must name the same file when both are supplied")
        args.mode, args.file = "video", args.video
    args.mode = args.mode or "retained"
    if args.gpio and not args.acknowledge_wiring:
        p.error("--gpio requires --acknowledge-wiring; check panel voltage/pins/topology first")
    if args.mode in ("shader", "video") and args.file is None:
        p.error("--mode shader/video requires --file (or use --video FILE)")
    if args.mode == "text" and args.font is None:
        p.error("text mode requires --font")
    if not 0 < args.seconds <= 86400:
        p.error("--seconds must be in (0, 86400]")
    if not 0 <= args.brightness <= 254:
        p.error("--brightness must be in 0..254")
    if args.gpio_cpu < -1:
        p.error("--gpio-cpu must be -1 or a nonnegative CPU index")
    if args.mode in ("video", "shader") and not args.file.is_file():
        p.error(f"{args.mode} requires an existing local file: {args.file}")
    caps = h.capabilities()
    if args.mode in ("video", "shader") and not caps.get(args.mode):
        p.error(f"{args.mode} support is not compiled into this Python extension. "
                "From the repository root rebuild it with: "
                "(cd python && HUB75_BACKEND=gpu python3 setup.py build_ext --inplace)")
    config = h.DisplayConfig(width=args.width, height=args.height, ports=args.ports,
                            chains=args.chains, panel_width=args.panel_width,
                            panel_height=args.panel_height, fps=args.fps,
                            brightness=args.brightness, offscreen=not args.gpio,
                            gpio_cpu=args.gpio_cpu,
                            video_preserve_aspect=args.preserve_aspect,
                            output_mode=h.OutputMode.WEIGHTED_BCM if args.output_mode == "weighted-bcm" else h.OutputMode.EQUAL_SLOT,
                            pwm_bits=args.pwm_bits, pwm_lsb_ns=args.pwm_lsb_ns,
                            color_curve=h.ColorCurve.CIE if args.color_curve == "cie" else h.ColorCurve.GAMMA,
                            gamma=args.gamma)
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
    print(caps)
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
