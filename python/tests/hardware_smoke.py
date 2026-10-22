"""Explicit GPIO smoke test. NEVER imported/discovered by the normal test suite.

HUB75_TEST_GPIO=1 python python/tests/hardware_smoke.py --acknowledge-wiring
Verify wiring, voltage, HAT and topology before authorizing physical output.
"""
import argparse
import os
import time
import hub75gpu as h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acknowledge-wiring", action="store_true")
    parser.add_argument("--ports", type=int, default=1)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--panel-width", type=int, default=64)
    parser.add_argument("--panel-height", type=int, default=64)
    args = parser.parse_args()
    if os.environ.get("HUB75_TEST_GPIO") != "1" or not args.acknowledge_wiring:
        parser.error("requires HUB75_TEST_GPIO=1 and --acknowledge-wiring; no GPIO activated")
    config = h.DisplayConfig(width=args.panel_width*args.chains, height=args.panel_height*args.ports,
                            panel_width=args.panel_width, panel_height=args.panel_height,
                            ports=args.ports, chains=args.chains, brightness=32, offscreen=False)
    with h.Display(config) as display:
        with display.frame(timeout=2) as frame:
            frame.clear((30, 5, 15))
        display.flush(2)
        deadline = time.monotonic() + 3
        while display.stats()["scanned"] == 0 and time.monotonic() < deadline:
            time.sleep(.01)
        assert display.stats()["scanned"] > 0, "scanner did not complete a full BCM scan"
        print(display.stats())
        # Deliberately hold the slot/scan for visual inspection; close blanks OE.
        time.sleep(1)


if __name__ == "__main__":
    main()
