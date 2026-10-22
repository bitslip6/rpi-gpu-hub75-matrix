"""Compatibility launcher for the maintained, offscreen-by-default CLI demo.

After installation prefer: python -m hub75gpu --help
"""
from pathlib import Path
import sys

# Do not shadow a clean installed package with an unbuilt checkout package.
if not list((Path(__file__).parent / "hub75gpu").glob("_native*.so")):
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).parent.resolve()]

from hub75gpu.__main__ import main

if __name__ == "__main__":
    main()
