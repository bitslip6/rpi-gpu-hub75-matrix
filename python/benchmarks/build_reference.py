"""Compile the direct-C reference against the EXACT installed native library.

This avoids comparing a target-tuned C renderer with a portable Python wheel.
Requires the source checkout/sdist, a C compiler and matching installed package.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import hub75gpu as h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/python-reference/example_runtime"))
    parser.add_argument("--libdir", type=Path, help="required only for system-library installs")
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    native = package / "native" if (package / "native").exists() else package.parent
    libdir = args.libdir or (Path(h.__file__).parent / ".libs")
    libdir = libdir.resolve()
    library = "rpihub75_gpu" if h.capabilities()["backend"] == "gpu" else "rpihub75"
    if not (libdir / ("lib" + library + ".so")).is_file():
        parser.error("selected native library not found; specify --libdir for managed system linkage")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = shlex.split(os.environ.get("CC", "cc")) + ["-std=gnu2x", "-O3", "-fno-fast-math",
        "-I" + str(native / "include"), "-I" + str(native / "examples"),
        str(native / "examples" / "runtime.c"), "-L" + str(libdir),
        "-Wl,-rpath," + str(libdir), "-l" + library, "-lpthread", "-lm", "-lrt", "-o", str(args.output)]
    subprocess.run(command, check=True)
    print(json.dumps({"binary": str(args.output.resolve()), "library": str(libdir / ("lib"+library+".so")),
                      "capabilities": h.capabilities(), "command": command}, indent=2))


if __name__ == "__main__":
    main()
