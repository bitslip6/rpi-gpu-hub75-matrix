"""Build sdist/wheel, then install in fresh venvs and run outside the checkout.

Requires `pip`, `build`, setuptools, wheel, native/Python development headers and
runtime dependencies. Source-install build isolation may download build tools.
No GPIO/EGL activation. Use --backend gpu to validate optional GPU packaging.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

PACKAGE = Path(__file__).resolve().parents[1]


def run(args, *, cwd, env):
    print("+", " ".join(map(str, args)), flush=True)
    try:
        return subprocess.check_output(list(map(str, args)), cwd=cwd, env=env, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        print(exc.output, flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--keep", type=Path, help="keep artifacts/results in this directory")
    args = parser.parse_args()
    env = os.environ.copy()
    for name in ("PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "HUB75_SYSTEM_LIBDIR", "HUB75_TEST_GPU"):
        env.pop(name, None)
    env["HUB75_BACKEND"] = args.backend
    with tempfile.TemporaryDirectory(prefix="hub75-install-") as td:
        work = Path(td)
        artifacts = work / "dist"
        build = [sys.executable, "-m", "build", "--outdir", artifacts, PACKAGE]
        if env.get("HUB75_TEST_NO_BUILD_ISOLATION") == "1":
            build.insert(3, "--no-isolation")
        print(run(build, cwd=work, env=env)[-3000:])
        wheels = list(artifacts.glob("*.whl"))
        sources = list(artifacts.glob("*.tar.gz"))
        assert len(wheels) == len(sources) == 1
        results = []
        for kind, artifact in (("wheel", wheels[0]), ("sdist", sources[0])):
            directory = work / ("venv-" + kind)
            venv.EnvBuilder(with_pip=False).create(directory)
            python = directory / "bin" / "python"
            # pip's --python can manage a venv without preinstalled pip.
            install = [sys.executable, "-m", "pip", "--python", python, "install", "--no-deps", artifact]
            print(run(install, cwd=work, env=env)[-2000:])
            code = r'''
import json, pathlib, sys
import hub75gpu as h
caps=h.capabilities()
with h.Display(h.DisplayConfig(width=8,height=8,offscreen=True)) as d:
    with d.frame() as f:f.clear((13,27,91))
    assert d.snapshot()==bytes((13,27,91,255))*64
assert caps['backend']==sys.argv[1]
if caps['backend']=='cpu':
    maps=pathlib.Path('/proc/self/maps').read_text()
    for name in ('libavcodec','libavformat','libavutil','libswscale','libEGL','libGLES','libgbm'):
        assert name not in maps, name
print(json.dumps({'caps':caps,'package':h.__file__,'extension':h._native.__file__}))
'''
            result = json.loads(run([python, "-X", "faulthandler", "-c", code, args.backend], cwd=work, env=env))
            assert str(directory) in result["package"]
            extension = Path(result["extension"])
            libs = extension.parent / ".libs"
            expected = "librpihub75_gpu.so" if args.backend == "gpu" else "librpihub75.so"
            assert sorted(p.name for p in libs.iterdir()) == [expected]
            dynamic = run(["readelf", "-d", extension], cwd=work, env=env)
            assert "$ORIGIN/.libs" in dynamic
            native_dynamic = run(["readelf", "-d", libs / expected], cwd=work, env=env)
            if args.backend == "cpu":
                assert not any(n in native_dynamic for n in ("libEGL", "libGLES", "libgbm", "libavcodec", "libswscale"))
            # ABI rejection must happen at import, before table-by-value use.
            bad = work / "badabi.c"
            bad.write_text("unsigned hub75_runtime_abi(void){return 999;}\n")
            library = work / "badabi.so"
            run([os.environ.get("CC", "cc"), "-shared", "-fPIC", bad, "-o", library], cwd=work, env=env)
            bad_env = dict(env, LD_PRELOAD=str(library))
            rejected = subprocess.run([str(python), "-c", "import hub75gpu"], cwd=work, env=bad_env,
                                      capture_output=True, text=True)
            assert rejected.returncode != 0 and "ABI" in rejected.stderr, rejected.stderr
            # Run the complete pin-free binding suite against the installed
            # package, not a source directory injected onto sys.path.
            tests = work / ("tests-" + kind)
            tests.mkdir()
            for source in (PACKAGE / "tests").glob("test_*.py"):
                shutil.copy2(source, tests)
            print(run([python, "-X", "faulthandler", "-m", "unittest", "discover", "-s", tests, "-v"],
                      cwd=work, env=env)[-1500:])
            results.append({"kind": kind, **result})
        if args.keep:
            args.keep.mkdir(parents=True, exist_ok=True)
            for artifact in artifacts.iterdir():
                shutil.copy2(artifact, args.keep)
            (args.keep / "results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
        print("clean wheel and sdist installation checks passed (no GPIO)")


if __name__ == "__main__":
    main()
