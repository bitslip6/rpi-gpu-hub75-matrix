"""PEP 517 integration. The renderer is built by the root Makefile, not duplicated.

HUB75_BACKEND=cpu|gpu (default cpu)
HUB75_SYSTEM_LIBDIR=/absolute/path selects deployment-managed libraries instead
of the supported private .libs layout. HUB75_CFLAGS_CPU defaults to portable code.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist

HERE = Path(__file__).resolve().parent
ROOT = HERE / "native" if (HERE / "native" / "Makefile").exists() else HERE.parent
BACKEND = os.environ.get("HUB75_BACKEND", "cpu")
if BACKEND not in ("cpu", "gpu"):
    raise RuntimeError("HUB75_BACKEND must be cpu or gpu")
LIBRARY = "rpihub75_gpu" if BACKEND == "gpu" else "rpihub75"
SYSTEM = os.environ.get("HUB75_SYSTEM_LIBDIR")
if SYSTEM and not Path(SYSTEM).is_absolute():
    raise RuntimeError("HUB75_SYSTEM_LIBDIR must be absolute")


class NativeBuild(build_ext):
    def run(self):
        if SYSTEM:
            libdir = Path(SYSTEM)
            # Do not accidentally ship an old private library when switching a
            # reused build directory to deployment-managed linkage.
            shutil.rmtree(Path(self.build_lib) / "hub75gpu" / ".libs", ignore_errors=True)
            if self.inplace:
                shutil.rmtree(HERE / "hub75gpu" / ".libs", ignore_errors=True)
        else:
            env = os.environ.copy()
            # Keep Python flags/outer make overrides out of the renderer build.
            for name in ("CFLAGS", "CPPFLAGS", "MAKEFLAGS", "MAKEOVERRIDES", "MFLAGS"):
                env.pop(name, None)
            cpu_flags = os.environ.get("HUB75_CFLAGS_CPU", "")
            settings = subprocess.check_output(
                ["make", "--no-print-directory", "-s", "-C", str(ROOT), "print-vars",
                 "BUILD=release", "CFLAGS_CPU=" + cpu_flags], env=env)
            # Distinct outputs, not only distinct object folders: otherwise a
            # newer library can hide old cached objects when switching back.
            extra = {name: env.get(name, "") for name in (
                "LDFLAGS", "LDLIBS_COMMON", "LDLIBS_FFMPEG", "LDLIBS_GPU", "LDLIBS_PNG", "LDLIBS_JPEG",
                "PATH", "CPATH", "C_INCLUDE_PATH", "LIBRARY_PATH", "COMPILER_PATH")}
            signature = hashlib.sha256(settings + (ROOT / "Makefile").read_bytes()
                                       + json.dumps(extra, sort_keys=True).encode()).hexdigest()[:16]
            libdir = Path(self.build_temp).resolve() / ("native-" + BACKEND + "-" + signature)
            libdir.mkdir(parents=True, exist_ok=True)
            output = libdir / ("lib" + LIBRARY + ".so")
            command = ["make", "-C", str(ROOT), "libgpu" if BACKEND == "gpu" else "lib",
                       "BUILD=release", "BUILDDIR=" + str(libdir / "objects"),
                       ("LIB_GPU=" if BACKEND == "gpu" else "LIB_NO_GPU=") + str(output),
                       "CFLAGS_CPU=" + cpu_flags,
                       "-j" + os.environ.get("HUB75_BUILD_JOBS", "2")]
            subprocess.check_call(command, env=env)
        for ext in self.extensions:
            ext.library_dirs = [str(libdir)]
            ext.runtime_library_dirs = [SYSTEM] if SYSTEM else ["$ORIGIN/.libs"]
        # Backend selection changes the linked library even when C sources did
        # not change; setuptools' timestamp cache cannot infer this dependency.
        self.force = True
        super().run()
        if not SYSTEM:
            dest = Path(self.build_lib) / "hub75gpu" / ".libs"
            dest.mkdir(parents=True, exist_ok=True)
            other = "librpihub75.so" if BACKEND == "gpu" else "librpihub75_gpu.so"
            (dest / other).unlink(missing_ok=True)
            shutil.copy2(libdir / ("lib" + LIBRARY + ".so"), dest)
            if self.inplace:
                dest = HERE / "hub75gpu" / ".libs"
                dest.mkdir(parents=True, exist_ok=True)
                (dest / other).unlink(missing_ok=True)
                shutil.copy2(libdir / ("lib" + LIBRARY + ".so"), dest)


class SourceDistribution(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        dest = Path(base_dir) / "native"
        dest.mkdir(exist_ok=True)
        for name in ("Makefile", "detect_flags.sh", "rpihub75.pc.in"):
            shutil.copy2(ROOT / name, dest / name)
        for name in ("src", "include"):
            shutil.copytree(ROOT / name, dest / name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*.o", "*.tmp", "__pycache__"))
        for name in ("examples/runtime.c", "examples/runtime_demo.h",
                     "tests/unit/test_runtime.c", "tests/unit/test_sampler.c"):
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)


setup(
    name="hub75gpu",
    version="0.3.0",
    description="Native Raspberry Pi HUB75 rendering with typed Python control",
    long_description=(HERE / "README.md").read_text(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=["hub75gpu"],
    package_data={"hub75gpu": ["py.typed", "*.pyi"]},
    ext_modules=[Extension("hub75gpu._native", ["hub75gpu/_native.c"],
                           include_dirs=[str(ROOT / "include")],
                           libraries=[LIBRARY], extra_compile_args=["-std=gnu2x", "-fno-fast-math"])],
    cmdclass={"build_ext": NativeBuild, "sdist": SourceDistribution},
    classifiers=["Programming Language :: Python :: 3", "Programming Language :: C",
                 "Operating System :: POSIX :: Linux"],
)
