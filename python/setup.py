try:
    from setuptools import setup, Extension
except Exception:
    # Fallback if setuptools.setup is unavailable
    from distutils.core import setup
    from distutils.extension import Extension
import os

# Build the CPython extension that wraps hub75gpu_t API from the C library
# We link against the CPU-only shared library (librpihub75.so) built by the Makefile.
# Ensure you've run `make` at repo root first, so the shared lib is available.

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
include_dir = os.path.join(repo_root, 'include')
lib_dir = repo_root

ext = Extension(
    name='hub75gpu',
    sources=['hub75gpu_py.c'],
    include_dirs=[include_dir],
    library_dirs=[lib_dir],
    libraries=['rpihub75'],  # CPU-only lib that contains the API table
    extra_link_args=['-Wl,-rpath,$ORIGIN/..'],  # so Python can find the .so at runtime from repo root
)

setup(
    name='hub75gpu',
    version='0.1.0',
    description='Python bindings exposing hub75gpu_t API for rpi-gpu-hub75-matrix',
    ext_modules=[ext],
)
