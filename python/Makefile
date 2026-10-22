# Convenience targets; PEP 517/setuptools owns extension packaging.
PYTHON ?= python3
.PHONY: all install wheel test clean
all:
	$(PYTHON) setup.py build_ext --inplace
install:
	$(PYTHON) -m pip install .
wheel:
	$(PYTHON) -m build
# CPU by default. EGL/GPIO tests have separate explicit opt-in guards.
test: all
	$(PYTHON) -X faulthandler -m unittest discover -s tests -v
clean:
	rm -rf build dist hub75gpu.egg-info hub75gpu/.libs
	rm -f hub75gpu/_native*.so
