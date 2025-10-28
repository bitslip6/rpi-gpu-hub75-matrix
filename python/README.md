# hub75gpu Python Extension

Simple CPython extension that exposes the hub75gpu_t function table from the rpi-gpu-hub75-matrix project.

Requirements:
- Build the C shared library first at repo root:
  - `make` (produces `librpihub75.so` and `librpihub75_gpu.so` in the repo root)
- Python 3 with setuptools

Build the extension in-place:

```
python3 setup.py build_ext --inplace
```

Usage (minimal smoke test):

```python
import hub75gpu as h

# Create a scene and initialize the API (sets current thread scene context)
scene = h.new_scene()
h.hub75_init(scene)

# 3D setup
cam = h.geo_camera()
os  = h.scene_new(0)
h.scene_set_current(os)
h.scene_set_ambient(0.1, 0.1, 0.1)
h.scene_add_directional(0, -1, -1, 1, 1, 1, 0.8, True)

cube = h.geo_cube(1, True)  # DRAW_FILLED=1
xform = h.geo_transform()

h.scene_add_object(cube, xform)

# Render a frame (assumes scene->image managed externally by your app)
h.frame_begin()
h.clear()
h.render_scene(cam, os)
h.frame_end()
```

Notes:
- Pointers are carried using Python capsules; the extension does not attempt to own or free most C objects (except camera/transform which are freed when their capsules are GC'd). For long-running apps, implement explicit free on the C side if needed.
- The bindings wrap the most used `hub75gpu_t` functions; feel free to extend as needed.
