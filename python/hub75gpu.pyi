from __future__ import annotations
from typing import List, Optional, Tuple

# This stub file provides IDE type information and autocompletion for the
# compiled extension module hub75gpu (hub75gpu.*.so).
# It doesn't affect runtime behavior but greatly improves editor experience.

# Capsule-backed opaque handles (runtime are PyCapsules) for 3D helpers
class Camera: ...
class Transform: ...
class Object: ...
class Scene3D: ...

class Display:
    """Hub75 display configuration and runtime state."""
    # Core dimensions and panel config
    width: int
    height: int
    panel_width: int
    panel_height: int
    stride: int
    num_ports: int
    num_chains: int
    bit_depth: int
    brightness: int
    panel_order: int
    # Timing
    fps: int
    auto_fps: bool
    # Color / tone
    gamma: float
    red_gamma: float
    green_gamma: float
    blue_gamma: float
    red_linear: float
    green_linear: float
    blue_linear: float
    dither: float
    tone_level: float
    quant_dither: bool
    jitter_brightness: bool
    motion_blur_frames: int
    show_fps: bool
    enhanced_debug: bool
    # Read-only
    @property
    def render_width(self) -> int: ...
    @property
    def render_height(self) -> int: ...
    @property
    def frame_index(self) -> int: ...
    @property
    def do_render(self) -> bool: ...
    # Shader/video source
    shader_file: Optional[str]

# -------- Display lifecycle --------

def new_scene() -> Display: ...

def parse_args(argv: List[str]) -> Display: ...

def hub75_init(scene: Display) -> None:
    """Initialize thread-local API helpers for the given scene (required for CPU 3D helpers)."""
    ...

def display_start(scene: Display) -> None: ...

def start_mapper(scene: Display) -> None: ...

def start_renderer(scene: Display) -> None:
    """Starts shader or video renderer thread based on scene.shader_file extension."""
    ...

def display_run(scene: Display) -> None:
    """Blocks while driving HUB75 GPIO output until shutdown is requested."""
    ...

def display_wait(scene: Display) -> None: ...

def request_shutdown(scene: Display) -> None: ...

def signal_handler_install() -> None: ...

# -------- CPU 2D/3D helpers (OO-style wrappers) --------

def frame_begin() -> None: ...

def frame_end() -> None: ...

def clear() -> None: ...

def pixel(x: int, y: int, rgb: Tuple[int, int, int]) -> None: ...

def line(x0: int, y0: int, x1: int, y1: int, rgb: Tuple[int, int, int]) -> None: ...

# Geometry factory

def geo_camera() -> Camera: ...

def geo_transform() -> Transform: ...

def geo_object(num_vertices: int, num_edges: int, num_faces: int) -> Object: ...

def geo_cube(mode: int, cull_backface: bool) -> Object: ...

def geo_tetrahedron() -> Object: ...

def geo_octahedron() -> Object: ...

def geo_pyramid() -> Object: ...

def geo_cylinder(segments: int) -> Object: ...

def geo_sphere(subdivisions: int) -> Object: ...

def geo_torus(major_segments: int, minor_segments: int) -> Object: ...

def geo_plane(width_segments: int, height_segments: int) -> Object: ...

# Scene3D management

def scene_new(count: int) -> Scene3D: ...

def scene_set_current(scene3d: Scene3D) -> None: ...

def scene_clear_current() -> None: ...

def scene_set_ambient(r: float, g: float, b: float) -> None: ...

def scene_add_directional(dx: float, dy: float, dz: float,
                          cr: float, cg: float, cb: float,
                          intensity: float, casts_shadows: bool) -> int: ...

def scene_add_object(obj: Object, xform: Transform) -> int: ...

def scene_get_object(object_id: int) -> Object: ...

def scene_get_transform(object_id: int) -> Transform: ...

# Rendering

def render_scene(camera: Camera, scene3d: Scene3D) -> None: ...

def render_wire(camera: Camera, obj: Object, xform: Transform) -> None: ...

def render_filled(camera: Camera, obj: Object, xform: Transform) -> None: ...

# Transform setters (for animation)

def transform_set_pos(xform: Transform, x: float, y: float, z: float) -> None: ...

def transform_set_rot(xform: Transform, x: float, y: float, z: float) -> None: ...

def transform_set_scale(xform: Transform, x: float, y: float, z: float) -> None: ...

# Camera setter

def camera_set_all(cam: Camera,
                   px: float, py: float, pz: float,
                   tx: float, ty: float, tz: float,
                   ux: float, uy: float, uz: float,
                   fov_y: float, aspect: float,
                   z_near: float, z_far: float) -> None: ...
