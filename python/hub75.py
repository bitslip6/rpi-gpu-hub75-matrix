from __future__ import annotations

# Keep this module as a convenience re-export for people preferring `import hub75 as h`.
# It intentionally avoids importing type names from the compiled module at runtime.

import hub75gpu as _ext

# Lifecycle and threading helpers
parse_args = _ext.parse_args
new_scene = _ext.new_scene
hub75_init = _ext.hub75_init
display_start = _ext.display_start
display_run = _ext.display_run
display_wait = _ext.display_wait
request_shutdown = _ext.request_shutdown
signal_handler_install = _ext.signal_handler_install
start_mapper = _ext.start_mapper
start_renderer = _ext.start_renderer

# 3D helpers re-export
frame_begin = _ext.frame_begin
frame_end = _ext.frame_end
clear = _ext.clear

geo_camera = _ext.geo_camera
geo_transform = _ext.geo_transform
geo_object = _ext.geo_object
geo_cube = _ext.geo_cube
geo_tetrahedron = _ext.geo_tetrahedron
geo_octahedron = _ext.geo_octahedron
geo_pyramid = _ext.geo_pyramid
geo_cylinder = _ext.geo_cylinder
geo_sphere = _ext.geo_sphere
geo_torus = _ext.geo_torus
geo_plane = _ext.geo_plane

scene_new = _ext.scene_new
scene_set_current = _ext.scene_set_current
scene_clear_current = _ext.scene_clear_current
scene_set_ambient = _ext.scene_set_ambient
scene_add_directional = _ext.scene_add_directional
scene_add_object = _ext.scene_add_object
scene_get_object = _ext.scene_get_object
scene_get_transform = _ext.scene_get_transform

render_scene = _ext.render_scene
render_wire = _ext.render_wire
render_filled = _ext.render_filled

transform_set_pos = _ext.transform_set_pos
transform_set_rot = _ext.transform_set_rot
transform_set_scale = _ext.transform_set_scale
camera_set_all = _ext.camera_set_all
