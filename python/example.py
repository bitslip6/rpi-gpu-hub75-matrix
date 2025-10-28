import hub75gpu as h

# Initialize scene and API
scene = h.new_scene()
h.hub75_init(scene)

# Create camera and scene-of-objects
cam = h.geo_camera()
os = h.scene_new(0)
h.scene_set_current(os)

# Lighting
h.scene_set_ambient(0.1, 0.1, 0.1)
h.scene_add_directional(0.0, -1.0, -1.0, 1.0, 1.0, 1.0, 0.8, True)

# Geometry
cube = h.geo_cube(1, True)  # DRAW_FILLED=1
xform = h.geo_transform()

# Add and render
h.scene_add_object(cube, xform)

# Begin/end frame and render scene using scene-owned lighting
h.frame_begin()
h.clear()
h.render_scene(cam, os)
h.frame_end()

print("Rendered one frame via hub75gpu Python bindings.")
