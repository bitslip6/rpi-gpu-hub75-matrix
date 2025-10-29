#!/usr/bin/env python3
import hub75gpu as h
import threading
import time
import math
import sys

"""
CPU 3D rendering demo (Python)
 - Creates a display scene, starts mapper and HUB75 output threads
 - Builds a simple 3D scene (camera + cube) with ambient + directional light
 - Animates the cube rotation and camera orbit using CPU renderer

Requires the extension to be built (see python/Makefile), and the top-level
shared libraries available in .. (built via the root Makefile).
"""


def main():
	# Configure a simple 128x128, 64-bit depth, 60 FPS scene from CLI-style args
	# scene = h.parse_args(["-x", "128", "-y", "128", "-d", "64", "-f", "60"])  # adjust to your setup
	scene = h.parse_args(sys.argv)  # adjust to your setup
	# Initialize display buffers and threads, and API for CPU 3D helpers
	h.display_start(scene)
	h.hub75_init(scene)
	h.signal_handler_install()  # C-level SIGINT/SIGTERM handler sets do_render=false

	# Start mapper thread (RGB->BCM) and HUB75 output loop in a background thread
	h.start_mapper(scene)
	disp_thread = threading.Thread(target=lambda: h.display_run(scene), daemon=True)
	disp_thread.start()

	# Build 3D camera + object scene (similar to example.c:render_3d)
	cam = h.geo_camera()
	cube = h.geo_cube(1, True)   # DRAW_FILLED=1, cull backfaces
	xform = h.geo_transform()

	os = h.scene_new(1)
	h.scene_set_current(os)
	h.scene_set_ambient(0.25, 0.25, 0.25)
	h.scene_add_directional(-0.5, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, False)
	h.scene_add_object(cube, xform)

	# Camera parameters
	width, height = 128, 128
	aspect = width / height
	fov = 1.2  # radians

	t0 = time.time()
	try:
		while True:
			t = time.time() - t0

			# Animate cube rotation
			h.transform_set_rot(xform, 0.7 * t, 1.14, 0.0)
			h.transform_set_scale(xform, 1.5, 1.5, 1.5)

			# Orbit camera around origin while looking at the cube
			cam_x = 5.0 * math.cos(0.3 * t)
			cam_z = 5.0 * math.sin(0.3 * t)
			h.camera_set_all(cam,
							 cam_x, 2.5, cam_z,      # position
							 0.0, 0.0, 0.0,          # target (origin)
							 0.0, 1.0, 0.0,          # up
							 fov, aspect, 0.1, 100.0)  # fov, aspect, znear, zfar

			# Render one frame to scene->image and push to rings
			h.frame_begin()
			h.clear()
			h.render_scene(cam, os)
			h.frame_end()

			# Modest pacing; HUB75 threads drive actual refresh
			time.sleep(0.001)
	except KeyboardInterrupt:
		pass
	finally:
		# Request shutdown and wait for threads to finish
		h.request_shutdown(scene)
		h.display_wait(scene)


if __name__ == "__main__":
	main()
