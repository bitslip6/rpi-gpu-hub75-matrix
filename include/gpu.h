/**
 * @brief render the shadertoy compatible shader source code in the 
 * file pointed to by scene->shader_file
 * 
 * exits if shader is unable to be loaded, compiled or rendered
 * 
 * loop exits and memory is freed if/when scene->do_render becomes false
 * 
 * frame delay is adaptive and updates to current scene->fps on each frame update
 * 
 * @param arg pointer to the current scene_info object
 */
void *main_render_shader(void *arg);