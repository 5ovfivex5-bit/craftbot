// Panel entry points. Each panel calls ImGui::Begin/End itself and honours its open flag.
#pragma once

#include <cstdint>

#include "app.h"

void draw_viewport(App& app, bool* p_open);
void draw_training(App& app, bool* p_open);
void draw_controls(App& app, bool* p_open);
void draw_agent(App& app, bool* p_open);
void draw_graphs(App& app, bool* p_open);
void draw_nn(App& app, bool* p_open);       // sets app.nn_visible
void draw_checkpoints(App& app, bool* p_open);
void draw_console(App& app, bool* p_open);
void draw_chat(App& app, bool* p_open);

// Creates or resizes an RGBA8 nearest-filtered texture and uploads pixels. Needs a current GL context.
void upload_texture(unsigned& tex, int w, int h, const unsigned char* rgba);
