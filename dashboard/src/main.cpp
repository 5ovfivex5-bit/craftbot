// MineMind Training Studio: SDL3 + OpenGL 3.3 core + Dear ImGui (docking).
// App shell only: window, theme, dock layout, menus, status bar, toasts, shortcuts.
// Panels live in panel_*.cpp; networking in net/.

#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl3.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_opengl.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#pragma GCC diagnostic pop

#include "app.h"
#include "gl_procs.h"
#include "net/client.h"
#include "panels.h"
#include "ui_util.h"

namespace {

constexpr const char* kLayoutDir = "layouts";
constexpr const char* kUserLayout = "layouts/user_layout.ini";

// ---------------------------------------------------------------- theme

void apply_theme() {
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowPadding = ImVec2(8, 6);
    s.FramePadding = ImVec2(6, 3);
    s.ItemSpacing = ImVec2(6, 4);
    s.ItemInnerSpacing = ImVec2(4, 3);
    s.IndentSpacing = 14;
    s.ScrollbarSize = 12;
    s.GrabMinSize = 9;
    s.WindowRounding = 5;
    s.ChildRounding = 4;
    s.FrameRounding = 3;
    s.PopupRounding = 4;
    s.ScrollbarRounding = 6;
    s.GrabRounding = 3;
    s.TabRounding = 4;
    s.WindowBorderSize = 1;
    s.FrameBorderSize = 0;
    s.TabBarBorderSize = 1;
    s.SeparatorTextBorderSize = 2;
    s.DockingSeparatorSize = 3;

    ImVec4* c = s.Colors;
    const ImVec4 bg(0.075f, 0.082f, 0.098f, 1.00f);
    const ImVec4 panel(0.105f, 0.115f, 0.137f, 1.00f);
    const ImVec4 frame(0.160f, 0.172f, 0.203f, 1.00f);
    const ImVec4 hover(0.215f, 0.232f, 0.275f, 1.00f);
    const ImVec4 accent(0.235f, 0.640f, 0.740f, 1.00f);
    const ImVec4 accent_dim(0.160f, 0.430f, 0.500f, 1.00f);
    const ImVec4 text(0.875f, 0.890f, 0.925f, 1.00f);
    const ImVec4 text_dim(0.500f, 0.530f, 0.590f, 1.00f);

    c[ImGuiCol_Text] = text;
    c[ImGuiCol_TextDisabled] = text_dim;
    c[ImGuiCol_WindowBg] = bg;
    c[ImGuiCol_ChildBg] = panel;
    c[ImGuiCol_PopupBg] = panel;
    c[ImGuiCol_Border] = ImVec4(0.200f, 0.215f, 0.255f, 1.0f);
    c[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_FrameBg] = frame;
    c[ImGuiCol_FrameBgHovered] = hover;
    c[ImGuiCol_FrameBgActive] = accent_dim;
    c[ImGuiCol_TitleBg] = panel;
    c[ImGuiCol_TitleBgActive] = frame;
    c[ImGuiCol_TitleBgCollapsed] = panel;
    c[ImGuiCol_MenuBarBg] = panel;
    c[ImGuiCol_ScrollbarBg] = bg;
    c[ImGuiCol_ScrollbarGrab] = frame;
    c[ImGuiCol_ScrollbarGrabHovered] = hover;
    c[ImGuiCol_ScrollbarGrabActive] = accent_dim;
    c[ImGuiCol_CheckMark] = accent;
    c[ImGuiCol_SliderGrab] = accent_dim;
    c[ImGuiCol_SliderGrabActive] = accent;
    c[ImGuiCol_Button] = frame;
    c[ImGuiCol_ButtonHovered] = hover;
    c[ImGuiCol_ButtonActive] = accent_dim;
    c[ImGuiCol_Header] = frame;
    c[ImGuiCol_HeaderHovered] = hover;
    c[ImGuiCol_HeaderActive] = accent_dim;
    c[ImGuiCol_Separator] = ImVec4(0.200f, 0.215f, 0.255f, 1.0f);
    c[ImGuiCol_SeparatorHovered] = accent;
    c[ImGuiCol_SeparatorActive] = accent;
    c[ImGuiCol_ResizeGrip] = frame;
    c[ImGuiCol_ResizeGripHovered] = hover;
    c[ImGuiCol_ResizeGripActive] = accent_dim;
    c[ImGuiCol_Tab] = panel;
    c[ImGuiCol_TabHovered] = hover;
    c[ImGuiCol_TabSelected] = frame;
    c[ImGuiCol_TabSelectedOverline] = accent;
    c[ImGuiCol_TabDimmed] = panel;
    c[ImGuiCol_TabDimmedSelected] = frame;
    c[ImGuiCol_TabDimmedSelectedOverline] = accent_dim;
    c[ImGuiCol_DockingPreview] = ImVec4(accent.x, accent.y, accent.z, 0.5f);
    c[ImGuiCol_DockingEmptyBg] = bg;
    c[ImGuiCol_PlotLines] = accent;
    c[ImGuiCol_PlotLinesHovered] = text;
    c[ImGuiCol_PlotHistogram] = accent;
    c[ImGuiCol_PlotHistogramHovered] = text;
    c[ImGuiCol_TableHeaderBg] = frame;
    c[ImGuiCol_TableBorderStrong] = ImVec4(0.200f, 0.215f, 0.255f, 1.0f);
    c[ImGuiCol_TableBorderLight] = ImVec4(0.150f, 0.160f, 0.190f, 1.0f);
    c[ImGuiCol_TableRowBgAlt] = ImVec4(1, 1, 1, 0.02f);
    c[ImGuiCol_TextSelectedBg] = ImVec4(accent.x, accent.y, accent.z, 0.35f);
    c[ImGuiCol_DragDropTarget] = accent;
    c[ImGuiCol_NavHighlight] = accent;
    c[ImGuiCol_NavWindowingHighlight] = ImVec4(1, 1, 1, 0.70f);
    c[ImGuiCol_NavWindowingDimBg] = ImVec4(0.2f, 0.2f, 0.2f, 0.2f);
    c[ImGuiCol_ModalWindowDimBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.55f);
}

// ---------------------------------------------------------------- layout

void build_default_layout(ImGuiID dock) {
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::DockBuilderRemoveNode(dock);
    ImGui::DockBuilderAddNode(dock, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dock, vp->WorkSize);

    ImGuiID left, rest, right, center, bottom, controls, training, agent, nn, viewport, lower;
    ImGui::DockBuilderSplitNode(dock, ImGuiDir_Left, 0.25f, &left, &rest);
    ImGui::DockBuilderSplitNode(rest, ImGuiDir_Right, 0.34f, &right, &center);
    ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, 0.32f, &bottom, &viewport);
    ImGui::DockBuilderSplitNode(left, ImGuiDir_Down, 0.48f, &controls, &training);
    ImGui::DockBuilderSplitNode(right, ImGuiDir_Down, 0.50f, &nn, &agent);
    lower = bottom;

    ImGui::DockBuilderDockWindow(win::kViewport, viewport);
    ImGui::DockBuilderDockWindow(win::kControls, controls);
    ImGui::DockBuilderDockWindow(win::kTraining, training);
    ImGui::DockBuilderDockWindow(win::kAgent, agent);
    ImGui::DockBuilderDockWindow(win::kNN, nn);
    ImGui::DockBuilderDockWindow(win::kGraphs, lower);
    ImGui::DockBuilderDockWindow(win::kCheckpoints, lower);
    ImGui::DockBuilderDockWindow(win::kConsole, lower);
    ImGui::DockBuilderDockWindow(win::kChat, lower);
    ImGui::DockBuilderFinish(dock);
    (void)center;
}

bool layout_is_built(ImGuiID dock) {
    ImGuiDockNode* node = ImGui::DockBuilderGetNode(dock);
    return node && node->IsSplitNode();
}

// ---------------------------------------------------------------- menus / bars

void draw_main_menu(App& app, bool& want_reset_layout, bool& want_quit) {
    if (!ImGui::BeginMainMenuBar()) return;
    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Save checkpoint", "Ctrl+S")) app.cmd("save_checkpoint");
        ImGui::Separator();
        if (ImGui::MenuItem("Quit", "Alt+F4")) want_quit = true;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
        ImGui::MenuItem(win::kViewport, nullptr, &app.open_viewport);
        ImGui::MenuItem(win::kControls, nullptr, &app.open_controls);
        ImGui::MenuItem(win::kTraining, nullptr, &app.open_training);
        ImGui::MenuItem(win::kAgent, nullptr, &app.open_agent);
        ImGui::MenuItem(win::kGraphs, nullptr, &app.open_graphs);
        ImGui::MenuItem(win::kNN, nullptr, &app.open_nn);
        ImGui::MenuItem(win::kCheckpoints, nullptr, &app.open_checkpoints);
        ImGui::MenuItem(win::kConsole, nullptr, &app.open_console);
        ImGui::MenuItem(win::kChat, nullptr, &app.open_chat);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Layout")) {
        if (ImGui::MenuItem("Save layout")) {
            std::error_code ec;
            std::filesystem::create_directories(kLayoutDir, ec);
            ImGui::SaveIniSettingsToDisk(kUserLayout);
            app.toast(ToastLevel::Info, std::string("Layout saved to ") + kUserLayout);
        }
        if (ImGui::MenuItem("Load layout")) {
            if (std::filesystem::exists(kUserLayout)) {
                ImGui::LoadIniSettingsFromDisk(kUserLayout);
                app.toast(ToastLevel::Info, "Layout loaded");
            } else {
                app.toast(ToastLevel::Warn, "No saved layout yet (use Save layout first)");
            }
        }
        if (ImGui::MenuItem("Reset layout")) want_reset_layout = true;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Help")) {
        if (ImGui::MenuItem("Help and glossary", "F1")) app.show_help = !app.show_help;
        ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
}

void draw_status_bar(App& app) {
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
    float h = ImGui::GetFrameHeight() + 8;
    // Begin() is always paired with End() for viewport side bars.
    if (!ImGui::BeginViewportSideBar("##status", vp, ImGuiDir_Down, h, flags)) {
        ImGui::End();
        return;
    }
    const mm::Frame* f = app.f.get();
    ImU32 dot;
    std::string conn_label;
    switch (f ? f->conn : mm::ConnState::Disconnected) {
        case mm::ConnState::Connected: dot = IM_COL32(90, 210, 120, 255); conn_label = "Connected"; break;
        case mm::ConnState::Connecting: dot = IM_COL32(230, 180, 70, 255); conn_label = "Connecting"; break;
        default: dot = IM_COL32(220, 80, 80, 255); conn_label = "Disconnected"; break;
    }
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    float r = 5;
    dl->AddCircleFilled(ImVec2(p.x + r + 2, p.y + ImGui::GetTextLineHeight() * 0.5f), r, dot);
    ImGui::Dummy(ImVec2(r * 2 + 6, ImGui::GetTextLineHeight()));
    ImGui::SameLine(0, 0);
    ImGui::Text("%s  %s", conn_label.c_str(), f ? f->endpoint.c_str() : "");
    if (f && f->since_rx_s >= 0 && f->conn == mm::ConnState::Connected) {
        ImGui::SameLine();
        ImGui::TextDisabled("(last msg %.1fs ago)", f->since_rx_s);
    }

    ImGui::SameLine(0, 24);
    if (!trainer_connected(app)) {
        ImGui::TextColored(ImVec4(0.95f, 0.70f, 0.30f, 1), "No data - trainer not connected");
    } else {
        const mm::Metrics& m = *f->metrics;
        ImGui::Text("Trainer: %s", m.state.c_str());
        ImGui::SameLine(0, 18);
        ImGui::Text("Mode: %s", m.mode.c_str());
        ImGui::SameLine(0, 18);
        ImGui::Text("SPS: %.0f", m.sps);
        ImGui::SameLine(0, 18);
        ImGui::Text("Env steps: %s", ui::fmt_count(m.env_steps).c_str());
        if (m.mem_warning) {
            ImGui::SameLine(0, 18);
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.35f, 1), "MEM: %s", m.mem_warning->c_str());
        }
    }
    ImGui::End();
}

void draw_toasts(App& app) {
    // Expire old toasts; newest at the bottom.
    while (!app.toasts.empty() && app.ui_time - app.toasts.front().created > 6.0) app.toasts.pop_front();
    if (app.toasts.empty()) return;
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImVec2 pos(vp->WorkPos.x + vp->WorkSize.x - 12, vp->WorkPos.y + vp->WorkSize.y - 12);
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always, ImVec2(1, 1));
    ImGui::SetNextWindowBgAlpha(0.92f);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                             ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize;
    if (!ImGui::Begin("##toasts", nullptr, flags)) {
        ImGui::End();
        return;
    }
    for (const Toast& t : app.toasts) {
        ImVec4 col = t.level == ToastLevel::Error  ? ImVec4(1.0f, 0.45f, 0.40f, 1)
                     : t.level == ToastLevel::Warn ? ImVec4(0.95f, 0.78f, 0.35f, 1)
                                                   : ImVec4(0.55f, 0.85f, 0.95f, 1);
        ImGui::PushStyleColor(ImGuiCol_Text, col);
        ImGui::TextUnformatted(t.text.c_str());
        ImGui::PopStyleColor();
    }
    ImGui::End();
}

void handle_shortcuts(App& app) {
    ImGuiIO& io = ImGui::GetIO();
    if (ImGui::IsKeyPressed(ImGuiKey_F1, false)) app.show_help = !app.show_help;
    if (io.WantTextInput) return;
    std::string state = trainer_state(app);
    if (ImGui::IsKeyPressed(ImGuiKey_Space, false) && trainer_connected(app)) {
        if (state == "running") app.cmd("pause");
        else if (state == "paused") app.cmd("resume");
    }
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S, false)) app.cmd("save_checkpoint");
    if (!io.KeyCtrl && !io.KeyAlt) {
        if (ImGui::IsKeyPressed(ImGuiKey_1, false)) app.cam_mode = CamMode::FirstPerson;
        if (ImGui::IsKeyPressed(ImGuiKey_2, false)) app.cam_mode = CamMode::ThirdPerson;
        if (ImGui::IsKeyPressed(ImGuiKey_3, false)) app.cam_mode = CamMode::Free;
    }
}

void draw_help_window(App& app) {
    if (!app.show_help) return;
    ImGui::SetNextWindowSize(ImVec2(560, 520), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Help and glossary", &app.show_help)) {
        ui::label_bold("Keyboard shortcuts");
        ImGui::BulletText("Space - pause / resume training");
        ImGui::BulletText("Ctrl+S - save a checkpoint now (at the next safe point)");
        ImGui::BulletText("F1 - this window");
        ImGui::BulletText("1 / 2 / 3 - viewport camera: first-person / third-person follow / free");
        ImGui::BulletText("Free camera: hold right mouse button to look, WASD to move, Q/E down/up");
        ImGui::Spacing();
        ui::label_bold("Glossary");
        ImGui::BulletText(
            "Environment: one independent copy of the voxel world the simulator runs. "
            "Many environments run in parallel, and the simulator is always headless.");
        ImGui::BulletText(
            "Agent: the player inside one environment. Each environment has exactly one agent, "
            "so the Agent Inspector shows the agent of the selected environment.");
        ImGui::BulletText(
            "Policy: the single neural network (GRU actor-critic) shared by all agents. "
            "There is one set of weights, not one network per environment.");
        ImGui::BulletText(
            "Workers: CPU threads used to step environments and compute observations. "
            "More workers means more parallel simulation, up to the number of cores.");
        ImGui::BulletText(
            "Steps: environment transitions summed over all environments. "
            "One step of one agent is one step; 16 envs for 10 steps each is 160 steps.");
        ImGui::BulletText("Update: one PPO optimizer iteration, performed once per rollout.");
        ImGui::Spacing();
        ui::label_bold("Modes");
        ImGui::BulletText("fast - no visualization or instrumentation, maximum throughput.");
        ImGui::BulletText("live - throttled snapshot and network visualization while training.");
        ImGui::BulletText("debug - fixed seeds, step-by-step debugging while paused.");
    }
    ImGui::End();
}

// ---------------------------------------------------------------- acks -> toasts

void pump_acks(App& app) {
    if (!app.f || !app.f->acks) return;
    double newest = app.last_ack_time;
    for (const mm::Ack& a : *app.f->acks) {
        if (a.received_at <= app.last_ack_time) continue;
        newest = std::max(newest, a.received_at);
        if (!a.ok) {
            app.toast(ToastLevel::Error, a.cmd + " failed: " + a.msg);
        } else if (a.cmd == "subscribe" || a.cmd == "list_checkpoints" || a.cmd == "get_history" ||
                   a.cmd == "select_env") {
            // Background plumbing: stay quiet.
        } else {
            app.toast(ToastLevel::Info, a.cmd + ": " + (a.msg.empty() ? "ok" : a.msg));
        }
    }
    app.last_ack_time = newest;
}

void print_usage() {
    std::printf("usage: minemind_studio [--host H] [--port P] [--visualize] [--camera first|third|free] [--screenshot out.png --frames N]\n");
}

}  // namespace

// ---------------------------------------------------------------- App methods

void App::toast(ToastLevel level, std::string text) {
    toasts.push_back(Toast{level, std::move(text), ui_time});
    while (toasts.size() > 8) toasts.pop_front();
}

int App::cmd(const std::string& name, nlohmann::json args) { return client.send(name, std::move(args)); }

void App::set_selected_env(int env) {
    selected_env = env;
    cmd("select_env", {{"env_id", env}});
}

void App::set_visualize(bool on) {
    viz_enabled = on;
    cmd("set_config", {{"values", {{"visualize", on}}}});
    client.set_snapshot_wanted(on);
}

std::string trainer_state(const App& app) {
    if (!app.f || !app.f->metrics) return "idle";
    return app.f->metrics->state;
}

bool trainer_connected(const App& app) {
    return app.f && app.f->conn == mm::ConnState::Connected && app.f->metrics != nullptr;
}

int agents_per_world(const App& app) {
    if (app.f && app.f->snapshot && app.f->snapshot->agents_per_world > 0) return app.f->snapshot->agents_per_world;
    if (app.f && app.f->metrics && app.f->metrics->agents && *app.f->metrics->agents > 0) return *app.f->metrics->agents;
    return 1;
}

int selectable_agents(const App& app) {
    int worlds = trainer_connected(app) ? app.f->metrics->num_envs : 0;
    return worlds * agents_per_world(app);
}

int global_agent_index(const App& app, const mm::TeamMember& m) {
    const mm::Snapshot* s = app.f ? app.f->snapshot.get() : nullptr;
    int k = agents_per_world(app);
    if (s && s->world_id >= 0 && k > 0) return s->world_id * k + m.agent;
    return m.agent;
}

std::string agent_label(const App& app, int index) {
    int k = agents_per_world(app);
    char buf[80];
    if (k > 1) std::snprintf(buf, sizeof buf, "Agent %d (world %d, bot %d)", index, index / k, index % k);
    else std::snprintf(buf, sizeof buf, "Env %d", index);
    return buf;
}

// ---------------------------------------------------------------- main

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = 7777;
    std::string shot_path;
    int shot_frames = 0;
    bool want_viz = false;
    std::string camera;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (a == "--host") host = next();
        else if (a == "--port") port = std::atoi(next().c_str());
        else if (a == "--screenshot") shot_path = next();
        else if (a == "--frames") shot_frames = std::atoi(next().c_str());
        else if (a == "--camera") camera = next();
        else if (a == "--visualize") want_viz = true;  // turn visualization on once connected
        else if (a == "-h" || a == "--help") { print_usage(); return 0; }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); print_usage(); return 2; }
    }

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 0);

    SDL_Window* window = SDL_CreateWindow("MineMind Training Studio", 1600, 900,
                                          SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY |
                                              SDL_WINDOW_HIDDEN);
    if (!window) {
        std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return 1;
    }
    SDL_GLContext gl = SDL_GL_CreateContext(window);
    if (!gl) {
        std::fprintf(stderr, "SDL_GL_CreateContext failed: %s\n", SDL_GetError());
        return 1;
    }
    SDL_GL_MakeCurrent(window, gl);
    SDL_GL_SetSwapInterval(1);  // vsync
    SDL_ShowWindow(window);
    if (!glp::load()) {
        std::fprintf(stderr, "required OpenGL 3.3 entry points are missing\n");
        return 1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags &= ~ImGuiConfigFlags_ViewportsEnable;
    io.IniFilename = "imgui.ini";  // dock + window layout is persisted here
    apply_theme();
    ImGui_ImplSDL3_InitForOpenGL(window, gl);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    mm::Client client(host, port);
    client.start();
    App app(client);
    if (camera == "third") app.cam_mode = CamMode::ThirdPerson;
    else if (camera == "free") app.cam_mode = CamMode::Free;
    client.set_nn_wanted(false);

    const ImGuiID dock_id = ImHashStr("MainDockSpace");  // no window context yet, so hash directly
    bool want_reset_layout = false;
    bool want_quit = false;
    bool running = true;
    long long frame_no = 0;
    bool screenshot_taken = false;
    bool viz_sent = false;

    while (running && !want_quit) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            ImGui_ImplSDL3_ProcessEvent(&ev);
            if (ev.type == SDL_EVENT_QUIT) running = false;
            if (ev.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && ev.window.windowID == SDL_GetWindowID(window))
                running = false;
        }
        if (!running) break;

        app.f = client.frame();
        app.ui_time = ImGui::GetTime();
        if (app.f && app.f->has_hello) mm::palette_set_names(app.f->hello.blocks);
        if (want_viz && !viz_sent && trainer_connected(app)) {
            app.set_visualize(true);
            viz_sent = true;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        handle_shortcuts(app);
        pump_acks(app);
        draw_main_menu(app, want_reset_layout, want_quit);

        ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::DockSpaceOverViewport(dock_id, vp, ImGuiDockNodeFlags_None);
        if (want_reset_layout || !layout_is_built(dock_id)) {
            build_default_layout(dock_id);
            want_reset_layout = false;
        }

        draw_status_bar(app);
        if (app.open_viewport) draw_viewport(app, &app.open_viewport);
        else client.set_snapshot_wanted(false);
        if (app.open_controls) draw_controls(app, &app.open_controls);
        if (app.open_training) draw_training(app, &app.open_training);
        if (app.open_agent) draw_agent(app, &app.open_agent);
        if (app.open_graphs) draw_graphs(app, &app.open_graphs);
        app.nn_visible = false;
        if (app.open_nn) draw_nn(app, &app.open_nn);
        if (app.open_checkpoints) draw_checkpoints(app, &app.open_checkpoints);
        if (app.open_console) draw_console(app, &app.open_console);
        if (app.open_chat) draw_chat(app, &app.open_chat);
        draw_help_window(app);
        draw_toasts(app);

        // NN instrumentation only while the NN panel is actually on screen (visible tab, not collapsed).
        client.set_nn_wanted(app.nn_visible);

        ImGui::Render();
        int fb_w = 0, fb_h = 0;
        SDL_GetWindowSizeInPixels(window, &fb_w, &fb_h);
        glViewport(0, 0, fb_w, fb_h);
        glClearColor(0.075f, 0.082f, 0.098f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        frame_no++;
        if (!shot_path.empty() && !screenshot_taken && frame_no >= shot_frames) {
            std::vector<unsigned char> px((size_t)fb_w * fb_h * 4);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadPixels(0, 0, fb_w, fb_h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
            // Flip rows: GL origin is bottom-left.
            std::vector<unsigned char> row((size_t)fb_w * 4);
            for (int y = 0; y < fb_h / 2; y++) {
                unsigned char* a = px.data() + (size_t)y * fb_w * 4;
                unsigned char* b = px.data() + (size_t)(fb_h - 1 - y) * fb_w * 4;
                std::memcpy(row.data(), a, row.size());
                std::memcpy(a, b, row.size());
                std::memcpy(b, row.data(), row.size());
            }
            if (stbi_write_png(shot_path.c_str(), fb_w, fb_h, 4, px.data(), fb_w * 4))
                std::printf("screenshot written: %s (%dx%d, %lld frames)\n", shot_path.c_str(), fb_w, fb_h, frame_no);
            else
                std::fprintf(stderr, "failed to write %s\n", shot_path.c_str());
            screenshot_taken = true;
            running = false;
        }

        SDL_GL_SwapWindow(window);
    }

    client.stop();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    SDL_GL_DestroyContext(gl);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
