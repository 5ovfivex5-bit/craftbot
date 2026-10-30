// Shared application context passed to every panel. Owned by main.cpp.
#pragma once

#include <deque>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "net/client.h"
#include "net/protocol.h"

enum class ToastLevel { Info, Warn, Error };

struct Toast {
    ToastLevel level;
    std::string text;
    double created;  // ImGui time, seconds
};

enum class CamMode { FirstPerson = 0, ThirdPerson = 1, Free = 2 };

// Window names double as dock/ini identifiers; keep them stable.
namespace win {
inline constexpr const char* kViewport = "Viewport";
inline constexpr const char* kTraining = "Training Stats";
inline constexpr const char* kControls = "Controls";
inline constexpr const char* kAgent = "Agent Inspector";
inline constexpr const char* kGraphs = "Graphs";
inline constexpr const char* kNN = "Neural Network";
inline constexpr const char* kCheckpoints = "Checkpoints";
inline constexpr const char* kConsole = "Console";
inline constexpr const char* kChat = "Chat";
}  // namespace win

struct App {
    explicit App(mm::Client& c) : client(c) {}

    mm::Client& client;
    std::shared_ptr<const mm::Frame> f;  // frame for the current UI frame
    double ui_time = 0;                  // ImGui time, seconds
    double last_ack_time = -1;           // newest ack already turned into a toast

    int selected_env = 0;                // shared by Viewport and Agent Inspector
    std::string selected_checkpoint;     // path picked in the Checkpoints panel
    bool viz_enabled = false;            // "Visualization enabled" toggle
    CamMode cam_mode = CamMode::FirstPerson;
    bool show_help = false;

    // Panel open flags (View menu / window close button).
    bool open_viewport = true, open_training = true, open_controls = true, open_agent = true;
    bool open_graphs = true, open_nn = true, open_checkpoints = true, open_console = true;
    bool open_chat = true;
    bool nn_visible = false;             // NN panel actually drawn this frame (drives subscribe nn)

    std::deque<Toast> toasts;

    void toast(ToastLevel level, std::string text);
    int cmd(const std::string& name, nlohmann::json args = nlohmann::json::object());
    void set_selected_env(int env);
    void set_visualize(bool on);
};

// Trainer state string from the latest metrics ("idle" when nothing received).
std::string trainer_state(const App& app);
bool trainer_connected(const App& app);
// v6 teams: agents per world (1 for older trainers), number of selectable agent indices,
// and the global agent index (world_id * agents_per_world + k) that select_env takes.
int agents_per_world(const App& app);
int selectable_agents(const App& app);
int global_agent_index(const App& app, const mm::TeamMember& m);
// Label for an agent index in selectors: "Env N" or "Agent N (world w, bot k)".
std::string agent_label(const App& app, int index);
