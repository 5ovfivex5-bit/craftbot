// Telemetry data model for the MineMind dashboard (see PROTOCOL.md section 2).
// Everything here is filled exclusively from messages received from the trainer.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace mm {

enum class ConnState { Disconnected, Connecting, Connected };

struct Hello {
    int schema = 0;
    std::string run_id;
    std::string state;
    nlohmann::json config;
    int world_size[3] = {0, 0, 0};
    std::vector<std::string> actions;
    std::vector<std::string> items;
    std::vector<std::string> blocks;
};

struct Metrics {
    double t = 0;
    std::string state = "idle";   // idle | running | paused | stopping | stopped
    std::string mode = "fast";    // fast | live | debug
    long long env_steps = 0;
    long long target_steps = 0;
    double sps = 0;
    long long episodes = 0;
    double avg_ep_len = 0;
    double avg_reward = 0;
    std::map<std::string, double> success_rate;
    long long updates = 0;
    double policy_loss = 0, value_loss = 0, entropy = 0, approx_kl = 0;
    double cpu_percent = 0, rss_mb = 0;
    std::optional<std::string> mem_warning;
    double elapsed_s = 0;
    std::optional<double> eta_s;  // nullopt -> "Calculating"
    std::optional<std::string> latest_checkpoint;
    std::optional<std::string> curriculum_stage;
    int num_envs = 0;
    nlohmann::json config;
    // v6 team metrics (absent from older trainers -> nullopt).
    std::optional<int> agents;               // agents per world
    std::optional<double> msg_entropy;
    std::optional<double> msg_rate;          // fraction of non-silent tokens
    std::optional<double> chest_deposits;    // per 1k agent steps
    std::optional<double> chest_withdraws;   // per 1k agent steps
    std::optional<double> gives;             // per 1k agent steps
    // v7 success breakdowns: family -> 0..1 and task id -> 0..1 (empty when absent).
    std::map<std::string, double> family_success;
    std::map<std::string, double> task_success;
};

struct Player {
    double x = 0, y = 0, z = 0;
    double vx = 0, vy = 0, vz = 0;
    double yaw = 0, pitch = 0;
    double health = 0;
    bool on_ground = false;
};

struct Target {
    int x = 0, y = 0, z = 0;
    std::string block;
};

// Mob or other entity from the snapshot's optional "entities" array (absent on older trainers).
struct Entity {
    std::string type;             // registry name, e.g. "zombie"
    double x = 0, y = 0, z = 0;   // feet position
    std::optional<double> health;
};

// v6: one teammate (or the selected agent) inside the snapshot's world.
struct TeamMember {
    int agent = 0;                // agent index k within its world
    double x = 0, y = 0, z = 0;
    double yaw = 0, pitch = 0;
    double health = 0;
    std::string last_action;
    int msg = 0;
    std::map<std::string, int> inventory;
};

struct ChestInfo {
    int x = 0, y = 0, z = 0;
    std::map<std::string, int> items;
};

struct ChatEntry {
    long long step = 0;
    int agent = 0;
    int msg = 0;
};

struct Snapshot {
    int env_id = -1;
    long long episode = 0;
    long long seed = 0;
    long long step = 0;
    Player player;
    std::string last_action;
    double last_reward = 0;
    double ep_reward = 0;
    std::string task;              // v7 task id; empty when the trainer does not send it
    std::string family;            // v7 task family ("gather", "social", ...); empty when absent
    std::optional<double> time_of_day;  // [0,1) for v7 trainers, 0 dawn, 0.25 noon, 0.5 dusk, 0.75 midnight
    std::optional<Target> target;
    std::map<std::string, int> inventory;
    std::vector<std::string> action_history;
    int origin[3] = {0, 0, 0};
    int size[3] = {0, 0, 0};
    std::vector<uint8_t> blocks;  // index (y*sz+z)*sx+x, relative to origin
    int obs_res = 0;
    std::vector<uint8_t> obs;     // res*res*2: [block id, depth]

    // v6 team fields. has_team is false for older trainers (hide the team UI then).
    bool has_team = false;
    int world_id = -1;
    int agent_id = -1;            // selected agent's k within its world
    int agents_per_world = 0;
    std::vector<TeamMember> team;  // every agent of the world, selected one included
    std::vector<ChestInfo> chests;
    std::vector<ChatEntry> chat;   // newest last
    std::vector<Entity> entities;  // mobs and others, empty when the snapshot has none
};

struct LayerAct {
    std::vector<int> shape;
    std::vector<float> sample;
    float mean = 0, std = 0, min = 0, max = 0;
    std::vector<float> hist_edges;
    std::vector<int> hist_counts;
};

struct Stat4 {
    float mean = 0, std = 0, min = 0, max = 0;
};

struct Activations {
    int env_id = -1;
    long long step = 0;
    std::vector<float> action_probs;
    float entropy = 0, value = 0;
    std::map<std::string, LayerAct> layers;
    bool has_gru = false;
    Stat4 gru;
    float gru_norm = 0;
    std::map<std::string, float> grad_norms;
    std::map<std::string, Stat4> weight_stats;
    bool nan_inf = false;
};

struct LayerInfo {
    std::string name, type;
    long long params = 0;
    std::vector<int> in_shape, out_shape;
    bool trainable = true;
};

struct ModelInfo {
    std::string name;
    long long params_total = 0;
    std::vector<int> img_shape;
    int vec_dim = 0;
    int gru_hidden = 0;
    std::vector<LayerInfo> layers;
    std::vector<std::pair<std::string, std::string>> edges;
};

struct LogLine {
    double t = 0;
    std::string level;  // info | warn | error
    std::string msg;
};

struct Checkpoint {
    std::string path;
    long long env_steps = 0;
    long long updates = 0;
    double time = 0;
    long long size_bytes = 0;
};

struct Ack {
    int id = 0;
    std::string cmd;
    bool ok = false;
    std::string msg;
    double received_at = 0;  // seconds since client start
};

// Immutable view handed to the UI thread. A new Frame is published whenever
// anything changes; the UI keeps the shared_ptr it got for the whole frame.
struct Frame {
    ConnState conn = ConnState::Disconnected;
    std::string conn_text;
    std::string endpoint;
    double since_rx_s = -1;      // seconds since last message (-1: never)
    bool has_hello = false;
    Hello hello;
    std::shared_ptr<const Metrics> metrics;
    std::shared_ptr<const std::vector<Metrics>> history;
    std::shared_ptr<const Snapshot> snapshot;
    std::shared_ptr<const Activations> acts;
    std::shared_ptr<const ModelInfo> model;
    std::shared_ptr<const std::vector<Checkpoint>> checkpoints;
    std::shared_ptr<const std::vector<LogLine>> logs;
    std::shared_ptr<const std::vector<Ack>> acks;
    std::map<int, std::string> pending;  // command id -> cmd name, not yet acked
    bool snapshot_wanted = false;
    bool nn_wanted = false;
    uint64_t gen = 0;
};

// Block palette (index == block id). Colours come from the generated registry table
// (src/gen_palette.h); the trainer's hello.blocks names pick the entry per id, and the v6
// table covers ids 0..19 of trainers whose names are not in the registry.
struct BlockStyle {
    unsigned char rgb[3];
    unsigned char speckle[3];
    bool has_speckle;
    bool pseudo = false;  // agent and mob ids (entities, never voxels)
};
const BlockStyle& block_style(int id);
// Rebuilds the id -> style table from the hello's block names. Cheap when unchanged. UI thread only.
void palette_set_names(const std::vector<std::string>& names);
int palette_size();

// Team colour for agent k (PROTOCOL.md 6.7: 0 blue, 1 red, 2 green, 3 yellow, 4 purple).
struct Rgb {
    unsigned char r, g, b;
};
Rgb bot_color(int k);

// Mob look: box size and colour for an entity type name (grey 0.6 x 1.8 when unknown).
struct MobLook {
    float width, height;
    Rgb color;
    bool known;
};
MobLook mob_look(const std::string& type);


}  // namespace mm
