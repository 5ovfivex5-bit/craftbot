// Agent Inspector: state of the selected environment's agent, observation images, debug steps.
#include <cmath>
#include <cstdio>
#include <vector>

#include <imgui.h>

#include "app.h"
#include "net/protocol.h"
#include "panels.h"
#include "ui_util.h"

namespace {

struct ObsTextures {
    const mm::Snapshot* src = nullptr;  // snapshot the textures were built from
    unsigned color = 0, depth = 0;
    int res = 0;
    long long debug_n = 1;
};

ObsTextures obs;

void build_obs_textures(const mm::Snapshot& s) {
    if (s.obs.empty() || s.obs_res <= 0) return;
    const int r = s.obs_res;
    std::vector<unsigned char> col((size_t)r * r * 4), dep((size_t)r * r * 4);
    for (int i = 0; i < r * r; i++) {
        int id = s.obs[i * 2];
        int d = s.obs[i * 2 + 1];
        const mm::BlockStyle& st = mm::block_style(id);
        float shade = 1.0f - 0.55f * (d / 255.0f);  // nearer surfaces are brighter
        for (int k = 0; k < 3; k++) col[i * 4 + k] = (unsigned char)(st.rgb[k] * shade);
        col[i * 4 + 3] = 255;
        unsigned char g = (unsigned char)(255 - d);  // near = bright, sky (255) = black
        dep[i * 4 + 0] = dep[i * 4 + 1] = dep[i * 4 + 2] = g;
        dep[i * 4 + 3] = 255;
    }
    upload_texture(obs.color, r, r, col.data());
    upload_texture(obs.depth, r, r, dep.data());
    obs.res = r;
}

void draw_snapshot_fields(const mm::Snapshot& s, const mm::Metrics* m) {
    ui::section("Identity");
    ui::kv("Environment id", std::to_string(s.env_id));
    ui::kv("Seed", std::to_string(s.seed));
    ui::kv("Episode", std::to_string(s.episode));
    ui::kv("Step in episode", std::to_string(s.step));
    ui::kv("Trainer state", m ? m->state : std::string("-"));

    ui::section("Agent");
    const mm::Player& p = s.player;
    ui::kv("Position (x, y, z)", ui::fmt_num(p.x, 2) + ", " + ui::fmt_num(p.y, 2) + ", " + ui::fmt_num(p.z, 2));
    ui::kv("Block cell", std::to_string((int)std::floor(p.x)) + ", " + std::to_string((int)std::floor(p.y)) + ", " +
                             std::to_string((int)std::floor(p.z)));
    ui::kv("Velocity", ui::fmt_num(p.vx, 2) + ", " + ui::fmt_num(p.vy, 2) + ", " + ui::fmt_num(p.vz, 2));
    ui::kv("Yaw / pitch", ui::fmt_num(p.yaw, 1) + " / " + ui::fmt_num(p.pitch, 1));
    ui::kv("Health", ui::fmt_num(p.health, 1));
    ui::kv("On ground", p.on_ground ? "yes" : "no");
    if (s.time_of_day) {
        const double t = *s.time_of_day;
        const char* phase = t < 0.1 ? "dawn" : t < 0.45 ? "day" : t < 0.6 ? "dusk" : "night";
        ui::kv("Time of day", ui::fmt_num(t, 3) + " (" + phase + ")");
    }
    if (!s.task.empty()) ui::kv("Task", s.task);
    if (!s.family.empty()) ui::kv("Family", s.family);
    if (s.target)
        ui::kv("Target block", s.target->block + " @ " + std::to_string(s.target->x) + ", " +
                                   std::to_string(s.target->y) + ", " + std::to_string(s.target->z));
    else
        ui::kv("Target block", "none in reach");

    if (!s.entities.empty()) {
        ui::section("Entities");
        for (const mm::Entity& e : s.entities) {
            mm::Rgb c = mm::mob_look(e.type).color;
            char line[160];
            std::snprintf(line, sizeof line, "%s  (%.1f, %.1f, %.1f)%s", e.type.c_str(), e.x, e.y, e.z,
                          e.health ? (std::string("  hp ") + ui::fmt_num(*e.health, 1)).c_str() : "");
            ImGui::TextColored(ImVec4(c.r / 255.0f * 0.5f + 0.5f, c.g / 255.0f * 0.5f + 0.5f, c.b / 255.0f * 0.5f + 0.5f, 1.0f), "%s", line);
        }
    }

    ui::section("Last transition");
    ui::kv("Last action", s.last_action.empty() ? std::string("-") : s.last_action);
    ui::kv("Last reward", ui::fmt_num(s.last_reward, 3));
    ui::kv("Episode reward", ui::fmt_num(s.ep_reward, 3));

    ui::section("Inventory");
    if (s.inventory.empty()) ImGui::TextDisabled("empty");
    for (const auto& [item, n] : s.inventory) ui::kv(item.c_str(), std::to_string(n));

    ui::section("Action history (newest first)");
    if (s.action_history.empty()) ImGui::TextDisabled("no actions yet");
    if (ImGui::BeginChild("##hist", ImVec2(0, 130), ImGuiChildFlags_Borders)) {
        for (auto it = s.action_history.rbegin(); it != s.action_history.rend(); ++it) {
            ImGui::TextUnformatted(it->c_str());
        }
    }
    ImGui::EndChild();
}

}  // namespace

void draw_agent(App& app, bool* p_open) {
    ImGui::SetNextWindowSize(ImVec2(430, 720), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(win::kAgent, p_open)) {
        ImGui::End();
        return;
    }
    const bool connected = trainer_connected(app);

    const int n_select = connected ? selectable_agents(app) : 0;
    if (n_select > 0) {
        ImGui::SetNextItemWidth(220);
        if (ImGui::BeginCombo("Environment", agent_label(app, app.selected_env).c_str())) {
            for (int i = 0; i < n_select; i++) {
                if (ImGui::Selectable(agent_label(app, i).c_str(), i == app.selected_env) && i != app.selected_env)
                    app.set_selected_env(i);
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        ui::help_marker("Each environment has one agent per team slot. Selecting one sends select_env to the trainer, "
                        "which snapshots and inspects that agent.");
    }

    if (!connected || !app.f->snapshot) {
        if (!connected) ui::no_data("Agent state arrives as snapshots while Visualization is enabled.");
        else if (!app.viz_enabled) ImGui::TextDisabled("Tick 'Visualization enabled' in the Viewport to receive agent snapshots.");
        else ImGui::TextDisabled("Waiting for the first snapshot of the selected environment...");
        ImGui::End();
        return;
    }
    const mm::Snapshot& s = *app.f->snapshot;
    if (obs.src != &s) {
        obs.src = &s;
        build_obs_textures(s);
    }
    draw_snapshot_fields(s, app.f->metrics.get());

    if (s.has_team) {
        ui::section("Team");
        ImGui::TextDisabled("world %d, %d agents. Click a bot to inspect it.", s.world_id, (int)s.team.size());
        for (const mm::TeamMember& m : s.team) {
            mm::Rgb c = mm::bot_color(m.agent);
            const bool sel = m.agent == s.agent_id;
            char label[48];
            std::snprintf(label, sizeof label, "bot %d%s##team%d", m.agent, sel ? "  (selected)" : "", m.agent);
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(c.r, c.g, c.b, 255));
            if (ImGui::Selectable(label, sel) && !sel) app.set_selected_env(global_agent_index(app, m));
            ImGui::PopStyleColor();
            ImGui::Indent(14);
            ImGui::TextDisabled("hp %.1f   last %s   msg m%d", m.health, m.last_action.empty() ? "-" : m.last_action.c_str(), m.msg);
            std::string inv;
            for (const auto& [item, n] : m.inventory) {
                if (!inv.empty()) inv += ", ";
                inv += item + " " + std::to_string(n);
            }
            ImGui::TextWrapped("%s", inv.empty() ? "inventory empty" : inv.c_str());
            ImGui::Unindent(14);
        }
    }

    ui::section("Observation (what the policy sees)");
    if (obs.color && obs.res > 0) {
        float side = 150.0f;
        ImGui::BeginGroup();
        ImGui::Image((ImTextureID)(intptr_t)obs.color, ImVec2(side, side), ImVec2(0, 0), ImVec2(1, 1));
        ImGui::TextDisabled("block id (colour, shaded by depth)");
        ImGui::EndGroup();
        ImGui::SameLine();
        ImGui::BeginGroup();
        ImGui::Image((ImTextureID)(intptr_t)obs.depth, ImVec2(side, side), ImVec2(0, 0), ImVec2(1, 1));
        ImGui::TextDisabled("depth (near = bright, sky = black)");
        ImGui::EndGroup();
        ImGui::TextDisabled("%dx%d rays, one pixel per ray; the policy input is this image plus a state vector.", obs.res, obs.res);
    } else {
        ImGui::TextDisabled("observation not included in this snapshot");
    }

    ui::section("Debug stepping (paused only)");
    bool paused = trainer_state(app) == "paused";
    ImGui::BeginDisabled(!paused);
    ImGui::SetNextItemWidth(90);
    ImGui::InputScalar("steps", ImGuiDataType_S64, &obs.debug_n, nullptr, nullptr, "%lld");
    if (obs.debug_n < 1) obs.debug_n = 1;
    if (obs.debug_n > 1000) obs.debug_n = 1000;
    if (ImGui::Button("Tick", ImVec2(100, 0))) app.cmd("debug_step", {{"n", obs.debug_n}, {"kind", "tick"}});
    ImGui::SameLine();
    if (ImGui::Button("Decision", ImVec2(100, 0))) app.cmd("debug_step", {{"n", obs.debug_n}, {"kind", "decision"}});
    ImGui::EndDisabled();
    ImGui::SameLine();
    ui::help_marker("Tick advances the simulator by one physics tick. Decision runs the policy once and applies its action. "
                    "Both act on a dedicated debug environment seeded from the selected episode; training is untouched.");
    if (!paused) ImGui::TextDisabled("Pause training to enable debug stepping.");

    ImGui::End();
}
