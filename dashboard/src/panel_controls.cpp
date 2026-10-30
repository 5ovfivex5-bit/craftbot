// Controls: run buttons (enabled per trainer state), config editor with queued-apply feedback.
#include <cstdio>
#include <string>
#include <vector>

#include <imgui.h>

#include "app.h"
#include "panels.h"
#include "ui_util.h"

namespace {

constexpr long long kEnvPresets[] = {1, 4, 8, 16, 32, 64, 128, 256, 512, 1000};
constexpr int kNumPresets = sizeof(kEnvPresets) / sizeof(kEnvPresets[0]);

struct Form {
    bool initialised = false;
    long long total_steps = 1000000;
    long long num_envs = 16;
    bool custom_envs = false;
    double lr = 3e-4;
    long long minibatch = 512;
    long long rollout_len = 128;
    long long ckpt_every = 50000;
    long long workers = 4;
    long long snapshot_hz = 10;
    long long nn_hz = 2;
    int mode = 0;  // 0 fast, 1 live, 2 debug
    int apply_id = -1;
    std::string apply_status;
    bool apply_error = false;
};

Form form;

void load_from_config(const nlohmann::json& c) {
    auto geti = [&](const char* k, long long& out) {
        if (c.contains(k) && c[k].is_number()) out = c[k].get<long long>();
    };
    geti("total_steps", form.total_steps);
    geti("num_envs", form.num_envs);
    geti("minibatch_size", form.minibatch);
    geti("rollout_len", form.rollout_len);
    geti("ckpt_every", form.ckpt_every);
    geti("workers", form.workers);
    geti("snapshot_hz", form.snapshot_hz);
    geti("nn_hz", form.nn_hz);
    if (c.contains("lr") && c["lr"].is_number()) form.lr = c["lr"].get<double>();
    if (c.contains("mode") && c["mode"].is_string()) {
        std::string m = c["mode"].get<std::string>();
        form.mode = m == "live" ? 1 : m == "debug" ? 2 : 0;
    }
    bool preset = false;
    for (auto p : kEnvPresets) preset |= p == form.num_envs;
    form.custom_envs = !preset;
}

void ensure_initialised(App& app) {
    if (form.initialised) return;
    if (app.f && app.f->metrics && !app.f->metrics->config.empty()) {
        load_from_config(app.f->metrics->config);
        form.initialised = true;
    } else if (app.f && app.f->has_hello && !app.f->hello.config.empty()) {
        load_from_config(app.f->hello.config);
        form.initialised = true;
    }
}

void run_button(App& app, const char* label, const char* cmd, bool enabled, const char* tip, float width) {
    ImGui::BeginDisabled(!enabled);
    if (ImGui::Button(label, ImVec2(width, 0))) app.cmd(cmd);
    ImGui::EndDisabled();
    if (!enabled && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("%s", tip);
}

}  // namespace

void draw_controls(App& app, bool* p_open) {
    ImGui::SetNextWindowSize(ImVec2(440, 600), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(win::kControls, p_open)) {
        ImGui::End();
        return;
    }
    const bool connected = trainer_connected(app);
    const std::string state = trainer_state(app);
    ensure_initialised(app);

    if (connected && app.f->metrics->mem_warning) {
        ImGui::TextColored(ImVec4(1, 0.45f, 0.35f, 1), "! %s", app.f->metrics->mem_warning->c_str());
    }

    // Run controls, enabled per trainer state.
    ui::section("Run");
    ImGui::Text("State: %s", connected ? state.c_str() : "no data");
    bool can_start = connected && (state == "idle" || state == "stopped");
    bool can_pause = connected && state == "running";
    bool can_resume = connected && state == "paused";
    bool can_stop = connected && (state == "running" || state == "paused");
    bool can_reset = connected && (state == "paused" || state == "stopped");
    bool can_save = connected && (state == "running" || state == "paused");
    bool can_load = connected && (state == "paused" || state == "stopped");
    const char* why = connected ? "not available in the current trainer state" : "trainer not connected";

    // Three buttons per row, each filling an equal share of the width.
    const float bw = (ImGui::GetContentRegionAvail().x - 2 * ImGui::GetStyle().ItemSpacing.x) / 3.0f;
    run_button(app, "Start", "start", can_start, why, bw);
    ImGui::SameLine();
    run_button(app, "Pause", "pause", can_pause, why, bw);
    ImGui::SameLine();
    run_button(app, "Resume", "resume", can_resume, why, bw);
    run_button(app, "Stop", "stop", can_stop, why, bw);
    ImGui::SameLine();
    run_button(app, "Reset", "reset", can_reset, "reset needs paused or stopped (fresh model and optimizer)", bw);
    ImGui::SameLine();
    run_button(app, "Save ckpt", "save_checkpoint", can_save, why, bw);
    bool can_load_sel = can_load && !app.selected_checkpoint.empty();
    ImGui::BeginDisabled(!can_load_sel);
    ImGui::SameLine();
    if (ImGui::Button("Load ckpt", ImVec2(bw, 0))) app.cmd("load_checkpoint", {{"path", app.selected_checkpoint}});
    ImGui::EndDisabled();
    ImGui::SameLine();
    ui::help_marker("Load restores model, optimizer, counters and RNG from the checkpoint selected in the "
                    "Checkpoints panel. It overwrites the training run and needs paused or stopped.");

    // Visualization + mode.
    ui::section("Visualization and mode");
    bool viz = app.viz_enabled;
    if (ImGui::Checkbox("Visualization enabled", &viz)) app.set_visualize(viz);
    ImGui::TextDisabled("mode");
    ImGui::SameLine();
    const char* modes[] = {"fast", "live", "debug"};
    for (int i = 0; i < 3; i++) {
        ImGui::SameLine();
        if (ImGui::RadioButton(modes[i], form.mode == i)) {
            form.mode = i;
            app.cmd("set_mode", {{"mode", modes[i]}});
        }
    }
    ui::help_marker("fast: no visualization or instrumentation. live: throttled visualization and network inspection. "
                    "debug: fixed seeds, step-by-step while paused. Mode changes never reset training.");

    // Config editor.
    ui::section("Configuration");
    ImGui::TextDisabled("Changes are queued and applied at the next rollout boundary.");
    ImGui::BeginDisabled(!connected && !form.initialised);
    if (ImGui::BeginTable("##cfg", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed, 150);
        ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);

        auto row = [](const char* name) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(name);
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-1);
        };

        row("Total steps");
        ImGui::InputScalar("##total", ImGuiDataType_S64, &form.total_steps, nullptr, nullptr, "%lld");

        row("Environments");
        {
            int preset_idx = -1;
            for (int i = 0; i < kNumPresets; i++)
                if (kEnvPresets[i] == form.num_envs) preset_idx = i;
            if (form.custom_envs) preset_idx = kNumPresets;
            char current[32] = "Custom";
            if (preset_idx >= 0 && preset_idx < kNumPresets) std::snprintf(current, sizeof current, "%lld", form.num_envs);
            ImGui::SetNextItemWidth(110);
            if (ImGui::BeginCombo("##envs", current)) {
                for (int i = 0; i < kNumPresets; i++) {
                    char item[16];
                    std::snprintf(item, sizeof item, "%lld", kEnvPresets[i]);
                    if (ImGui::Selectable(item, preset_idx == i)) {
                        form.num_envs = kEnvPresets[i];
                        form.custom_envs = false;
                    }
                }
                if (ImGui::Selectable("Custom...", form.custom_envs)) form.custom_envs = true;
                ImGui::EndCombo();
            }
            if (form.custom_envs) {
                ImGui::SameLine();
                ImGui::SetNextItemWidth(90);
                ImGui::InputScalar("##envs_custom", ImGuiDataType_S64, &form.num_envs, nullptr, nullptr, "%lld");
                if (form.num_envs < 1) form.num_envs = 1;
            }
            ImGui::SameLine();
            ui::help_marker(
                "Environments: independent copies of the world, stepped in parallel.\n"
                "Agents: one per environment; they all share one policy (one set of weights).\n"
                "Policies: the single network that every agent uses; there are never 1000 networks.");
        }

        row("Learning rate");
        ImGui::InputDouble("##lr", &form.lr, 0, 0, "%.6g");

        row("Minibatch size");
        ImGui::InputScalar("##mb", ImGuiDataType_S64, &form.minibatch, nullptr, nullptr, "%lld");

        row("Rollout length");
        ImGui::InputScalar("##roll", ImGuiDataType_S64, &form.rollout_len, nullptr, nullptr, "%lld");

        row("Checkpoint every");
        ImGui::InputScalar("##ckpt", ImGuiDataType_S64, &form.ckpt_every, nullptr, nullptr, "%lld steps");

        row("Workers");
        ImGui::InputScalar("##workers", ImGuiDataType_S64, &form.workers, nullptr, nullptr, "%lld");
        ImGui::SameLine();
        ui::help_marker("Workers: CPU threads that step environments and compute observations. "
                        "More workers means more parallel simulation, up to the number of cores.");

        row("Snapshot rate (Hz)");
        ImGui::InputScalar("##snaphz", ImGuiDataType_S64, &form.snapshot_hz, nullptr, nullptr, "%lld");

        row("Network rate (Hz)");
        ImGui::InputScalar("##nnhz", ImGuiDataType_S64, &form.nn_hz, nullptr, nullptr, "%lld");

        ImGui::EndTable();
    }
    if (form.total_steps < 1) form.total_steps = 1;
    if (form.minibatch < 1) form.minibatch = 1;
    if (form.rollout_len < 1) form.rollout_len = 1;
    if (form.workers < 0) form.workers = 0;
    ImGui::EndDisabled();

    if (ImGui::Button("Apply", ImVec2(120, 0))) {
        nlohmann::json values = {
            {"total_steps", form.total_steps},   {"num_envs", form.num_envs},     {"lr", form.lr},
            {"minibatch_size", form.minibatch},  {"rollout_len", form.rollout_len}, {"ckpt_every", form.ckpt_every},
            {"workers", form.workers},           {"snapshot_hz", form.snapshot_hz}, {"nn_hz", form.nn_hz},
            {"mode", modes[form.mode]},
        };
        form.apply_id = app.cmd("set_config", {{"values", values}});
        form.apply_status = "queued";
        form.apply_error = false;
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(!connected || !app.f || !app.f->metrics);
    if (ImGui::Button("Reload from trainer")) {
        form.initialised = false;
        ensure_initialised(app);
    }
    ImGui::EndDisabled();

    // Feedback for the last apply: queued until the ack arrives.
    if (form.apply_id >= 0 && app.f) {
        if (app.f->pending.count(form.apply_id)) {
            ImGui::TextColored(ImVec4(0.95f, 0.8f, 0.4f, 1), "Apply: queued (waiting for trainer ack)");
        } else if (app.f->acks) {
            for (auto it = app.f->acks->rbegin(); it != app.f->acks->rend(); ++it) {
                if (it->id != form.apply_id) continue;
                if (it->ok)
                    ImGui::TextColored(ImVec4(0.6f, 0.9f, 0.6f, 1), "Apply: %s", it->msg.empty() ? "accepted" : it->msg.c_str());
                else
                    ImGui::TextColored(ImVec4(1, 0.5f, 0.4f, 1), "Apply failed: %s", it->msg.c_str());
                break;
            }
        }
    }

    if (connected && app.f->metrics && !app.f->metrics->config.empty()) {
        ImGui::Spacing();
        if (ImGui::TreeNode("Config reported by trainer")) {
            for (auto& [k, v] : app.f->metrics->config.items()) {
                ImGui::TextDisabled("%s", k.c_str());
                ImGui::SameLine(170);
                ImGui::TextUnformatted(v.dump().c_str());
            }
            ImGui::TreePop();
        }
    }
    ImGui::End();
}
