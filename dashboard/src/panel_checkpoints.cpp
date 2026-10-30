// Checkpoints browser: list (list_checkpoints), load (overwrites run) and evaluate (separate run).
#include <cstdio>
#include <string>

#include <imgui.h>

#include "app.h"
#include "panels.h"
#include "ui_util.h"

namespace {

int g_eval_episodes = 10;
bool g_confirm_load = false;

}  // namespace

void draw_checkpoints(App& app, bool* p_open) {
    ImGui::SetNextWindowSize(ImVec2(620, 380), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(win::kCheckpoints, p_open)) {
        ImGui::End();
        return;
    }
    const bool connected = trainer_connected(app);
    const std::string state = trainer_state(app);

    if (ImGui::Button("Refresh list")) app.cmd("list_checkpoints");
    ImGui::SameLine();
    ImGui::BeginDisabled(!connected || !(state == "running" || state == "paused"));
    if (ImGui::Button("Save now")) app.cmd("save_checkpoint");
    ImGui::EndDisabled();
    ImGui::SameLine();
    ui::help_marker("Checkpoints are written atomically (model, optimizer, config, counters, curriculum, RNG). "
                    "Load overwrites the current training run. Evaluate runs a separate policy on a separate environment "
                    "and never touches the run.");

    const mm::Frame* f = app.f.get();
    if (!f || !f->checkpoints) {
        ImGui::Spacing();
        ui::no_data("Press Refresh list once connected to read the checkpoint directory.");
        ImGui::End();
        return;
    }
    const auto& items = *f->checkpoints;
    if (items.empty()) ImGui::TextDisabled("No checkpoints yet. Checkpoints are written every 'ckpt every' steps.");

    if (ImGui::BeginTable("##ckpts", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
                                         ImGuiTableFlags_Resizable,
                          ImVec2(0, 200))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Path", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Env steps", ImGuiTableColumnFlags_WidthFixed, 110);
        ImGui::TableSetupColumn("Updates", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Saved", ImGuiTableColumnFlags_WidthFixed, 150);
        ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableHeadersRow();
        for (size_t i = items.size(); i-- > 0;) {  // newest first
            const mm::Checkpoint& c = items[i];
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            bool sel = app.selected_checkpoint == c.path;
            ImGui::PushID((int)i);
            if (ImGui::Selectable(c.path.c_str(), sel, ImGuiSelectableFlags_SpanAllColumns)) app.selected_checkpoint = c.path;
            ImGui::PopID();
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(ui::fmt_count(c.env_steps).c_str());
            ImGui::TableSetColumnIndex(2);
            ImGui::TextUnformatted(ui::fmt_count(c.updates).c_str());
            ImGui::TableSetColumnIndex(3);
            ImGui::TextUnformatted(ui::fmt_time(c.time).c_str());
            ImGui::TableSetColumnIndex(4);
            ImGui::TextUnformatted(ui::fmt_bytes((double)c.size_bytes).c_str());
        }
        ImGui::EndTable();
    }

    ui::section("Selected checkpoint");
    if (app.selected_checkpoint.empty()) {
        ImGui::TextDisabled("Select a checkpoint above.");
    } else {
        ImGui::TextWrapped("%s", app.selected_checkpoint.c_str());
    }
    bool can_load = connected && (state == "paused" || state == "stopped") && !app.selected_checkpoint.empty();
    ImGui::BeginDisabled(!can_load);
    if (ImGui::Button("Load into training run...")) {
        g_confirm_load = true;
        ImGui::OpenPopup("Confirm load##ckpt");
    }
    ImGui::EndDisabled();
    if (!can_load && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Needs a selected checkpoint and the trainer paused or stopped.");

    ImGui::SameLine();
    ImGui::BeginDisabled(!connected || app.selected_checkpoint.empty());
    ImGui::SetNextItemWidth(70);
    ImGui::InputInt("episodes", &g_eval_episodes);
    if (g_eval_episodes < 1) g_eval_episodes = 1;
    if (g_eval_episodes > 1000) g_eval_episodes = 1000;
    ImGui::SameLine();
    if (ImGui::Button("Evaluate")) {
        app.cmd("eval_checkpoint", {{"path", app.selected_checkpoint}, {"episodes", g_eval_episodes}});
        app.toast(ToastLevel::Info, "Evaluation requested - results appear in the Console");
    }
    ImGui::EndDisabled();
    ImGui::TextDisabled("Evaluate does not overwrite the training run; results are reported in the Console.");

    if (ImGui::BeginPopupModal("Confirm load##ckpt", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped("Load this checkpoint into the training run?\nThe current model, optimizer and counters will be replaced.");
        ImGui::TextDisabled("%s", app.selected_checkpoint.c_str());
        ImGui::Spacing();
        if (ImGui::Button("Load", ImVec2(120, 0))) {
            app.cmd("load_checkpoint", {{"path", app.selected_checkpoint}});
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
    (void)g_confirm_load;
    ImGui::End();
}
