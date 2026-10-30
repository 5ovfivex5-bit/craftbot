// Training Stats: every field of the metrics message, progress bar, ETA, losses, system.
#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include <imgui.h>

#include "app.h"
#include "panels.h"
#include "ui_util.h"

namespace {

constexpr int kTopBottomN = 8;
int g_task_view = 1;  // 0 all, 1 top N, 2 bottom N (task table only)

// Name -> success table. Rows are sortable by clicking the headers. `view` (may be null) limits the
// rows to the best or worst n by success, which only applies to large sets such as tasks.
void success_table(const char* id, const char* key_header, const std::map<std::string, double>& data, const int* view,
                   int n) {
    std::vector<std::pair<std::string, double>> rows(data.begin(), data.end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    std::vector<std::pair<std::string, double>> shown = rows;
    if (view && *view != 0 && (int)rows.size() > n) {
        if (*view == 1) shown.assign(rows.begin(), rows.begin() + n);
        else shown.assign(rows.end() - n, rows.end());
    }
    const ImGuiTableFlags flags = ImGuiTableFlags_Sortable | ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                  ImGuiTableFlags_SizingStretchProp;
    if (!ImGui::BeginTable(id, 2, flags)) return;
    ImGui::TableSetupColumn(key_header, ImGuiTableColumnFlags_None, 3.0f);
    ImGui::TableSetupColumn("Success", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_PreferSortDescending, 1.2f);
    ImGui::TableHeadersRow();
    if (ImGuiTableSortSpecs* specs = ImGui::TableGetSortSpecs(); specs && specs->SpecsCount > 0) {
        const ImGuiTableColumnSortSpecs& cs = specs->Specs[0];
        const bool desc = cs.SortDirection == ImGuiSortDirection_Descending;
        std::stable_sort(shown.begin(), shown.end(), [&](const auto& a, const auto& b) {
            if (cs.ColumnIndex == 1) return desc ? a.second > b.second : a.second < b.second;
            return desc ? a.first > b.first : a.first < b.first;
        });
    }
    for (const auto& [name, v] : shown) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted(name.c_str());
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%s %%", ui::fmt_num(v * 100.0, 1).c_str());
    }
    ImGui::EndTable();
}

}  // namespace

void draw_training(App& app, bool* p_open) {
    ImGui::SetNextWindowSize(ImVec2(430, 560), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(win::kTraining, p_open)) {
        ImGui::End();
        return;
    }
    if (!trainer_connected(app)) {
        ui::no_data("Training statistics appear when the trainer sends its first metrics message (2 Hz).");
        ImGui::End();
        return;
    }
    const mm::Metrics& m = *app.f->metrics;

    if (m.mem_warning) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(120, 30, 30, 255));
        ImGui::BeginChild("##membanner", ImVec2(0, ImGui::GetFrameHeight() * 2.2f), ImGuiChildFlags_Borders);
        ImGui::TextColored(ImVec4(1, 0.8f, 0.7f, 1), "MEMORY WARNING");
        ImGui::TextWrapped("%s", m.mem_warning->c_str());
        ImGui::EndChild();
        ImGui::PopStyleColor();
    }

    // Progress: steps are environment transitions, summed over all environments.
    double frac = m.target_steps > 0 ? (double)m.env_steps / (double)m.target_steps : 0.0;
    if (frac > 1.0) frac = 1.0;
    char overlay[160];
    std::snprintf(overlay, sizeof overlay, "Training: %s / %s env steps (%.1f%%)", ui::fmt_count(m.env_steps).c_str(),
                  ui::fmt_count(m.target_steps).c_str(), frac * 100.0);
    ImGui::ProgressBar((float)frac, ImVec2(-1, 0), overlay);
    ImGui::TextWrapped("steps = environment transitions (summed over all envs), not PPO updates");

    ImGui::Text("ETA:");
    ImGui::SameLine();
    if (m.eta_s) ImGui::Text("%s", ui::fmt_hms(*m.eta_s).c_str());
    else ImGui::TextDisabled("Calculating");
    ui::help_marker("ETA = (target - env steps) / smoothed steps-per-second. Shown as 'Calculating' until enough samples exist.");

    ui::section("Run");
    ui::kv("State", m.state);
    ui::kv("Mode", m.mode);
    ui::kv("Elapsed", ui::fmt_hms(m.elapsed_s));
    ui::kv("Curriculum stage", m.curriculum_stage ? *m.curriculum_stage : std::string("-"));
    ui::kv("Latest checkpoint", m.latest_checkpoint ? *m.latest_checkpoint : std::string("none yet"));
    ui::kv("Environments", std::to_string(m.num_envs));
    if (app.f->has_hello && app.f->hello.world_size[0] > 0)
        ui::kv("World size (x, y, z)", std::to_string(app.f->hello.world_size[0]) + " x " +
                                           std::to_string(app.f->hello.world_size[1]) + " x " +
                                           std::to_string(app.f->hello.world_size[2]));

    ui::section("Throughput");
    ui::kv("Env steps", ui::fmt_count(m.env_steps));
    ui::kv("Target steps", ui::fmt_count(m.target_steps));
    ui::kv("Steps / second", ui::fmt_num(m.sps, 0));
    ui::kv("PPO updates", ui::fmt_count(m.updates));

    ui::section("Episodes and success");
    ui::kv("Episodes finished", ui::fmt_count(m.episodes));
    ui::kv("Avg episode length", ui::fmt_num(m.avg_ep_len, 1) + " steps");
    ui::kv("Avg episode reward", ui::fmt_num(m.avg_reward, 3));
    if (m.success_rate.empty()) {
        ImGui::TextDisabled("success rate: no items reported yet");
    } else {
        for (const auto& [item, rate] : m.success_rate) {
            char label[96];
            std::snprintf(label, sizeof label, "Success: %s", item.c_str());
            ui::kv(label, ui::fmt_num(rate * 100.0, 1) + " %");
        }
    }

    // v7 breakdowns: per task family and per task. Hidden when the trainer does not send them.
    if (!m.family_success.empty()) {
        ui::section("Success by family");
        success_table("##family_success", "Family", m.family_success, nullptr, 0);
    }
    if (!m.task_success.empty()) {
        ui::section("Success by task");
        const char* views[] = {"All tasks", "Top 8", "Bottom 8"};
        ImGui::SetNextItemWidth(130);
        ImGui::Combo("Show##task_view", &g_task_view, views, 3);
        success_table("##task_success", "Task", m.task_success, &g_task_view, kTopBottomN);
    }

    // v6 team metrics: only shown when the trainer reports them.
    if (m.agents || m.msg_entropy || m.msg_rate || m.chest_deposits || m.chest_withdraws || m.gives) {
        ui::section("Team");
        if (m.agents) ui::kv("Agents per world", std::to_string(*m.agents));
        if (m.msg_entropy) ui::kv("Message entropy", ui::fmt_num(*m.msg_entropy, 4));
        if (m.msg_rate) ui::kv("Message rate", ui::fmt_num(*m.msg_rate * 100.0, 1) + " % non-silent");
        if (m.chest_deposits) ui::kv("Chest deposits / 1k steps", ui::fmt_num(*m.chest_deposits, 2));
        if (m.chest_withdraws) ui::kv("Chest withdraws / 1k steps", ui::fmt_num(*m.chest_withdraws, 2));
        if (m.gives) ui::kv("Gives / 1k steps", ui::fmt_num(*m.gives, 2));
    }

    ui::section("Losses and policy");
    ui::kv("Policy loss", ui::fmt_num(m.policy_loss, 5));
    ui::kv("Value loss", ui::fmt_num(m.value_loss, 5));
    ui::kv("Entropy", ui::fmt_num(m.entropy, 4));
    ui::kv("Approx KL", ui::fmt_num(m.approx_kl, 5));

    ui::section("System");
    ui::kv("CPU", ui::fmt_num(m.cpu_percent, 1) + " %");
    ui::kv("Resident memory (RSS)", ui::fmt_num(m.rss_mb, 1) + " MB");
    ui::kv("Memory warning", m.mem_warning ? *m.mem_warning : std::string("none"),
           m.mem_warning ? IM_COL32(255, 120, 100, 255) : 0);

    ImGui::End();
}
