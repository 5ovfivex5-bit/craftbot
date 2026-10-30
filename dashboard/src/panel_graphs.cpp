// Graphs: rolling metric history (bounded ring in the client), selectable time window.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <imgui.h>

#include "app.h"
#include "panels.h"
#include "ui_util.h"

namespace {

struct Range {
    const char* name;
    double seconds;  // < 0 means everything
};
const Range kRanges[] = {{"1 min", 60.0}, {"5 min", 300.0}, {"30 min", 1800.0}, {"All", -1.0}};

int g_range = 1;
int g_success_item = 0;  // 0 = mean across items, otherwise index into hello.items + 1

struct Series {
    std::vector<float> xs, ys;
};

// Plot points are downsampled to at most kMaxPoints so drawing stays cheap.
constexpr size_t kMaxPoints = 1500;

template <typename Fn>
Series extract(const std::vector<mm::Metrics>& h, size_t first, Fn value) {
    Series s;
    size_t n = h.size() - first;
    size_t step = std::max<size_t>(1, n / kMaxPoints);
    double t_last = h.back().t;
    for (size_t i = first; i < h.size(); i += step) {
        double v = value(h[i]);
        s.xs.push_back((float)(h[i].t - t_last));
        s.ys.push_back(std::isfinite(v) ? (float)v : NAN);
    }
    // Always include the newest point so the readout matches the last sample.
    if ((h.size() - 1) % step != 0) {
        s.xs.push_back(0.0f);
        s.ys.push_back((float)value(h.back()));
    }
    return s;
}

void plot_block(const char* title, const char* id, const std::vector<mm::Metrics>& h, size_t first,
                double (*pick)(const mm::Metrics&), ImU32 color, const char* unit, const char* fmt) {
    ui::label_bold(title);
    Series s = extract(h, first, pick);
    double last = pick(h.back());
    ImGui::SameLine();
    ImGui::TextDisabled("latest");
    ImGui::SameLine();
    char buf[64];
    std::snprintf(buf, sizeof buf, fmt, last);
    ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(color), "%s %s", buf, unit);
    ui::line_plot(id, {{title, color, s.xs, s.ys}}, ImVec2(-1, 110), unit);
}

double pick_reward(const mm::Metrics& m) { return m.avg_reward; }
double pick_sps(const mm::Metrics& m) { return m.sps; }
double pick_ploss(const mm::Metrics& m) { return m.policy_loss; }
double pick_vloss(const mm::Metrics& m) { return m.value_loss; }
double pick_entropy(const mm::Metrics& m) { return m.entropy; }
double pick_eplen(const mm::Metrics& m) { return m.avg_ep_len; }
double pick_cpu(const mm::Metrics& m) { return m.cpu_percent; }
double pick_rss(const mm::Metrics& m) { return m.rss_mb; }
double pick_success_mean(const mm::Metrics& m) {
    if (m.success_rate.empty()) return NAN;
    double s = 0;
    for (auto& [k, v] : m.success_rate) s += v;
    return 100.0 * s / m.success_rate.size();
}

}  // namespace

void draw_graphs(App& app, bool* p_open) {
    ImGui::SetNextWindowSize(ImVec2(900, 420), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(win::kGraphs, p_open)) {
        ImGui::End();
        return;
    }
    const std::vector<mm::Metrics>* hist = app.f ? app.f->history.get() : nullptr;
    if (!hist || hist->empty()) {
        ui::no_data("Graphs fill from the metrics stream (2 Hz).");
        ImGui::End();
        return;
    }

    ImGui::SetNextItemWidth(110);
    if (ImGui::BeginCombo("Time range", kRanges[g_range].name)) {
        for (int i = 0; i < 4; i++)
            if (ImGui::Selectable(kRanges[i].name, g_range == i)) g_range = i;
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%zu samples held (bounded rolling history)", hist->size());

    // First index inside the selected window (history is time-ordered).
    size_t first = 0;
    double window = kRanges[g_range].seconds;
    if (window > 0) {
        double cutoff = hist->back().t - window;
        while (first < hist->size() - 1 && (*hist)[first].t < cutoff) first++;
    }

    ImGui::BeginChild("##graphs", ImVec2(0, 0));
    if (ImGui::BeginTable("##gt", 2, ImGuiTableFlags_SizingStretchSame)) {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    plot_block("Reward (avg episode)", "##rew", *hist, first, pick_reward, IM_COL32(120, 220, 140, 255), "", "%.3f");
    ImGui::TableSetColumnIndex(1);
    plot_block("Steps per second", "##sps", *hist, first, pick_sps, IM_COL32(120, 180, 255, 255), "", "%.0f");

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    plot_block("Policy loss", "##pl", *hist, first, pick_ploss, IM_COL32(255, 160, 100, 255), "", "%.4f");
    ImGui::TableSetColumnIndex(1);
    plot_block("Value loss", "##vl", *hist, first, pick_vloss, IM_COL32(230, 130, 230, 255), "", "%.4f");

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    plot_block("Entropy", "##ent", *hist, first, pick_entropy, IM_COL32(255, 220, 110, 255), "", "%.3f");
    ImGui::TableSetColumnIndex(1);
    // Success rate: mean across items, or one item chosen by the combo.
    ui::label_bold("Success rate");
    ImGui::SameLine();
    std::string sel = "mean";
    const auto& items = app.f->hello.items;
    if (g_success_item > 0 && g_success_item <= (int)items.size()) sel = items[g_success_item - 1];
    ImGui::SetNextItemWidth(130);
    if (ImGui::BeginCombo("##succ", sel.c_str())) {
        if (ImGui::Selectable("mean", g_success_item == 0)) g_success_item = 0;
        for (int i = 0; i < (int)items.size(); i++)
            if (ImGui::Selectable(items[i].c_str(), g_success_item == i + 1)) g_success_item = i + 1;
        ImGui::EndCombo();
    }
    {
        Series s;
        if (g_success_item == 0) {
            s = extract(*hist, first, pick_success_mean);
        } else {
            const std::string key = items[g_success_item - 1];
            s = extract(*hist, first, [&](const mm::Metrics& m) {
                auto it = m.success_rate.find(key);
                return it == m.success_rate.end() ? NAN : 100.0 * it->second;
            });
        }
        ui::line_plot("##succ_plot", {{"success %", IM_COL32(150, 230, 210, 255), s.xs, s.ys}}, ImVec2(-1, 110), "%");
    }

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    plot_block("Episode length", "##eplen", *hist, first, pick_eplen, IM_COL32(200, 200, 120, 255), "steps", "%.1f");
    ImGui::TableSetColumnIndex(1);
    plot_block("CPU", "##cpu", *hist, first, pick_cpu, IM_COL32(255, 120, 120, 255), "%", "%.1f");

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    plot_block("Resident memory", "##rss", *hist, first, pick_rss, IM_COL32(160, 160, 255, 255), "MB", "%.0f");
    ImGui::EndTable();
    }
    ImGui::EndChild();
    ImGui::End();
}
