// Neural Network inspector. Four clearly separated views:
//   Architecture  - static graph from the `model` message (not live)
//   Activations   - per-layer forward activations of the selected env (heatmap, histogram)
//   Outputs       - action probabilities, entropy, value estimate, GRU hidden-state stats
//   Gradients     - gradient norms (last PPO update) and weight statistics (parameters)
// The activations message is only requested while this panel is on screen.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include <imgui.h>
#include <imgui_internal.h>

#include "app.h"
#include "net/protocol.h"
#include "panels.h"
#include "ui_util.h"

namespace {

struct NNUi {
    std::string selected;      // selected layer in the architecture view
    std::string act_layer;     // selected layer in the activation view
};
NNUi st;

std::string shape_str(const std::vector<int>& s) {
    if (s.empty()) return "-";
    std::string out = "[";
    for (size_t i = 0; i < s.size(); i++) {
        if (i) out += ", ";
        out += std::to_string(s[i]);
    }
    return out + "]";
}

ImU32 type_color(const std::string& t) {
    auto has = [&](const char* k) { return t.find(k) != std::string::npos; };
    if (has("Conv")) return IM_COL32(70, 120, 200, 255);
    if (has("GRU") || has("LSTM") || has("Recur")) return IM_COL32(140, 90, 200, 255);
    if (has("Linear") || has("Dense")) return IM_COL32(40, 150, 140, 255);
    if (has("Head") || has("Out")) return IM_COL32(210, 130, 60, 255);
    if (has("Input")) return IM_COL32(90, 90, 100, 255);
    if (has("Norm")) return IM_COL32(100, 130, 90, 255);
    return IM_COL32(70, 80, 100, 255);
}

// Longest-path layering over the model's edge list (Kahn's algorithm). Cycles fall back to level 0.
std::map<std::string, int> layer_levels(const mm::ModelInfo& m) {
    std::map<std::string, int> level, indeg;
    std::map<std::string, std::vector<std::string>> adj;
    for (const auto& l : m.layers) {
        level[l.name] = 0;
        indeg[l.name] = 0;
    }
    for (const auto& [a, b] : m.edges) {
        if (!level.count(a) || !level.count(b)) continue;
        adj[a].push_back(b);
        indeg[b]++;
    }
    std::vector<std::string> queue;
    for (const auto& [name, d] : indeg)
        if (d == 0) queue.push_back(name);
    for (size_t qi = 0; qi < queue.size(); qi++) {
        std::string u = queue[qi];
        for (const auto& v : adj[u]) {
            level[v] = std::max(level[v], level[u] + 1);
            if (--indeg[v] == 0) queue.push_back(v);
        }
    }
    return level;
}

void draw_architecture(const mm::ModelInfo& m, const mm::Activations* acts) {
    const float NW = 176, NH = 52, GX = 56, GY = 18, PAD = 12;
    auto level = layer_levels(m);
    int max_level = 0;
    for (auto& [n, l] : level) max_level = std::max(max_level, l);
    std::vector<int> per_level(max_level + 1, 0);
    std::map<std::string, ImVec2> pos;
    for (const auto& l : m.layers) {
        int lv = level[l.name];
        pos[l.name] = ImVec2(PAD + lv * (NW + GX), PAD + per_level[lv] * (NH + GY));
        per_level[lv]++;
    }
    int rows = *std::max_element(per_level.begin(), per_level.end());
    ImVec2 canvas(PAD * 2 + (max_level + 1) * (NW + GX), PAD * 2 + std::max(1, rows) * (NH + GY));

    ImGui::TextDisabled("Static architecture from the model message. Click a layer for details.");
    if (ImGui::BeginChild("##arch", ImVec2(0, 300), ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_HorizontalScrollbar)) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 org = ImGui::GetCursorScreenPos();
        ImGui::Dummy(canvas);
        // Edges first, so nodes sit on top.
        for (const auto& [a, b] : m.edges) {
            if (!pos.count(a) || !pos.count(b)) continue;
            ImVec2 pa(org.x + pos[a].x + NW, org.y + pos[a].y + NH * 0.5f);
            ImVec2 pb(org.x + pos[b].x, org.y + pos[b].y + NH * 0.5f);
            float dx = std::max(30.0f, (pb.x - pa.x) * 0.5f);
            dl->AddBezierCubic(pa, ImVec2(pa.x + dx, pa.y), ImVec2(pb.x - dx, pb.y), pb, IM_COL32(150, 165, 190, 200), 1.6f);
            dl->AddTriangleFilled(ImVec2(pb.x, pb.y), ImVec2(pb.x - 7, pb.y - 4), ImVec2(pb.x - 7, pb.y + 4),
                                  IM_COL32(150, 165, 190, 220));
        }
        for (const auto& l : m.layers) {
            ImVec2 p0(org.x + pos[l.name].x, org.y + pos[l.name].y);
            ImVec2 p1(p0.x + NW, p0.y + NH);
            bool hovered = ImGui::IsMouseHoveringRect(p0, p1) && ImGui::IsWindowHovered();
            if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) st.selected = l.name;
            bool sel = st.selected == l.name;
            ImU32 col = type_color(l.type);
            dl->AddRectFilled(p0, p1, col, 6.0f);
            dl->AddRect(p0, p1, sel ? IM_COL32(255, 230, 120, 255) : (hovered ? IM_COL32(230, 235, 245, 255) : IM_COL32(20, 22, 28, 255)),
                        6.0f, 0, sel ? 3.0f : 1.5f);
            dl->AddText(ImVec2(p0.x + 8, p0.y + 5), IM_COL32(255, 255, 255, 255), l.name.c_str());
            std::string sub = l.type + "  " + ui::fmt_count(l.params) + " params" + (l.trainable ? "" : " (frozen)");
            dl->AddText(ImVec2(p0.x + 8, p0.y + 27), IM_COL32(225, 230, 240, 220), sub.c_str());
        }
    }
    ImGui::EndChild();

    if (st.selected.empty()) {
        ImGui::TextDisabled("Select a layer in the graph.");
        return;
    }
    const mm::LayerInfo* li = nullptr;
    for (const auto& l : m.layers)
        if (l.name == st.selected) li = &l;
    if (!li) return;
    ui::section(("Layer: " + li->name).c_str());
    ui::kv("Type", li->type);
    ui::kv("Parameters", ui::fmt_count(li->params));
    ui::kv("Trainable", li->trainable ? "yes" : "no (frozen)");
    ui::kv("Input shape", shape_str(li->in_shape));
    ui::kv("Output shape", shape_str(li->out_shape));
    if (acts) {
        auto it = acts->layers.find(li->name);
        if (it != acts->layers.end()) {
            ImGui::TextDisabled("Activation (forward output, selected env, step %lld)", acts->step);
            ui::kv("  mean / std", ui::fmt_num(it->second.mean, 4) + " / " + ui::fmt_num(it->second.std, 4));
            ui::kv("  min / max", ui::fmt_num(it->second.min, 4) + " / " + ui::fmt_num(it->second.max, 4));
        }
        auto g = acts->grad_norms.find(li->name);
        if (g != acts->grad_norms.end()) ui::kv("Gradient norm (last update)", ui::fmt_num(g->second, 5));
        auto w = acts->weight_stats.find(li->name);
        if (w != acts->weight_stats.end()) {
            ui::kv("Weights mean / std", ui::fmt_num(w->second.mean, 4) + " / " + ui::fmt_num(w->second.std, 4));
            ui::kv("Weights min / max", ui::fmt_num(w->second.min, 4) + " / " + ui::fmt_num(w->second.max, 4));
        }
    }
}

void draw_activations(const mm::ModelInfo* m, const mm::Activations* acts, const std::vector<std::string>& actions) {
    (void)actions;
    if (!acts || acts->layers.empty()) {
        ImGui::TextDisabled("No activation data yet. Activations stream while this panel is open and training is connected.");
        return;
    }
    // Layer choice: model order, restricted to layers that reported activations.
    std::vector<std::string> names;
    if (m)
        for (const auto& l : m->layers)
            if (acts->layers.count(l.name)) names.push_back(l.name);
    if (names.empty())
        for (const auto& [k, v] : acts->layers) names.push_back(k);
    if (st.act_layer.empty() || !acts->layers.count(st.act_layer)) st.act_layer = names.front();

    ImGui::SetNextItemWidth(200);
    if (ImGui::BeginCombo("Layer", st.act_layer.c_str())) {
        for (const auto& n : names)
            if (ImGui::Selectable(n.c_str(), n == st.act_layer)) st.act_layer = n;
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("env %d, step %lld", acts->env_id, acts->step);

    const mm::LayerAct& la = acts->layers.at(st.act_layer);
    ui::section("Activation heatmap (sampled units, colour = value)");
    ui::kv("Shape", shape_str(la.shape));
    ui::kv("Samples shown", std::to_string(la.sample.size()) + (la.sample.size() >= 256 ? " (capped)" : ""));
    if (!la.sample.empty()) {
        const size_t n = la.sample.size();
        const int cols = std::max(1, (int)std::ceil(std::sqrt((double)n)));
        const int rows = (int)((n + cols - 1) / cols);
        float avail = std::min(ImGui::GetContentRegionAvail().x, 320.0f);
        float cell = std::max(2.0f, std::floor(avail / cols));
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 p0 = ImGui::GetCursorScreenPos();
        ImGui::Dummy(ImVec2(cell * cols, cell * rows));
        float lo = la.min, hi = la.max;
        for (size_t i = 0; i < n; i++) {
            float t = hi > lo ? (la.sample[i] - lo) / (hi - lo) : 0.5f;
            int cx = (int)(i % cols), cy = (int)(i / cols);
            ImVec2 a(p0.x + cx * cell, p0.y + cy * cell);
            dl->AddRectFilled(a, ImVec2(a.x + cell, a.y + cell), ui::heat_color(t));
        }
        ImGui::TextDisabled("Activation values of this layer (not weights). Colour bar:");
        ui::colorbar("##heatcb", la.min, la.max, ui::heat_color, ImVec2(std::min(avail, 260.0f), 12));
    }

    ui::section("Activation histogram (all values of the layer)");
    if (!la.hist_counts.empty()) {
        ImVec2 p0 = ImGui::GetCursorScreenPos();
        float W = std::min(ImGui::GetContentRegionAvail().x, 360.0f), H = 90;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImGui::Dummy(ImVec2(W, H + 16));
        int mx = 1;
        for (int c : la.hist_counts) mx = std::max(mx, c);
        int bins = (int)la.hist_counts.size();
        for (int i = 0; i < bins; i++) {
            float h = H * (float)la.hist_counts[i] / (float)mx;
            float x0 = p0.x + W * i / bins, x1 = p0.x + W * (i + 1) / bins - 1;
            dl->AddRectFilled(ImVec2(x0, p0.y + H - h), ImVec2(x1, p0.y + H), IM_COL32(90, 170, 210, 255));
        }
        dl->AddRect(p0, ImVec2(p0.x + W, p0.y + H), IM_COL32(70, 75, 90, 255));
        char lo[32], hi[32];
        if (!la.hist_edges.empty()) {
            std::snprintf(lo, sizeof lo, "%.3g", la.hist_edges.front());
            std::snprintf(hi, sizeof hi, "%.3g", la.hist_edges.back());
        } else {
            std::snprintf(lo, sizeof lo, "%.3g", la.min);
            std::snprintf(hi, sizeof hi, "%.3g", la.max);
        }
        dl->AddText(ImVec2(p0.x, p0.y + H + 2), IM_COL32(180, 185, 200, 255), lo);
        ImVec2 hs = ImGui::CalcTextSize(hi);
        dl->AddText(ImVec2(p0.x + W - hs.x, p0.y + H + 2), IM_COL32(180, 185, 200, 255), hi);
        ImGui::TextDisabled("16 bins between min and max; bar height = count");
    }

    ui::section("Layer summary (activations)");
    ui::kv("Mean", ui::fmt_num(la.mean, 5));
    ui::kv("Std", ui::fmt_num(la.std, 5));
    ui::kv("Min / max", ui::fmt_num(la.min, 4) + " / " + ui::fmt_num(la.max, 4));
}

void draw_outputs(const mm::Hello& hello, const mm::Activations* acts) {
    if (!acts) {
        ImGui::TextDisabled("No policy output yet. Activations stream while this panel is open.");
        return;
    }
    ui::section("Action probabilities (policy output, current observation)");
    if (acts->action_probs.empty()) ImGui::TextDisabled("no probabilities reported");
    int best = 0;
    for (size_t i = 0; i < acts->action_probs.size(); i++)
        if (acts->action_probs[i] > acts->action_probs[best]) best = (int)i;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float label_w = 150, bar_w = 180;
    for (size_t i = 0; i < acts->action_probs.size(); i++) {
        std::string name = i < hello.actions.size() ? hello.actions[i] : "action " + std::to_string(i);
        float p = acts->action_probs[i];
        ImVec2 cur = ImGui::GetCursorScreenPos();
        ImGui::Dummy(ImVec2(label_w + bar_w + 70, ImGui::GetTextLineHeight() + 2));
        dl->AddText(ImVec2(cur.x, cur.y + 1), (int)i == best ? IM_COL32(255, 230, 140, 255) : IM_COL32(200, 205, 215, 255), name.c_str());
        ImVec2 b0(cur.x + label_w, cur.y + 3), b1(b0.x + bar_w * std::clamp(p, 0.0f, 1.0f), cur.y + ImGui::GetTextLineHeight() - 1);
        dl->AddRectFilled(ImVec2(b0.x, b0.y), ImVec2(b0.x + bar_w, b1.y), IM_COL32(35, 40, 52, 255), 3.0f);
        dl->AddRectFilled(b0, b1, (int)i == best ? IM_COL32(230, 190, 80, 255) : IM_COL32(90, 170, 210, 255), 3.0f);
        char v[32];
        std::snprintf(v, sizeof v, "%5.1f %%", p * 100.0f);
        dl->AddText(ImVec2(b0.x + bar_w + 8, cur.y + 1), IM_COL32(220, 225, 235, 255), v);
    }
    ImGui::TextDisabled("Bars are the softmax probabilities of the policy head.");

    ui::section("Policy and value heads");
    ui::kv("Entropy (policy, nats)", ui::fmt_num(acts->entropy, 4));
    ui::kv("Value estimate V(s)", ui::fmt_num(acts->value, 4));
    ImGui::TextDisabled("Entropy is the policy's uncertainty; the value head predicts the discounted return.");

    ui::section("GRU hidden state (recurrent memory, activation statistics)");
    if (!acts->has_gru) {
        ImGui::TextDisabled("no GRU statistics reported");
    } else {
        ui::kv("Mean / std", ui::fmt_num(acts->gru.mean, 4) + " / " + ui::fmt_num(acts->gru.std, 4));
        ui::kv("Min / max", ui::fmt_num(acts->gru.min, 4) + " / " + ui::fmt_num(acts->gru.max, 4));
        ui::kv("L2 norm", ui::fmt_num(acts->gru_norm, 4));
    }
}

void draw_grads_weights(const mm::Activations* acts) {
    if (!acts) {
        ImGui::TextDisabled("No gradient or weight data yet. It arrives with the activations stream.");
        return;
    }
    ui::section("Gradient norms (most recent PPO update; training signal, not activations)");
    if (acts->grad_norms.empty()) {
        ImGui::TextDisabled("no gradient norms reported yet (available after the first update)");
    } else {
        float mx = 1e-12f;
        for (auto& [k, v] : acts->grad_norms) mx = std::max(mx, v);
        if (ImGui::BeginTable("##grads", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Layer");
            ImGui::TableSetupColumn("Grad norm");
            ImGui::TableSetupColumn("Relative");
            ImGui::TableHeadersRow();
            for (auto& [k, v] : acts->grad_norms) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(k.c_str());
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%.5g", v);
                ImGui::TableSetColumnIndex(2);
                ImGui::ProgressBar(v / mx, ImVec2(-1, 0), "");
            }
            ImGui::EndTable();
        }
    }

    ui::section("Weight statistics (learned parameters, not activations)");
    if (acts->weight_stats.empty()) {
        ImGui::TextDisabled("no weight statistics reported yet");
    } else if (ImGui::BeginTable("##weights", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Layer");
        ImGui::TableSetupColumn("Mean");
        ImGui::TableSetupColumn("Std");
        ImGui::TableSetupColumn("Min");
        ImGui::TableSetupColumn("Max");
        ImGui::TableHeadersRow();
        for (auto& [k, w] : acts->weight_stats) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(k.c_str());
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.4g", w.mean);
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%.4g", w.std);
            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%.4g", w.min);
            ImGui::TableSetColumnIndex(4);
            ImGui::Text("%.4g", w.max);
        }
        ImGui::EndTable();
    }
}

}  // namespace

void draw_nn(App& app, bool* p_open) {
    ImGui::SetNextWindowSize(ImVec2(520, 640), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(win::kNN, p_open)) {
        ImGui::End();
        return;
    }
    app.nn_visible = true;  // Begin() returned true: visible on screen (not collapsed, tab selected)

    const bool connected = trainer_connected(app);
    const mm::Activations* acts = app.f ? app.f->acts.get() : nullptr;
    const mm::ModelInfo* model = app.f ? app.f->model.get() : nullptr;
    if (!connected) {
        ui::no_data("Model and activations arrive once the trainer is connected.");
    }

    if (acts && acts->nan_inf) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(140, 25, 25, 255));
        ImGui::BeginChild("##nanwarn", ImVec2(0, ImGui::GetFrameHeight() * 2.4f), ImGuiChildFlags_Borders);
        ImGui::TextColored(ImVec4(1, 0.85f, 0.8f, 1), "WARNING: NaN or Inf detected in network tensors");
        ImGui::TextWrapped("env %d, step %lld. Training may be diverging; consider lowering the learning rate.",
                           acts->env_id, acts->step);
        ImGui::EndChild();
        ImGui::PopStyleColor();
    }

    if (ImGui::BeginTabBar("##nntabs")) {
        if (ImGui::BeginTabItem("Architecture")) {
            if (model && !model->layers.empty()) {
                ImGui::TextDisabled("%s  |  %s params  |  input img [%s]  vec %d  GRU hidden %d", model->name.c_str(),
                                    ui::fmt_count(model->params_total).c_str(), shape_str(model->img_shape).c_str(),
                                    model->vec_dim, model->gru_hidden);
                draw_architecture(*model, acts);
            } else {
                ui::no_data("Model description has not been received yet.");
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Activations")) {
            ImGui::TextDisabled("Forward activations of the selected environment (not weights).");
            draw_activations(model, acts, app.f && app.f->has_hello ? app.f->hello.actions : std::vector<std::string>{});
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Outputs")) {
            draw_outputs(app.f ? app.f->hello : mm::Hello{}, acts);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Grads/weights")) {
            draw_grads_weights(acts);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}
