// Console: trainer log lines with level filter, text filter, autoscroll and copy.
#include <cctype>
#include <cstdio>
#include <string>
#include <vector>

#include <imgui.h>

#include "app.h"
#include "panels.h"
#include "ui_util.h"

namespace {

bool g_show_info = true, g_show_warn = true, g_show_error = true;
bool g_autoscroll = true;
char g_filter[128] = "";

ImU32 level_color(const std::string& l) {
    if (l == "error") return IM_COL32(255, 110, 100, 255);
    if (l == "warn") return IM_COL32(245, 200, 90, 255);
    return IM_COL32(200, 210, 225, 255);
}

bool passes(const mm::LogLine& l) {
    if (l.level == "error" && !g_show_error) return false;
    if (l.level == "warn" && !g_show_warn) return false;
    if (l.level != "error" && l.level != "warn" && !g_show_info) return false;
    if (g_filter[0] && l.msg.find(g_filter) == std::string::npos) {
        // Case-insensitive fallback.
        std::string a = l.msg, b = g_filter;
        for (auto& c : a) c = (char)std::tolower((unsigned char)c);
        for (auto& c : b) c = (char)std::tolower((unsigned char)c);
        if (a.find(b) == std::string::npos) return false;
    }
    return true;
}

}  // namespace

void draw_console(App& app, bool* p_open) {
    ImGui::SetNextWindowSize(ImVec2(760, 320), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(win::kConsole, p_open)) {
        ImGui::End();
        return;
    }
    ImGui::Checkbox("info", &g_show_info);
    ImGui::SameLine();
    ImGui::Checkbox("warn", &g_show_warn);
    ImGui::SameLine();
    ImGui::Checkbox("error", &g_show_error);
    ImGui::SameLine();
    ImGui::Checkbox("autoscroll", &g_autoscroll);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(220);
    ImGui::InputTextWithHint("##filter", "filter text", g_filter, sizeof g_filter);
    ImGui::SameLine();

    const mm::Frame* f = app.f.get();
    const std::vector<mm::LogLine> empty;
    const std::vector<mm::LogLine>& logs = f && f->logs ? *f->logs : empty;

    if (ImGui::SmallButton("Copy visible")) {
        std::string all;
        for (size_t i = 0; i < logs.size(); i++) {
            if (!passes(logs[i])) continue;
            char t[32];
            std::snprintf(t, sizeof t, "[%8.2f] ", logs[i].t);
            all += t;
            all += logs[i].level + "  " + logs[i].msg + "\n";
        }
        ImGui::SetClipboardText(all.c_str());
        app.toast(ToastLevel::Info, "Console copied to clipboard");
    }

    ImGui::Separator();
    if (logs.empty()) {
        ui::no_data("Trainer log lines appear here.");
        ImGui::End();
        return;
    }
    // Filtered index list so the clipper only sees visible lines (no gaps).
    std::vector<size_t> shown;
    shown.reserve(logs.size());
    for (size_t i = 0; i < logs.size(); i++)
        if (passes(logs[i])) shown.push_back(i);

    static size_t last_count = 0;
    if (ImGui::BeginChild("##logs", ImVec2(0, 0), ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar)) {
        ImGuiListClipper clipper;
        clipper.Begin((int)shown.size());
        while (clipper.Step()) {
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++) {
                const mm::LogLine& l = logs[shown[i]];
                ImGui::PushStyleColor(ImGuiCol_Text, level_color(l.level));
                ImGui::Text("[%8.2f] %-5s %s", l.t, l.level.c_str(), l.msg.c_str());
                ImGui::PopStyleColor();
            }
        }
        if (g_autoscroll && logs.size() != last_count) ImGui::SetScrollHereY(1.0f);
        last_count = logs.size();
    }
    ImGui::EndChild();
    ImGui::End();
}
