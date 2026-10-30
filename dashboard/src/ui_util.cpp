#include "ui_util.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <ctime>

#include "net/protocol.h"

namespace ui {

void no_data(const char* what) {
    ImGui::TextColored(ImVec4(0.95f, 0.70f, 0.30f, 1.0f), "No data - trainer not connected");
    if (what) ImGui::TextDisabled("%s", what);
}

void help_marker(const char* text) {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::BeginItemTooltip()) {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 34.0f);
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

void kv(const char* key, const std::string& value, ImU32 color) {
    ImGui::TextDisabled("%s", key);
    // Label column takes half the width; long values wrap instead of overlapping.
    ImGui::SameLine(ImGui::GetCursorPosX() + std::max(96.0f, ImGui::GetContentRegionAvail().x * 0.5f));
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x);
    if (color)
        ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(color), "%s", value.c_str());
    else
        ImGui::TextUnformatted(value.c_str());
}

void kv_fmt(const char* key, const char* fmt, ...) {
    char buf[256];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, args);
    va_end(args);
    kv(key, buf);
}

void label_bold(const char* text) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.90f, 1.0f, 1.0f));
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
}

void section(const char* text) {
    ImGui::Spacing();
    label_bold(text);
    ImGui::Separator();
}

std::string fmt_count(long long n) {
    std::string s = std::to_string(n < 0 ? -n : n);
    std::string out;
    int c = 0;
    for (auto it = s.rbegin(); it != s.rend(); ++it) {
        if (c && c % 3 == 0) out.insert(out.begin(), ' ');
        out.insert(out.begin(), *it);
        c++;
    }
    return n < 0 ? "-" + out : out;
}

std::string fmt_hms(double seconds) {
    if (!(seconds >= 0)) return "-";
    long long s = (long long)seconds;
    char buf[64];
    if (s >= 3600)
        std::snprintf(buf, sizeof buf, "%lldh %02lldm %02llds", s / 3600, (s / 60) % 60, s % 60);
    else if (s >= 60)
        std::snprintf(buf, sizeof buf, "%lldm %02llds", s / 60, s % 60);
    else
        std::snprintf(buf, sizeof buf, "%llds", s);
    return buf;
}

std::string fmt_bytes(double bytes) {
    char buf[64];
    if (bytes >= 1024.0 * 1024 * 1024)
        std::snprintf(buf, sizeof buf, "%.2f GiB", bytes / (1024.0 * 1024 * 1024));
    else if (bytes >= 1024.0 * 1024)
        std::snprintf(buf, sizeof buf, "%.1f MiB", bytes / (1024.0 * 1024));
    else if (bytes >= 1024.0)
        std::snprintf(buf, sizeof buf, "%.1f KiB", bytes / 1024.0);
    else
        std::snprintf(buf, sizeof buf, "%.0f B", bytes);
    return buf;
}

std::string fmt_time(double unix_s) {
    if (unix_s <= 0) return "-";
    std::time_t t = (std::time_t)unix_s;
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[64];
    std::strftime(buf, sizeof buf, "%Y-%m-%d %H:%M:%S", &tm);
    return buf;
}

std::string fmt_num(double v, int prec) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*f", prec, v);
    return buf;
}

void line_plot(const char* id, const std::vector<PlotSeries>& series, ImVec2 size, const char* y_unit) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    if (size.x <= 0) size.x = ImGui::GetContentRegionAvail().x;
    ImVec2 p1(p0.x + size.x, p0.y + size.y);
    ImGui::InvisibleButton(id, size);
    bool hovered = ImGui::IsItemHovered();

    dl->AddRectFilled(p0, p1, IM_COL32(16, 18, 24, 255), 4.0f);
    dl->AddRect(p0, p1, IM_COL32(60, 66, 80, 255), 4.0f);

    float xmin = 1e30f, xmax = -1e30f, ymin = 1e30f, ymax = -1e30f;
    for (const auto& s : series) {
        size_t n = std::min(s.xs.size(), s.ys.size());
        for (size_t i = 0; i < n; i++) {
            if (!std::isfinite(s.ys[i])) continue;
            xmin = std::min(xmin, s.xs[i]);
            xmax = std::max(xmax, s.xs[i]);
            ymin = std::min(ymin, s.ys[i]);
            ymax = std::max(ymax, s.ys[i]);
        }
    }
    if (xmin > xmax) {
        dl->AddText(ImVec2(p0.x + 8, p0.y + size.y * 0.5f - 7), IM_COL32(120, 120, 130, 255), "waiting for data...");
        return;
    }
    if (xmax - xmin < 1e-6f) xmax = xmin + 1.0f;
    if (ymax - ymin < 1e-9f) {
        float pad = std::max(1e-3f, std::fabs(ymax) * 0.1f);
        ymin -= pad;
        ymax += pad;
    }
    float pad = (ymax - ymin) * 0.08f;
    ymin -= pad;
    ymax += pad;

    // Zero line when visible.
    if (ymin < 0 && ymax > 0) {
        float y0 = p1.y - (0 - ymin) / (ymax - ymin) * size.y;
        dl->AddLine(ImVec2(p0.x, y0), ImVec2(p1.x, y0), IM_COL32(70, 70, 80, 255));
    }

    for (const auto& s : series) {
        size_t n = std::min(s.xs.size(), s.ys.size());
        ImVec2 prev(0, 0);
        bool have_prev = false;
        for (size_t i = 0; i < n; i++) {
            if (!std::isfinite(s.ys[i])) {
                have_prev = false;
                continue;
            }
            float px = p0.x + (s.xs[i] - xmin) / (xmax - xmin) * size.x;
            float py = p1.y - (s.ys[i] - ymin) / (ymax - ymin) * size.y;
            ImVec2 cur(px, py);
            if (have_prev) dl->AddLine(prev, cur, s.color, 1.5f);
            prev = cur;
            have_prev = true;
        }
    }

    // Legend + y-range.
    float ly = p0.y + 4;
    for (const auto& s : series) {
        dl->AddRectFilled(ImVec2(p0.x + 8, ly + 3), ImVec2(p0.x + 16, ly + 11), s.color);
        dl->AddText(ImVec2(p0.x + 20, ly), IM_COL32(200, 205, 215, 255), s.label);
        ly += ImGui::GetTextLineHeight() + 1;
    }
    char range[128];
    std::snprintf(range, sizeof range, "%s%s  [%.4g .. %.4g]", "y", y_unit ? y_unit : "", ymin + pad, ymax - pad);
    ImVec2 ts = ImGui::CalcTextSize(range);
    dl->AddText(ImVec2(p1.x - ts.x - 6, p0.y + 4), IM_COL32(130, 135, 150, 255), range);

    if (hovered) {
        ImVec2 m = ImGui::GetIO().MousePos;
        float frac = std::clamp((m.x - p0.x) / size.x, 0.0f, 1.0f);
        float xv = xmin + frac * (xmax - xmin);
        dl->AddLine(ImVec2(m.x, p0.y), ImVec2(m.x, p1.y), IM_COL32(200, 200, 200, 90));
        ImGui::SetTooltip("t = %.1f s", xv);
    }
}

ImU32 heat_color(float t) {
    // Compact viridis approximation: dark purple -> teal -> yellow.
    static const float stops[5][3] = {
        {0.267f, 0.005f, 0.329f}, {0.230f, 0.322f, 0.546f}, {0.128f, 0.567f, 0.551f},
        {0.369f, 0.789f, 0.383f}, {0.993f, 0.906f, 0.144f},
    };
    t = std::clamp(t, 0.0f, 1.0f) * 4.0f;
    int i = std::min(3, (int)t);
    float f = t - i;
    float r = stops[i][0] + (stops[i + 1][0] - stops[i][0]) * f;
    float g = stops[i][1] + (stops[i + 1][1] - stops[i][1]) * f;
    float b = stops[i][2] + (stops[i + 1][2] - stops[i][2]) * f;
    return IM_COL32((int)(r * 255), (int)(g * 255), (int)(b * 255), 255);
}

ImU32 diverge_color(float t) {
    t = std::clamp(t, -1.0f, 1.0f);
    if (t < 0) {
        float k = 1.0f + t;  // 0 at -1 (blue), 1 at 0 (white)
        return IM_COL32((int)(60 + 195 * k), (int)(90 + 165 * k), 255, 255);
    }
    float k = 1.0f - t;  // 1 at 0 (white), 0 at +1 (red)
    return IM_COL32(255, (int)(90 + 165 * k), (int)(60 + 195 * k), 255);
}

void colorbar(const char* id, float min_v, float max_v, ImU32 (*ramp)(float), ImVec2 size) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImVec2 p1(p0.x + size.x, p0.y + size.y);
    ImGui::InvisibleButton(id, size);
    const int steps = 64;
    for (int i = 0; i < steps; i++) {
        float a = (float)i / steps, b = (float)(i + 1) / steps;
        ImVec2 q0(p0.x + a * size.x, p0.y), q1(p0.x + b * size.x, p1.y);
        dl->AddRectFilled(q0, q1, ramp(a));
    }
    dl->AddRect(p0, p1, IM_COL32(90, 95, 110, 255));
    char lo[64], hi[64];
    std::snprintf(lo, sizeof lo, "min %.4g", min_v);
    std::snprintf(hi, sizeof hi, "max %.4g", max_v);
    dl->AddText(ImVec2(p0.x, p1.y + 2), IM_COL32(190, 195, 205, 255), lo);
    ImVec2 ts = ImGui::CalcTextSize(hi);
    dl->AddText(ImVec2(p1.x - ts.x, p1.y + 2), IM_COL32(190, 195, 205, 255), hi);
    ImGui::Dummy(ImVec2(0, ImGui::GetTextLineHeight() + 2));
}

ImU32 block_color(int id, int face_shade_percent) {
    const mm::BlockStyle& s = mm::block_style(id);
    int k = face_shade_percent;
    return IM_COL32(s.rgb[0] * k / 100, s.rgb[1] * k / 100, s.rgb[2] * k / 100, 255);
}

}  // namespace ui
