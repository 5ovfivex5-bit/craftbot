// Small reusable widgets: no-data notice, KPI rows, line plots, colour ramps.
#pragma once

#include <string>
#include <vector>

#include <imgui.h>

namespace ui {

// "No data — trainer not connected" when there is nothing received yet.
void no_data(const char* what = nullptr);
void help_marker(const char* text);
void kv(const char* key, const std::string& value, ImU32 color = 0);
void kv_fmt(const char* key, const char* fmt, ...) IM_FMTARGS(2);
void label_bold(const char* text);
void section(const char* text);

std::string fmt_count(long long n);      // 1 234 567
std::string fmt_hms(double seconds);     // 1h 02m 03s
std::string fmt_bytes(double bytes);     // 1.2 MiB
std::string fmt_time(double unix_s);     // local HH:MM:SS date-time
std::string fmt_num(double v, int prec = 4);

// Line plot inside the current window. xs/ys are equally sized.
struct PlotSeries {
    const char* label;
    ImU32 color;
    std::vector<float> xs;
    std::vector<float> ys;
};
void line_plot(const char* id, const std::vector<PlotSeries>& series, ImVec2 size, const char* y_unit = nullptr);

// Viridis-like ramp for heatmaps, t in [0, 1].
ImU32 heat_color(float t);
// Diverging blue-white-red ramp for signed values, t in [-1, 1].
ImU32 diverge_color(float t);

// Draws a horizontal colour bar with min/max labels.
void colorbar(const char* id, float min_v, float max_v, ImU32 (*ramp)(float), ImVec2 size);

// Palette colour for a block id as ImU32 (opaque).
ImU32 block_color(int id, int face_shade_percent = 100);

}  // namespace ui
