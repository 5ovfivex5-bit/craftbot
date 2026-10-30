// Chat: the selected world's latest non-silent messages (PROTOCOL.md 6.8 `chat`).
// Shown only for v6 trainers (snapshot carries `team`); older trainers get a hint.
#include <cstdio>

#include <imgui.h>

#include "app.h"
#include "net/protocol.h"
#include "panels.h"
#include "ui_util.h"

void draw_chat(App& app, bool* p_open) {
    ImGui::SetNextWindowSize(ImVec2(380, 320), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(win::kChat, p_open)) {
        ImGui::End();
        return;
    }
    if (!trainer_connected(app)) {
        ui::no_data("Chat arrives with snapshots while Visualization is enabled.");
        ImGui::End();
        return;
    }
    const mm::Snapshot* s = app.f->snapshot.get();
    if (!s) {
        if (!app.viz_enabled) ImGui::TextDisabled("Tick 'Visualization enabled' in the Viewport to receive snapshots.");
        else ImGui::TextDisabled("Waiting for the first snapshot...");
    } else if (!s->has_team) {
        ImGui::TextDisabled("This trainer does not report team messages (no 'team' in snapshots).");
    } else {
        ImGui::Text("World %d   %zu messages", s->world_id, s->chat.size());
        ImGui::SameLine();
        ui::help_marker("Each agent emits one message token per step (0 = silent, shown here as nothing). "
                        "Tokens have no fixed meaning; teammates see them in their next observation.");
        static const mm::Snapshot* last_src = nullptr;
        static bool was_at_bottom = true;
        const bool new_snapshot = s != last_src;
        last_src = s;
        if (ImGui::BeginChild("##chatlog", ImVec2(0, 0), ImGuiChildFlags_Borders)) {
            if (s->chat.empty()) ImGui::TextDisabled("no messages yet");
            for (const mm::ChatEntry& e : s->chat) {
                mm::Rgb c = mm::bot_color(e.agent);
                ImGui::TextColored(ImVec4(c.r / 255.0f, c.g / 255.0f, c.b / 255.0f, 1.0f),
                                   "step %lld  bot %d: m%d", e.step, e.agent, e.msg);
            }
            if (new_snapshot && was_at_bottom) ImGui::SetScrollHereY(1.0f);
            was_at_bottom = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4.0f;
        }
        ImGui::EndChild();
    }
    ImGui::End();
}
