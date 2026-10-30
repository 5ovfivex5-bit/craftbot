#include "protocol.h"

#include <cstring>
#include <string>
#include <unordered_map>

#include "gen_palette.h"

namespace mm {
namespace {

// v6 palette (ids 0..19), kept as the fallback for trainers whose block names are not in the
// generated registry. Mirrors the old COLORS / SPECKLE table in src/craftsim.cpp.
constexpr int kLegacyBlocks = 20;
const BlockStyle kLegacy[kLegacyBlocks] = {
    {{135, 190, 255}, {0, 0, 0}, false},      // 0 AIR (sky)
    {{95, 159, 53}, {0, 0, 0}, false},        // 1 GRASS
    {{134, 96, 67}, {0, 0, 0}, false},        // 2 DIRT
    {{125, 125, 125}, {0, 0, 0}, false},      // 3 STONE
    {{105, 105, 105}, {0, 0, 0}, false},      // 4 COBBLE
    {{102, 81, 51}, {0, 0, 0}, false},        // 5 LOG
    {{58, 120, 38}, {0, 0, 0}, false},        // 6 LEAVES
    {{180, 144, 90}, {0, 0, 0}, false},       // 7 PLANKS
    {{150, 98, 56}, {0, 0, 0}, false},        // 8 TABLE
    {{80, 80, 80}, {0, 0, 0}, false},         // 9 FURNACE
    {{125, 125, 125}, {25, 25, 25}, true},    // 10 COAL_ORE
    {{125, 125, 125}, {216, 175, 147}, true}, // 11 IRON_ORE
    {{125, 125, 125}, {90, 230, 230}, true},  // 12 DIAMOND_ORE
    {{45, 45, 45}, {0, 0, 0}, false},         // 13 BEDROCK
    {{219, 207, 163}, {0, 0, 0}, false},      // 14 SAND
    {{136, 126, 126}, {0, 0, 0}, false},      // 15 GRAVEL
    {{52, 95, 218}, {0, 0, 0}, false},        // 16 WATER
    {{77, 77, 81}, {0, 0, 0}, false},         // 17 DEEPSLATE
    {{118, 78, 40}, {80, 50, 25}, true},      // 18 CHEST (brown wood)
    {{255, 70, 230}, {0, 0, 0}, false, true}, // 19 AGENT (pseudo id, bright marker)
};

const BlockStyle kUnknown = {{128, 128, 128}, {0, 0, 0}, false, false};

// Table indexed by the trainer's block id.
std::vector<BlockStyle> g_table;
std::vector<std::string> g_names;  // names the table was built from (cache key)
bool g_built = false;

BlockStyle from_gen(const gen::GenBlock& b) {
    BlockStyle s{};
    for (int k = 0; k < 3; k++) {
        s.rgb[k] = b.rgb[k];
        s.speckle[k] = b.speckle[k];
    }
    s.has_speckle = b.has_speckle;
    s.pseudo = b.pseudo;
    return s;
}

const gen::GenBlock* find_gen_block(const std::string& name) {
    static const std::unordered_map<std::string, const gen::GenBlock*> index = [] {
        std::unordered_map<std::string, const gen::GenBlock*> m;
        for (const auto& b : gen::kBlocks) m[b.name] = &b;
        return m;
    }();
    auto it = index.find(name);
    return it == index.end() ? nullptr : it->second;
}

void rebuild(const std::vector<std::string>& names) {
    g_table.clear();
    const int n = names.empty() ? gen::kBlockCount : (int)names.size();
    g_table.reserve(n);
    for (int id = 0; id < n; id++) {
        const gen::GenBlock* gb = nullptr;
        if (names.empty()) {
            gb = &gen::kBlocks[id];  // no names sent: registry order is the id order
        } else {
            gb = find_gen_block(names[id]);
        }
        if (gb) {
            g_table.push_back(from_gen(*gb));
        } else if (id < kLegacyBlocks) {
            g_table.push_back(kLegacy[id]);  // older trainer, v6 id
        } else {
            g_table.push_back(kUnknown);
        }
    }
}

}  // namespace

void palette_set_names(const std::vector<std::string>& names) {
    if (g_built && names == g_names) return;
    g_names = names;
    rebuild(names);
    g_built = true;
}

int palette_size() {
    if (!g_built) palette_set_names({});
    return (int)g_table.size();
}

// Mirrors COLORS / SPECKLE in src/craftsim.cpp. Index == block id.
const BlockStyle& block_style(int id) {
    if (!g_built) palette_set_names({});
    if (id < 0 || id >= (int)g_table.size()) return kUnknown;
    return g_table[id];
}

Rgb bot_color(int k) {
    static const Rgb kBots[5] = {
        {60, 120, 255},   // 0 blue
        {235, 65, 65},    // 1 red
        {70, 200, 90},    // 2 green
        {240, 215, 60},   // 3 yellow
        {165, 85, 235},   // 4 purple
    };
    if (k < 0 || k >= 5) return Rgb{180, 180, 180};
    return kBots[k];
}

MobLook mob_look(const std::string& raw) {
    std::string type = raw.rfind("minecraft:", 0) == 0 ? raw.substr(10) : raw;
    for (const auto& m : gen::kMobs) {
        if (type != m.name) continue;
        const BlockStyle& st = block_style(m.pseudo_id);
        return MobLook{m.width, m.height, Rgb{st.rgb[0], st.rgb[1], st.rgb[2]}, true};
    }
    return MobLook{0.6f, 1.8f, Rgb{150, 150, 150}, false};
}

}  // namespace mm
