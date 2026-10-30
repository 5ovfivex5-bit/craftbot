// Survival (PROTOCOL 12.4, SIM-2). See survival.h. Mechanics follow vanilla FoodData / drowning / lava
// where noted; tick granularity is one decision (4 physics ticks), so timers advance by the ticks elapsed.
#include "survival.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include "sim.h"

namespace craftsim::survival {
namespace {

// Food table (data/registry/items.json "food", nutrition / saturation). The generated registry does not carry
// nutrition, so it is mirrored here by item name; tests/test_survival.py checks it against the JSON.
struct Food { const char* name; int nutrition; float saturation; };
constexpr Food FOODS[] = {
    {"apple", 4, 2.4f},          {"bread", 5, 6.0f},          {"beef", 3, 1.8f},
    {"cooked_beef", 8, 12.8f},   {"porkchop", 3, 1.8f},       {"cooked_porkchop", 8, 12.8f},
    {"mutton", 2, 1.2f},         {"cooked_mutton", 6, 9.6f},  {"chicken", 2, 1.2f},
    {"cooked_chicken", 6, 7.2f}, {"rotten_flesh", 4, 0.8f},
};

// Per-item food lookup (nutrition 0 = not food), built once from the names above.
const std::array<const Food*, reg::NUM_ITEMS>& foodTable() {
    static const std::array<const Food*, reg::NUM_ITEMS> table = [] {
        std::array<const Food*, reg::NUM_ITEMS> t{};
        for (int i = 0; i < reg::NUM_ITEMS; i++)
            for (const Food& f : FOODS)
                if (std::strcmp(reg::ITEMS[i].name, f.name) == 0) t[i] = &f;
        return t;
    }();
    return table;
}

// Per-block light emission (registry light_emission) and opacity for sky light.
inline int emissionOf(int b) { return b >= 0 && b < reg::NUM_BLOCKS ? reg::BLOCKS[b].light : 0; }
inline bool opaqueToSky(int b) { return reg::BLOCKS[b].solid && !reg::BLOCKS[b].transparent; }

// Block id at world cell (x, y, z); out of the world: air above, bedrock-like wall at the sides (matches sim get()).
inline int blockAt(const World& W, int x, int y, int z) {
    if (y >= WY) return reg::BLK_AIR;
    if (x < 0 || x >= WX || z < 0 || z >= WZ || y < 0) return reg::BLK_BEDROCK;
    return W.w[(y * WZ + z) * WX + x];
}

// Sky light level (0..15) by day tick: full day, dusk ramp to the night ambient 4, dawn ramp back up.
float skyLevel(int d) {
    if (d < 12000) return 15.f;
    if (d < NIGHT_START) return 15.f - 11.f * (d - 12000) / float(NIGHT_START - 12000);
    if (d < 22000) return 4.f;
    return 4.f + 11.f * (d - 22000) / float(DAY_TICKS - 22000);
}

// Deterministic per-world start time: a morning time in [0, 2000) game ticks from the world seed (no RNG draw).
int startTick(uint64_t seed) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (int)(z % 2000u);
}

// Block light: max over light-emitting blocks within Manhattan distance `emission` of (emission - distance).
int blockLight(const World& W, int x, int y, int z) {
    constexpr int R = 14;   // torch 14 is the largest emission the registry uses below glowstone/lava (15)
    int best = 0;
    for (int dy = -R; dy <= R; dy++) {
        const int ay = std::abs(dy);
        if (ay > R) continue;
        const int ry = R - ay;
        for (int dz = -ry; dz <= ry; dz++) {
            const int rz = ry - std::abs(dz);
            for (int dx = -rz; dx <= rz; dx++) {
                const int b = blockAt(W, x + dx, y + dy, z + dz);
                const int e = emissionOf(b);
                if (e == 0) continue;
                const int lvl = e - (std::abs(dx) + ay + std::abs(dz));
                if (lvl > best) best = lvl;
            }
        }
    }
    return best;
}

constexpr double EYE_H = 1.62;   // player eye height (sim.cpp EYE)

// Feet / body / eye samples of an agent (cell coordinates).
struct Cells { int x, z, feet, body, eye; };
inline Cells cellsOf(const Agent& a) {
    return {(int)std::floor(a.px), (int)std::floor(a.pz), (int)std::floor(a.py), (int)std::floor(a.py + 0.9),
            (int)std::floor(a.py + EYE_H)};
}

inline void addExhaustion(AgentState& s, float amount) { s.exhaustion = std::min(s.exhaustion + amount, 40.f); }

inline void heal(Agent& a, AgentState& s, float amount) {
    if (a.health >= MAX_FOOD) { s.healAcc = 0.f; return; }
    s.healAcc += amount;
    while (s.healAcc >= 1.f && a.health < MAX_FOOD) { a.health++; s.healAcc -= 1.f; }
    if (a.health >= MAX_FOOD) s.healAcc = 0.f;
}

}  // namespace

int dayTick(const World& W) {
    const int64_t t = (int64_t)startTick(W.seed) + (int64_t)W.t * TICKS_PER_DECISION + W.dayOffset;
    return (int)(t % DAY_TICKS);
}

bool isNight(const World& W) { return dayTick(W) >= NIGHT_START; }

float timeOfDay(const World& W) { return dayTick(W) / float(DAY_TICKS); }

float lightAt(const World& W, int x, int y, int z) {
    // Sky light only where nothing opaque is above the cell.
    float sky = 0.f;
    bool open = true;
    for (int yy = y + 1; yy < WY; yy++) {
        if (opaqueToSky(blockAt(W, x, yy, z))) { open = false; break; }
    }
    if (open) sky = skyLevel(dayTick(W));
    const float blk = (float)blockLight(W, x, y, z);
    return std::clamp(std::max(sky, blk) / 15.f, 0.f, 1.f);
}

bool eat(Agent& a, int item) {
    if (item < 0 || item >= reg::NUM_ITEMS) return false;
    const Food* f = foodTable()[item];
    if (!f || a.inv[item] <= 0) return false;
    AgentState& s = a.surv;
    if (s.hunger >= MAX_FOOD) return false;   // vanilla canEat: food < 20 (no always-edible food in the registry)
    a.inv[item]--;
    s.hunger = std::min(MAX_FOOD, s.hunger + f->nutrition);
    // FoodData.add: saturation clamped to [0, food].
    s.saturation = std::clamp(s.saturation + f->saturation, 0.f, float(s.hunger));
    return true;
}

bool sleep(World& W, Agent& a) {
    if (!isNight(W)) return false;
    // Nearest 3 mobs (obs layout: present, dx, dy, dz (/16), hostile, health). Any hostile within 8 blocks blocks sleep.
    float buf[18] = {};
    mobs::obs(W, a, buf);
    for (int k = 0; k < 3; k++) {
        const float* m = buf + 6 * k;
        if (m[0] < 0.5f || m[4] < 0.5f) continue;
        const double dx = m[1] * 16.0, dy = m[2] * 16.0, dz = m[3] * 16.0;
        if (std::sqrt(dx * dx + dy * dy + dz * dz) <= 8.0) return false;
    }
    // Skip the world clock to the next sunrise (dayTick 0), like vanilla when everyone sleeps.
    W.dayOffset += DAY_TICKS - dayTick(W);
    return true;
}

void tick(World& W, Agent& a) {
    if (a.health <= 0) return;
    AgentState& s = a.surv;
    const int dt = std::max(0, a.ticks - s.lastTicks);
    s.lastTicks = a.ticks;
    const Cells c = cellsOf(a);
    const bool eyeWater = blockAt(W, c.x, c.eye, c.z) == reg::BLK_WATER;
    const bool inWater = blockAt(W, c.x, c.feet, c.z) == reg::BLK_WATER || blockAt(W, c.x, c.body, c.z) == reg::BLK_WATER;
    const bool inLava = blockAt(W, c.x, c.feet, c.z) == reg::BLK_LAVA || blockAt(W, c.x, c.body, c.z) == reg::BLK_LAVA;

    int mined = 0, killed = 0;
    for (int i = 0; i < reg::NUM_BLOCKS; i++) mined += a.st.mined[i];
    for (int i = 0; i < reg::NUM_MOBS; i++) killed += a.st.killed[i];

    // ---- exhaustion (vanilla values): sprint 0.1/m on ground, swim 0.01/m, jump 0.05, mining 0.005/block, kill 0.1.
    if (s.started) {
        const double hd = std::hypot(a.px - s.lastX, a.pz - s.lastZ);
        const double dy = a.py - s.lastY;
        const double d3 = std::sqrt(hd * hd + dy * dy);
        const bool sprint = a.lastAction == PR_SPRINT_FWD && a.onGround && !inWater;
        if (eyeWater) addExhaustion(s, float(0.01 * d3));
        else if (inWater) addExhaustion(s, float(0.01 * hd));
        else if (sprint) addExhaustion(s, float(0.1 * hd));
        if ((a.lastAction == PR_JUMP || a.lastAction == PR_FWD_JUMP) && s.prevOnGround) addExhaustion(s, 0.05f);
        if (mined > s.minedSum) addExhaustion(s, 0.005f * (mined - s.minedSum));
        if (killed > s.killedSum) addExhaustion(s, 0.1f * (killed - s.killedSum));
    }
    s.lastX = a.px;
    s.lastY = a.py;
    s.lastZ = a.pz;
    s.minedSum = mined;
    s.killedSum = killed;
    s.prevOnGround = a.onGround;
    s.started = true;

    // ---- FoodData.tick: exhaustion drains saturation first, then food.
    while (s.exhaustion > 4.f) {
        s.exhaustion -= 4.f;
        if (s.saturation > 0.f) s.saturation = std::max(s.saturation - 1.f, 0.f);
        else s.hunger = std::max(s.hunger - 1, 0);
    }
    // Natural regeneration (gamerule on, Normal difficulty): fast when saturated, slow when food >= 18, starvation at 0.
    const bool hurt = a.health < MAX_FOOD;
    if (s.saturation > 0.f && hurt && s.hunger >= MAX_FOOD) {
        s.foodTimer += dt;
        while (s.foodTimer >= 10) {
            s.foodTimer -= 10;
            const float spent = std::min(s.saturation, 6.f);
            heal(a, s, spent / 6.f);
            addExhaustion(s, spent);
        }
    } else if (s.hunger >= 18 && hurt) {
        s.foodTimer += dt;
        while (s.foodTimer >= 80) {
            s.foodTimer -= 80;
            heal(a, s, 1.f);
            addExhaustion(s, 6.f);
        }
    } else if (s.hunger <= 0) {
        s.foodTimer += dt;
        while (s.foodTimer >= 80) {
            s.foodTimer -= 80;
            if (a.health > 1) a.health--;   // Normal difficulty: starvation stops at 1 health
        }
    } else {
        s.foodTimer = 0;
    }

    // ---- drowning: breath 300 ticks with the eyes in water, then 2 damage every 20 ticks; refills at 4/tick.
    if (eyeWater) {
        s.air -= dt;
        while (s.air <= -20) { s.air += 20; a.health -= 2; }
    } else {
        s.air = std::min(MAX_AIR, s.air + 4 * dt);
    }

    // ---- lava: sets on fire for 300 ticks and hurts 4 every 10 ticks (cooldown approximated); water puts fire out;
    // burning hurts 1 every 20 ticks.
    s.lavaCd = std::max(0, s.lavaCd - dt);
    if (inLava) {
        s.fireTicks = std::max(s.fireTicks, 300);
        if (s.lavaCd <= 0) { a.health -= 4; s.lavaCd = 10; }
    }
    if (inWater) s.fireTicks = 0;
    if (s.fireTicks > 0) {
        s.fireTicks = std::max(0, s.fireTicks - dt);
        s.fireTimer += dt;
        while (s.fireTimer >= 20) { s.fireTimer -= 20; a.health -= 1; }
    } else {
        s.fireTimer = 0;
    }

    // ---- survive_night: a night that ends with the agent still alive (dusk -> dawn).
    const bool night = isNight(W);
    if (s.wasNight && !night && a.health > 0) a.st.surviveNight = true;
    s.wasNight = night;

    if (a.health < 0) a.health = 0;
}

}  // namespace craftsim::survival
