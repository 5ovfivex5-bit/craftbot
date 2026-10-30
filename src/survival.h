// Survival (PROTOCOL 12.4, SIM-2): day/night clock, sky + block light, hunger/food/regen/starvation,
// sleep, drowning, lava and burning. Owned by the survival phase-2 agent.
// survival::AgentState is embedded in Agent and serialized as raw bytes by save_state: it must stay POD.
// Time model: the world clock is derived (no stored world field): dayTick = startTick(seed) + 4 * W.t, where
// W.t counts decisions (4 ticks each) and startTick is a seed-derived morning time. 0 = sunrise.
// Sleep cannot skip the clock yet (needs a World field, see src/README_v7.md notes in the report).
#pragma once
#include <cstdint>

namespace craftsim {
struct World;
struct Agent;
namespace survival {

constexpr int DAY_TICKS = 24000;          // game ticks per day (vanilla)
constexpr int TICKS_PER_DECISION = 4;     // one decision = 4 physics ticks (sim default `ticks`)
constexpr int NIGHT_START = 14000;        // night = dayTick in [14000, 24000) (vanilla 13000 shifted to sunrise = 0)
constexpr int MAX_AIR = 300;              // breath, game ticks (vanilla)
constexpr int MAX_FOOD = 20;

// Per-agent survival state (embedded in Agent; must be trivially copyable for save_state).
struct AgentState {
    int hunger = 20;          // food level 0..20 (obs scalar 1 = hunger / 20)
    float saturation = 5.f;   // saturation 0..hunger (vanilla FoodData)
    float exhaustion = 0.f;   // 0..40, every 4 units drains one saturation or one food
    int foodTimer = 0;        // ticks toward natural regen / starvation damage
    float healAcc = 0.f;      // fractional healing carry (health is an int in the sim)
    int air = MAX_AIR;        // breath; below 0 drowning damage every 20 ticks
    int fireTicks = 0;        // ticks left burning (lava contact sets 300)
    int fireTimer = 0;        // ticks toward the next burn damage
    int lavaCd = 0;           // ticks until the next lava contact damage
    int lastTicks = 0;        // Agent::ticks at the last tick()
    int minedSum = 0;         // sum of TaskStats::mined at the last tick() (mining exhaustion)
    int killedSum = 0;        // sum of TaskStats::killed at the last tick() (attack exhaustion)
    double lastX = 0, lastY = 0, lastZ = 0;
    bool started = false;     // first tick() done (positions / sums valid)
    bool prevOnGround = false;
    bool wasNight = false;    // night state at the last tick() (dawn detection for survive_night)
};

// Called once per agent per decision, after its action and physics (v7 only).
void tick(World& W, Agent& a);
// `use` on a food item (held): consumes one and applies nutrition. True when eaten. Only when hungry (food < 20).
bool eat(Agent& a, int item);
// `use` on a bed item: sleeps when it is night and no hostile mob is within 8 blocks. True when the agent slept.
bool sleep(World& W, Agent& a);
// Light level 0..1 at block (x, y, z): max(sky light by time when the column is open to the sky, block light
// from light-emitting blocks with Manhattan falloff) / 15. Used for the obs scalar and mob spawning.
float lightAt(const World& W, int x, int y, int z);
// Time of day 0..1 (0 = sunrise). Used for the obs time sin/cos.
float timeOfDay(const World& W);
// Game tick of the day 0..DAY_TICKS-1 (0 = sunrise).
int dayTick(const World& W);
// True while it is night (dayTick >= NIGHT_START).
bool isNight(const World& W);

}  // namespace survival
}  // namespace craftsim
