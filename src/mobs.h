// Mobs (PROTOCOL 12.4, SIM-1). Owned by the mobs phase-2 agent; CORE only calls these hooks.
// Module state lives in mobs::State, which World embeds, so this file is the only place the mob
// state layout is defined. Mob data (health, damage, speed, spawn rules, attack type) comes from
// reg::MOBS (data/registry/mobs.json); drop tables are mirrored in mobs.cpp (not in the generated header).
#pragma once
#include <cstdint>

namespace craftsim {
struct World;
struct Agent;
namespace mobs {

constexpr int MAX_MOBS = 24;     // fixed capacity per world (no heap allocation per step)

// One live mob (or a free slot when type < 0). Positions are the feet centre in blocks; timers in ticks
// (20 per second; the mob module advances 4 ticks per world step, the CraftSim default `ticks`).
struct Mob {
    int8_t type = -1;            // index into reg::MOBS, -1 = free slot
    int8_t state = 0;            // AI state (debug / obs), see mobs.cpp
    bool ground = false;         // standing on a block after the last physics tick
    bool blocked = false;        // horizontal move cut short by a block in the last physics tick
    int8_t stuck = 0;            // consecutive decisions blocked while moving (random walk after a few)
    float hp = 0.f;
    double x = 0, y = 0, z = 0;
    double vx = 0, vy = 0, vz = 0;
    double wdx = 0, wdz = 0;     // wander direction (unit, or 0 when idle)
    double fx = 0, fz = 0;       // position of the last attacker (flee source)
    int16_t cd = 0;              // melee / shot cooldown
    int16_t fuse = 0;            // creeper fuse
    int16_t flee = 0;            // remaining flee time after a hit
    int16_t burn = 0;            // remaining sun burn time
    int16_t burnT = 0;           // ticks until the next burn damage
    int16_t wander = 0;          // remaining time of the current wander leg
    int32_t age = 0;             // decisions since spawn
};

// World-level mob state (embedded in World).
struct State {
    Mob mobs[MAX_MOBS];
    bool started = false;        // initial passive population spawned (needs the first agents)
    int passiveT = 0;            // decisions until the next passive spawn attempt
    int hostileT = 0;            // decisions until the next hostile spawn attempt
    // rayHit memo: the mobs that can be hit from one ray origin (image rays of an agent share it). Rebuilt when
    // the origin, the max distance or the mob state (`gen`, bumped by tick / attack) changes. Not serialized.
    uint32_t gen = 0;
    mutable uint32_t memoGen = 0xFFFFFFFFu;
    mutable double memoO[3] = {0, 0, 0};
    mutable double memoMax = -1;
    mutable bool memoCone = false;       // memo filtered to the view cone of memoF (beginView)
    mutable double memoF[3] = {0, 0, 1};
    mutable int memoN = 0;
    mutable int memoIdx[MAX_MOBS];
    mutable double memoC[MAX_MOBS][3];   // mob centre minus origin
    mutable double memoV[MAX_MOBS];      // |centre - origin|^2
    mutable double memoR2[MAX_MOBS];     // bounding sphere radius^2
    mutable double memoRad[MAX_MOBS];    // bounding sphere radius
};

// Called once per world generation, after terrain, trees and ores. Clears the mob state (no RNG draws).
void init(World& W);
// Called once per world step (decision), after the agents acted and before the observations are written.
// Uses W.rng. Agents are passed in: `ag` points to the world's K agents (K may be 0). Core change needed:
// sim.cpp stepWorld must call `mobs::tick(W, ag, K)` (the World-only overload runs no agent-dependent logic).
void tick(World& W, Agent* ag, int K);
inline void tick(World& W) { tick(W, nullptr, 0); }
// Optional, for speed: called once per agent image (writeObs) before its pixel rays. Pre-selects the mobs in the
// agent's view cone so that rayHit (called per pixel) only tests a few candidates. Without this call rayHit is
// still correct, just slower.
void beginView(const World& W, const Agent& a);
// Nearest-mob ray hook, called by raycast() for observation rays: true when a mob box is hit before maxDist
// along the ray (origin o, unit direction d). tHit = distance, blockId = the mob's pseudo block id.
bool rayHit(const World& W, const double o[3], const double d[3], double maxDist, float& tHit, uint8_t& blockId);
// Attack from agent `attacker` along its look direction, reach 3.0 blocks, only when no block is nearer.
// damage < 0 = use the attacker's best weapon (sword / axe, else fist 1). True when a mob was hit. On a kill:
// drops go straight into attacker.inv (the core's obtain() is static, so no team reward is paid for mob
// drops) and attacker.st.killed[mob] is incremented.
bool attack(World& W, Agent& attacker, float damage);
// Writes the 3 nearest mobs of agent `a` into out[18]: per mob [present, dx/16, dy/16, dz/16 (yaw frame),
// hostile, health/max]. Zeros when absent.
void obs(const World& W, const Agent& a, float* out18);

}  // namespace mobs
}  // namespace craftsim
