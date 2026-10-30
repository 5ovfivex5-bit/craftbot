// Builds runtime + worldgen variety (PROTOCOL 12.4, SIM-4). Blueprints come from data/builds/blueprints.json
// (generated into src/gen/blueprints.h; blueprint index = build id used by build_complete predicates).
#pragma once
#include <cstdint>
#include <vector>

namespace craftsim {
struct World;
struct Agent;
namespace build {

// One cell of the world's build target: world coordinates, wanted block (0 = required air) and wanted item.
struct TCell {
    int16_t x, y, z;
    uint8_t block, item;
};

// World-level build state (embedded in World). The target is placed lazily, on the first obs/progress call of an
// episode whose task has a blueprint (see build.cpp). Fields are mutable because obs/progress take const World&.
struct State {
    mutable uint64_t key = 0;            // hash of (seed, task, task overrides) the target was placed for
    mutable bool placed = false;         // placement was attempted for `key` (blueprint -1: task has none)
    mutable int blueprint = -1;          // blueprint index of the target (-1 none)
    mutable int rot = 0;                 // rotation 0..3 (quarter turns about y)
    mutable int ox = 0, oy = 0, oz = 0;  // world position of the footprint's minimum corner and bottom layer
    mutable int nSolid = 0;              // cells with a non-air wanted block
    mutable std::vector<TCell> cells;    // solid cells first, then required-air cells (block 0)
};

// Worldgen variety, run by initWorld after the core terrain on every world generation (any action mode): biomes
// (plains / forest / desert), dense oak in forests, desert sand + no trees, boulders, near-surface caves with exposed
// coal and iron, and 0..2 converted village houses or ruins with a chest. Deterministic in `seed`. The spawn box
// (+-26 columns around the centre) stays plains with grass and is never built on. variety = false keeps the
// core terrain unchanged (used by v6 if the caller asks for it).
void worldgen(World& W, uint64_t seed, bool variety = true);
// Sets the world's build target to blueprint `blueprint` for the current episode: (x, z) = centre cell, y = bottom
// layer, rotation 0. The episode's task does not need to reference it (test/debug hook).
void placeTarget(World& W, int blueprint, int x, int y, int z);
// Progress 0..1 of the episode's build target: (matched solid cells - violated required-air cells) / solid cells.
// 0 when the task has no blueprint or the target is not placed yet.
float progress(const World& W);
// Writes out[33]: 8 nearest unfilled solid cells x [dx/8, dy/8, dz/8 (yaw frame), wanted item / NUM_ITEMS], then
// progress. Places the target on the first call of an episode (position of the agent = team spawn at reset).
// Zeros when the task has no blueprint.
void obs(const World& W, const Agent& a, float* out33);

}  // namespace build
}  // namespace craftsim
