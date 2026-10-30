// Nether + portal (PROTOCOL 12.4, SIM-3). Owned by the nether phase-2 agent.
//
// Dimensions: the core keeps ONE live block array per world (World::w). This module keeps the other
// dimension's array in State::otherW and swaps the two (O(1)) when the team travels. Chests and the
// mobs state are swapped the same way, so they never leak across dimensions. The nether array is
// generated lazily, on the team's first trip (nether::generate only records the seed).
//
// Portals: the portal interior is AIR in the block array (there is no nether_portal block in the
// registry). Interior cells are tracked in State::portals / State::portalAt per dimension.
#pragma once
#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

#include "gen/registry.h"
#include "mobs.h"

namespace craftsim {
struct World;
struct Agent;
namespace nether {

constexpr int DIM_OVERWORLD = 0;
constexpr int DIM_NETHER = 1;
constexpr int PORTAL_DECISIONS = 20;   // consecutive decisions inside a lit portal before the team travels (80 ticks)

// Chest contents while the other dimension is active (same layout as craftsim::Chest).
struct StashedChest {
    int count[reg::NUM_ITEMS];
    uint8_t depositMask;
};

// A portal in one dimension. Interior cell (u, v) is (ox+u, oy+v, oz) for axis 0 (interior along x,
// plane z = oz) and (ox, oy+v, oz+u) for axis 1 (interior along z, plane x = ox). (ox, oy, oz) is the
// bottom-left interior cell. The frame is obsidian around the w x h interior.
struct Portal {
    int axis = 0;
    int ox = 0, oy = 0, oz = 0;
    int w = 2, h = 3;
    int link = -1;      // index of the partner portal in the other dimension, -1 none
};

// World-level nether state (embedded in World).
struct State {
    int dim = DIM_OVERWORLD;                 // dimension of the live block array (World::w)
    uint64_t netherSeed = 0;                 // seed of this world's nether (set by generate)
    std::vector<uint8_t> otherW;             // block array of the inactive dimension (empty: nether not built yet)
    std::map<int, StashedChest> otherChests; // chests of the inactive dimension
    mobs::State otherMobs;                   // mobs of the inactive dimension
    std::vector<Portal> portals[2];          // portals per dimension
    std::unordered_map<int, int> portalAt[2];   // interior cell index -> index into portals[dim]
    std::vector<Agent*> team;                // agents seen by travel() (team members, call order)
    std::vector<int> timers;                 // decisions in a portal, parallel to team
};

// Called once per world generation (after the overworld), with seed = world seed ^ 0x9E37... Resets the
// nether state; the nether block array is built on the first trip.
void generate(World& W, uint64_t seed);
// `use` flint_and_steel at the crosshair block (x, y, z): lights a portal frame whose interior touches an
// air face of that block. True when a portal was lit (CORE sets portal_lit).
bool tryLightPortal(World& W, Agent& a, int x, int y, int z);
// Called once per agent per decision. When this agent has stood in a lit portal for PORTAL_DECISIONS
// decisions, the whole team (every live agent seen by travel) moves to the other dimension together.
void travel(World& W, Agent& a);

}  // namespace nether
}  // namespace craftsim
