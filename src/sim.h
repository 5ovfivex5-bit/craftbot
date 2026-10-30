// craftsim: headless, batched, Minecraft-like voxel simulator (public C++ API), v7.
// Worlds hold the blocks, chests, RNG, task and team achievements; agents (K per world) hold body,
// inventory, health, mining, task statistics and novelty state. Mechanics are implemented in sim.cpp.
// Data (blocks, items, recipes, mobs, tasks) is generated into gen/registry.h by tools/gen_registry.py.
// Two action modes: "v7" (factored actions, data-driven registry, tasks) and "v6" (legacy 28 actions,
// v6 ids/tables and the 102-float obs of PROTOCOL 6.5). See src/README_v7.md.
#pragma once

#include <cstdint>
#include <cstddef>
#include <map>
#include <random>
#include <array>
#include <vector>

#include "build.h"
#include "gen/registry.h"
#include "mobs.h"
#include "nether.h"
#include "survival.h"

namespace craftsim {

constexpr int WX = 128, WY = 64, WZ = 128;   // world size in blocks (x, y, z)
constexpr int SEA_LEVEL = 30;                // water fills terrain up to this y (MC sea 63 scaled)
constexpr int DEEPSLATE_Y = 8;               // stone below this y is deepslate
constexpr int MAX_AGENTS = 5;                // agents per world (1..5)
constexpr int NUM_MSGS = 8;                  // message tokens (0 = silent)

constexpr int NUM_BLOCKS = reg::NUM_BLOCKS;      // 59 (ids 0..19 = v6 blocks, mobs are pseudo blocks)
constexpr int NUM_ITEMS = reg::NUM_ITEMS;        // 97 (ids 0..14 = v6 items)
constexpr int NUM_RECIPES = reg::NUM_RECIPES;
constexpr int NUM_MOBS = reg::NUM_MOBS;
constexpr int NUM_TASKS = reg::NUM_TASKS;
constexpr int NUM_FAMILIES = reg::NUM_FAMILIES;

// v6 sizes (action_mode "v6": 15 items and 20 blocks are visible, 28 actions, 102-float obs)
constexpr int V6_NUM_BLOCKS = 20;
constexpr int V6_NUM_ITEMS = 15;
constexpr int V6_NUM_ACTIONS = 28;

enum Block : uint8_t {
    AIR, GRASS, DIRT, STONE, COBBLE, LOG, LEAVES, PLANKS, TABLE, FURNACE,
    COAL_ORE, IRON_ORE, DIAMOND_ORE, BEDROCK, SAND, GRAVEL, WATER, DEEPSLATE,
    CHEST,   // 18
    AGENT,   // 19: pseudo block, only ever seen in observations (never stored in the world)
};

enum Item {
    I_LOG, I_PLANKS, I_STICK, I_TABLE, I_WOOD_PICK, I_COBBLE, I_STONE_PICK, I_FURNACE,
    I_COAL, I_IRON_ORE, I_IRON_INGOT, I_IRON_PICK, I_DIAMOND, I_DIRT, I_CHEST,
};

// v6 action ids (action_mode "v6" only; mapped to primary/recipe/item internally).
enum Action {
    A_NOOP, A_FWD, A_BACK, A_LEFT, A_RIGHT, A_JUMP, A_FWD_JUMP,
    A_TURN_L, A_TURN_R, A_LOOK_UP, A_LOOK_DOWN,
    A_ATTACK, A_PLACE_BLOCK, A_PLACE_TABLE, A_PLACE_FURNACE,
    A_CRAFT_PLANKS, A_CRAFT_STICKS, A_CRAFT_TABLE, A_CRAFT_WOOD_PICK,
    A_CRAFT_STONE_PICK, A_CRAFT_FURNACE, A_SMELT_IRON, A_CRAFT_IRON_PICK,
    A_CRAFT_CHEST, A_PLACE_CHEST, A_DEPOSIT, A_WITHDRAW, A_GIVE,
};

// v7 primary actions: action column 0. Column 1 = craft/smelt recipe (0 none, r+1 for recipe r).
// Column 2 = item select (0 keep held item, i+1 select item i). Column 3 = message token 0..7.
enum Primary : uint8_t {
    PR_NOOP, PR_FWD, PR_BACK, PR_LEFT, PR_RIGHT, PR_JUMP, PR_FWD_JUMP, PR_SNEAK, PR_SPRINT_FWD,
    PR_TURN_L, PR_TURN_R, PR_LOOK_UP, PR_LOOK_DOWN, PR_ATTACK, PR_USE, PR_CRAFT, PR_SMELT,
    PR_DEPOSIT, PR_WITHDRAW, PR_GIVE, PR_DROP,
    NUM_PRIMARY,
};
constexpr int NUM_ACTION_COLS = 4;

enum ActionMode { MODE_V6 = 6, MODE_V7 = 7 };

// ---- v6 obs layout (VEC_DIM_V6 = 102, PROTOCOL 6.5): inventory 15, 11 scalars, goal 15, 4 teammates
// x 12, nearest chest 4, crosshair chest contents 9 (items 0..8).
constexpr int VEC_DIM_V6 = V6_NUM_ITEMS + 11 + V6_NUM_ITEMS + 4 * 12 + 4 + 9;   // 102
constexpr int GOAL_OFFSET = V6_NUM_ITEMS + 11;                                    // 26
constexpr int TEAM_OFFSET = GOAL_OFFSET + V6_NUM_ITEMS;                           // 41
constexpr int CHEST_OFFSET = TEAM_OFFSET + 4 * 12;                                // 89
constexpr int XHAIR_OFFSET = CHEST_OFFSET + 4;                                    // 93

// ---- v7 obs layout (VEC_DIM = VEC7_DIM). Names are generated in craftsim.VEC_NAMES.
constexpr int V7_INV = 0;                                   // NUM_ITEMS: log2(1+n)/7
constexpr int V7_NUM_SCALARS = 20;
constexpr int V7_SCALARS = V7_INV + NUM_ITEMS;              // see sim.cpp writeObsV7 for the order
constexpr int V7_TASK = V7_SCALARS + V7_NUM_SCALARS;        // NUM_TASKS one-hot
constexpr int V7_TASK_PARAM = V7_TASK + NUM_TASKS;          // 2: target item / NUM_ITEMS, count / 64
constexpr int V7_TEAM = V7_TASK_PARAM + 2;                  // 4 slots x 12 (as v6)
constexpr int V7_CHEST = V7_TEAM + 4 * 12;                  // nearest chest 4
constexpr int V7_XHAIR = V7_CHEST + 4;                      // crosshair chest contents, 9 valuable items
constexpr int V7_MOBS = V7_XHAIR + 9;                       // 3 mobs x 6 (mobs::obs)
constexpr int V7_BUILD = V7_MOBS + 18;                      // 8 cells x 4 + progress (build::obs)
constexpr int VEC7_DIM = V7_BUILD + 33;
constexpr int VEC_DIM = VEC7_DIM;                           // default (v7) obs width

constexpr uint32_t GOAL_BIT = 1u << 31;                     // ach bit: goal (v6) / task success (v7)
constexpr int MAX_MILESTONES = reg::MAX_MILESTONES;

constexpr int NUM_CELLS = (WX / 4) * (WY / 4) * (WZ / 4);   // 16384 novelty cells
constexpr int VISITED_WORDS = NUM_CELLS / 64;               // 256

// Per-agent, per-episode task statistics (v7). Zeroed at every episode start.
struct TaskStats {
    int mined[NUM_BLOCKS] = {};
    int placed[NUM_BLOCKS] = {};
    int crafted[NUM_ITEMS] = {};
    int eaten[NUM_ITEMS] = {};
    int killed[NUM_MOBS] = {};
    int ate = 0;
    bool slept = false, surviveNight = false, portalLit = false;
    double dist = 0;                 // horizontal blocks travelled
    double minY = 1e9;               // lowest feet y reached (reached_y: minY <= max_y)
    int near[4] = {};                // decisions with a teammate within 2 / 4 / 8 / 16 blocks
    int look = 0, follow = 0, crouch = 0, gave = 0, shared = 0, msgNear = 0;
    uint32_t msDone = 0;             // milestone bits of the current task
};

// One agent: body, inventory, health, mining, novelty bookkeeping, kit, task statistics.
struct Agent {
    double px = 0, py = 0, pz = 0, vx = 0, vy = 0, vz = 0;
    float yaw = 0, pitch = 0;
    bool onGround = false;
    double fallDist = 0;
    int health = 20;
    int inv[NUM_ITEMS] = {};
    int fuel = 0;
    int mx = -1, my = -1, mz = -1;   // block being mined
    double prog = 0;                 // mining progress 0..1
    int ticks = 0;                   // physics ticks this episode (attack can span many per step)
    int lastAction = 0;              // last action id applied (v6 id or v7 primary; 0 = noop after reset)
    int msg = 0;                     // message token sent on the last step (0 = silent)
    bool kit = false;                // episode started with a start kit
    uint64_t kitGot = 0;             // items the kit provided: no shaping, no ach bits this episode (v6: <= 14)
    // v7 state
    int held = -1;                   // selected item (-1 none), set by the item-select column / v6 placement
    bool sneak = false;              // sneak toggle state (primary sneak flips it)
    int dim = 0;                     // dimension (0 overworld; nether not implemented yet)
    int sneakT[2] = {-1000000, -1000000};   // world step of the last two sneak toggles
    uint64_t shapeMask[2] = {0, 0};  // knowledge shaping paid this episode (item bitmask, NUM_ITEMS <= 128)
    TaskStats st;
    survival::AgentState surv;
    // Curiosity bookkeeping (per agent, per episode).
    uint64_t visited[VISITED_WORDS] = {};   // 4x4x4 cells entered
    uint64_t seenBlocks = 0;         // block ids seen in the obs image (bit = block id)
    int bestDepthStep = 0;           // deepest 2-block step below SEA_LEVEL reached so far
};

// Chest storage: unlimited counts per item.
struct Chest {
    int count[NUM_ITEMS] = {};
    uint8_t depositMask = 0;         // agents (bit k) that deposited into it this episode (shared_chest)
};

// One world: blocks, chests, RNG streams, goal, task, team achievements, episode counters.
struct World {
    std::vector<uint8_t> w;          // blocks, index (y*WZ + z)*WX + x
    std::map<int, Chest> chests;     // key = block cell index of the chest
    std::mt19937_64 rng;             // world generation, goal, kit, spawn (reseeded per world)
    std::mt19937_64 seedRng;         // stream of world seeds for auto-resets
    uint64_t seed = 0;               // seed used by the last world generation
    int goal = 0;                    // goal item index for this episode (v6 goal mechanism)
    uint32_t got = 0;                // team achievement mask (first-time item pickups) + GOAL_BIT
    int t = 0;                       // world steps taken this episode
    int64_t dayOffset = 0;           // game ticks skipped by sleeping (survival day clock)
    int episode = 0;                 // episode index
    bool restart = false;            // episode was restored by VecEnv::loadState (cleared on new world)
    // v7 task (one per episode). taskW = per-world sampling weights (config, not saved), taskP0/P1 =
    // one-off overrides from set_task (-1 none) for the current episode.
    int task = 0;
    int taskP0 = -1, taskP1 = -1;
    int maxT = 0;                    // episode length limit in steps (min of max_steps and the task's)
    std::vector<float> taskW;
    mobs::State mobs;
    nether::State nether;
    build::State build;
};

// Read-only view of one agent (plus its world's episode fields), as returned by VecEnv::state().
struct Snapshot {
    double x = 0, y = 0, z = 0, vx = 0, vy = 0, vz = 0;
    float yaw = 0, pitch = 0;
    bool onGround = false;
    int health = 0, t = 0, ticks = 0;
    uint64_t seed = 0;
    int episode = 0, lastAction = 0, msg = 0;
    double miningProgress = 0;
    int goal = 0;
    bool kit = false;
    bool restart = false;
    bool hasTarget = false;          // block under the crosshair within reach
    int tx = 0, ty = 0, tz = 0;
    uint8_t tblock = AIR;
    int held = -1, dim = 0, task = 0, taskMaxT = 0;
    bool sneak = false;
};

// Block window around the player: full world height, XZ clipped to world bounds.
// blocks[((y - oy) * sz + (z - oz)) * sx + (x - ox)]; origin (ox, 0, oz).
struct Region {
    int ox = 0, oy = 0, oz = 0;
    int sx = 0, sy = 0, sz = 0;
    std::vector<uint8_t> blocks;
};

struct ChestView {
    int x = 0, y = 0, z = 0;
    int count[NUM_ITEMS] = {};
};

// Batched sim: W worlds x K agents. Agent index i = w*K + k. Per-agent buffers have N = W*K rows.
// Actions: mode v7 -> int32[N, 4] (primary, recipe, item select, msg); mode v6 -> int32[N].
// Obs per agent: img [res,res,2] = (block id hit by ray, depth 0..255), vec [vecDim()].
// Episodes are per world: all K agents are done on the same step; the world then auto-resets
// with a fresh seed drawn from its own seed stream.
class VecEnv {
public:
    // threads <= 0 -> OpenMP default thread count. Output buffers are caller-owned.
    VecEnv(int numWorlds, int agentsPerWorld, uint64_t seed, int res, int maxSteps, int ticks, int threads,
           int mode = MODE_V7);

    int size() const { return numAgents_; }              // N = W * K agent rows
    int numWorlds() const { return numWorlds_; }
    int agentsPerWorld() const { return K_; }
    int res() const { return res_; }
    int mode() const { return mode_; }
    int vecDim() const { return mode_ == MODE_V6 ? VEC_DIM_V6 : VEC7_DIM; }
    int actionCols() const { return mode_ == MODE_V6 ? 1 : NUM_ACTION_COLS; }

    void reset(uint8_t* img, float* vec);
    // acts: N * actionCols() values. msgs may be null (v6 only: all silent; v7 uses column 3 if null).
    void step(const int32_t* actions, const int32_t* msgs, uint8_t* img, float* vec, float* rew,
              bool* done, uint32_t* ach);
    // Fresh world from seed (world w); writes the K obs rows of world w (rows w*K .. w*K+K-1).
    void resetWorld(int w, uint64_t seed, uint8_t* img, float* vec);

    // World snapshots (Go-Explore). saveState(w): all agents, blocks, chests, RNG streams.
    // loadState(w, ...): restores world w (goal < 0 keeps the saved goal), sets t = 0 and
    // restart = true, writes the K obs rows of world w.
    std::vector<uint8_t> saveState(int w) const;
    void loadState(int w, const uint8_t* data, size_t len, int goal, uint8_t* img, float* vec);

    void render(int i, int width, int height, uint8_t* rgb) const;
    void inventory(int i, int32_t* out) const;
    void setInventory(int i, int item, int count);
    Snapshot state(int i) const;
    void setPose(int i, float yaw, float pitch);
    uint8_t block(int i, int x, int y, int z) const;
    Region region(int i, int radius) const;
    std::vector<ChestView> chests(int w) const;
    void lastMsgs(int32_t* out) const;

    // Goals (v6 mechanism; per world). Weights apply to future world generations.
    void setGoalWeights(const float* w);   // NUM_ITEMS non-negative weights, sum > 0
    int goal(int i) const;                 // goal of agent i's world
    void setGoal(int i, int item);         // sets the goal of agent i's world

    // Start kits: table[goal * NUM_ITEMS + item] counts (>= 0, rounded; the goal's own entry is
    // ignored), prob[goal] in [0, 1]. Applies to future world generations (every agent gets it).
    void setStartKits(const float* table, const float* prob);

    // Novelty scale (>= 0; 0 = off). Applies to every agent from the next step.
    void setNovelty(float scale);
    float novelty() const { return novScale_; }
    // Novelty part of each agent's last step (already multiplied by the scale). Zeros after reset.
    void lastNovelty(float* out) const;

    // Team share in [0, 1] (default 0.5): fraction of each achievement/goal reward paid to all agents.
    void setTeamShare(float share);
    float teamShare() const { return teamShare_; }

    // v7 tasks. Weights per task for future episodes of world w (non-negative, sum > 0).
    void setTaskWeights(int w, const float* weights);
    // Sets the task of world w for its current episode (resets its agents' task statistics). p0/p1 override
    // the success predicate's a/b (-1 keeps the data value).
    void setTask(int w, int task, int p0, int p1);
    int task(int i) const;                 // task of agent i's world
    // Knowledge shaping: while task `task` is active, the first time an agent obtains one of `items`
    // it gets 0.05 * value (ITEMS[item].value). Replaces the table of that task.
    void setShapingTable(int task, const int* items, int n);

    // World-level time and entities (phase-2 hooks; neutral until filled).
    float timeOfDay(int w) const;
    // Live mobs of world w: {type (reg::MOBS index), x, y, z (feet), health}.
    struct EntityView { int type; double x, y, z; float hp; };
    std::vector<EntityView> entities(int w) const;
    // Build target of world w's episode (empty until placed on the first obs of a build episode):
    // blueprint index, rotation, and cells {x, y, z, block, item} (block 0 = required air).
    struct BuildView { int blueprint = -1, rot = 0; std::vector<std::array<int, 5>> cells; };
    BuildView buildTarget(int w) const;
    // Seed of the world's current generation (overworld; nether derives from it).
    uint64_t worldSeed(int w) const;

private:
    void initWorldAt(int w, uint64_t seed);
    uint64_t nextSeed(int w);
    void writeAgentObs(int i, uint8_t* img, float* vec, uint64_t* seenOut) const;
    void stepWorld(int w, const int32_t* actions, const int32_t* msgs, uint8_t* img, float* vec,
                   float* rew, bool* done, uint32_t* ach);
    int nthreads() const;

    int numWorlds_, K_, numAgents_, res_, maxSteps_, ticks_, threads_, mode_;
    float goalW_[NUM_ITEMS];
    int kitCount_[NUM_ITEMS][NUM_ITEMS] = {};
    float kitProb_[NUM_ITEMS] = {};
    uint8_t shaping_[NUM_TASKS][NUM_ITEMS] = {};
    float novScale_ = 0.f;
    float teamShare_ = 0.5f;
    std::vector<float> lastNov_;
    std::vector<World> worlds_;
    std::vector<Agent> agents_;
};

} // namespace craftsim
