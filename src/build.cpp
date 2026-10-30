// Builds: lazily placed blueprint targets (build_complete, obs slots) and worldgen variety (SIM-4, PROTOCOL 12.4/12.6).
// Only this file and build.h belong to SIM-4; the core calls build::worldgen in initWorld and build::obs/progress
// from writeObs and the build_complete predicate.
#include "build.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <mutex>
#include <random>

#include "gen/blueprints.h"
#include "gen/registry.h"
#include "sim.h"

namespace craftsim::build {
namespace {

constexpr double PI = 3.14159265358979323846;
constexpr int SPAWN_BOX = 26;   // half-width of the no-build box around the world centre (spawn is within +-20)
constexpr int CROWN = 2;        // extra cells counted as obstacles above a target

enum Biome { PLAINS, FOREST, DESERT };

inline int cellIdx(int x, int y, int z) { return (y * WZ + z) * WX + x; }
inline bool inWorld(int x, int y, int z) { return x >= 0 && x < WX && y >= 0 && y < WY && z >= 0 && z < WZ; }

inline uint64_t mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

inline bool inSpawnBox(int x, int z) {
    return std::abs(x - WX / 2) <= SPAWN_BOX && std::abs(z - WZ / 2) <= SPAWN_BOX;
}

// ---------------------------------------------------------------- blueprint targets

// Key of the episode's target: changes with the world seed (new episode) and with set_task overrides.
uint64_t keyOf(const World& W) {
    uint64_t k = mix64(W.seed ^ mix64((uint64_t)W.task + 1) ^ mix64((uint64_t)(W.taskP0 + 2) * 31 + 7) ^
                       mix64((uint64_t)(W.taskP1 + 2) * 131 + 11));
    return k ? k : 1;
}

// Blueprint of the task: the success predicate if it is build_complete (taskP0 overrides its blueprint), else the
// first build_complete milestone (e.g. nether_light_portal targets the nether_portal frame). -1 if none.
int targetBlueprint(const World& W) {
    const reg::TaskDef& t = reg::TASKS[W.task];
    if (t.success.type == reg::P_BUILD_COMPLETE) {
        const int bp = W.taskP0 >= 0 ? W.taskP0 : t.success.a;
        return bp >= 0 && bp < bp::NUM_BLUEPRINTS ? bp : -1;
    }
    for (int j = 0; j < t.nMilestones; j++) {
        const reg::Pred& p = t.ms[j].pred;
        if (p.type == reg::P_BUILD_COMPLETE && p.a >= 0 && p.a < bp::NUM_BLUEPRINTS) return p.a;
    }
    return -1;
}

inline void rotDims(const bp::Blueprint& b, int rot, int& w, int& d) {
    if (rot & 1) { w = b.sz; d = b.sx; }
    else { w = b.sx; d = b.sz; }
}

// Blueprint-local (x, z) -> footprint offset (u, v) for a quarter turn.
inline void rotXZ(const bp::Blueprint& b, int rot, int x, int z, int& u, int& v) {
    switch (rot & 3) {
    case 0: u = x; v = z; break;
    case 1: u = b.sz - 1 - z; v = x; break;
    case 2: u = b.sx - 1 - x; v = b.sz - 1 - z; break;
    default: u = z; v = b.sx - 1 - x; break;
    }
}

// Highest non-leaf, non-log, non-air block of a column (ground for placement); -1 if none. blk = its block.
int groundY(const World& W, int x, int z, uint8_t& blk) {
    for (int y = WY - 1; y >= 0; y--) {
        const uint8_t b = W.w[cellIdx(x, y, z)];
        if (b == AIR || b == LOG || b == LEAVES || b == CHEST) continue;
        blk = b;
        return y;
    }
    return -1;
}

void fillTarget(const State& S, const World& W, int bp, int rot, int ox, int oy, int oz) {
    (void)W;
    const bp::Blueprint& b = bp::BLUEPRINTS[bp];
    S.cells.clear();
    S.nSolid = 0;
    S.cells.reserve(b.count);
    for (int pass = 0; pass < 2; pass++)
        for (int i = b.first; i < b.first + b.count; i++) {
            const bp::Cell& c = bp::CELLS[i];
            const bool solid = c.block != 0;
            if ((pass == 0) != solid) continue;
            int u, v;
            rotXZ(b, rot, c.x, c.z, u, v);
            S.cells.push_back(TCell{(int16_t)(ox + u), (int16_t)(oy + c.y), (int16_t)(oz + v), c.block, c.item});
            if (solid) S.nSolid++;
        }
}

// Target site: a footprint 6..14 blocks from the agent (the team spawn at reset) on ground that is dry and as level
// as possible (range 0 preferred; range > 2 rejected so no cell floats), with few obstacles in the volume above.
// Deterministic (no RNG; ties go to the first candidate).
bool chooseSpot(const World& W, const Agent& a, int bp, int& rot, int& ox, int& oy, int& oz) {
    const bp::Blueprint& b = bp::BLUEPRINTS[bp];
    const int cx0 = (int)std::floor(a.px), cz0 = (int)std::floor(a.pz);
    // Ground cache for a window around the agent (-1: outside the world or no ground).
    constexpr int R = 22, S = 2 * R + 1;
    std::vector<int> gy(S * S, -1);
    std::vector<uint8_t> gb(S * S, AIR);
    for (int dz = -R; dz <= R; dz++)
        for (int dx = -R; dx <= R; dx++) {
            const int x = cx0 + dx, z = cz0 + dz;
            if (x < 0 || x >= WX || z < 0 || z >= WZ) continue;
            uint8_t blk = AIR;
            gy[(dz + R) * S + dx + R] = groundY(W, x, z, blk);
            gb[(dz + R) * S + dx + R] = blk;
        }
    int bestScore = INT_MAX;
    bool found = false;
    for (int r : {6, 8, 10, 12, 14})
        for (int ang = 0; ang < 360; ang += 10)
            for (int q = 0; q < 4; q++) {
                const double t = ang * PI / 180.0;
                const int cx = cx0 + (int)std::lround(r * std::cos(t));
                const int cz = cz0 + (int)std::lround(r * std::sin(t));
                int w, d;
                rotDims(b, q, w, d);
                const int x0 = cx - w / 2, z0 = cz - d / 2;
                if (x0 < 1 || z0 < 1 || x0 + w > WX - 1 || z0 + d > WZ - 1) continue;
                int lo = WY, hi = -1;
                bool bad = false;
                for (int z = z0; z < z0 + d && !bad; z++)
                    for (int x = x0; x < x0 + w && !bad; x++) {
                        const int dx = x - cx0, dz = z - cz0;
                        if (std::abs(dx) > R || std::abs(dz) > R) { bad = true; break; }
                        const int i = (dz + R) * S + dx + R;
                        if (gy[i] < 0 || gb[i] == WATER) bad = true;
                        else { lo = std::min(lo, gy[i]); hi = std::max(hi, gy[i]); }
                    }
                if (bad || hi - lo > 2) continue;
                const int y0 = hi + 1;
                if (y0 + b.sy + CROWN >= WY) continue;
                int obstacles = 0;
                for (int z = z0; z < z0 + d; z++)
                    for (int x = x0; x < x0 + w; x++)
                        for (int y = y0; y < y0 + b.sy; y++)
                            if (W.w[cellIdx(x, y, z)] != AIR) obstacles++;
                const int score = (hi - lo) * 20 + obstacles;
                if (score < bestScore) {
                    bestScore = score;
                    found = true;
                    rot = q; ox = x0; oy = y0; oz = z0;
                }
            }
    return found;
}

// Placements remembered per (seed, task, overrides): a world restored by load_state (build::State is not part of the
// snapshot) gets the same target as before instead of a search on its changed blocks. Ring buffer, process-wide,
// guarded by a mutex (worlds step in parallel). Entries are a pure function of the key, so a miss only costs a search.
struct Memo { uint64_t key; int bp, rot, ox, oy, oz; };
constexpr int MEMO_SIZE = 4096;
std::mutex memoMu;
Memo memo[MEMO_SIZE];
int memoNext = 0;

bool memoFind(uint64_t key, Memo& out) {
    std::lock_guard<std::mutex> lock(memoMu);
    for (const Memo& m : memo)
        if (m.key == key) { out = m; return true; }
    return false;
}

void memoPut(const Memo& m) {
    std::lock_guard<std::mutex> lock(memoMu);
    memo[memoNext] = m;
    memoNext = (memoNext + 1) % MEMO_SIZE;
}

// Places the episode's target if not yet done for this key. Lazy: called from obs (has the agent) only.
bool ensurePlaced(const World& W, const Agent& a) {
    const uint64_t k = keyOf(W);
    if (W.build.placed && W.build.key == k) return W.build.blueprint >= 0;
    W.build.key = k;
    W.build.placed = true;
    W.build.blueprint = -1;
    W.build.cells.clear();
    W.build.nSolid = 0;
    const int bp = targetBlueprint(W);
    if (bp < 0) return false;
    Memo m{k, bp, 0, 0, 0, 0};
    if (!memoFind(k, m)) {
        if (!chooseSpot(W, a, bp, m.rot, m.ox, m.oy, m.oz)) return false;
        m.key = k;
        m.bp = bp;
        memoPut(m);
    }
    fillTarget(W.build, W, bp, m.rot, m.ox, m.oy, m.oz);
    W.build.blueprint = bp;
    W.build.rot = m.rot;
    W.build.ox = m.ox;
    W.build.oy = m.oy;
    W.build.oz = m.oz;
    return true;
}

// ---------------------------------------------------------------- worldgen helpers

inline uint32_t hash2(int x, int z, uint32_t s) {
    uint32_t h = s ^ (uint32_t)x * 0x85EBCA6Bu ^ (uint32_t)z * 0x27D4EB2Fu;
    h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
    return h;
}

// Value noise in [0, 1] at lattice scale 1 (x, z already scaled).
float valueNoise(float x, float z, uint32_t s) {
    const int xi = (int)std::floor(x), zi = (int)std::floor(z);
    float fx = x - xi, fz = z - zi;
    fx = fx * fx * (3 - 2 * fx);
    fz = fz * fz * (3 - 2 * fz);
    const float a = hash2(xi, zi, s) * (1.0f / 4294967296.0f), b = hash2(xi + 1, zi, s) * (1.0f / 4294967296.0f);
    const float c = hash2(xi, zi + 1, s) * (1.0f / 4294967296.0f), d = hash2(xi + 1, zi + 1, s) * (1.0f / 4294967296.0f);
    return a + (b - a) * fx + (c - a) * fz + (a - b - c + d) * fx * fz;
}

// Two octaves, in [0, 1] (about 0.5 centred), wavelength in blocks = 1 / f.
float biomeNoise(int x, int z, uint32_t s, float f) {
    const float a = valueNoise(x * f, z * f, s);
    const float b = valueNoise(x * f * 2.f, z * f * 2.f, s + 101u);
    return (a + 0.5f * b) / 1.5f;
}

// Oak with a blob crown (same shape as the core's oak). Caller checks the space.
void oakTree(World& W, std::mt19937_64& g, int x, int h, int z) {
    std::uniform_int_distribution<int> u(0, 2);
    const int H = 4 + u(g);
    for (int i = 1; i <= H; i++) W.w[cellIdx(x, h + i, z)] = LOG;
    const int top = h + H;
    for (int yo = 0; yo >= -3; yo--) {
        const int r = std::max(2 - 1 - yo / 2, 0);
        for (int dz = -r; dz <= r; dz++)
            for (int dx = -r; dx <= r; dx++) {
                const bool corner = std::abs(dx) == r && std::abs(dz) == r;
                if (corner && (u(g) % 2)) continue;
                const int xx = x + dx, yy = top + yo, zz = z + dz;
                if (inWorld(xx, yy, zz) && W.w[cellIdx(xx, yy, zz)] == AIR) W.w[cellIdx(xx, yy, zz)] = LEAVES;
            }
    }
    if (inWorld(x, top + 1, z) && W.w[cellIdx(x, top + 1, z)] == AIR) W.w[cellIdx(x, top + 1, z)] = LEAVES;
}

// Dense oak on forest columns (the core's trees stay), desert columns get sand and no trees.
void biomesAndTrees(World& W, std::mt19937_64& g, uint32_t s, const std::vector<int>& top, std::vector<uint8_t>& biome) {
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    for (int z = 0; z < WZ; z++)
        for (int x = 0; x < WX; x++) {
            Biome bi = PLAINS;
            if (!inSpawnBox(x, z)) {
                const float humid = biomeNoise(x, z, s + 1, 1.f / 40.f);
                const float temp = biomeNoise(x, z, s + 2, 1.f / 52.f);
                if (temp > 0.56f && humid < 0.44f) bi = DESERT;
                else if (humid > 0.60f) bi = FOREST;
            }
            biome[z * WX + x] = (uint8_t)bi;
            const int h = top[z * WX + x];
            if (h < 0 || bi == PLAINS) continue;
            if (bi == DESERT) {
                if (h <= SEA_LEVEL || W.w[cellIdx(x, h, z)] != GRASS) continue;
                for (int y = h; y >= std::max(1, h - 3); y--)
                    if (W.w[cellIdx(x, y, z)] == GRASS || W.w[cellIdx(x, y, z)] == DIRT) W.w[cellIdx(x, y, z)] = SAND;
                for (int y = h + 1; y <= std::min(WY - 1, h + 14); y++) {
                    uint8_t& c = W.w[cellIdx(x, y, z)];
                    if (c == LOG || c == LEAVES) c = AIR;
                }
            } else if (W.w[cellIdx(x, h, z)] == GRASS && u01(g) < 0.10f) {
                bool clear = h + 9 < WY;
                for (int y = h + 1; y <= h + 8 && clear; y++) clear = W.w[cellIdx(x, y, z)] == AIR;
                if (clear && x > 2 && z > 2 && x < WX - 3 && z < WZ - 3) oakTree(W, g, x, h, z);
            }
        }
}

// Near-surface cave pockets (carve stone only, never the 3-block crust), and coal/iron veins exposed in their walls.
void shallowCaves(World& W, std::mt19937_64& g, const std::vector<int>& top) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::uniform_int_distribution<int> pockets(4, 7);
    std::vector<int> carved;
    const int n = pockets(g);
    for (int p = 0; p < n; p++) {
        double x = std::uniform_int_distribution<int>(10, WX - 11)(g);
        double z = std::uniform_int_distribution<int>(10, WZ - 11)(g);
        const int t0 = top[(int)z * WX + (int)x];
        if (t0 < SEA_LEVEL + 4) continue;   // not under water or the beach
        double y = t0 - 6.0;
        double ang = u(g) * 2 * PI, pitch = (u(g) - 0.5) * 0.3;
        const int len = std::uniform_int_distribution<int>(18, 34)(g);
        const double rad = 1.4 + u(g) * 0.7;
        for (int s = 0; s < len; s++) {
            x += std::cos(ang) * 0.8;
            z += std::sin(ang) * 0.8;
            y += std::sin(pitch) * 0.8;
            ang += (u(g) - 0.5) * 0.8;
            const int tc = top[std::clamp((int)z, 0, WZ - 1) * WX + std::clamp((int)x, 0, WX - 1)];
            const double lo = tc - 9.0, hi = tc - 5.0;
            if (y < lo) { pitch = 0.25; y = lo; }
            if (y > hi) { pitch = -0.25; y = hi; }
            pitch = std::clamp(pitch + (u(g) - 0.5) * 0.2, -0.4, 0.4);
            const int r = (int)std::ceil(rad);
            for (int by = (int)y - r; by <= (int)y + r; by++)
                for (int bz = (int)z - r; bz <= (int)z + r; bz++)
                    for (int bx = (int)x - r; bx <= (int)x + r; bx++) {
                        if (!inWorld(bx, by, bz) || by < 2 || by > top[bz * WX + bx] - 4) continue;
                        const double dx = bx + 0.5 - x, dy = by + 0.5 - y, dz = bz + 0.5 - z;
                        if (dx * dx + dy * dy + dz * dz > rad * rad) continue;
                        uint8_t& c = W.w[cellIdx(bx, by, bz)];
                        if (c == STONE || c == DEEPSLATE) {
                            c = AIR;
                            carved.push_back(cellIdx(bx, by, bz));
                        }
                    }
        }
    }
    // Walls of the carved cells, 6..(-) blocks below the surface: a few coal (common) and iron (rare) veins.
    const int dirs[6] = {1, -1, WX, -WX, WX * WZ, -WX * WZ};
    for (int idx : carved) {
        const int x = idx % WX, z = (idx / WX) % WZ, y = idx / (WX * WZ);
        for (int d : dirs) {
            const int nidx = idx + d;
            if (nidx < 0 || nidx >= (int)W.w.size()) continue;
            const int nx = nidx % WX, nz = (nidx / WX) % WZ, ny = nidx / (WX * WZ);
            if (std::abs(nx - x) + std::abs(nz - z) + std::abs(ny - y) != 1) continue;   // no wrap across rows
            if (W.w[nidx] != STONE) continue;
            if (ny < 2 || ny > top[nz * WX + nx] - 3 || ny < top[nz * WX + nx] - 10) continue;
            const float r = u(g);
            if (r < 0.05) W.w[nidx] = COAL_ORE;
            else if (r < 0.065) W.w[nidx] = IRON_ORE;
        }
    }
}

// Underground lava pools (vanilla lava lakes, y 4..14): a flattened ellipsoid in stone, lower half lava, upper
// half air. The first pool (and each later one with p = 0.4) is "quenched" like where cave water met it: its
// surface layer is obsidian. That makes obsidian (nether portal frames) obtainable with a diamond pickaxe.
void lavaPools(World& W, std::mt19937_64& g, const std::vector<int>& top) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const int n = std::uniform_int_distribution<int>(3, 5)(g);
    for (int p = 0; p < n; p++) {
        const int cx = std::uniform_int_distribution<int>(8, WX - 9)(g);
        const int cz = std::uniform_int_distribution<int>(8, WZ - 9)(g);
        const int cy = std::uniform_int_distribution<int>(5, 13)(g);
        if (top[cz * WX + cx] < cy + 8) continue;   // keep at least a few blocks of rock overhead
        const double rx = 2.5 + u(g) * 1.5, rz = 2.5 + u(g) * 1.5, ry = 1.6;
        const bool quench = p == 0 || u(g) < 0.4;
        for (int by = cy - 2; by <= cy + 2; by++)
            for (int bz = cz - 4; bz <= cz + 4; bz++)
                for (int bx = cx - 4; bx <= cx + 4; bx++) {
                    if (!inWorld(bx, by, bz) || by < 1) continue;
                    const double dx = (bx + 0.5 - cx) / rx, dy = (by + 0.5 - cy) / ry, dz = (bz + 0.5 - cz) / rz;
                    if (dx * dx + dy * dy + dz * dz > 1.0) continue;
                    uint8_t& c = W.w[cellIdx(bx, by, bz)];
                    if (c == BEDROCK) continue;
                    if (by < cy) c = (uint8_t)reg::BLK_LAVA;
                    else if (by == cy) c = quench ? (uint8_t)reg::BLK_OBSIDIAN : (uint8_t)reg::BLK_LAVA;
                    else c = AIR;
                }
    }
}

// Round boulders of cobblestone on dry grass (outside the spawn box), resting on the ground.
void boulders(World& W, std::mt19937_64& g, const std::vector<int>& top, const std::vector<uint8_t>& biome) {
    std::uniform_int_distribution<int> count(3, 6);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const int n = count(g);
    for (int b = 0; b < n; b++) {
        for (int tries = 0; tries < 30; tries++) {
            const int x = std::uniform_int_distribution<int>(4, WX - 5)(g);
            const int z = std::uniform_int_distribution<int>(4, WZ - 5)(g);
            const int h = top[z * WX + x];
            if (inSpawnBox(x, z) || h <= SEA_LEVEL + 1 || biome[z * WX + x] == DESERT) continue;
            if (W.w[cellIdx(x, h, z)] != GRASS) continue;
            const double rad = 1.5 + u(g) * 0.9;
            const uint8_t stone = u(g) < 0.3 ? STONE : COBBLE;
            const int cy = h + 1, r = (int)std::ceil(rad);
            for (int by = cy - r; by <= cy + r; by++)
                for (int bz = z - r; bz <= z + r; bz++)
                    for (int bx = x - r; bx <= x + r; bx++) {
                        if (!inWorld(bx, by, bz) || by <= h - 1) continue;
                        const double dx = bx - x, dy = by - cy, dz = bz - z;
                        if (dx * dx + dy * dy + dz * dz > rad * rad) continue;
                        uint8_t& c = W.w[cellIdx(bx, by, bz)];
                        if (c == AIR) c = stone;
                    }
            break;
        }
    }
}

// ---------------------------------------------------------------- villages and ruins

// Places a blueprint with its footprint at (ox, oy, oz) rotation rot: clears the volume, fills a cobblestone
// foundation down to the ground, writes the solid cells and (ruin) knocks out part of them. A chest with a few
// items sits in the first free ground-floor cell. Returns false (nothing written) if a column has no ground.
void placeStructure(World& W, std::mt19937_64& g, int bp, int rot, int ox, int oy, int oz, bool ruin) {
    const bp::Blueprint& b = bp::BLUEPRINTS[bp];
    int w, d;
    rotDims(b, rot, w, d);
    for (int z = oz; z < oz + d; z++)
        for (int x = ox; x < ox + w; x++) {
            uint8_t blk = AIR;
            const int gy = groundY(W, x, z, blk);
            for (int y = gy + 1; y < oy; y++) W.w[cellIdx(x, y, z)] = COBBLE;
            for (int y = oy; y < oy + b.sy; y++) W.w[cellIdx(x, y, z)] = AIR;
        }
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<int> chestCells;
    for (int i = b.first; i < b.first + b.count; i++) {
        const bp::Cell& c = bp::CELLS[i];
        if (c.block == 0) continue;
        int u0, v0;
        rotXZ(b, rot, c.x, c.z, u0, v0);
        const int x = ox + u0, y = oy + c.y, z = oz + v0;
        if (ruin && c.y > 0 && u(g) < 0.45) continue;
        W.w[cellIdx(x, y, z)] = c.block;
        if (c.block == CHEST) chestCells.push_back(cellIdx(x, y, z));
    }
    static const uint8_t loot[] = {I_COAL, I_COBBLE, I_PLANKS, I_DIRT, I_IRON_ORE};
    auto fill = [&](int idx) {
        Chest& ch = W.chests[idx];
        std::uniform_int_distribution<int> pick(0, 4), amount(2, 6);
        for (int k = 0; k < 3; k++) ch.count[loot[pick(g)]] += amount(g);
    };
    for (int idx : chestCells) fill(idx);
    if (chestCells.empty()) {
        for (int z = oz; z < oz + d && chestCells.empty(); z++)
            for (int x = ox; x < ox + w && chestCells.empty(); x++) {
                const int idx = cellIdx(x, oy, z);
                if (W.w[idx] != AIR) continue;
                W.w[idx] = CHEST;
                fill(idx);
                chestCells.push_back(idx);
            }
    }
}

// Up to two houses (vanilla-converted, tier 1-2) per world, mostly whole, sometimes ruins. Sites: dry, level
// within 2 blocks, outside the spawn box and not overlapping an earlier structure.
void villages(World& W, std::mt19937_64& g) {
    std::vector<int> houses;
    for (int i = 0; i < bp::NUM_BLUEPRINTS; i++) {
        const bp::Blueprint& b = bp::BLUEPRINTS[i];
        if (b.vanilla && b.family == 1 && b.tier <= 2) houses.push_back(i);
    }
    if (houses.empty()) return;
    std::uniform_int_distribution<int> roll(0, 99);
    const int r = roll(g);
    const int n = r < 45 ? 0 : r < 85 ? 1 : 2;
    struct Rect { int x0, z0, x1, z1; };
    std::vector<Rect> taken;
    for (int s = 0; s < n; s++) {
        for (int tries = 0; tries < 40; tries++) {
            const int bp = houses[std::uniform_int_distribution<int>(0, (int)houses.size() - 1)(g)];
            const bp::Blueprint& b = bp::BLUEPRINTS[bp];
            const int rot = std::uniform_int_distribution<int>(0, 3)(g);
            int w, d;
            rotDims(b, rot, w, d);
            const int cx = std::uniform_int_distribution<int>(8, WX - 9)(g);
            const int cz = std::uniform_int_distribution<int>(8, WZ - 9)(g);
            const int x0 = cx - w / 2, z0 = cz - d / 2;
            if (x0 < 2 || z0 < 2 || x0 + w > WX - 2 || z0 + d > WZ - 2) continue;
            const Rect R{x0 - 2, z0 - 2, x0 + w + 2, z0 + d + 2};
            if (R.x0 < WX / 2 + SPAWN_BOX && R.x1 > WX / 2 - SPAWN_BOX && R.z0 < WZ / 2 + SPAWN_BOX &&
                R.z1 > WZ / 2 - SPAWN_BOX)
                continue;
            bool clash = false;
            for (const Rect& t : taken)
                if (R.x0 < t.x1 && t.x0 < R.x1 && R.z0 < t.z1 && t.z0 < R.z1) clash = true;
            if (clash) continue;
            int lo = WY, hi = -1;
            bool ok = true;
            for (int z = z0; z < z0 + d && ok; z++)
                for (int x = x0; x < x0 + w && ok; x++) {
                    uint8_t blk = AIR;
                    const int gy = groundY(W, x, z, blk);
                    if (gy < 0 || blk == WATER || gy <= SEA_LEVEL) ok = false;
                    else { lo = std::min(lo, gy); hi = std::max(hi, gy); }
                }
            if (!ok || hi - lo > 2 || hi + 1 + b.sy >= WY) continue;
            const bool ruin = std::uniform_int_distribution<int>(0, 99)(g) < 35;
            placeStructure(W, g, bp, rot, x0, hi + 1, z0, ruin);
            taken.push_back(R);
            break;
        }
    }
}

}  // namespace

// ---------------------------------------------------------------- public API

void worldgen(World& W, uint64_t seed, bool variety) {
    // Any previous target belonged to the previous episode.
    W.build.placed = false;
    W.build.key = 0;
    W.build.blueprint = -1;
    W.build.nSolid = 0;
    W.build.cells.clear();
    if (!variety) return;

    const uint32_t s = (uint32_t)mix64(seed ^ 0xB17D5EEDull);
    std::mt19937_64 g(mix64(seed ^ 0x5EEDB10Cull));

    // Ground per column (before any change below; the core terrain and trees are in place).
    std::vector<int> top(WX * WZ, -1);
    for (int z = 0; z < WZ; z++)
        for (int x = 0; x < WX; x++) {
            uint8_t blk = AIR;
            top[z * WX + x] = groundY(W, x, z, blk);
        }
    std::vector<uint8_t> biome(WX * WZ, PLAINS);
    biomesAndTrees(W, g, s, top, biome);
    shallowCaves(W, g, top);
    boulders(W, g, top, biome);
    villages(W, g);
    std::mt19937_64 gl(mix64(seed ^ 0x1A7A0B51ull));   // own stream: the layout above stays unchanged
    lavaPools(W, gl, top);
}

void placeTarget(World& W, int blueprint, int x, int y, int z) {
    if (blueprint < 0 || blueprint >= bp::NUM_BLUEPRINTS) return;
    const bp::Blueprint& b = bp::BLUEPRINTS[blueprint];
    W.build.key = keyOf(W);
    W.build.placed = true;
    W.build.blueprint = blueprint;
    W.build.rot = 0;
    W.build.ox = x - b.sx / 2;
    W.build.oy = y;
    W.build.oz = z - b.sz / 2;
    fillTarget(W.build, W, blueprint, 0, W.build.ox, W.build.oy, W.build.oz);
}

float progress(const World& W) {
    const State& S = W.build;
    if (!S.placed || S.key != keyOf(W) || S.blueprint < 0 || S.nSolid == 0) return 0.f;
    int ok = 0, bad = 0;
    for (const TCell& c : S.cells) {
        const uint8_t w = W.w[cellIdx(c.x, c.y, c.z)];
        if (c.block != 0) ok += w == c.block;
        else bad += w != AIR;
    }
    const float p = (float)(ok - bad) / (float)S.nSolid;
    return std::clamp(p, 0.f, 1.f);
}

void obs(const World& W, const Agent& a, float* out33) {
    for (int i = 0; i < 33; i++) out33[i] = 0.f;
    if (!ensurePlaced(W, a)) return;
    const State& S = W.build;
    // 8 nearest solid cells that do not yet hold their wanted block (insertion into a sorted array of 8).
    struct Near { double d2; const TCell* c; };
    Near best[8];
    int nb = 0;
    for (const TCell& c : S.cells) {
        if (c.block == 0 || W.w[cellIdx(c.x, c.y, c.z)] == c.block) continue;
        const double dx = c.x + 0.5 - a.px, dy = c.y + 0.5 - a.py, dz = c.z + 0.5 - a.pz;
        const double d2 = dx * dx + dy * dy + dz * dz;
        if (nb == 8 && d2 >= best[7].d2) continue;
        int k = nb < 8 ? nb++ : 7;
        while (k > 0 && best[k - 1].d2 > d2) { best[k] = best[k - 1]; k--; }
        best[k] = Near{d2, &c};
    }
    const double yr = a.yaw * PI / 180.0;
    const double fx = -std::sin(yr), fz = std::cos(yr), rx = -std::cos(yr), rz = -std::sin(yr);
    for (int j = 0; j < nb; j++) {
        const TCell& c = *best[j].c;
        const double dx = c.x + 0.5 - a.px, dy = c.y + 0.5 - a.py, dz = c.z + 0.5 - a.pz;
        out33[4 * j + 0] = (float)((dx * rx + dz * rz) / 8.0);
        out33[4 * j + 1] = (float)(dy / 8.0);
        out33[4 * j + 2] = (float)((dx * fx + dz * fz) / 8.0);
        out33[4 * j + 3] = (float)c.item / (float)NUM_ITEMS;
    }
    out33[32] = progress(W);
}

}  // namespace craftsim::build
