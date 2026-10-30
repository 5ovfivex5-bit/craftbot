// craftsim: mechanics. Mirrors vanilla Minecraft where it matters for learning:
//   * 20 ticks/s, player AABB 0.6 x 1.8, eye height 1.62, reach 4.5
//   * gravity 0.08 b/t^2, vertical drag 0.98, jump impulse 0.42 (apex ~1.25 blocks)
//   * ground slipperiness 0.6 * 0.91, walk accel 0.1 * 0.98  -> 4.317 m/s walking
//   * water (LivingEntity.travelInWater): swim accel 0.02, drag 0.8 on x/y/z, fluid gravity
//     0.08/16 per tick, jump in liquid +0.04/tick, jumpOutOfFluid -> vy 0.3 against a wall
//   * block breaking (BlockBehaviour.getDestroyProgress): progress/tick =
//       toolSpeed / hardness / (canHarvest ? 30 : 100), divided by 5 when airborne (Player.getDestroySpeed)
//       and by 5 when eyes are in water (SUBMERGED_MINING_SPEED 0.2)
//   * tool speeds wood/stone/iron 2/4/6 (ToolMaterial), hardness from Blocks.java
//   * vanilla recipes: log->4 planks, 2 planks->4 sticks, 4 planks->table, pickaxes = 3 head + 2 sticks
//   * fall damage = ceil(fallDistance - 3); no fall damage while in water
// Rewards: first time the team obtains each item, 0.1 * (1 + log2(MineRL value)); goal item +10.
// Team split (v6): team_share of each reward to every agent, the rest to the obtaining agent.
// Worldgen: ore height ranges from data/worldgen/placement/OrePlacements.java scaled to our world,
// vein shapes from world/level/levelgen/feature/OreFeature (doPlace), trees from TreeFeatures.
// Worlds are 128 x 64 x 128: counts of features are scaled by area (4x the v5 64 x 64 world).

#include "sim.h"

#include <omp.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace craftsim {
namespace {

constexpr double PI = 3.14159265358979323846;

constexpr double GRAVITY = 0.08;
constexpr double DRAG = 0.98;
constexpr double JUMP_VEL = 0.42;
constexpr double GROUND_FRICTION = 0.6 * 0.91;
constexpr double AIR_FRICTION = 0.91;
constexpr double GROUND_ACCEL = 0.1 * 0.98;
constexpr double AIR_ACCEL = 0.02 * 0.98;
constexpr double WATER_ACCEL = 0.02;        // LivingEntity.travelInWater: moveRelative(0.02)
constexpr double WATER_DRAG = 0.8;          // LivingEntity.WATER_DRAG / getWaterSlowDown()
constexpr double WATER_JUMP = 0.04;         // jumpInLiquid
constexpr double HALF_W = 0.3, HEIGHT = 1.8, EYE = 1.62;
constexpr double REACH = 4.5;
constexpr float TURN_DEG = 15.f;
constexpr double VIEW_DIST = 24.0;
constexpr double FOV_DEG = 70.0;
constexpr double SPAWN_RING = 2.0;          // teammates spawn within +-2 columns of the spawn column

struct BlockInfo { float hardness; int harvestTier; bool pickaxe; int drop; };

// Hardness = Blocks.java strength(hardness, resistance). Tier: 0 hand, 1 wood, 2 stone, 3 iron.
const BlockInfo BINFO[V6_NUM_BLOCKS] = {
    {0.0f, 0, false, -1},          // AIR
    {0.6f, 0, false, I_DIRT},      // GRASS      strength(0.6)
    {0.5f, 0, false, I_DIRT},      // DIRT       strength(0.5)
    {1.5f, 1, true,  I_COBBLE},    // STONE      strength(1.5), requiresCorrectToolForDrops
    {2.0f, 1, true,  I_COBBLE},    // COBBLE     strength(2.0)
    {2.0f, 0, false, I_LOG},       // LOG        logProperties strength(2.0)
    {0.2f, 0, false, -1},          // LEAVES     leavesProperties strength(0.2)
    {2.0f, 0, false, I_PLANKS},    // PLANKS     strength(2.0)
    {2.5f, 0, false, I_TABLE},     // TABLE      strength(2.5)
    {3.5f, 1, true,  I_FURNACE},   // FURNACE    strength(3.5)
    {3.0f, 1, true,  I_COAL},      // COAL_ORE   strength(3.0)
    {3.0f, 2, true,  I_IRON_ORE},  // IRON_ORE   strength(3.0)
    {3.0f, 3, true,  I_DIAMOND},   // DIAMOND_ORE strength(3.0)
    {-1.f, 99, false, -1},         // BEDROCK    strength(-1)
    {0.5f, 0, false, -1},          // SAND       strength(0.5), drops nothing in our item set
    {0.6f, 0, false, -1},          // GRAVEL     strength(0.6), drops nothing in our item set
    {-1.f, 99, false, -1},         // WATER      unbreakable
    {3.0f, 1, true,  I_COBBLE},    // DEEPSLATE  strength(3.0, 6.0), wooden pickaxe, drops cobble
    {2.5f, 0, false, I_CHEST},     // CHEST      strength(2.5) like the table, drops the chest item
    {-1.f, 99, false, -1},         // AGENT      pseudo block (never stored)
};


// MineRL ObtainDiamond item values; reward = 0.1 * (1 + log2(value)) the first time.
const int ITEM_VALUE_V6[V6_NUM_ITEMS] = {1, 2, 4, 4, 8, 16, 32, 32, 16, 64, 128, 256, 1024, 0, 8};

inline bool isTool(int item) { return reg::ITEMS[item].kind == reg::K_TOOL; }

inline int cell(int x, int y, int z) { return (y * WZ + z) * WX + x; }

inline uint8_t get(const World& W, int x, int y, int z) {
    if (y >= WY) return AIR;
    if (x < 0 || x >= WX || z < 0 || z >= WZ || y < 0) return BEDROCK;
    return W.w[cell(x, y, z)];
}

inline void put(World& W, int x, int y, int z, uint8_t b) {
    if (x < 0 || x >= WX || z < 0 || z >= WZ || y < 0 || y >= WY) return;
    W.w[cell(x, y, z)] = b;
}

// Collision: air and water are passable.
inline bool solid(uint8_t b) { return b != AIR && b != WATER && b != reg::BLK_LAVA; }   // fluids are passable

// ---------------------------------------------------------------- noise / worldgen

inline uint32_t hash3(int x, int y, int z, uint32_t s) {
    uint32_t h = s ^ (uint32_t)x * 0x85EBCA6Bu ^ (uint32_t)y * 0xC2B2AE35u ^ (uint32_t)z * 0x27D4EB2Fu;
    h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
    return h;
}

inline float h01(int x, int z, uint32_t s) { return hash3(x, 0, z, s) * (1.0f / 4294967296.0f); }

float vnoise(float x, float z, uint32_t s) {
    int xi = (int)std::floor(x), zi = (int)std::floor(z);
    float fx = x - xi, fz = z - zi;
    fx = fx * fx * (3 - 2 * fx);
    fz = fz * fz * (3 - 2 * fz);
    float a = h01(xi, zi, s), b = h01(xi + 1, zi, s), c = h01(xi, zi + 1, s), d = h01(xi + 1, zi + 1, s);
    return a + (b - a) * fx + (c - a) * fz + (a - b - c + d) * fx * fz;
}

float fbm(float x, float z, uint32_t s) {
    float sum = 0, amp = 1, norm = 0, f = 1.f / 32.f;
    for (int o = 0; o < 4; o++) {
        sum += amp * vnoise(x * f, z * f, s + o * 1013u);
        norm += amp;
        amp *= 0.5f;
        f *= 2.f;
    }
    return sum / norm;
}

using Rng = std::mt19937_64;

// Uniform integer in [lo, hi].
int uniformInt(Rng& rng, int lo, int hi) {
    std::uniform_int_distribution<int> d(lo, hi);
    return d(rng);
}

// Triangle-distributed integer in [lo, hi] with mode `mode` (inverse CDF; mode clamped).
// Used for MC HeightRangePlacement.triangle after mapping its y anchors to our world.
int triangleInt(Rng& rng, int lo, int hi, int mode) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double a = lo, b = hi + 1.0, c = std::clamp(mode, lo, hi) + 0.5;
    if (b <= a) return lo;
    double p = u(rng);
    double fc = (c - a) / (b - a);
    double x = p < fc ? a + std::sqrt(p * (b - a) * (c - a)) : b - std::sqrt((1 - p) * (b - a) * (b - c));
    return std::clamp((int)std::floor(x), lo, hi);
}

// OreFeature.doPlace: a blob of `size` blocks made of spheres along a random segment.
// Exposed ores (touching air or water) are dropped with probability `discard`.
void oreBlob(World& W, uint8_t ore, int x, int y, int z, int size, double discard) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double f = u(W.rng) * PI;
    double x0 = x + std::sin(f) * size / 8.0, x1 = x - std::sin(f) * size / 8.0;
    double z0 = z + std::cos(f) * size / 8.0, z1 = z - std::cos(f) * size / 8.0;
    double y0 = y + uniformInt(W.rng, -2, 0), y1 = y + uniformInt(W.rng, -2, 0);
    double d[32 * 4];   // size <= 32 (coal veins use 17)
    for (int i = 0; i < size; i++) {
        double step = (double)i / size;
        double ss = u(W.rng) * size / 16.0;
        d[i * 4 + 0] = x0 + (x1 - x0) * step;
        d[i * 4 + 1] = y0 + (y1 - y0) * step;
        d[i * 4 + 2] = z0 + (z1 - z0) * step;
        d[i * 4 + 3] = ((std::sin(PI * step) + 1.0) * ss + 1.0) / 2.0;
    }
    for (int i1 = 0; i1 < size - 1; i1++) {
        if (!(d[i1 * 4 + 3] <= 0.0))
            for (int i2 = i1 + 1; i2 < size; i2++) {
                if (d[i2 * 4 + 3] <= 0.0) continue;
                double dx = d[i1 * 4] - d[i2 * 4], dy = d[i1 * 4 + 1] - d[i2 * 4 + 1];
                double dz = d[i1 * 4 + 2] - d[i2 * 4 + 2], dr = d[i1 * 4 + 3] - d[i2 * 4 + 3];
                if (dr * dr > dx * dx + dy * dy + dz * dz) {
                    if (dr > 0.0) d[i2 * 4 + 3] = -1.0;
                    else d[i1 * 4 + 3] = -1.0;
                }
            }
    }
    for (int i = 0; i < size; i++) {
        double r = d[i * 4 + 3];
        if (r < 0.0) continue;
        double xx = d[i * 4], yy = d[i * 4 + 1], zz = d[i * 4 + 2];
        int x0i = (int)std::floor(xx - r), x1i = (int)std::floor(xx + r);
        int y0i = (int)std::floor(yy - r), y1i = (int)std::floor(yy + r);
        int z0i = (int)std::floor(zz - r), z1i = (int)std::floor(zz + r);
        for (int bx = x0i; bx <= x1i; bx++) {
            double xd = (bx + 0.5 - xx) / r;
            if (xd * xd >= 1.0) continue;
            for (int by = y0i; by <= y1i; by++) {
                double yd = (by + 0.5 - yy) / r;
                if (xd * xd + yd * yd >= 1.0) continue;
                for (int bz = z0i; bz <= z1i; bz++) {
                    double zd = (bz + 0.5 - zz) / r;
                    if (xd * xd + yd * yd + zd * zd >= 1.0) continue;
                    if (bx < 0 || bx >= WX || bz < 0 || bz >= WZ || by < 0 || by >= WY) continue;
                    uint8_t b = get(W, bx, by, bz);
                    if (b != STONE && b != DEEPSLATE) continue;
                    if (discard > 0) {
                        bool exposed = get(W, bx + 1, by, bz) == AIR || get(W, bx - 1, by, bz) == AIR ||
                                       get(W, bx, by + 1, bz) == AIR || get(W, bx, by - 1, bz) == AIR ||
                                       get(W, bx, by, bz + 1) == AIR || get(W, bx, by, bz - 1) == AIR;
                        if (exposed && u(W.rng) < discard) continue;
                    }
                    put(W, bx, by, bz, ore);
                }
            }
        }
    }
}

// Ore placements: MC OrePlacements.java y-ranges mapped to our 64-high world with
// y' = (y + 64) * 30 / 127 (MC bottom -64 -> 0, MC sea 63 -> our SEA_LEVEL 30).
// Counts are per world: v5 counts x4 for the 128 x 128 world (same density per column).
void placeOres(World& W) {
    auto& r = W.rng;
    // coal: COAL_UPPER count 30 uniform [136,top] -> [47,63]; COAL_LOWER count 20 triangle [0,192] -> [15,60]
    for (int i = 0; i < 100; i++) oreBlob(W, COAL_ORE, uniformInt(r, 0, WX - 1), uniformInt(r, 47, 63), uniformInt(r, 0, WZ - 1), 17, 0.0);
    for (int i = 0; i < 140; i++) oreBlob(W, COAL_ORE, uniformInt(r, 0, WX - 1), triangleInt(r, 15, 60, 38), uniformInt(r, 0, WZ - 1), 17, 0.0);
    // iron: UPPER 90 triangle [80,384] -> [34,63]; MIDDLE 10 triangle [-24,56] -> [9,28];
    // SMALL 10 uniform [bottom,72] -> [0,32]. Veins of 9 (small: 4).
    for (int i = 0; i < 60; i++) oreBlob(W, IRON_ORE, uniformInt(r, 0, WX - 1), triangleInt(r, 34, 63, 63), uniformInt(r, 0, WZ - 1), 9, 0.0);
    for (int i = 0; i < 60; i++) oreBlob(W, IRON_ORE, uniformInt(r, 0, WX - 1), triangleInt(r, 9, 28, 19), uniformInt(r, 0, WZ - 1), 9, 0.0);
    for (int i = 0; i < 40; i++) oreBlob(W, IRON_ORE, uniformInt(r, 0, WX - 1), uniformInt(r, 0, 32), uniformInt(r, 0, WZ - 1), 4, 0.0);
    // diamond: MC triangle(aboveBottom(-80), aboveBottom(80)) mode bottom -> [0,19] mode 0.
    // small 28 (size 4, discard .5), large 4 (size 12, discard .7), buried 16 (size 8, discard 1),
    // medium 8 uniform [-64,-4] -> [0,14] (size 8, discard .5).
    for (int i = 0; i < 28; i++) oreBlob(W, DIAMOND_ORE, uniformInt(r, 0, WX - 1), triangleInt(r, 0, 19, 0), uniformInt(r, 0, WZ - 1), 4, 0.5);
    for (int i = 0; i < 4; i++) oreBlob(W, DIAMOND_ORE, uniformInt(r, 0, WX - 1), triangleInt(r, 0, 19, 0), uniformInt(r, 0, WZ - 1), 12, 0.7);
    for (int i = 0; i < 16; i++) oreBlob(W, DIAMOND_ORE, uniformInt(r, 0, WX - 1), triangleInt(r, 0, 19, 0), uniformInt(r, 0, WZ - 1), 8, 1.0);
    for (int i = 0; i < 8; i++) oreBlob(W, DIAMOND_ORE, uniformInt(r, 0, WX - 1), uniformInt(r, 0, 14), uniformInt(r, 0, WZ - 1), 8, 0.5);
}

// Random-walk caves carved through stone/deepslate (never within 3 blocks of the surface).
// 16..28 worms: the v5 count (4..7) scaled to the larger area.
void carveCaves(World& W, const std::vector<int>& hmap) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    int worms = 16 + uniformInt(W.rng, 0, 12);
    for (int wm = 0; wm < worms; wm++) {
        double x = uniformInt(W.rng, 4, WX - 5), z = uniformInt(W.rng, 4, WZ - 5);
        double y = uniformInt(W.rng, 4, SEA_LEVEL - 4);
        double ang = u(W.rng) * 2 * PI, pitch = (u(W.rng) - 0.5) * 0.4;
        int len = uniformInt(W.rng, 30, 70);
        double rad = 1.2 + u(W.rng) * 1.2;
        for (int s = 0; s < len; s++) {
            double stepLen = 0.8;
            x += std::cos(ang) * std::cos(pitch) * stepLen;
            z += std::sin(ang) * std::cos(pitch) * stepLen;
            y += std::sin(pitch) * stepLen;
            ang += (u(W.rng) - 0.5) * 0.7;
            pitch = std::clamp(pitch + (u(W.rng) - 0.5) * 0.2, -0.6, 0.6);
            if (y < 2 || y > SEA_LEVEL - 4) pitch = -pitch;
            int r = (int)std::ceil(rad);
            for (int by = (int)y - r; by <= (int)y + r; by++)
                for (int bz = (int)z - r; bz <= (int)z + r; bz++)
                    for (int bx = (int)x - r; bx <= (int)x + r; bx++) {
                        if (bx < 0 || bx >= WX || bz < 0 || bz >= WZ || by < 1 || by >= WY) continue;
                        double dx = bx + 0.5 - x, dy = by + 0.5 - y, dz = bz + 0.5 - z;
                        if (dx * dx + dy * dy + dz * dz > rad * rad) continue;
                        if (by > hmap[bz * WX + bx] - 3) continue;
                        uint8_t b = get(W, bx, by, bz);
                        if (b == STONE || b == DEEPSLATE) put(W, bx, by, bz, AIR);
                    }
        }
    }
}

// Tree shapes (TreeFeatures): trunk = StraightTrunkPlacer, leaves = BlobFoliagePlacer (oak/birch,
// radius 2, height 3) or SpruceFoliagePlacer (conical). Corners of a blob row are skipped half the time.
void leafCell(World& W, int x, int y, int z) {
    if (get(W, x, y, z) == AIR) put(W, x, y, z, LEAVES);
}

// spruce: corners of a row with radius > 0 are always skipped; blob: skipped half the time.
void leafRow(World& W, int cx, int y, int cz, int r, bool spruce) {
    for (int dz = -r; dz <= r; dz++)
        for (int dx = -r; dx <= r; dx++) {
            bool corner = std::abs(dx) == r && std::abs(dz) == r;
            if (corner && (spruce ? r > 0 : uniformInt(W.rng, 0, 1) == 0)) continue;
            leafCell(W, cx + dx, y, cz + dz);
        }
}

void blobTree(World& W, int x, int h, int z, int trunkBase, int trunkRand, uint8_t log) {
    int H = trunkBase + uniformInt(W.rng, 0, trunkRand);
    for (int i = 1; i <= H; i++) put(W, x, h + i, z, log);
    int top = h + H;
    // BlobFoliagePlacer(radius 2, offset 0, height 3): rows yo = 0..-3, radius = max(2 - 1 - yo/2, 0)
    for (int yo = 0; yo >= -3; yo--) {
        int r = std::max(2 - 1 - yo / 2, 0);
        leafRow(W, x, top + yo, z, r, false);
    }
    leafCell(W, x, top + 1, z);   // tip
}

// Spruce: StraightTrunkPlacer(5, 2, 1) and SpruceFoliagePlacer(radius 2..3, offset 0..2,
// trunkHeight 1..2): foliageHeight = max(4, H - trunkHeight), rows yo = offset..-foliageHeight.
void spruceTree(World& W, int x, int h, int z, uint8_t log) {
    int H = 5 + uniformInt(W.rng, 0, 2) + uniformInt(W.rng, 0, 1);
    for (int i = 1; i <= H; i++) put(W, x, h + i, z, log);
    int top = h + H;
    int offset = uniformInt(W.rng, 0, 2);
    int foliageHeight = std::max(4, H - uniformInt(W.rng, 1, 2));
    int leafRadius = uniformInt(W.rng, 2, 3);
    // Radius cycles 0/1 up to leafRadius, from the top layer down.
    int cur = uniformInt(W.rng, 0, 1), maxR = 1, minR = 0;
    for (int yo = offset; yo >= -foliageHeight; yo--) {
        leafRow(W, x, top + yo, z, cur, true);
        if (cur >= maxR) {
            cur = minR;
            minR = 1;
            maxR = std::min(maxR + 1, leafRadius);
        } else {
            cur++;
        }
    }
}

void generate(World& W, std::vector<int>& hmap) {
    uint32_t s = (uint32_t)W.rng();
    W.w.assign((size_t)WX * WY * WZ, AIR);
    hmap.assign(WX * WZ, 0);
    std::uniform_real_distribution<float> u01(0.f, 1.f);

    // Heights: fbm centred a little above sea level, so roughly a third of columns end up as
    // lakes/ocean floor. Range about 16..50.
    for (int z = 0; z < WZ; z++)
        for (int x = 0; x < WX; x++) {
            float n = fbm((float)x, (float)z, s);
            int h = SEA_LEVEL + 5 + (int)std::lround((n - 0.5f) * 44.f);
            hmap[z * WX + x] = std::clamp(h, 6, WY - 16);
        }

    // Two to four meandering rivers (v5: one or two per 64-wide world): 5-wide channels 2 blocks
    // below sea level, running along x across the whole map.
    int rivers = 2 + (int)(W.rng() % 3);
    for (int k = 0; k < rivers; k++) {
        float z0 = 8 + u01(W.rng) * (WZ - 16), ph = u01(W.rng) * 2 * (float)PI;
        float amp = 3 + u01(W.rng) * 4, freq = 0.08f + u01(W.rng) * 0.06f;
        for (int x = 0; x < WX; x++) {
            int zc = (int)std::lround(z0 + amp * std::sin(x * freq + ph));
            for (int z = zc - 2; z <= zc + 2; z++)
                if (z >= 0 && z < WZ) hmap[z * WX + x] = std::min(hmap[z * WX + x], SEA_LEVEL - 2);
        }
    }

    // Columns: surface block, filler, water above submerged floors, deepslate below DEEPSLATE_Y.
    for (int z = 0; z < WZ; z++)
        for (int x = 0; x < WX; x++) {
            int h = hmap[z * WX + x];
            bool wet = h < SEA_LEVEL;
            bool nearWater = false;
            for (int dz = -3; dz <= 3 && !nearWater; dz++)
                for (int dx = -3; dx <= 3 && !nearWater; dx++) {
                    int nx = x + dx, nz = z + dz;
                    if (nx >= 0 && nx < WX && nz >= 0 && nz < WZ && hmap[nz * WX + nx] < SEA_LEVEL) nearWater = true;
                }
            uint8_t top, sub;
            if (wet) {
                // Submerged floor: gravel patches in sand.
                top = sub = vnoise(x * 0.17f, z * 0.17f, s + 77) > 0.62f ? GRAVEL : SAND;
            } else if (h <= SEA_LEVEL + 2 && nearWater) {
                top = sub = SAND;   // beach
            } else {
                top = GRASS;
                sub = DIRT;
            }
            for (int y = 0; y <= h; y++) {
                uint8_t b;
                if (y == 0) b = BEDROCK;
                else if (y == h) b = top;
                else if (y >= h - 3) b = sub;
                else b = y < DEEPSLATE_Y ? DEEPSLATE : STONE;
                put(W, x, y, z, b);
            }
            if (wet)
                for (int y = h + 1; y <= SEA_LEVEL; y++) put(W, x, y, z, WATER);
        }

    carveCaves(W, hmap);
    placeOres(W);

    // Trees on grass above sea level. Forest patches are dense, the rest sparse.
    for (int z = 3; z < WZ - 3; z++)
        for (int x = 3; x < WX - 3; x++) {
            int h = hmap[z * WX + x];
            if (get(W, x, h, z) != GRASS) continue;
            float forest = vnoise(x * 0.07f, z * 0.07f, s + 11);
            float chance = forest > 0.55f ? 0.12f : 0.012f;
            if (u01(W.rng) > chance) continue;
            if (h + 12 >= WY) continue;
            bool clear = true;
            for (int y = h + 1; y <= h + 11 && clear; y++) clear = get(W, x, y, z) == AIR;
            if (!clear) continue;
            float grove = vnoise(x * 0.1f, z * 0.1f, s + 23);
            if (h >= SEA_LEVEL + 12) spruceTree(W, x, h, z, LOG);
            else if (grove > 0.62f) blobTree(W, x, h, z, 5, 2, LOG);   // birch: same blob shape
            else blobTree(W, x, h, z, 4, 2, LOG);                       // oak
        }
}

// Sample a goal from the weights (uniform over the cumulative sum).
int sampleGoal(Rng& rng, const float* w) {
    double total = 0;
    for (int i = 0; i < NUM_ITEMS; i++) total += w[i];
    std::uniform_real_distribution<double> u(0.0, total);
    double p = u(rng), acc = 0;
    for (int i = 0; i < NUM_ITEMS; i++) {
        acc += w[i];
        if (p < acc && w[i] > 0) return i;
    }
    for (int i = NUM_ITEMS - 1; i >= 0; i--)
        if (w[i] > 0) return i;
    return 0;
}

// Depth step below sea level: 0 at or above SEA_LEVEL, then one per 2 blocks.
inline int depthStep(int fy) { return fy < SEA_LEVEL ? (SEA_LEVEL - fy) / 2 : 0; }

// 4x4x4 cell index of integer block coordinates (32 x 16 x 32 cells).
inline int cellOf(int fx, int fy, int fz) {
    fx = std::clamp(fx, 0, WX - 1); fy = std::clamp(fy, 0, WY - 1); fz = std::clamp(fz, 0, WZ - 1);
    return ((fy >> 2) * (WZ / 4) + (fz >> 2)) * (WX / 4) + (fx >> 2);
}
static_assert(NUM_CELLS == (WX / 4) * (WY / 4) * (WZ / 4), "cell count");

inline void feetBlock(const Agent& a, int& fx, int& fy, int& fz) {
    fx = (int)std::floor(a.px); fy = (int)std::floor(a.py); fz = (int)std::floor(a.pz);
}

void markVisited(Agent& a) {
    int fx, fy, fz;
    feetBlock(a, fx, fy, fz);
    int c = cellOf(fx, fy, fz);
    a.visited[c >> 6] |= 1ull << (c & 63);
    a.bestDepthStep = depthStep(fy);
}

// Movement novelty (raw, before the scale): +1 for a first-time 4x4x4 cell, +3 per new depth step.
float noveltyMove(Agent& a) {
    int fx, fy, fz;
    feetBlock(a, fx, fy, fz);
    int c = cellOf(fx, fy, fz);
    float n = 0;
    uint64_t bit = 1ull << (c & 63);
    if (!(a.visited[c >> 6] & bit)) { a.visited[c >> 6] |= bit; n += 1.f; }
    int ds = depthStep(fy);
    if (ds > a.bestDepthStep) { n += 3.f * (ds - a.bestDepthStep); a.bestDepthStep = ds; }
    return n;
}

// View novelty (raw): +2 per block type not seen yet this episode. Updates the seen mask.
float noveltyBlocks(Agent& a, uint64_t seen) {
    uint64_t fresh = seen & ~a.seenBlocks;
    a.seenBlocks |= seen;
    return 2.f * (float)std::popcount(fresh);
}

// Teammate box (feet at position, 0.6 x 1.8 x 0.6) hit by the ray? Returns the entry time.
// Only used for the <= 4 teammates of a world, so it stays a plain slab test.
inline bool boxHit(const Agent& o, double ox, double oy, double oz, double dx, double dy, double dz,
                   double maxd, double& tEnter, int& face) {
    const double mn[3] = {o.px - HALF_W, o.py, o.pz - HALF_W};
    const double mx[3] = {o.px + HALF_W, o.py + HEIGHT, o.pz + HALF_W};
    const double org[3] = {ox, oy, oz}, dir[3] = {dx, dy, dz};
    double t0 = -1e30, t1 = 1e30;
    int f = -1;
    for (int a = 0; a < 3; a++) {
        if (std::abs(dir[a]) < 1e-12) {
            if (org[a] < mn[a] || org[a] > mx[a]) return false;
            continue;
        }
        double inv = 1.0 / dir[a];
        double ta = (mn[a] - org[a]) * inv, tb = (mx[a] - org[a]) * inv;
        if (ta > tb) std::swap(ta, tb);
        if (ta > t0) { t0 = ta; f = a; }
        if (tb < t1) t1 = tb;
        if (t0 > t1) return false;
    }
    if (t1 < 0 || t0 > maxd) return false;
    tEnter = std::max(t0, 0.0);
    face = f < 0 ? 1 : f;
    return true;
}

// ---------------------------------------------------------------- raycast

struct Hit {
    bool hit = false;
    int x = 0, y = 0, z = 0;    // block hit (for AGENT: the teammate's feet block)
    int px = 0, py = 0, pz = 0; // empty cell before it (placement target)
    int face = -1;              // axis crossed to enter the hit cell (0=x,1=y,2=z)
    double dist = 0;
    uint8_t b = AIR;
    int agent = -1;             // teammate index k when b == AGENT
};

// Water stops rays unless `passWater` (a viewer submerged in water sees through it).
// `ag` (K agents, `self` excluded) can be hit before any block; ag == nullptr ignores agents.
Hit raycast(const World& W, const Agent* ag, int K, int self, double ox, double oy, double oz,
            double dx, double dy, double dz, double maxd, bool passWater) {
    Hit h;
    double aT = 1e30;
    int aK = -1, aFace = -1;
    uint8_t mobId = AIR;
    if (ag) {
        for (int k = 0; k < K; k++) {
            if (k == self) continue;
            double te;
            int f;
            if (boxHit(ag[k], ox, oy, oz, dx, dy, dz, maxd, te, f) && te < aT) { aT = te; aK = k; aFace = f; }
        }
        // Mobs are seen by the same (observation) rays as teammates: aK == -2 marks a mob hit.
        const double o[3] = {ox, oy, oz}, d[3] = {dx, dy, dz};
        float mt;
        uint8_t mid;
        if (mobs::rayHit(W, o, d, maxd, mt, mid) && mt < aT) { aT = mt; aK = -2; aFace = -1; mobId = mid; }
    }
    auto agentHit = [&]() {
        h.hit = true;
        h.dist = aT;
        h.face = aFace;
        if (aK == -2) {
            h.b = mobId;
            h.agent = -1;
            h.x = (int)std::floor(ox + dx * aT); h.y = (int)std::floor(oy + dy * aT); h.z = (int)std::floor(oz + dz * aT);
        } else {
            h.b = AGENT;
            h.agent = aK;
            h.x = (int)std::floor(ag[aK].px); h.y = (int)std::floor(ag[aK].py); h.z = (int)std::floor(ag[aK].pz);
        }
        return h;
    };
    int x = (int)std::floor(ox), y = (int)std::floor(oy), z = (int)std::floor(oz);
    int sx = dx > 0 ? 1 : -1, sy = dy > 0 ? 1 : -1, sz = dz > 0 ? 1 : -1;
    double tdx = dx != 0 ? std::abs(1.0 / dx) : 1e30;
    double tdy = dy != 0 ? std::abs(1.0 / dy) : 1e30;
    double tdz = dz != 0 ? std::abs(1.0 / dz) : 1e30;
    double tmx = dx != 0 ? (dx > 0 ? x + 1 - ox : ox - x) * tdx : 1e30;
    double tmy = dy != 0 ? (dy > 0 ? y + 1 - oy : oy - y) * tdy : 1e30;
    double tmz = dz != 0 ? (dz > 0 ? z + 1 - oz : oz - z) * tdz : 1e30;
    int lx = x, ly = y, lz = z, face = -1;
    double t = 0;
    for (;;) {
        if (aK != -1 && aT <= t) return agentHit();
        uint8_t b = get(W, x, y, z);
        if (b != AIR && !(passWater && b == WATER)) {
            h.hit = true;
            h.x = x; h.y = y; h.z = z;
            h.px = lx; h.py = ly; h.pz = lz;
            h.face = face; h.dist = t; h.b = b;
            return h;
        }
        if (y >= WY && dy >= 0) {
            if (aK != -1 && aT <= maxd) return agentHit();
            return h;
        }
        lx = x; ly = y; lz = z;
        if (tmx < tmy && tmx < tmz) { t = tmx; x += sx; tmx += tdx; face = 0; }
        else if (tmy < tmz)         { t = tmy; y += sy; tmy += tdy; face = 1; }
        else                        { t = tmz; z += sz; tmz += tdz; face = 2; }
        if (t > maxd) {
            if (aK != -1 && aT <= maxd) return agentHit();
            return h;
        }
    }
}

inline void lookDir(float yaw, float pitch, double& dx, double& dy, double& dz) {
    double y = yaw * PI / 180.0, p = pitch * PI / 180.0;
    dx = -std::sin(y) * std::cos(p);
    dy = -std::sin(p);
    dz = std::cos(y) * std::cos(p);
}

inline bool eyeInWater(const World& W, const Agent& a) {
    return get(W, (int)std::floor(a.px), (int)std::floor(a.py + EYE), (int)std::floor(a.pz)) == WATER;
}

// Crosshair targeting: blocks only (agents never block targeting rays).
inline Hit target(const World& W, const Agent& a) {
    double dx, dy, dz;
    lookDir(a.yaw, a.pitch, dx, dy, dz);
    return raycast(W, nullptr, 0, -1, a.px, a.py + EYE, a.pz, dx, dy, dz, REACH, eyeInWater(W, a));
}

// ---------------------------------------------------------------- physics

// Clip a move of `d` along axis `a` against solid blocks (Minecraft-style axis sweep).
double sweep(const World& W, const double mn[3], const double mx[3], int a, double d) {
    if (d == 0) return 0;
    int o1 = (a + 1) % 3, o2 = (a + 2) % 3;
    int lo1 = (int)std::floor(mn[o1] + 1e-9), hi1 = (int)std::floor(mx[o1] - 1e-9);
    int lo2 = (int)std::floor(mn[o2] + 1e-9), hi2 = (int)std::floor(mx[o2] - 1e-9);
    auto slabSolid = [&](int c) {
        int p[3];
        p[a] = c;
        for (int i = lo1; i <= hi1; i++)
            for (int j = lo2; j <= hi2; j++) {
                p[o1] = i; p[o2] = j;
                if (solid(get(W, p[0], p[1], p[2]))) return true;
            }
        return false;
    };
    if (d > 0) {
        int c0 = (int)std::floor(mx[a]), c1 = (int)std::floor(mx[a] + d);
        for (int c = c0; c <= c1; c++)
            if (slabSolid(c)) return std::max(0.0, c - mx[a] - 1e-7);
    } else {
        int c0 = (int)std::floor(mn[a]), c1 = (int)std::floor(mn[a] + d);
        for (int c = c0; c >= c1; c--)
            if (slabSolid(c)) return std::min(0.0, c + 1 - mn[a] + 1e-7);
    }
    return d;
}

// True if the player's box, shifted up by dy, overlaps no solid block.
bool freeAbove(const World& W, const Agent& e, double dy) {
    int x0 = (int)std::floor(e.px - HALF_W + 1e-9), x1 = (int)std::floor(e.px + HALF_W - 1e-9);
    int z0 = (int)std::floor(e.pz - HALF_W + 1e-9), z1 = (int)std::floor(e.pz + HALF_W - 1e-9);
    int y0 = (int)std::floor(e.py + dy + 1e-9), y1 = (int)std::floor(e.py + dy + HEIGHT - 1e-9);
    for (int y = y0; y <= y1; y++)
        for (int z = z0; z <= z1; z++)
            for (int x = x0; x <= x1; x++)
                if (solid(get(W, x, y, z))) return false;
    return true;
}

// Player box overlaps any water block (LivingEntity.isInWater).
bool inWater(const World& W, const Agent& e) {
    int x0 = (int)std::floor(e.px - HALF_W + 1e-9), x1 = (int)std::floor(e.px + HALF_W - 1e-9);
    int z0 = (int)std::floor(e.pz - HALF_W + 1e-9), z1 = (int)std::floor(e.pz + HALF_W - 1e-9);
    int y0 = (int)std::floor(e.py + 1e-9), y1 = (int)std::floor(e.py + HEIGHT - 1e-9);
    for (int y = y0; y <= y1; y++)
        for (int z = z0; z <= z1; z++)
            for (int x = x0; x <= x1; x++)
                if (get(W, x, y, z) == WATER) return true;
    return false;
}

// Moves the box by (dx, dy, dz) with collision. Returns true if it was stopped horizontally.
bool moveEntity(const World& W, Agent& e, double dx, double dy, double dz) {
    double mn[3] = {e.px - HALF_W, e.py, e.pz - HALF_W};
    double mx[3] = {e.px + HALF_W, e.py + HEIGHT, e.pz + HALF_W};
    double ay = sweep(W, mn, mx, 1, dy); mn[1] += ay; mx[1] += ay;
    double ax = sweep(W, mn, mx, 0, dx); mn[0] += ax; mx[0] += ax;
    double az = sweep(W, mn, mx, 2, dz);
    e.px += ax; e.py += ay; e.pz += az;

    bool hitY = ay != dy;
    e.onGround = hitY && dy < 0;
    if (hitY) e.vy = 0;
    if (ax != dx) e.vx = 0;
    if (az != dz) e.vz = 0;

    if (e.onGround) {
        if (e.fallDist > 3.0) e.health -= (int)std::ceil(e.fallDist - 3.0);
        e.fallDist = 0;
    } else if (ay < 0) {
        e.fallDist -= ay;
    }
    return ax != dx || az != dz;
}

// Team achievement state shared by the K agents of one world during one step.
// rew[k] accumulates the step reward of agent k of this world.
struct Team {
    World& W;
    Agent* ag;
    int K;
    float ts;              // team share
    float* rew;
    int mode;              // MODE_V6 or MODE_V7
    const uint8_t* shape;  // v7 knowledge shaping table of the world's task (nullptr = none)
};

// Mining data of block b for the active mode (v6: legacy table; v7: registry).
struct MineInfo { float hardness; int tier; int tool; int drop; int dropCount; };

MineInfo mineInfo(int mode, int b) {
    if (mode == MODE_V6) {
        const BlockInfo& bi = BINFO[b];
        return {bi.hardness, bi.harvestTier, bi.pickaxe ? (int)reg::T_PICKAXE : (int)reg::T_NONE, bi.drop, 1};
    }
    const reg::BlockDef& d = reg::BLOCKS[b];
    return {d.hardness, d.minTier, d.bestTool, d.drop, d.dropCount};
}

// Items of each tool type (index = reg::ToolType), built once.
const std::vector<int>& toolItems(int type) {
    static const std::vector<std::vector<int>> lists = [] {
        std::vector<std::vector<int>> L(6);
        for (int i = 0; i < NUM_ITEMS; i++)
            if (reg::ITEMS[i].kind == reg::K_TOOL) L[reg::ITEMS[i].toolType].push_back(i);
        return L;
    }();
    return lists[type];
}

// Best tool of `type` the agent holds: its tier, and the speed multiplier while mining with it
// (hand or no matching tool: tier 0, speed 1).
void bestTool(const Agent& a, int type, int& tier, double& speed) {
    tier = 0;
    speed = 1.0;
    if (type == reg::T_NONE) return;
    for (int item : toolItems(type))
        if (a.inv[item] > 0) {
            tier = std::max(tier, (int)reg::ITEMS[item].toolTier);
            speed = std::max(speed, (double)reg::ITEMS[item].toolSpeed);
        }
}

// Smelts per fuel item: v6 = coal 8, planks 1 (legacy); v7 = fuel ticks / 200 (one smelt = 200 ticks).
int fuelSmelts(int mode, int item) {
    if (mode == MODE_V6) return item == I_COAL ? 8 : item == I_PLANKS ? 1 : 0;
    return reg::ITEMS[item].fuelTicks / 200;
}

// Obtained by agent k (mined, crafted or smelted): counts for the team achievement mask. v6 pays the
// MineRL shaping split by team_share (kit items pay nothing). v7 pays no item shaping; the knowledge
// table of the task pays 0.05 * value once per agent and item.
void obtain(Team& T, int k, int item, int n) {
    Agent& a = T.ag[k];
    a.inv[item] += n;
    const bool kitItem = (a.kitGot >> item) & 1;
    if (T.mode == MODE_V6) {
        const uint32_t bit = 1u << item;
        if (!((T.W.got | (uint32_t)a.kitGot) & bit) && ITEM_VALUE_V6[item] > 0) {
            T.W.got |= bit;
            const float r = 0.1f * (1.0f + std::log2((float)ITEM_VALUE_V6[item]));
            for (int j = 0; j < T.K; j++) T.rew[j] += T.ts * r;
            T.rew[k] += (1.f - T.ts) * r;
        }
        return;
    }
    if (item < 31 && !kitItem) T.W.got |= 1u << item;
    if (T.shape && T.shape[item] && !kitItem && !((a.shapeMask[item >> 6] >> (item & 63)) & 1)) {
        a.shapeMask[item >> 6] |= 1ull << (item & 63);
        T.rew[k] += 0.05f * (float)reg::ITEMS[item].value;
    }
}

// Attack is a "hold the button" macro: it keeps mining the same block until it breaks,
// the target changes, or this many ticks pass. Without it a policy must pick attack 15
// decisions in a row to break one log by hand, which random exploration never does.
constexpr int MAX_ATTACK_TICKS = 80;

void attackTick(Team& T, int k) {
    Agent& a = T.ag[k];
    Hit h = target(T.W, a);
    if (!h.hit) { a.prog = 0; a.mx = -1; return; }
    if (h.x != a.mx || h.y != a.my || h.z != a.mz) {
        a.prog = 0;
        a.mx = h.x; a.my = h.y; a.mz = h.z;
    }
    const MineInfo mi = mineInfo(T.mode, h.b);
    if (mi.hardness < 0) return;
    int tier;
    double speed;
    bestTool(a, mi.tool, tier, speed);
    bool can = tier >= mi.tier;
    if (!a.onGround) speed /= 5.0;              // Player.getDestroySpeed: airborne
    if (eyeInWater(T.W, a)) speed /= 5.0;       // SUBMERGED_MINING_SPEED 0.2
    a.prog += mi.hardness == 0 ? 1.0 : speed / mi.hardness / (can ? 30.0 : 100.0);
    if (a.prog >= 1.0 - 1e-6) {
        put(T.W, h.x, h.y, h.z, AIR);
        a.st.mined[h.b]++;
        if (can && mi.drop >= 0) obtain(T, k, mi.drop, mi.dropCount);
        if (h.b == CHEST) {
            // Breaking a chest spills its contents into the miner's inventory (not achievements).
            auto it = T.W.chests.find(cell(h.x, h.y, h.z));
            if (it != T.W.chests.end()) {
                for (int i = 0; i < NUM_ITEMS; i++) a.inv[i] += it->second.count[i];
                T.W.chests.erase(it);
            }
        }
        a.prog = 0;
        a.mx = -1;
    }
}

// One physics tick. `mul` scales the ground acceleration (sprint).
void tick(Team& T, int k, double fwd, double strafe, bool jump, bool attack, double mul = 1.0) {
    const World& W = T.W;
    Agent& a = T.ag[k];
    double len = std::sqrt(fwd * fwd + strafe * strafe);
    if (len > 1) { fwd /= len; strafe /= len; }
    double yr = a.yaw * PI / 180.0;
    double fx = -std::sin(yr), fz = std::cos(yr);   // forward
    double rx = -std::cos(yr), rz = -std::sin(yr);  // right

    if (inWater(W, a)) {
        // LivingEntity.travelInWater.
        if (jump) a.vy += WATER_JUMP;
        a.vx += (fwd * fx + strafe * rx) * WATER_ACCEL;
        a.vz += (fwd * fz + strafe * rz) * WATER_ACCEL;
        bool blocked = moveEntity(W, a, a.vx, a.vy, a.vz);
        a.vx *= WATER_DRAG;
        a.vy *= WATER_DRAG;
        a.vz *= WATER_DRAG;
        a.vy -= GRAVITY / 16.0;                     // getFluidFallingAdjustedMovement
        // jumpOutOfFluid: pushing into a wall with room above climbs out.
        if (blocked && freeAbove(W, a, 0.6)) a.vy = 0.3;
        a.fallDist = 0;                             // no fall damage in water
    } else {
        if (jump && a.onGround) a.vy = JUMP_VEL;

        double accel = (a.onGround ? GROUND_ACCEL : AIR_ACCEL) * (a.onGround ? mul : 1.0);
        a.vx += (fwd * fx + strafe * rx) * accel;
        a.vz += (fwd * fz + strafe * rz) * accel;

        double fric = a.onGround ? GROUND_FRICTION : AIR_FRICTION;
        moveEntity(W, a, a.vx, a.vy, a.vz);
        a.vy = (a.vy - GRAVITY) * DRAG;
        a.vx *= fric;
        a.vz *= fric;
    }

    if (attack) attackTick(T, k);
    else { a.prog = 0; a.mx = -1; }
}

// ---------------------------------------------------------------- interaction

bool nearBlock(const World& W, const Agent& e, uint8_t b) {
    int cx = (int)std::floor(e.px), cy = (int)std::floor(e.py), cz = (int)std::floor(e.pz);
    for (int dy = -3; dy <= 4; dy++)
        for (int dz = -3; dz <= 3; dz++)
            for (int dx = -3; dx <= 3; dx++)
                if (get(W, cx + dx, cy + dy, cz + dz) == b) return true;
    return false;
}

// Places `b` on the block face under the crosshair. On success (x, y, z) is the placed cell.
bool placeBlock(World& W, Agent& e, uint8_t b, int& ox, int& oy, int& oz) {
    Hit h = target(W, e);
    if (!h.hit || h.face < 0) return false;
    int x = h.px, y = h.py, z = h.pz;
    if (x < 0 || x >= WX || z < 0 || z >= WZ || y < 1 || y >= WY) return false;
    if (get(W, x, y, z) != AIR) return false;
    if (x < e.px + HALF_W && x + 1 > e.px - HALF_W && y < e.py + HEIGHT && y + 1 > e.py &&
        z < e.pz + HALF_W && z + 1 > e.pz - HALF_W)
        return false;
    put(W, x, y, z, b);
    e.prog = 0;
    e.mx = -1;
    ox = x; oy = y; oz = z;
    return true;
}

// Crafting from the registry (needs: inventory 2x2, crafting table within reach, or furnace = smelt only).
bool craftRecipe(Team& T, int k, int r) {
    if (r < 0 || r >= NUM_RECIPES) return false;
    const reg::RecipeDef& R = reg::RECIPES[r];
    Agent& a = T.ag[k];
    if (R.needs == reg::N_FURNACE) return false;
    if (R.needs == reg::N_TABLE && !nearBlock(T.W, a, TABLE)) return false;
    for (int i = 0; i < R.nIng; i++)
        if (a.inv[R.ingItem[i]] < R.ingCount[i]) return false;
    for (int i = 0; i < R.nIng; i++) a.inv[R.ingItem[i]] -= R.ingCount[i];
    obtain(T, k, R.result, R.count);
    a.st.crafted[R.result]++;
    return true;
}

// Smelting from the registry (needs a furnace in reach). Fuel: the best fuel item in the inventory is
// burnt when the furnace is cold; each smelt uses one unit of fuel time.
bool smeltRecipe(Team& T, int k, int r) {
    if (r < 0 || r >= NUM_RECIPES) return false;
    const reg::RecipeDef& R = reg::RECIPES[r];
    Agent& a = T.ag[k];
    if (R.needs != reg::N_FURNACE || R.nIng != 1) return false;
    if (!nearBlock(T.W, a, FURNACE)) return false;
    const int in = R.ingItem[0];
    if (a.inv[in] < R.ingCount[0]) return false;
    if (a.fuel == 0) {
        int best = -1, bestS = 0;
        for (int i = 0; i < NUM_ITEMS; i++) {
            if (a.inv[i] <= 0) continue;
            int s = fuelSmelts(T.mode, i);
            if (s > bestS) { bestS = s; best = i; }
        }
        if (best < 0) return false;
        a.inv[best]--;
        a.fuel = bestS;
    }
    a.fuel--;
    a.inv[in] -= R.ingCount[0];
    obtain(T, k, R.result, R.count);
    return true;
}

// `use` with the selected item: place a block item on the crosshair face; flint_and_steel lights a
// portal; food is eaten (survival); a bed sleeps (survival). Ignored on crafting table/furnace/chest in v7.
void useItem(Team& T, int k) {
    World& W = T.W;
    Agent& a = T.ag[k];
    const int item = a.held;
    if (item < 0 || item >= NUM_ITEMS || a.inv[item] <= 0) return;
    const reg::ItemDef& D = reg::ITEMS[item];
    if (T.mode == MODE_V7 && std::string(D.name) == "white_bed" && survival::sleep(W, a)) {
        a.st.slept = true;   // a bed sleeps at night; otherwise it is placed below like any block item
        return;
    }
    if (D.places >= 0) {
        if (T.mode == MODE_V7) {
            Hit h = target(W, a);
            if (h.hit && (h.b == TABLE || h.b == FURNACE || h.b == CHEST)) return;
        }
        int x, y, z;
        const uint8_t b = (uint8_t)D.places;
        if (placeBlock(W, a, b, x, y, z)) {
            a.inv[item]--;
            a.st.placed[b]++;
            if (b == CHEST) W.chests[cell(x, y, z)] = Chest{};
        }
        return;
    }
    if (T.mode != MODE_V7) return;
    if (D.kind == reg::K_FOOD) {
        if (survival::eat(a, item)) { a.st.ate++; a.st.eaten[item]++; }
        return;
    }
    const std::string name = D.name;
    if (name == "flint_and_steel") {
        Hit h = target(W, a);
        if (h.hit && nether::tryLightPortal(W, a, h.x, h.y, h.z)) a.st.portalLit = true;
    }
}

// Nearest teammate within 4 blocks (feet distance), lowest index on ties; -1 if none.
int giveTarget(const Team& T, int k) {
    const Agent& a = T.ag[k];
    int best = -1;
    double bestD = 16.0;
    for (int j = 0; j < T.K; j++) {
        if (j == k) continue;
        const Agent& o = T.ag[j];
        double dx = o.px - a.px, dy = o.py - a.py, dz = o.pz - a.pz;
        double d2 = dx * dx + dy * dy + dz * dz;
        if (d2 <= 16.0 && d2 < bestD) { bestD = d2; best = j; }
    }
    return best;
}

// One decoded decision for one agent (v7 columns, or the v6 id mapped onto them).
struct Cmd {
    int prim = PR_NOOP;
    int recipe = -1;   // craft/smelt recipe index or -1
    int item = -1;     // item to select before acting, or -1 (keep)
};

// Decodes the action row of agent i (v6: 1 column, v7: 4 columns). v6 ids map onto primaries; the
// placement item is chosen from the agent's inventory at decision time (as in v5/v6).
Cmd decodeAction(const Agent& a, int mode, const int32_t* row) {
    Cmd c;
    if (mode == MODE_V7) {
        c.prim = std::clamp(row[0], 0, (int)NUM_PRIMARY - 1);
        c.recipe = std::clamp(row[1], 0, NUM_RECIPES) - 1;
        c.item = std::clamp(row[2], 0, NUM_ITEMS) - 1;
        return c;
    }
    const int act = std::clamp(row[0], 0, V6_NUM_ACTIONS - 1);
    switch (act) {
    case A_NOOP: c.prim = PR_NOOP; break;
    case A_FWD: c.prim = PR_FWD; break;
    case A_BACK: c.prim = PR_BACK; break;
    case A_LEFT: c.prim = PR_LEFT; break;
    case A_RIGHT: c.prim = PR_RIGHT; break;
    case A_JUMP: c.prim = PR_JUMP; break;
    case A_FWD_JUMP: c.prim = PR_FWD_JUMP; break;
    case A_TURN_L: c.prim = PR_TURN_L; break;
    case A_TURN_R: c.prim = PR_TURN_R; break;
    case A_LOOK_UP: c.prim = PR_LOOK_UP; break;
    case A_LOOK_DOWN: c.prim = PR_LOOK_DOWN; break;
    case A_ATTACK: c.prim = PR_ATTACK; break;
    case A_PLACE_BLOCK: {
        int item = a.inv[I_COBBLE] ? I_COBBLE : a.inv[I_DIRT] ? I_DIRT : a.inv[I_PLANKS] ? I_PLANKS : -1;
        c.prim = item >= 0 ? PR_USE : PR_NOOP;
        c.item = item;
        break;
    }
    case A_PLACE_TABLE: c.prim = a.inv[I_TABLE] ? PR_USE : PR_NOOP; c.item = I_TABLE; break;
    case A_PLACE_FURNACE: c.prim = a.inv[I_FURNACE] ? PR_USE : PR_NOOP; c.item = I_FURNACE; break;
    case A_PLACE_CHEST: c.prim = a.inv[I_CHEST] ? PR_USE : PR_NOOP; c.item = I_CHEST; break;
    case A_CRAFT_PLANKS: c.prim = PR_CRAFT; c.recipe = reg::REC_V6_CRAFT_PLANKS; break;
    case A_CRAFT_STICKS: c.prim = PR_CRAFT; c.recipe = reg::REC_V6_CRAFT_STICKS; break;
    case A_CRAFT_TABLE: c.prim = PR_CRAFT; c.recipe = reg::REC_V6_CRAFT_TABLE; break;
    case A_CRAFT_WOOD_PICK: c.prim = PR_CRAFT; c.recipe = reg::REC_V6_CRAFT_WOOD_PICK; break;
    case A_CRAFT_STONE_PICK: c.prim = PR_CRAFT; c.recipe = reg::REC_V6_CRAFT_STONE_PICK; break;
    case A_CRAFT_FURNACE: c.prim = PR_CRAFT; c.recipe = reg::REC_V6_CRAFT_FURNACE; break;
    case A_CRAFT_IRON_PICK: c.prim = PR_CRAFT; c.recipe = reg::REC_V6_CRAFT_IRON_PICK; break;
    case A_CRAFT_CHEST: c.prim = PR_CRAFT; c.recipe = reg::REC_V6_CRAFT_CHEST; break;
    case A_SMELT_IRON: c.prim = PR_SMELT; c.recipe = reg::REC_V6_SMELT_IRON; break;
    case A_DEPOSIT: c.prim = PR_DEPOSIT; break;
    case A_WITHDRAW: c.prim = PR_WITHDRAW; break;
    case A_GIVE: c.prim = PR_GIVE; break;
    default: c.prim = PR_NOOP; break;
    }
    return c;
}

void interact(Team& T, int k, const Cmd& c) {
    World& W = T.W;
    Agent& a = T.ag[k];
    switch (c.prim) {
    case PR_USE: useItem(T, k); break;
    case PR_CRAFT: craftRecipe(T, k, c.recipe); break;
    case PR_SMELT: smeltRecipe(T, k, c.recipe); break;
    case PR_DEPOSIT: {
        // Crosshair on a chest within reach: all non-tool items go into it.
        Hit h = target(W, a);
        if (h.hit && h.b == CHEST) {
            Chest& ch = W.chests[cell(h.x, h.y, h.z)];
            bool any = false;
            for (int i = 0; i < NUM_ITEMS; i++) {
                if (isTool(i) || a.inv[i] <= 0) continue;
                ch.count[i] += a.inv[i];
                a.inv[i] = 0;
                any = true;
            }
            if (any) ch.depositMask |= (uint8_t)(1u << k);
        }
        break;
    }
    case PR_WITHDRAW: {
        // Crosshair on a chest within reach: all of its items come out (transfer, no reward).
        Hit h = target(W, a);
        if (h.hit && h.b == CHEST) {
            auto it = W.chests.find(cell(h.x, h.y, h.z));
            if (it != W.chests.end()) {
                bool had = false;
                for (int i = 0; i < NUM_ITEMS; i++) {
                    had = had || it->second.count[i] > 0;
                    a.inv[i] += it->second.count[i];
                    it->second.count[i] = 0;
                }
                // shared_chest: someone else deposited here -> both sides count it.
                const uint8_t others = it->second.depositMask & (uint8_t)~(1u << k);
                if (had && others) {
                    a.st.shared++;
                    for (int j = 0; j < T.K; j++)
                        if ((others >> j) & 1) T.ag[j].st.shared++;
                }
                it->second.depositMask = 0;
            }
        }
        break;
    }
    case PR_GIVE: {
        // Nearest teammate within 4 blocks gets half (rounded up) of each non-tool stack (no reward).
        int j = giveTarget(T, k);
        if (j < 0) break;
        Agent& o = T.ag[j];
        bool any = false;
        for (int i = 0; i < NUM_ITEMS; i++) {
            if (isTool(i) || a.inv[i] <= 0) continue;
            int n = (a.inv[i] + 1) / 2;
            a.inv[i] -= n;
            o.inv[i] += n;
            any = true;
        }
        if (any) a.st.gave++;
        break;
    }
    case PR_DROP:
        if (a.held >= 0) a.inv[a.held] = 0;
        break;
    default:
        break;
    }
}

// One agent step: one primary applied, then `ticks` physics ticks (longer while mining).
void stepAgent(Team& T, int k, const Cmd& c, int ticks) {
    Agent& a = T.ag[k];
    double fwd = 0, strafe = 0, mul = 1.0;
    bool jump = false, attack = false;
    switch (c.prim) {
    case PR_FWD: fwd = 1; break;
    case PR_BACK: fwd = -1; break;
    case PR_LEFT: strafe = -1; break;
    case PR_RIGHT: strafe = 1; break;
    case PR_JUMP: jump = true; break;
    case PR_FWD_JUMP: fwd = 1; jump = true; break;
    case PR_SPRINT_FWD: fwd = 1; mul = 1.3; break;
    case PR_SNEAK:
        // Toggle; the last two toggle steps are kept for crouch_greet.
        a.sneak = !a.sneak;
        a.sneakT[1] = a.sneakT[0];
        a.sneakT[0] = T.W.t;
        break;
    case PR_TURN_L: a.yaw = std::fmod(a.yaw - TURN_DEG + 360.f, 360.f); break;
    case PR_TURN_R: a.yaw = std::fmod(a.yaw + TURN_DEG, 360.f); break;
    case PR_LOOK_UP: a.pitch = std::max(-90.f, a.pitch - TURN_DEG); break;
    case PR_LOOK_DOWN: a.pitch = std::min(90.f, a.pitch + TURN_DEG); break;
    case PR_ATTACK:
        // A mob under the crosshair (within reach, in front of any block) takes the hit instead of mining;
        // damage < 0 asks mobs::attack to use the attacker's best weapon (sword/axe damage, else fist 1).
        if (!mobs::attack(T.W, a, -1.f)) attack = true;
        break;
    default: interact(T, k, c); break;
    }
    if (a.sneak && (fwd != 0 || strafe != 0)) { fwd *= 0.3; strafe *= 0.3; }
    for (int j = 0; a.health > 0 && (j < ticks || (attack && a.mx >= 0 && j < MAX_ATTACK_TICKS)); j++) {
        tick(T, k, fwd, strafe, jump, attack, mul);
        a.ticks++;
    }
}

// ---------------------------------------------------------------- v7 task predicates + per-decision stats

// Teammate o inside a 90 degree cone in front of agent a (eye to body centre), distance in [minD, maxD].
bool inView(const Agent& a, const Agent& o, double minD, double maxD) {
    double dx = o.px - a.px, dy = (o.py + 0.9) - (a.py + EYE), dz = o.pz - a.pz;
    double d = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (d < minD || d > maxD || d < 1e-6) return false;
    double f[3];
    lookDir(a.yaw, a.pitch, f[0], f[1], f[2]);
    return (dx * f[0] + dy * f[1] + dz * f[2]) / d >= std::cos(45.0 * PI / 180.0);
}

// Distance bucket of the near-teammate counters: 0 <= 2, 1 <= 4, 2 <= 8, 3 <= 16 blocks.
inline int nearBucket(double dist) { return dist <= 2 ? 0 : dist <= 4 ? 1 : dist <= 8 ? 2 : 3; }

// Per-decision statistics of agent k (after all agents acted): distance, lowest y, teammate
// proximity, looking at a teammate, following, crouch greeting, messages.
void decisionStats(World& W, Agent* ag, int K, int k, const Cmd& c, double moved) {
    Agent& a = ag[k];
    TaskStats& s = a.st;
    s.dist += moved;
    s.minY = std::min(s.minY, a.py);
    if (a.health <= 0 || K <= 1) return;
    double dmin = 1e30;
    for (int o = 0; o < K; o++) {
        if (o == k) continue;
        const Agent& t = ag[o];
        double dx = t.px - a.px, dy = t.py - a.py, dz = t.pz - a.pz;
        dmin = std::min(dmin, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    if (dmin <= 2) s.near[0]++;
    if (dmin <= 4) s.near[1]++;
    if (dmin <= 8) s.near[2]++;
    if (dmin <= 16) s.near[3]++;
    if (a.msg > 0 && dmin <= 8) s.msgNear++;
    // Looking at a teammate: the crosshair ray (agents included) hits the AGENT pseudo block.
    double dx, dy, dz;
    lookDir(a.yaw, a.pitch, dx, dy, dz);
    Hit h = raycast(W, ag, K, k, a.px, a.py + EYE, a.pz, dx, dy, dz, REACH, eyeInWater(W, a));
    if (h.hit && h.b == AGENT) s.look++;
    // Following: moved forwards with a teammate in view 3..8 blocks away.
    if (moved > 1e-3 && (c.prim == PR_FWD || c.prim == PR_SPRINT_FWD || c.prim == PR_FWD_JUMP)) {
        for (int o = 0; o < K; o++)
            if (o != k && inView(a, ag[o], 3.0, 8.0)) { s.follow++; break; }
    }
    // Crouch greeting: a sneak toggle that follows another within 10 decisions, teammate within 4 in view.
    if (c.prim == PR_SNEAK && a.sneakT[1] > -1000000 && a.sneakT[0] - a.sneakT[1] <= 10) {
        for (int o = 0; o < K; o++)
            if (o != k && inView(a, ag[o], 0.0, 4.0)) {
                s.crouch++;
                a.sneakT[1] = -1000000;   // the next greet needs two new toggles
                break;
            }
    }
}

// The task success predicate of the world (set_task overrides applied).
reg::Pred successPred(const World& W) {
    reg::Pred p = reg::TASKS[W.task].success;
    if (W.taskP0 >= 0) p.a = W.taskP0;
    if (W.taskP1 >= 0) p.b = W.taskP1;
    return p;
}

// Truth of one predicate for agent a (counters are per agent and episode).
bool predTrue(const World& W, const Agent& a, const reg::Pred& p) {
    const TaskStats& s = a.st;
    const bool itemOk = p.a >= 0 && p.a < NUM_ITEMS;
    const bool blockOk = p.a >= 0 && p.a < NUM_BLOCKS;
    const bool mobOk = p.a >= 0 && p.a < NUM_MOBS;
    switch (p.type) {
    case reg::P_HAS_ITEM: return itemOk && a.inv[p.a] >= p.b;
    case reg::P_CRAFTED: return itemOk && s.crafted[p.a] >= 1;
    case reg::P_MINED: return blockOk && s.mined[p.a] >= p.b;
    case reg::P_KILLED: return mobOk && s.killed[p.a] >= p.b;
    case reg::P_ATE:
        if (p.a < 0) return s.ate >= 1;
        return itemOk && s.eaten[p.a] >= 1;
    case reg::P_SLEPT: return s.slept;
    case reg::P_SURVIVE_NIGHT: return s.surviveNight;
    case reg::P_PLACED: return blockOk && s.placed[p.a] >= p.b;
    case reg::P_BUILD_COMPLETE: return build::progress(W) >= p.b / 1000.f;
    case reg::P_PORTAL_LIT: return s.portalLit;
    case reg::P_IN_DIMENSION: return a.dim == p.a;
    case reg::P_REACHED_Y: return s.minY <= p.a;
    case reg::P_DISTANCE_TRAVELLED: return s.dist >= p.a;
    case reg::P_HEALTH_ABOVE: return a.health > p.a;
    case reg::P_NEAR_TEAMMATE: return s.near[nearBucket(p.a)] >= p.b;
    case reg::P_LOOK_AT_TEAMMATE: return s.look >= p.b;
    case reg::P_CROUCH_GREET: return s.crouch >= p.b;
    case reg::P_FOLLOW_TEAMMATE: return s.follow >= p.b;
    case reg::P_GAVE_ITEM: return s.gave >= p.b;
    case reg::P_SHARED_CHEST: return s.shared >= p.b;
    case reg::P_MSG_NEAR_TEAMMATE: return s.msgNear >= p.b;
    default: return false;
    }
}

// ---------------------------------------------------------------- observations

struct Camera { double ox, oy, oz, f[3], r[3], u[3], tanH; };

Camera camera(const Agent& e) {
    Camera c;
    c.ox = e.px; c.oy = e.py + EYE; c.oz = e.pz;
    lookDir(e.yaw, e.pitch, c.f[0], c.f[1], c.f[2]);
    double yr = e.yaw * PI / 180.0;
    c.r[0] = -std::cos(yr); c.r[1] = 0; c.r[2] = -std::sin(yr);
    c.u[0] = c.r[1] * c.f[2] - c.r[2] * c.f[1];
    c.u[1] = c.r[2] * c.f[0] - c.r[0] * c.f[2];
    c.u[2] = c.r[0] * c.f[1] - c.r[1] * c.f[0];
    c.tanH = std::tan(FOV_DEG * 0.5 * PI / 180.0);
    return c;
}

inline Hit castPixel(const World& W, const Agent* ag, int K, int self, const Camera& c, int i, int j,
                     int Wd, int H, double d[3], bool passWater) {
    double aspect = (double)Wd / H;
    double u = ((i + 0.5) / Wd * 2 - 1) * c.tanH * aspect;
    double v = (1 - (j + 0.5) / H * 2) * c.tanH;
    for (int k = 0; k < 3; k++) d[k] = c.f[k] + c.r[k] * u + c.u[k] * v;
    double n = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    for (int k = 0; k < 3; k++) d[k] /= n;
    return raycast(W, ag, K, self, c.ox, c.oy, c.oz, d[0], d[1], d[2], VIEW_DIST, passWater);
}

// Teammate-relative offset rotated into the agent's yaw frame: x = right, z = forward.
inline void yawFrame(const Agent& a, double dx, double dz, double& xl, double& zl) {
    double yr = a.yaw * PI / 180.0;
    double fx = -std::sin(yr), fz = std::cos(yr);
    double rx = -std::cos(yr), rz = -std::sin(yr);
    xl = dx * rx + dz * rz;
    zl = dx * fx + dz * fz;
}

// Per pixel: [block id (0 = sky), depth 0..255]. Returns the bitmask of block ids hit by the image rays
// (bit = block id; sky and the AGENT pseudo block excluded). Vector obs: v6 layout (6.5) or v7 layout.
uint64_t writeObs(const World& W, const Agent* ag, int K, int self, int res, int maxSteps, int mode, uint8_t* img,
                  float* vec) {
    const Agent& a = ag[self];
    Camera c = camera(a);
    int R = res;
    double d[3];
    bool pass = eyeInWater(W, a);
    uint64_t seen = 0;
    mobs::beginView(W, a);   // pre-filter mobs to this agent's view so per-pixel rayHit stays cheap
    for (int j = 0; j < R; j++)
        for (int i = 0; i < R; i++) {
            Hit h = castPixel(W, ag, K, self, c, i, j, R, R, d, pass);
            uint8_t* p = img + (j * R + i) * 2;
            p[0] = h.hit ? h.b : 0;
            p[1] = h.hit ? (uint8_t)std::min(255.0, h.dist / VIEW_DIST * 255.0) : 255;
            if (h.hit) seen |= 1ull << h.b;
        }
    seen &= ~(1ull << AGENT);

    // Shared pieces: teammate slots (ascending index, k' != self) and the nearest chest.
    auto writeTeam = [&](float* base) {
        for (int i = 0; i < 4 * 12; i++) base[i] = 0.f;
        int slot = 0;
        for (int o = 0; o < K; o++) {
            if (o == self) continue;
            const Agent& t = ag[o];
            float* s = base + slot * 12;
            double xl, zl;
            yawFrame(a, t.px - a.px, t.pz - a.pz, xl, zl);
            s[0] = 1.f;
            s[1] = (float)(xl / 32.0);
            s[2] = (float)((t.py - a.py) / 32.0);
            s[3] = (float)(zl / 32.0);
            s[4 + std::clamp(t.msg, 0, NUM_MSGS - 1)] = 1.f;
            slot++;
        }
    };
    auto writeChest = [&](float* cv) {
        cv[0] = cv[1] = cv[2] = cv[3] = 0.f;
        double bestD = 256.0;
        int bx = 0, by = 0, bz = 0;
        bool found = false;
        for (const auto& kv : W.chests) {
            int key = kv.first;
            int y = key / (WX * WZ), z = (key % (WX * WZ)) / WX, x = key % WX;
            double dx = x + 0.5 - a.px, dy = y + 0.5 - a.py, dz = z + 0.5 - a.pz;
            double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 <= bestD) { bestD = d2; bx = x; by = y; bz = z; found = true; }
        }
        if (found) {
            double xl, zl;
            yawFrame(a, bx + 0.5 - a.px, bz + 0.5 - a.pz, xl, zl);
            cv[0] = 1.f;
            cv[1] = (float)(xl / 16.0);
            cv[2] = (float)((by + 0.5 - a.py) / 16.0);
            cv[3] = (float)(zl / 16.0);
        }
    };

    if (mode == MODE_V6) {
        float* out = vec;
        for (int i = 0; i < V6_NUM_ITEMS; i++) out[i] = std::log1p((float)a.inv[i]) / std::log1p(64.f);
        int k = V6_NUM_ITEMS;
        double yr = a.yaw * PI / 180.0;
        out[k++] = a.health / 20.f;
        out[k++] = a.onGround ? 1.f : 0.f;
        out[k++] = a.pitch / 90.f;
        out[k++] = (float)std::sin(yr);
        out[k++] = (float)std::cos(yr);
        out[k++] = (float)a.prog;
        out[k++] = (float)((a.py - SEA_LEVEL) / 32.0);
        out[k++] = nearBlock(W, a, TABLE) ? 1.f : 0.f;
        out[k++] = nearBlock(W, a, FURNACE) ? 1.f : 0.f;
        out[k++] = std::min(a.fuel, 8) / 8.f;
        out[k++] = (float)W.t / maxSteps;
        for (int i = 0; i < V6_NUM_ITEMS; i++) out[GOAL_OFFSET + i] = i == W.goal ? 1.f : 0.f;
        writeTeam(out + TEAM_OFFSET);
        writeChest(out + CHEST_OFFSET);
        float* xv = out + XHAIR_OFFSET;
        for (int i = 0; i < 9; i++) xv[i] = 0.f;
        Hit th = target(W, a);
        if (th.hit && th.b == CHEST) {
            auto it = W.chests.find(cell(th.x, th.y, th.z));
            if (it != W.chests.end())
                for (int i = 0; i < 9; i++) xv[i] = std::log2(1.f + (float)it->second.count[i]) / 6.f;
        }
        return seen;
    }

    // ---- v7 layout (see src/README_v7.md and V7_* in sim.h)
    for (int i = 0; i < NUM_ITEMS; i++) vec[V7_INV + i] = std::log2(1.f + (float)a.inv[i]) / 7.f;
    float* s = vec + V7_SCALARS;
    const double yr = a.yaw * PI / 180.0;
    const reg::Pred succ = successPred(W);
    const double tTime = survival::timeOfDay(W) * 2.0 * PI;
    const int fx = (int)std::floor(a.px), fy = (int)std::floor(a.py), fz = (int)std::floor(a.pz);
    s[0] = a.health / 20.f;                        // 0 health
    s[1] = a.surv.hunger / 20.f;                   // 1 hunger
    s[2] = a.surv.saturation / 20.f;               // 2 saturation (0..20, survival FoodData)
    s[3] = a.onGround ? 1.f : 0.f;                 // 3 on_ground
    s[4] = a.pitch / 90.f;                         // 4 pitch
    s[5] = (float)std::sin(yr);                    // 5 sin_yaw
    s[6] = (float)std::cos(yr);                    // 6 cos_yaw
    s[7] = (float)((a.py - SEA_LEVEL) / 32.0);     // 7 height
    s[8] = inWater(W, a) ? 1.f : 0.f;              // 8 in_water
    s[9] = survival::lightAt(W, fx, fy, fz);       // 9 light_level
    s[10] = (float)std::sin(tTime);                // 10 time_sin
    s[11] = (float)std::cos(tTime);                // 11 time_cos
    s[12] = (float)a.dim;                          // 12 dimension (0 overworld, 1 nether)
    s[13] = 0.f;                                   // 13 armor (phase 2)
    s[14] = (float)a.prog;                         // 14 mining_progress
    s[15] = std::min(a.fuel, 8) / 8.f;             // 15 fuel
    s[16] = W.maxT > 0 ? (float)std::clamp(1.0 - (double)W.t / W.maxT, 0.0, 1.0) : 0.f;   // 16 steps_left
    s[17] = a.sneak ? 1.f : 0.f;                   // 17 sneak
    s[18] = (float)(a.held + 1) / NUM_ITEMS;       // 18 held item index (0 = none)
    s[19] = (float)(std::sqrt(a.vx * a.vx + a.vz * a.vz) / 0.3);   // 19 horizontal speed (walk = 1)

    // Task: one-hot over NUM_TASKS, then the target item / count of the success predicate.
    float* tk = vec + V7_TASK;
    for (int i = 0; i < NUM_TASKS; i++) tk[i] = i == W.task ? 1.f : 0.f;
    const bool itemTarget = succ.type == reg::P_HAS_ITEM || succ.type == reg::P_CRAFTED;
    vec[V7_TASK_PARAM] = itemTarget && succ.a >= 0 ? (float)succ.a / NUM_ITEMS : 0.f;
    vec[V7_TASK_PARAM + 1] = (float)std::clamp(succ.b, 0, 64) / 64.f;

    writeTeam(vec + V7_TEAM);
    writeChest(vec + V7_CHEST);
    float* xv = vec + V7_XHAIR;
    for (int i = 0; i < 9; i++) xv[i] = 0.f;
    Hit th = target(W, a);
    if (th.hit && th.b == CHEST) {
        auto it = W.chests.find(cell(th.x, th.y, th.z));
        if (it != W.chests.end())
            for (int i = 0; i < 9; i++)
                xv[i] = std::log2(1.f + (float)it->second.count[reg::XHAIR_ITEMS[i]]) / 6.f;
    }
    mobs::obs(W, a, vec + V7_MOBS);
    build::obs(W, a, vec + V7_BUILD);
    return seen;
}

// Block colours come from the registry (blocks.json "color"); coal, iron and diamond ore get speckles.
inline const uint8_t* blockColor(int b) { return reg::BLOCKS[b].color; }
inline const uint8_t* speckleOf(int b) {
    static const uint8_t COAL[3] = {25, 25, 25}, IRON[3] = {216, 175, 147}, DIA[3] = {90, 230, 230};
    if (b == COAL_ORE) return COAL;
    if (b == IRON_ORE) return IRON;
    if (b == DIAMOND_ORE) return DIA;
    return nullptr;
}
// Teammate boxes by agent index k: blue, red, green, yellow, purple.
const uint8_t AGENT_COLORS[MAX_AGENTS][3] = {
    {60, 120, 255}, {230, 60, 60}, {60, 200, 80}, {240, 220, 60}, {160, 80, 220},
};
const uint8_t WATER_TINT[3] = {40, 90, 200};

inline double frac(double v) { return v - std::floor(v); }

void renderRGB(const World& W, const Agent* ag, int K, int self, int Wd, int H, uint8_t* out) {
    const Agent& a = ag[self];
    Camera c = camera(a);
    double d[3];
    bool pass = eyeInWater(W, a);
    for (int j = 0; j < H; j++)
        for (int i = 0; i < Wd; i++) {
            Hit h = castPixel(W, ag, K, self, c, i, j, Wd, H, d, pass);
            uint8_t* p = out + (j * Wd + i) * 3;
            if (!h.hit) { std::memcpy(p, blockColor(AIR), 3); continue; }

            // Water is drawn as a tint over whatever is behind it.
            bool viaWater = false;
            if (h.b == WATER) {
                Hit under = castPixel(W, ag, K, self, c, i, j, Wd, H, d, true);
                if (under.hit && under.b != WATER) { h = under; viaWater = true; }
            }

            double hx = c.ox + d[0] * h.dist, hy = c.oy + d[1] * h.dist, hz = c.oz + d[2] * h.dist;
            double u, v;
            if (h.face == 0)      { u = frac(hz); v = frac(hy); }
            else if (h.face == 1) { u = frac(hx); v = frac(hz); }
            else                  { u = frac(hx); v = frac(hy); }
            int tu = std::min(7, (int)(u * 8)), tv = std::min(7, (int)(v * 8));
            uint32_t th = hash3(h.x * 8 + tu, h.y * 8 + tv, h.z * 8 + tu * 3 + tv, 1337u + h.face);
            float tex = 0.88f + 0.24f * (th & 255) / 255.f;

            const uint8_t* col = h.b == AGENT ? AGENT_COLORS[h.agent % MAX_AGENTS] : blockColor(h.b);
            if (h.b == GRASS && h.face != 1 && v < 0.8) col = blockColor(DIRT);
            if (const uint8_t* sp = speckleOf(h.b); sp && ((th >> 8) & 3) == 0) col = sp;

            float shade = h.face == 1 ? (d[1] < 0 ? 1.0f : 0.5f) : h.face == 0 ? 0.8f : h.face == 2 ? 0.68f : 1.0f;
            float fog = (float)std::min(1.0, (h.dist / VIEW_DIST) * (h.dist / VIEW_DIST));
            for (int k = 0; k < 3; k++) {
                float cv = col[k] * shade * tex;
                if (viaWater) cv = cv * 0.55f + WATER_TINT[k] * 0.45f;
                cv = cv * (1 - fog) + blockColor(AIR)[k] * fog;
                p[k] = (uint8_t)std::clamp(cv, 0.f, 255.f);
            }
        }
    // crosshair
    for (int k = -3; k <= 3; k++) {
        uint8_t* a1 = out + ((H / 2) * Wd + Wd / 2 + k) * 3;
        uint8_t* b1 = out + ((H / 2 + k) * Wd + Wd / 2) * 3;
        a1[0] = a1[1] = a1[2] = 255;
        b1[0] = b1[1] = b1[2] = 255;
    }
}

// Start a world: generation from `seed`, K agents spawned on dry grass within SPAWN_RING columns,
// goal sampled (v6), start kit applied with the kit probability, task sampled (v7).
void initWorld(World& W, Agent* ag, int K, uint64_t seed, const float* goalW, const int* kitCount,
               const float* kitProb, int mode, int maxSteps) {
    W.seed = seed;
    W.rng.seed(seed);
    W.chests.clear();
    W.got = 0;
    W.t = 0;
    W.dayOffset = 0;
    W.restart = false;
    std::vector<int> hmap;
    generate(W, hmap);
    mobs::init(W);
    build::worldgen(W, seed, mode == MODE_V7);   // biome/cave/structure variety only in v7 (v6 terrain unchanged)
    // The nether of this world is generated from a seed derived from the world seed (never reused).
    nether::generate(W, seed ^ 0x9E3779B97F4A7C15ull);

    auto clearAbove = [&](int x, int h, int z) {
        for (int y = h + 1; y < WY; y++)
            if (get(W, x, y, z) != AIR) return false;
        return true;
    };
    // Spawn column near the middle whose whole column above is air, searched in a ring first,
    // then over the whole map.
    int sx = WX / 2, sz = WZ / 2;
    double spawnY = -1;
    for (int tries = 0; tries < 400 && spawnY < 0; tries++) {
        int x = uniformInt(W.rng, WX / 2 - 20, WX / 2 + 20), z = uniformInt(W.rng, WZ / 2 - 20, WZ / 2 + 20);
        if (x < 1 || x >= WX - 1 || z < 1 || z >= WZ - 1) continue;
        int h = hmap[z * WX + x];
        if (h <= SEA_LEVEL || get(W, x, h, z) != GRASS || !clearAbove(x, h, z)) continue;
        sx = x; sz = z; spawnY = h + 1.0;
    }
    for (int z = 1; z < WZ - 1 && spawnY < 0; z++)
        for (int x = 1; x < WX - 1 && spawnY < 0; x++) {
            int h = hmap[z * WX + x];
            if (h > SEA_LEVEL && get(W, x, h, z) == GRASS && clearAbove(x, h, z)) {
                sx = x; sz = z; spawnY = h + 1.0;
            }
        }
    if (spawnY < 0) {
        // Last resort (no dry grass at all): highest solid column.
        int y = WY - 1;
        while (y > 0 && get(W, sx, y, sz) == AIR) y--;
        spawnY = y + 1.0;
    }

    for (int k = 0; k < K; k++) {
        int x = sx, z = sz;
        double y = spawnY;
        if (k > 0)
            for (int tries = 0; tries < 64; tries++) {
                int xx = sx + uniformInt(W.rng, -(int)SPAWN_RING, (int)SPAWN_RING);
                int zz = sz + uniformInt(W.rng, -(int)SPAWN_RING, (int)SPAWN_RING);
                if (xx < 1 || xx >= WX - 1 || zz < 1 || zz >= WZ - 1) continue;
                int h = hmap[zz * WX + xx];
                if (h > SEA_LEVEL && get(W, xx, h, zz) == GRASS && clearAbove(xx, h, zz)) {
                    x = xx; z = zz; y = h + 1.0;
                    break;
                }
            }
        Agent& a = ag[k];
        a = Agent{};
        a.px = x + 0.5; a.py = y; a.pz = z + 0.5;
        a.yaw = TURN_DEG * (float)(W.rng() % 24);
        a.pitch = 0;
        markVisited(a);
        if (mode == MODE_V7) {
            // Start partly hungry (food 12..20, from the seed; no RNG draw) so eating is possible early:
            // vanilla refuses food at a full bar, and a full bar barely drops within one episode.
            const uint64_t hsh = (seed ^ (0x9E3779B97F4A7C15ull * (uint64_t)(k + 1))) * 0xBF58476D1CE4E5B9ull;
            a.surv.hunger = 12 + (int)((hsh >> 40) % 9);
            a.surv.saturation = std::min(a.surv.saturation, (float)a.surv.hunger);
        }
    }

    W.goal = sampleGoal(W.rng, goalW);

    // Start kit: with probability prob[goal], the goal's row of the table (never the goal itself).
    const float p = kitProb[W.goal];
    bool apply = false;
    if (p > 0.f) {
        apply = p >= 1.f;
        if (!apply) apply = std::uniform_real_distribution<float>(0.f, 1.f)(W.rng) < p;
    }
    if (apply)
        for (int k = 0; k < K; k++) {
            Agent& a = ag[k];
            for (int item = 0; item < NUM_ITEMS; item++) {
                int c = kitCount[W.goal * NUM_ITEMS + item];
                if (item == W.goal || c <= 0) continue;
                a.inv[item] = c;
                a.kitGot |= 1ull << item;
                a.kit = true;
            }
        }

    // v7: one task per episode, drawn from the world's task weights (after the goal and kit draws).
    W.taskP0 = W.taskP1 = -1;
    if (mode == MODE_V7) {
        double total = 0;
        for (float w : W.taskW) total += w;
        int task = 0;
        if (total > 0) {
            std::uniform_real_distribution<double> u(0.0, total);
            double pr = u(W.rng), acc = 0;
            for (int i = 0; i < NUM_TASKS; i++) {
                acc += W.taskW[i];
                if (pr < acc && W.taskW[i] > 0) { task = i; break; }
                task = i;
            }
        }
        W.task = task;
        W.maxT = std::min(maxSteps, reg::TASKS[task].maxSteps);
    } else {
        W.maxT = maxSteps;
    }
}

} // namespace

namespace {

Snapshot snapshotOf(const World& W, const Agent& a, int mode) {
    Snapshot s;
    s.x = a.px; s.y = a.py; s.z = a.pz;
    s.vx = a.vx; s.vy = a.vy; s.vz = a.vz;
    s.yaw = a.yaw; s.pitch = a.pitch;
    s.onGround = a.onGround;
    s.health = a.health; s.t = W.t; s.ticks = a.ticks;
    s.seed = W.seed;
    s.episode = W.episode; s.lastAction = a.lastAction; s.msg = a.msg;
    s.miningProgress = a.prog;
    s.goal = W.goal;
    s.kit = a.kit;
    s.restart = W.restart;
    Hit h = target(W, a);
    s.hasTarget = h.hit;
    s.tx = h.x; s.ty = h.y; s.tz = h.z; s.tblock = h.b;
    s.held = a.held; s.dim = a.dim; s.task = W.task; s.taskMaxT = W.maxT; s.sneak = a.sneak;
    (void)mode;
    return s;
}

Region regionOf(const World& W, const Agent& a, int radius) {
    radius = std::max(0, radius);
    int cx = (int)std::floor(a.px), cz = (int)std::floor(a.pz);
    Region r;
    r.ox = std::max(0, cx - radius);
    r.oz = std::max(0, cz - radius);
    int x1 = std::min(WX - 1, cx + radius), z1 = std::min(WZ - 1, cz + radius);
    r.sx = std::max(0, x1 - r.ox + 1);
    r.sz = std::max(0, z1 - r.oz + 1);
    r.sy = WY;
    r.blocks.resize((size_t)r.sx * r.sy * r.sz);
    for (int y = 0; y < r.sy; y++)
        for (int z = 0; z < r.sz; z++)
            for (int x = 0; x < r.sx; x++)
                r.blocks[((size_t)y * r.sz + z) * r.sx + x] = W.w[cell(r.ox + x, y, r.oz + z)];
    return r;
}

} // namespace

// ---------------------------------------------------------------- VecEnv

VecEnv::VecEnv(int numWorlds, int agentsPerWorld, uint64_t seed, int res, int maxSteps, int ticks, int threads,
               int mode)
    : numWorlds_(numWorlds), K_(agentsPerWorld), numAgents_(numWorlds * agentsPerWorld), res_(res),
      maxSteps_(maxSteps), ticks_(ticks), threads_(threads), mode_(mode) {
    if (numWorlds < 1) throw std::invalid_argument("num_envs (worlds) must be >= 1");
    if (agentsPerWorld < 1 || agentsPerWorld > MAX_AGENTS) throw std::invalid_argument("agents must be in 1..5");
    if (res < 1) throw std::invalid_argument("res must be >= 1");
    if (mode != MODE_V6 && mode != MODE_V7) throw std::invalid_argument("action_mode must be 'v6' or 'v7'");
    // Default goal weights (v6): uniform over the v6 items 0..12 (dirt, chest excluded).
    for (int i = 0; i < NUM_ITEMS; i++) goalW_[i] = i < I_DIRT ? 1.f : 0.f;
    lastNov_.assign(numAgents_, 0.f);
    worlds_.resize(numWorlds_);
    agents_.resize(numAgents_);
    std::mt19937_64 seeder(seed);
    for (auto& W : worlds_) {
        W.seedRng.seed(seeder());
        W.rng.seed(seeder());
        W.w.assign((size_t)WX * WY * WZ, AIR);   // empty world until reset()
        W.taskW.assign(NUM_TASKS, 1.f);
        W.maxT = maxSteps;
    }
}

void VecEnv::setGoalWeights(const float* w) {
    double total = 0;
    for (int i = 0; i < NUM_ITEMS; i++) {
        if (!(w[i] >= 0.f)) throw std::invalid_argument("goal weights must be non-negative");
        total += w[i];
    }
    if (!(total > 0)) throw std::invalid_argument("goal weights must not all be zero");
    std::copy(w, w + NUM_ITEMS, goalW_);
}

int VecEnv::goal(int i) const {
    if (i < 0 || i >= numAgents_) throw std::out_of_range("agent index out of range");
    return worlds_[i / K_].goal;
}

void VecEnv::setGoal(int i, int item) {
    if (i < 0 || i >= numAgents_) throw std::out_of_range("agent index out of range");
    if (item < 0 || item >= NUM_ITEMS) throw std::out_of_range("item index out of range");
    worlds_[i / K_].goal = item;
}

void VecEnv::setTeamShare(float share) {
    if (!(share >= 0.f && share <= 1.f)) throw std::invalid_argument("team share must be in [0, 1]");
    teamShare_ = share;
}

void VecEnv::setTaskWeights(int w, const float* weights) {
    if (w < 0 || w >= numWorlds_) throw std::out_of_range("world index out of range");
    double total = 0;
    for (int i = 0; i < NUM_TASKS; i++) {
        if (!(weights[i] >= 0.f)) throw std::invalid_argument("task weights must be non-negative");
        total += weights[i];
    }
    if (!(total > 0)) throw std::invalid_argument("task weights must not all be zero");
    worlds_[w].taskW.assign(weights, weights + NUM_TASKS);
}

void VecEnv::setTask(int w, int task, int p0, int p1) {
    if (mode_ != MODE_V7) throw std::invalid_argument("tasks need action_mode 'v7'");
    if (w < 0 || w >= numWorlds_) throw std::out_of_range("world index out of range");
    if (task < 0 || task >= NUM_TASKS) throw std::out_of_range("task index out of range");
    World& W = worlds_[w];
    W.task = task;
    W.taskP0 = p0;
    W.taskP1 = p1;
    W.maxT = std::min(maxSteps_, reg::TASKS[task].maxSteps);
    // Restart the episode's task bookkeeping: counters and milestones start from zero.
    W.t = 0;
    for (int k = 0; k < K_; k++) agents_[(size_t)w * K_ + k].st = TaskStats{};
}

int VecEnv::task(int i) const {
    if (i < 0 || i >= numAgents_) throw std::out_of_range("agent index out of range");
    return worlds_[i / K_].task;
}

void VecEnv::setShapingTable(int task, const int* items, int n) {
    if (task < 0 || task >= NUM_TASKS) throw std::out_of_range("task index out of range");
    std::fill(shaping_[task], shaping_[task] + NUM_ITEMS, uint8_t(0));
    for (int i = 0; i < n; i++) {
        if (items[i] < 0 || items[i] >= NUM_ITEMS) throw std::out_of_range("item index out of range");
        shaping_[task][items[i]] = 1;
    }
}

uint64_t VecEnv::worldSeed(int w) const {
    if (w < 0 || w >= numWorlds_) throw std::out_of_range("world index out of range");
    return worlds_[w].seed;
}

VecEnv::BuildView VecEnv::buildTarget(int w) const {
    if (w < 0 || w >= numWorlds_) throw std::out_of_range("world index out of range");
    const build::State& S = worlds_[w].build;
    BuildView v;
    v.blueprint = S.blueprint;
    v.rot = S.rot;
    for (const build::TCell& c : S.cells) v.cells.push_back({c.x, c.y, c.z, c.block, c.item});
    return v;
}

std::vector<VecEnv::EntityView> VecEnv::entities(int w) const {
    if (w < 0 || w >= numWorlds_) throw std::out_of_range("world index out of range");
    std::vector<EntityView> out;
    for (const mobs::Mob& m : worlds_[w].mobs.mobs)
        if (m.type >= 0) out.push_back({m.type, m.x, m.y, m.z, m.hp});
    return out;
}

float VecEnv::timeOfDay(int w) const {
    if (w < 0 || w >= numWorlds_) throw std::out_of_range("world index out of range");
    return survival::timeOfDay(worlds_[w]);
}

// Never more threads than worlds (avoids fork/join overhead for small N).
int VecEnv::nthreads() const {
    int t = threads_ > 0 ? threads_ : omp_get_max_threads();
    return std::max(1, std::min(t, numWorlds_));
}

void VecEnv::initWorldAt(int w, uint64_t seed) {
    initWorld(worlds_[w], &agents_[(size_t)w * K_], K_, seed, goalW_, &kitCount_[0][0], kitProb_, mode_, maxSteps_);
}

// A world seed never equal to the previous one, drawn from the world's own seed stream.
uint64_t VecEnv::nextSeed(int w) {
    World& W = worlds_[w];
    uint64_t s;
    do { s = W.seedRng(); } while (s == W.seed);
    return s;
}

void VecEnv::writeAgentObs(int i, uint8_t* img, float* vec, uint64_t* seenOut) const {
    const int w = i / K_, k = i % K_;
    uint64_t seen = writeObs(worlds_[w], &agents_[(size_t)w * K_], K_, k, res_, maxSteps_, mode_, img, vec);
    if (seenOut) *seenOut = seen;
}

void VecEnv::reset(uint8_t* img, float* vec) {
    const int nw = numWorlds_;
    const int V = vecDim();
    #pragma omp parallel for schedule(dynamic) num_threads(nthreads())
    for (int w = 0; w < nw; w++) {
        initWorldAt(w, nextSeed(w));
        worlds_[w].episode = 0;
        for (int k = 0; k < K_; k++) {
            const int i = w * K_ + k;
            uint64_t seen = 0;
            writeAgentObs(i, img + (size_t)i * res_ * res_ * 2, vec + (size_t)i * V, &seen);
            agents_[i].seenBlocks = seen;
            lastNov_[i] = 0.f;
        }
    }
}

void VecEnv::step(const int32_t* actions, const int32_t* msgs, uint8_t* img, float* vec, float* rew,
                  bool* done, uint32_t* ach) {
    const int nw = numWorlds_;
    #pragma omp parallel for schedule(dynamic) num_threads(nthreads())
    for (int w = 0; w < nw; w++) stepWorld(w, actions, msgs, img, vec, rew, done, ach);
}

// One world step: all K agents act in index order (team rewards accumulate per world). v6: the goal
// check. v7: per-decision stats, task milestones (paid to the agent) and task success (team split).
// Then the episode end check, observations, and the world auto-reset when done.
void VecEnv::stepWorld(int w, const int32_t* actions, const int32_t* msgs, uint8_t* img, float* vec,
                       float* rew, bool* done, uint32_t* ach) {
    World& W = worlds_[w];
    Agent* ag = &agents_[(size_t)w * K_];
    const int K = K_, base = w * K_;
    const int V = vecDim();
    const float scale = novScale_;
    const bool v7 = mode_ == MODE_V7;
    float r[MAX_AGENTS] = {};
    float nov[MAX_AGENTS] = {};
    Team T{W, ag, K, teamShare_, r, mode_, v7 ? shaping_[W.task] : nullptr};
    const int cols = actionCols();

    W.t++;
    Cmd cmds[MAX_AGENTS];
    double moved[MAX_AGENTS] = {};
    for (int k = 0; k < K; k++) {
        Agent& a = ag[k];
        const int32_t* row = actions + (size_t)(base + k) * cols;
        a.msg = (a.health > 0) ? (v7 ? (msgs ? std::clamp((int)msgs[base + k], 0, NUM_MSGS - 1)
                                             : std::clamp((int)row[3], 0, NUM_MSGS - 1))
                                     : (msgs ? std::clamp((int)msgs[base + k], 0, NUM_MSGS - 1) : 0))
                                : 0;
        const Cmd c = decodeAction(a, mode_, row);
        cmds[k] = c;
        a.lastAction = mode_ == MODE_V6 ? std::clamp((int)row[0], 0, V6_NUM_ACTIONS - 1) : c.prim;
        if (a.health <= 0) continue;   // dead agents are frozen until the world resets
        if (c.item >= 0) a.held = c.item;
        const double x0 = a.px, z0 = a.pz;
        stepAgent(T, k, c, ticks_);
        moved[k] = std::hypot(a.px - x0, a.pz - z0);
    }

    bool goalNow = false;
    if (v7) {
        mobs::tick(W, ag, K);
        for (int k = 0; k < K; k++) {
            Agent& a = ag[k];
            if (a.health <= 0) continue;
            survival::tick(W, a);
            nether::travel(W, a);
            decisionStats(W, ag, K, k, cmds[k], moved[k]);
        }
        // Milestones pay the agent that reached them (once per episode); success ends the world.
        const reg::Pred succ = successPred(W);
        const reg::TaskDef& task = reg::TASKS[W.task];
        for (int k = 0; k < K; k++) {
            Agent& a = ag[k];
            if (a.health <= 0) continue;
            for (int j = 0; j < task.nMilestones; j++) {
                const uint32_t bit = 1u << j;
                if (!(a.st.msDone & bit) && predTrue(W, a, task.ms[j].pred)) {
                    a.st.msDone |= bit;
                    r[k] += task.ms[j].reward;
                }
            }
        }
        for (int k = 0; k < K && !goalNow; k++) {
            if (ag[k].health <= 0 || !predTrue(W, ag[k], succ)) continue;
            W.got |= GOAL_BIT;
            for (int j = 0; j < K; j++) r[j] += teamShare_ * 10.f;
            r[k] += (1.f - teamShare_) * 10.f;
            goalNow = true;
        }
    } else if (!(W.got & GOAL_BIT)) {
        // v6 goal: the first agent (index order) to hold the goal item -> +10 split by team share.
        for (int k = 0; k < K && !goalNow; k++) {
            if (ag[k].inv[W.goal] <= 0) continue;
            W.got |= GOAL_BIT;
            for (int j = 0; j < K; j++) r[j] += teamShare_ * 10.f;
            r[k] += (1.f - teamShare_) * 10.f;
            goalNow = true;
        }
    }

    bool alive = false;
    for (int k = 0; k < K; k++) alive = alive || ag[k].health > 0;
    const bool d = goalNow || W.t >= W.maxT || !alive;
    for (int k = 0; k < K; k++) nov[k] = noveltyMove(ag[k]);   // before any reset: the step's own position

    const uint32_t mask = W.got;
    for (int k = 0; k < K; k++) {
        ach[base + k] = mask;
        done[base + k] = d;
    }
    if (d) {
        // The obs belongs to the new episode: its view initialises the seen masks, no pay.
        initWorldAt(w, nextSeed(w));
        W.episode++;
        for (int k = 0; k < K; k++) {
            const int i = base + k;
            uint64_t seen = 0;
            writeAgentObs(i, img + (size_t)i * res_ * res_ * 2, vec + (size_t)i * V, &seen);
            agents_[i].seenBlocks = seen;
        }
        for (int k = 0; k < K; k++) {
            const float rn = scale * nov[k];
            lastNov_[base + k] = rn;
            rew[base + k] = r[k] + rn;
        }
    } else {
        for (int k = 0; k < K; k++) {
            const int i = base + k;
            uint64_t seen = 0;
            writeAgentObs(i, img + (size_t)i * res_ * res_ * 2, vec + (size_t)i * V, &seen);
            nov[k] += noveltyBlocks(ag[k], seen);
            const float rn = scale * nov[k];
            lastNov_[i] = rn;
            rew[i] = r[k] + rn;
        }
    }
}

void VecEnv::resetWorld(int w, uint64_t seed, uint8_t* img, float* vec) {
    if (w < 0 || w >= numWorlds_) throw std::out_of_range("world index out of range");
    initWorldAt(w, seed);
    worlds_[w].episode = 0;
    const int V = vecDim();
    for (int k = 0; k < K_; k++) {
        const int i = w * K_ + k;
        uint64_t seen = 0;
        writeAgentObs(i, img + (size_t)k * res_ * res_ * 2, vec + (size_t)k * V, &seen);
        agents_[i].seenBlocks = seen;
        lastNov_[i] = 0.f;
    }
}

void VecEnv::setStartKits(const float* table, const float* prob) {
    for (int g = 0; g < NUM_ITEMS; g++) {
        if (!(prob[g] >= 0.f && prob[g] <= 1.f)) throw std::invalid_argument("start kit prob must be in [0, 1]");
        kitProb_[g] = prob[g];
        for (int k = 0; k < NUM_ITEMS; k++) {
            float v = table[g * NUM_ITEMS + k];
            if (!(v >= 0.f && v < 1e6f)) throw std::invalid_argument("start kit counts must be finite and >= 0");
            kitCount_[g][k] = (int)std::lround(v);
        }
    }
}

void VecEnv::setNovelty(float scale) {
    if (!(scale >= 0.f) || !std::isfinite(scale)) throw std::invalid_argument("novelty scale must be finite and >= 0");
    novScale_ = scale;
}

void VecEnv::lastNovelty(float* out) const {
    std::copy(lastNov_.begin(), lastNov_.end(), out);
}

void VecEnv::lastMsgs(int32_t* out) const {
    for (int i = 0; i < numAgents_; i++) out[i] = agents_[i].msg;
}

// ---- world snapshots. Blob layout: magic, version, world blocks (raw), world fields, RNG streams
// as text, chests, agent count, then every agent field. Module-level state (mobs/nether/build) is not
// saved yet: phase-2 modules must serialize their State in their own code when they need it.
namespace {
constexpr uint32_t SNAP_MAGIC = 0x32565343u;   // "CSV2"
constexpr uint32_t SNAP_VERSION = 3;

struct Writer {
    std::vector<uint8_t>& b;
    template <class T> void put(const T& v) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        b.insert(b.end(), p, p + sizeof(T));
    }
    void bytes(const void* p, size_t n) {
        const uint8_t* q = static_cast<const uint8_t*>(p);
        b.insert(b.end(), q, q + n);
    }
    void rng(const std::mt19937_64& g) {
        std::ostringstream os;
        os << g;
        std::string s = os.str();
        uint32_t n = (uint32_t)s.size();
        put(n);
        bytes(s.data(), s.size());
    }
};

struct Reader {
    const uint8_t* p;
    size_t n, off = 0;
    template <class T> void get(T& v) {
        if (off + sizeof(T) > n) throw std::invalid_argument("world snapshot truncated");
        std::memcpy(&v, p + off, sizeof(T));
        off += sizeof(T);
    }
    void bytes(void* dst, size_t k) {
        if (off + k > n) throw std::invalid_argument("world snapshot truncated");
        std::memcpy(dst, p + off, k);
        off += k;
    }
    void rng(std::mt19937_64& g) {
        uint32_t n2 = 0;
        get(n2);
        if (n2 > 1u << 20) throw std::invalid_argument("world snapshot has a bad rng state");
        std::string s(n2, '\0');
        bytes(s.data(), n2);
        std::istringstream is(s);
        is >> g;
        if (!is) throw std::invalid_argument("world snapshot has a bad rng state");
    }
};

static_assert(std::is_trivially_copyable<TaskStats>::value, "TaskStats must be POD");
static_assert(std::is_trivially_copyable<survival::AgentState>::value, "AgentState must be POD");

void writeAgent(Writer& w, const Agent& a) {
    w.put(a.px); w.put(a.py); w.put(a.pz);
    w.put(a.vx); w.put(a.vy); w.put(a.vz);
    w.put(a.yaw); w.put(a.pitch);
    uint8_t og = a.onGround, kit = a.kit, sn = a.sneak;
    w.put(og); w.put(kit);
    w.put(a.fallDist);
    w.put(a.health);
    for (int k = 0; k < NUM_ITEMS; k++) w.put(a.inv[k]);
    w.put(a.kitGot);
    w.put(a.fuel);
    w.put(a.mx); w.put(a.my); w.put(a.mz);
    w.put(a.prog);
    w.put(a.ticks); w.put(a.lastAction); w.put(a.msg);
    w.bytes(a.visited, sizeof(a.visited));
    w.put(a.seenBlocks); w.put(a.bestDepthStep);
    w.put(a.held); w.put(sn); w.put(a.dim);
    w.put(a.sneakT[0]); w.put(a.sneakT[1]);
    w.put(a.shapeMask[0]); w.put(a.shapeMask[1]);
    w.bytes(&a.st, sizeof(a.st));
    w.bytes(&a.surv, sizeof(a.surv));
}

void readAgent(Reader& r, Agent& a) {
    r.get(a.px); r.get(a.py); r.get(a.pz);
    r.get(a.vx); r.get(a.vy); r.get(a.vz);
    r.get(a.yaw); r.get(a.pitch);
    uint8_t og = 0, kit = 0, sn = 0;
    r.get(og); r.get(kit);
    a.onGround = og; a.kit = kit;
    r.get(a.fallDist);
    r.get(a.health);
    for (int k = 0; k < NUM_ITEMS; k++) r.get(a.inv[k]);
    r.get(a.kitGot);
    r.get(a.fuel);
    r.get(a.mx); r.get(a.my); r.get(a.mz);
    r.get(a.prog);
    r.get(a.ticks); r.get(a.lastAction); r.get(a.msg);
    r.bytes(a.visited, sizeof(a.visited));
    r.get(a.seenBlocks); r.get(a.bestDepthStep);
    r.get(a.held); r.get(sn); r.get(a.dim);
    a.sneak = sn;
    r.get(a.sneakT[0]); r.get(a.sneakT[1]);
    r.get(a.shapeMask[0]); r.get(a.shapeMask[1]);
    r.bytes(&a.st, sizeof(a.st));
    r.bytes(&a.surv, sizeof(a.surv));
}
} // namespace

std::vector<uint8_t> VecEnv::saveState(int wi) const {
    if (wi < 0 || wi >= numWorlds_) throw std::out_of_range("world index out of range");
    const World& W = worlds_[wi];
    std::vector<uint8_t> b;
    b.reserve((size_t)WX * WY * WZ + 8192 * (size_t)K_ + 16384);
    Writer wr{b};
    wr.put(SNAP_MAGIC);
    wr.put(SNAP_VERSION);
    uint64_t wn = W.w.size();
    wr.put(wn);
    wr.bytes(W.w.data(), W.w.size());
    wr.put(W.seed); wr.put(W.goal); wr.put(W.got); wr.put(W.t); wr.put(W.episode);
    uint8_t rs = W.restart;
    wr.put(rs);
    wr.put(W.task); wr.put(W.taskP0); wr.put(W.taskP1); wr.put(W.maxT);
    wr.rng(W.rng);
    wr.rng(W.seedRng);
    uint32_t nc = (uint32_t)W.chests.size();
    wr.put(nc);
    for (const auto& kv : W.chests) {
        int key = kv.first;
        wr.put(key);
        wr.bytes(kv.second.count, sizeof(kv.second.count));
        wr.put(kv.second.depositMask);
    }
    uint32_t K = (uint32_t)K_;
    wr.put(K);
    for (int k = 0; k < K_; k++) writeAgent(wr, agents_[(size_t)wi * K_ + k]);
    return b;
}

void VecEnv::loadState(int wi, const uint8_t* data, size_t len, int goal, uint8_t* img, float* vec) {
    if (wi < 0 || wi >= numWorlds_) throw std::out_of_range("world index out of range");
    if (goal >= NUM_ITEMS) throw std::out_of_range("item index out of range");
    Reader r{data, len};
    uint32_t magic = 0, version = 0;
    r.get(magic); r.get(version);
    if (magic != SNAP_MAGIC || version != SNAP_VERSION) throw std::invalid_argument("not a world snapshot");
    World t;
    uint64_t wn = 0;
    r.get(wn);
    if (wn != (uint64_t)WX * WY * WZ) throw std::invalid_argument("world snapshot has wrong world size");
    t.w.resize(wn);
    r.bytes(t.w.data(), wn);
    r.get(t.seed); r.get(t.goal); r.get(t.got); r.get(t.t); r.get(t.episode);
    uint8_t rs = 0;
    r.get(rs);
    t.restart = rs;
    r.get(t.task); r.get(t.taskP0); r.get(t.taskP1); r.get(t.maxT);
    if (t.task < 0 || t.task >= NUM_TASKS) throw std::invalid_argument("world snapshot has a bad task");
    r.rng(t.rng);
    r.rng(t.seedRng);
    uint32_t nc = 0;
    r.get(nc);
    if (nc > (uint32_t)WX * WY * WZ) throw std::invalid_argument("world snapshot has a bad chest count");
    for (uint32_t c = 0; c < nc; c++) {
        int key = 0;
        r.get(key);
        Chest ch;
        r.bytes(ch.count, sizeof(ch.count));
        r.get(ch.depositMask);
        t.chests[key] = ch;
    }
    uint32_t K = 0;
    r.get(K);
    if (K != (uint32_t)K_) throw std::invalid_argument("world snapshot has a different agent count");
    std::vector<Agent> ta(K_);
    for (int k = 0; k < K_; k++) readAgent(r, ta[k]);
    if (r.off != len) throw std::invalid_argument("world snapshot has trailing bytes");

    if (goal >= 0) t.goal = goal;
    t.t = 0;
    t.restart = true;
    t.taskW = worlds_[wi].taskW;   // task weights are configuration, not episode state
    worlds_[wi] = std::move(t);
    for (int k = 0; k < K_; k++) agents_[(size_t)wi * K_ + k] = ta[k];
    const int V = vecDim();
    for (int k = 0; k < K_; k++) {
        const int i = wi * K_ + k;
        lastNov_[i] = 0.f;
        uint64_t seen = 0;
        writeAgentObs(i, img + (size_t)k * res_ * res_ * 2, vec + (size_t)k * V, &seen);
    }
}

void VecEnv::render(int i, int width, int height, uint8_t* rgb) const {
    const int w = i / K_, k = i % K_;
    renderRGB(worlds_[w], &agents_[(size_t)w * K_], K_, k, width, height, rgb);
}

void VecEnv::inventory(int i, int32_t* out) const {
    const Agent& a = agents_.at(i);
    for (int k = 0; k < NUM_ITEMS; k++) out[k] = a.inv[k];
}

void VecEnv::setInventory(int i, int item, int count) {
    if (item < 0 || item >= NUM_ITEMS) throw std::out_of_range("item index out of range");
    agents_.at(i).inv[item] = count;
}

Snapshot VecEnv::state(int i) const {
    const int w = i / K_;
    return snapshotOf(worlds_[w], agents_.at(i), mode_);
}

void VecEnv::setPose(int i, float yaw, float pitch) {
    Agent& a = agents_.at(i);
    a.yaw = yaw;
    a.pitch = std::clamp(pitch, -90.f, 90.f);
}

uint8_t VecEnv::block(int i, int x, int y, int z) const {
    return get(worlds_[i / K_], x, y, z);
}

Region VecEnv::region(int i, int radius) const {
    return regionOf(worlds_[i / K_], agents_.at(i), radius);
}

std::vector<ChestView> VecEnv::chests(int wi) const {
    if (wi < 0 || wi >= numWorlds_) throw std::out_of_range("world index out of range");
    std::vector<ChestView> out;
    for (const auto& kv : worlds_[wi].chests) {
        ChestView v;
        int key = kv.first;
        v.y = key / (WX * WZ);
        v.z = (key % (WX * WZ)) / WX;
        v.x = key % WX;
        std::copy(kv.second.count, kv.second.count + NUM_ITEMS, v.count);
        out.push_back(v);
    }
    return out;
}

} // namespace craftsim
