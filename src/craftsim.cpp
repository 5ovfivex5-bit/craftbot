// craftsim — a fast, batched, Minecraft-like voxel simulator for training RL agents.
//
// Mechanics follow vanilla Minecraft where it matters for learning:
//   * 20 ticks/s, player AABB 0.6 x 1.8, eye height 1.62, reach 4.5
//   * gravity 0.08 b/t^2, vertical drag 0.98, jump impulse 0.42 (apex ~1.25 blocks)
//   * ground slipperiness 0.6 * 0.91, walk accel 0.1 * 0.98  -> 4.317 m/s walking
//   * block breaking: progress/tick = toolSpeed / hardness / (canHarvest ? 30 : 100)
//   * tool tiers wood/stone/iron (speed 2/4/6), harvest levels for stone/iron/diamond
//   * vanilla recipes: log->4 planks, 2 planks->4 sticks, 4 planks->table, pickaxes = 3 head + 2 sticks
//   * fall damage = ceil(fallDistance - 3)
// Rewards follow MineRL ObtainDiamond (first time each item is obtained), log2-scaled.
//
// Exposed through a plain C ABI so Python (ctypes) can drive many envs at once.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int WX = 64, WY = 48, WZ = 64;
constexpr double PI = 3.14159265358979323846;

constexpr double GRAVITY = 0.08;
constexpr double DRAG = 0.98;
constexpr double JUMP_VEL = 0.42;
constexpr double GROUND_FRICTION = 0.6 * 0.91;
constexpr double AIR_FRICTION = 0.91;
constexpr double GROUND_ACCEL = 0.1 * 0.98;
constexpr double AIR_ACCEL = 0.02 * 0.98;
constexpr double HALF_W = 0.3, HEIGHT = 1.8, EYE = 1.62;
constexpr double REACH = 4.5;
constexpr float TURN_DEG = 15.f;
constexpr double VIEW_DIST = 24.0;
constexpr double FOV_DEG = 70.0;

enum Block : uint8_t {
    AIR, GRASS, DIRT, STONE, COBBLE, LOG, LEAVES, PLANKS, TABLE, FURNACE,
    COAL_ORE, IRON_ORE, DIAMOND_ORE, BEDROCK, NUM_BLOCKS
};

enum Item {
    I_LOG, I_PLANKS, I_STICK, I_TABLE, I_WOOD_PICK, I_COBBLE, I_STONE_PICK, I_FURNACE,
    I_COAL, I_IRON_ORE, I_IRON_INGOT, I_IRON_PICK, I_DIAMOND, I_DIRT, NUM_ITEMS
};

enum Action {
    A_NOOP, A_FWD, A_BACK, A_LEFT, A_RIGHT, A_JUMP, A_FWD_JUMP,
    A_TURN_L, A_TURN_R, A_LOOK_UP, A_LOOK_DOWN,
    A_ATTACK, A_PLACE_BLOCK, A_PLACE_TABLE, A_PLACE_FURNACE,
    A_CRAFT_PLANKS, A_CRAFT_STICKS, A_CRAFT_TABLE, A_CRAFT_WOOD_PICK,
    A_CRAFT_STONE_PICK, A_CRAFT_FURNACE, A_SMELT_IRON, A_CRAFT_IRON_PICK,
    NUM_ACTIONS
};

constexpr int VEC_DIM = NUM_ITEMS + 11;

struct BlockInfo { float hardness; int harvestTier; bool pickaxe; int drop; };

const BlockInfo BINFO[NUM_BLOCKS] = {
    {0.0f, 0, false, -1},          // AIR
    {0.6f, 0, false, I_DIRT},      // GRASS
    {0.5f, 0, false, I_DIRT},      // DIRT
    {1.5f, 1, true,  I_COBBLE},    // STONE
    {2.0f, 1, true,  I_COBBLE},    // COBBLE
    {2.0f, 0, false, I_LOG},       // LOG
    {0.2f, 0, false, -1},          // LEAVES
    {2.0f, 0, false, I_PLANKS},    // PLANKS
    {2.5f, 0, false, I_TABLE},     // TABLE
    {3.5f, 1, true,  I_FURNACE},   // FURNACE
    {3.0f, 1, true,  I_COAL},      // COAL_ORE
    {3.0f, 2, true,  I_IRON_ORE},  // IRON_ORE
    {3.0f, 3, true,  I_DIAMOND},   // DIAMOND_ORE
    {-1.f, 99, false, -1},         // BEDROCK (unbreakable)
};

const double TOOL_SPEED[4] = {1.0, 2.0, 4.0, 6.0};

// MineRL ObtainDiamond item values; reward = 1 + log2(value) the first time.
const int ITEM_VALUE[NUM_ITEMS] = {1, 2, 4, 4, 8, 16, 32, 32, 16, 64, 128, 256, 1024, 0};

struct Env {
    std::vector<uint8_t> w;
    double px, py, pz, vx, vy, vz;
    float yaw, pitch;
    bool onGround;
    double fallDist;
    int health;
    int inv[NUM_ITEMS];
    uint32_t got;
    int fuel;
    int mx, my, mz;
    double prog;
    int t;
    std::mt19937_64 rng;
};

struct Sim {
    int n, res, maxSteps, ticks;
    std::vector<Env> envs;
};

inline int cell(int x, int y, int z) { return (y * WZ + z) * WX + x; }

inline uint8_t get(const Env& e, int x, int y, int z) {
    if (y >= WY) return AIR;
    if (x < 0 || x >= WX || z < 0 || z >= WZ || y < 0) return BEDROCK;
    return e.w[cell(x, y, z)];
}

inline void put(Env& e, int x, int y, int z, uint8_t b) {
    if (x < 0 || x >= WX || z < 0 || z >= WZ || y < 0 || y >= WY) return;
    e.w[cell(x, y, z)] = b;
}

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

void vein(Env& e, uint8_t ore, int count, int size, int ymin, int ymax) {
    std::uniform_int_distribution<int> rx(0, WX - 1), rz(0, WZ - 1), ry(ymin, ymax), step(-1, 1);
    for (int c = 0; c < count; c++) {
        int x = rx(e.rng), y = ry(e.rng), z = rz(e.rng);
        for (int i = 0; i < size; i++) {
            if (get(e, x, y, z) == STONE) put(e, x, y, z, ore);
            x += step(e.rng); y += step(e.rng); z += step(e.rng);
        }
    }
}

void generate(Env& e, std::vector<int>& hmap) {
    uint32_t s = (uint32_t)e.rng();
    e.w.assign(WX * WY * WZ, AIR);
    hmap.assign(WX * WZ, 0);
    for (int z = 0; z < WZ; z++)
        for (int x = 0; x < WX; x++) {
            int h = 20 + (int)std::lround((fbm((float)x, (float)z, s) - 0.5f) * 24.f);
            h = std::clamp(h, 8, WY - 12);
            hmap[z * WX + x] = h;
            for (int y = 0; y <= h; y++) {
                uint8_t b = y == 0 ? BEDROCK : y < h - 3 ? STONE : y < h ? DIRT : GRASS;
                put(e, x, y, z, b);
            }
        }
    vein(e, COAL_ORE, 50, 8, 4, 36);
    vein(e, IRON_ORE, 40, 5, 2, 24);
    vein(e, DIAMOND_ORE, 14, 4, 1, 10);

    std::uniform_real_distribution<float> u01(0.f, 1.f);
    for (int z = 2; z < WZ - 2; z++)
        for (int x = 2; x < WX - 2; x++) {
            if (u01(e.rng) > 0.02f) continue;
            int h = hmap[z * WX + x];
            if (get(e, x, h, z) != GRASS) continue;
            int th = 4 + (int)(e.rng() % 3);
            if (h + th + 2 >= WY) continue;
            put(e, x, h, z, DIRT);
            int top = h + th;
            for (int dy = -2; dy <= 1; dy++) {
                int r = dy >= 0 ? 1 : 2;
                for (int dz = -r; dz <= r; dz++)
                    for (int dx = -r; dx <= r; dx++) {
                        if (r == 2 && std::abs(dx) == 2 && std::abs(dz) == 2 && u01(e.rng) < 0.5f) continue;
                        if (dy == 1 && std::abs(dx) + std::abs(dz) > 1) continue;
                        if (get(e, x + dx, top + dy, z + dz) == AIR) put(e, x + dx, top + dy, z + dz, LEAVES);
                    }
            }
            for (int i = 1; i <= th; i++) put(e, x, h + i, z, LOG);
        }
}

void resetEnv(Env& e, uint64_t seed) {
    e.rng.seed(seed);
    std::vector<int> hmap;
    generate(e, hmap);

    // Spawn on an open grass column near the middle of the map.
    std::uniform_int_distribution<int> off(-8, 8);
    int sx = WX / 2, sz = WZ / 2;
    for (int tries = 0; tries < 64; tries++) {
        int x = WX / 2 + off(e.rng), z = WZ / 2 + off(e.rng);
        int h = hmap[z * WX + x];
        if (get(e, x, h, z) == GRASS && get(e, x, h + 1, z) == AIR && get(e, x, h + 2, z) == AIR) {
            sx = x; sz = z;
            break;
        }
    }
    int y = WY - 1;
    while (y > 0 && get(e, sx, y, sz) == AIR) y--;
    e.px = sx + 0.5; e.py = y + 1.0; e.pz = sz + 0.5;
    e.vx = e.vy = e.vz = 0;
    e.yaw = TURN_DEG * (float)(e.rng() % 24);
    e.pitch = 0;
    e.onGround = false;
    e.fallDist = 0;
    e.health = 20;
    std::memset(e.inv, 0, sizeof(e.inv));
    e.got = 0;
    e.fuel = 0;
    e.mx = e.my = e.mz = -1;
    e.prog = 0;
    e.t = 0;
}

// ---------------------------------------------------------------- raycast

struct Hit {
    bool hit = false;
    int x = 0, y = 0, z = 0;    // block hit
    int px = 0, py = 0, pz = 0; // empty cell before it (placement target)
    int face = -1;              // axis crossed to enter the hit cell (0=x,1=y,2=z)
    double dist = 0;
    uint8_t b = AIR;
};

Hit raycast(const Env& e, double ox, double oy, double oz, double dx, double dy, double dz, double maxd) {
    Hit h;
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
        uint8_t b = get(e, x, y, z);
        if (b != AIR) {
            h.hit = true;
            h.x = x; h.y = y; h.z = z;
            h.px = lx; h.py = ly; h.pz = lz;
            h.face = face; h.dist = t; h.b = b;
            return h;
        }
        if (y >= WY && dy >= 0) return h;
        lx = x; ly = y; lz = z;
        if (tmx < tmy && tmx < tmz) { t = tmx; x += sx; tmx += tdx; face = 0; }
        else if (tmy < tmz)         { t = tmy; y += sy; tmy += tdy; face = 1; }
        else                        { t = tmz; z += sz; tmz += tdz; face = 2; }
        if (t > maxd) return h;
    }
}

inline void lookDir(float yaw, float pitch, double& dx, double& dy, double& dz) {
    double y = yaw * PI / 180.0, p = pitch * PI / 180.0;
    dx = -std::sin(y) * std::cos(p);
    dy = -std::sin(p);
    dz = std::cos(y) * std::cos(p);
}

inline Hit target(const Env& e) {
    double dx, dy, dz;
    lookDir(e.yaw, e.pitch, dx, dy, dz);
    return raycast(e, e.px, e.py + EYE, e.pz, dx, dy, dz, REACH);
}

// ---------------------------------------------------------------- physics

// Clip a move of `d` along axis `a` against solid blocks (Minecraft-style axis sweep).
double sweep(const Env& e, const double mn[3], const double mx[3], int a, double d) {
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
                if (get(e, p[0], p[1], p[2]) != AIR) return true;
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

void moveEntity(Env& e, double dx, double dy, double dz) {
    double mn[3] = {e.px - HALF_W, e.py, e.pz - HALF_W};
    double mx[3] = {e.px + HALF_W, e.py + HEIGHT, e.pz + HALF_W};
    double ay = sweep(e, mn, mx, 1, dy); mn[1] += ay; mx[1] += ay;
    double ax = sweep(e, mn, mx, 0, dx); mn[0] += ax; mx[0] += ax;
    double az = sweep(e, mn, mx, 2, dz);
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
}

int pickTier(const Env& e) {
    if (e.inv[I_IRON_PICK] > 0) return 3;
    if (e.inv[I_STONE_PICK] > 0) return 2;
    if (e.inv[I_WOOD_PICK] > 0) return 1;
    return 0;
}

void addItem(Env& e, int item, int n, float& r) {
    e.inv[item] += n;
    if (!(e.got & (1u << item)) && ITEM_VALUE[item] > 0) {
        e.got |= 1u << item;
        r += 1.0f + std::log2((float)ITEM_VALUE[item]);
    }
}

void attackTick(Env& e, float& r) {
    Hit h = target(e);
    if (!h.hit) { e.prog = 0; e.mx = -1; return; }
    if (h.x != e.mx || h.y != e.my || h.z != e.mz) {
        e.prog = 0;
        e.mx = h.x; e.my = h.y; e.mz = h.z;
    }
    const BlockInfo& bi = BINFO[h.b];
    if (bi.hardness < 0) return;
    int tier = pickTier(e);
    bool can = tier >= bi.harvestTier;
    double speed = (bi.pickaxe && tier > 0) ? TOOL_SPEED[tier] : 1.0;
    e.prog += bi.hardness == 0 ? 1.0 : speed / bi.hardness / (can ? 30.0 : 100.0);
    if (e.prog >= 1.0 - 1e-6) {
        put(e, h.x, h.y, h.z, AIR);
        if (can && bi.drop >= 0) addItem(e, bi.drop, 1, r);
        e.prog = 0;
        e.mx = -1;
    }
}

void tick(Env& e, double fwd, double strafe, bool jump, bool attack, float& r) {
    if (jump && e.onGround) e.vy = JUMP_VEL;

    double len = std::sqrt(fwd * fwd + strafe * strafe);
    if (len > 1) { fwd /= len; strafe /= len; }
    double accel = e.onGround ? GROUND_ACCEL : AIR_ACCEL;
    double yr = e.yaw * PI / 180.0;
    double fx = -std::sin(yr), fz = std::cos(yr);   // forward
    double rx = -std::cos(yr), rz = -std::sin(yr);  // right
    e.vx += (fwd * fx + strafe * rx) * accel;
    e.vz += (fwd * fz + strafe * rz) * accel;

    double fric = e.onGround ? GROUND_FRICTION : AIR_FRICTION;
    moveEntity(e, e.vx, e.vy, e.vz);
    e.vy = (e.vy - GRAVITY) * DRAG;
    e.vx *= fric;
    e.vz *= fric;

    if (attack) attackTick(e, r);
    else { e.prog = 0; e.mx = -1; }
}

// ---------------------------------------------------------------- interaction

bool nearBlock(const Env& e, uint8_t b) {
    int cx = (int)std::floor(e.px), cy = (int)std::floor(e.py), cz = (int)std::floor(e.pz);
    for (int dy = -3; dy <= 4; dy++)
        for (int dz = -3; dz <= 3; dz++)
            for (int dx = -3; dx <= 3; dx++)
                if (get(e, cx + dx, cy + dy, cz + dz) == b) return true;
    return false;
}

bool placeBlock(Env& e, uint8_t b) {
    Hit h = target(e);
    if (!h.hit || h.face < 0) return false;
    int x = h.px, y = h.py, z = h.pz;
    if (x < 0 || x >= WX || z < 0 || z >= WZ || y < 1 || y >= WY) return false;
    if (get(e, x, y, z) != AIR) return false;
    if (x < e.px + HALF_W && x + 1 > e.px - HALF_W && y < e.py + HEIGHT && y + 1 > e.py &&
        z < e.pz + HALF_W && z + 1 > e.pz - HALF_W)
        return false;
    put(e, x, y, z, b);
    e.prog = 0;
    e.mx = -1;
    return true;
}

// 3 x head material + 2 sticks at a crafting table.
void craftPick(Env& e, int head, int out, float& r) {
    if (nearBlock(e, TABLE) && e.inv[head] >= 3 && e.inv[I_STICK] >= 2) {
        e.inv[head] -= 3;
        e.inv[I_STICK] -= 2;
        addItem(e, out, 1, r);
    }
}

void interact(Env& e, int a, float& r) {
    switch (a) {
    case A_PLACE_BLOCK: {
        int item = e.inv[I_COBBLE] ? I_COBBLE : e.inv[I_DIRT] ? I_DIRT : e.inv[I_PLANKS] ? I_PLANKS : -1;
        uint8_t b = item == I_COBBLE ? COBBLE : item == I_DIRT ? DIRT : PLANKS;
        if (item >= 0 && placeBlock(e, b)) e.inv[item]--;
        break;
    }
    case A_PLACE_TABLE:
        if (e.inv[I_TABLE] && placeBlock(e, TABLE)) e.inv[I_TABLE]--;
        break;
    case A_PLACE_FURNACE:
        if (e.inv[I_FURNACE] && placeBlock(e, FURNACE)) e.inv[I_FURNACE]--;
        break;
    case A_CRAFT_PLANKS:
        if (e.inv[I_LOG] >= 1) { e.inv[I_LOG]--; addItem(e, I_PLANKS, 4, r); }
        break;
    case A_CRAFT_STICKS:
        if (e.inv[I_PLANKS] >= 2) { e.inv[I_PLANKS] -= 2; addItem(e, I_STICK, 4, r); }
        break;
    case A_CRAFT_TABLE:
        if (e.inv[I_PLANKS] >= 4) { e.inv[I_PLANKS] -= 4; addItem(e, I_TABLE, 1, r); }
        break;
    case A_CRAFT_WOOD_PICK:  craftPick(e, I_PLANKS, I_WOOD_PICK, r); break;
    case A_CRAFT_STONE_PICK: craftPick(e, I_COBBLE, I_STONE_PICK, r); break;
    case A_CRAFT_IRON_PICK:  craftPick(e, I_IRON_INGOT, I_IRON_PICK, r); break;
    case A_CRAFT_FURNACE:
        if (nearBlock(e, TABLE) && e.inv[I_COBBLE] >= 8) { e.inv[I_COBBLE] -= 8; addItem(e, I_FURNACE, 1, r); }
        break;
    case A_SMELT_IRON:
        // Coal smelts 8 items, a plank smelts 1 (vanilla: 1.5).
        if (nearBlock(e, FURNACE) && e.inv[I_IRON_ORE] >= 1) {
            if (e.fuel == 0 && e.inv[I_COAL]) { e.inv[I_COAL]--; e.fuel = 8; }
            else if (e.fuel == 0 && e.inv[I_PLANKS]) { e.inv[I_PLANKS]--; e.fuel = 1; }
            if (e.fuel > 0) {
                e.fuel--;
                e.inv[I_IRON_ORE]--;
                addItem(e, I_IRON_INGOT, 1, r);
            }
        }
        break;
    }
}

float stepEnv(const Sim& s, Env& e, int a, bool& done) {
    float r = 0;
    double fwd = 0, strafe = 0;
    bool jump = false, attack = false;
    switch (a) {
    case A_FWD: fwd = 1; break;
    case A_BACK: fwd = -1; break;
    case A_LEFT: strafe = -1; break;
    case A_RIGHT: strafe = 1; break;
    case A_JUMP: jump = true; break;
    case A_FWD_JUMP: fwd = 1; jump = true; break;
    case A_TURN_L: e.yaw = std::fmod(e.yaw - TURN_DEG + 360.f, 360.f); break;
    case A_TURN_R: e.yaw = std::fmod(e.yaw + TURN_DEG, 360.f); break;
    case A_LOOK_UP: e.pitch = std::max(-90.f, e.pitch - TURN_DEG); break;
    case A_LOOK_DOWN: e.pitch = std::min(90.f, e.pitch + TURN_DEG); break;
    case A_ATTACK: attack = true; break;
    default: interact(e, a, r); break;
    }
    for (int k = 0; k < s.ticks && e.health > 0; k++) tick(e, fwd, strafe, jump, attack, r);
    e.t++;
    done = e.health <= 0 || e.t >= s.maxSteps;
    return r;
}

// ---------------------------------------------------------------- observations

struct Camera { double ox, oy, oz, f[3], r[3], u[3], tanH; };

Camera camera(const Env& e) {
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

inline Hit castPixel(const Env& e, const Camera& c, int i, int j, int W, int H, double d[3]) {
    double aspect = (double)W / H;
    double u = ((i + 0.5) / W * 2 - 1) * c.tanH * aspect;
    double v = (1 - (j + 0.5) / H * 2) * c.tanH;
    for (int k = 0; k < 3; k++) d[k] = c.f[k] + c.r[k] * u + c.u[k] * v;
    double n = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    for (int k = 0; k < 3; k++) d[k] /= n;
    return raycast(e, c.ox, c.oy, c.oz, d[0], d[1], d[2], VIEW_DIST);
}

// Per pixel: [block id (0 = sky), depth 0..255].
void writeObs(const Sim& s, const Env& e, uint8_t* img, float* vec) {
    Camera c = camera(e);
    int R = s.res;
    double d[3];
    for (int j = 0; j < R; j++)
        for (int i = 0; i < R; i++) {
            Hit h = castPixel(e, c, i, j, R, R, d);
            uint8_t* p = img + (j * R + i) * 2;
            p[0] = h.hit ? h.b : 0;
            p[1] = h.hit ? (uint8_t)std::min(255.0, h.dist / VIEW_DIST * 255.0) : 255;
        }

    for (int i = 0; i < NUM_ITEMS; i++) vec[i] = std::log1p((float)e.inv[i]) / std::log1p(64.f);
    int k = NUM_ITEMS;
    double yr = e.yaw * PI / 180.0;
    vec[k++] = e.health / 20.f;
    vec[k++] = e.onGround ? 1.f : 0.f;
    vec[k++] = e.pitch / 90.f;
    vec[k++] = (float)std::sin(yr);
    vec[k++] = (float)std::cos(yr);
    vec[k++] = (float)e.prog;
    vec[k++] = (float)(e.py / WY);
    vec[k++] = nearBlock(e, TABLE) ? 1.f : 0.f;
    vec[k++] = nearBlock(e, FURNACE) ? 1.f : 0.f;
    vec[k++] = std::min(e.fuel, 8) / 8.f;
    vec[k++] = (float)e.t / s.maxSteps;
}

const uint8_t COLORS[NUM_BLOCKS][3] = {
    {135, 190, 255}, {95, 159, 53}, {134, 96, 67}, {125, 125, 125}, {105, 105, 105},
    {102, 81, 51},   {58, 120, 38}, {180, 144, 90}, {150, 98, 56}, {80, 80, 80},
    {125, 125, 125}, {125, 125, 125}, {125, 125, 125}, {45, 45, 45},
};
const uint8_t SPECKLE[NUM_BLOCKS][3] = {
    {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
    {0, 0, 0}, {0, 0, 0}, {25, 25, 25}, {216, 175, 147}, {90, 230, 230}, {0, 0, 0},
};

inline double frac(double v) { return v - std::floor(v); }

void renderRGB(const Env& e, int W, int H, uint8_t* out) {
    Camera c = camera(e);
    double d[3];
    for (int j = 0; j < H; j++)
        for (int i = 0; i < W; i++) {
            Hit h = castPixel(e, c, i, j, W, H, d);
            uint8_t* p = out + (j * W + i) * 3;
            if (!h.hit) { std::memcpy(p, COLORS[AIR], 3); continue; }

            double hx = c.ox + d[0] * h.dist, hy = c.oy + d[1] * h.dist, hz = c.oz + d[2] * h.dist;
            double u, v;
            if (h.face == 0)      { u = frac(hz); v = frac(hy); }
            else if (h.face == 1) { u = frac(hx); v = frac(hz); }
            else                  { u = frac(hx); v = frac(hy); }
            int tu = std::min(7, (int)(u * 8)), tv = std::min(7, (int)(v * 8));
            uint32_t th = hash3(h.x * 8 + tu, h.y * 8 + tv, h.z * 8 + tu * 3 + tv, 1337u + h.face);
            float tex = 0.88f + 0.24f * (th & 255) / 255.f;

            const uint8_t* col = COLORS[h.b];
            if (h.b == GRASS && h.face != 1 && v < 0.8) col = COLORS[DIRT];
            if (SPECKLE[h.b][0] && ((th >> 8) & 3) == 0) col = SPECKLE[h.b];

            float shade = h.face == 1 ? (d[1] < 0 ? 1.0f : 0.5f) : h.face == 0 ? 0.8f : h.face == 2 ? 0.68f : 1.0f;
            float fog = (float)std::min(1.0, (h.dist / VIEW_DIST) * (h.dist / VIEW_DIST));
            for (int k = 0; k < 3; k++) {
                float cv = col[k] * shade * tex;
                cv = cv * (1 - fog) + COLORS[AIR][k] * fog;
                p[k] = (uint8_t)std::clamp(cv, 0.f, 255.f);
            }
        }
    // crosshair
    for (int k = -3; k <= 3; k++) {
        uint8_t* a = out + ((H / 2) * W + W / 2 + k) * 3;
        uint8_t* b = out + ((H / 2 + k) * W + W / 2) * 3;
        a[0] = a[1] = a[2] = 255;
        b[0] = b[1] = b[2] = 255;
    }
}

} // namespace

// ---------------------------------------------------------------- C API

extern "C" {

int cs_num_actions() { return NUM_ACTIONS; }
int cs_num_items() { return NUM_ITEMS; }
int cs_num_blocks() { return NUM_BLOCKS; }
int cs_vec_dim() { return VEC_DIM; }

void* cs_create(int n, uint64_t seed, int res, int max_steps, int ticks) {
    Sim* s = new Sim;
    s->n = n; s->res = res; s->maxSteps = max_steps; s->ticks = ticks;
    s->envs.resize(n);
    std::mt19937_64 seeder(seed);
    for (auto& e : s->envs) e.rng.seed(seeder());
    return s;
}

void cs_destroy(void* h) { delete (Sim*)h; }

void cs_reset(void* h, uint8_t* img, float* vec) {
    Sim& s = *(Sim*)h;
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < s.n; i++) {
        Env& e = s.envs[i];
        resetEnv(e, e.rng());
        writeObs(s, e, img + (size_t)i * s.res * s.res * 2, vec + (size_t)i * VEC_DIM);
    }
}

// Auto-resets finished envs; ach[i] holds the achievement bitmask of the step's episode.
void cs_step(void* h, const int32_t* act, uint8_t* img, float* vec, float* rew, uint8_t* done, uint32_t* ach) {
    Sim& s = *(Sim*)h;
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < s.n; i++) {
        Env& e = s.envs[i];
        bool d = false;
        int a = std::clamp((int)act[i], 0, NUM_ACTIONS - 1);
        rew[i] = stepEnv(s, e, a, d);
        done[i] = d;
        ach[i] = e.got;
        if (d) resetEnv(e, e.rng());
        writeObs(s, e, img + (size_t)i * s.res * s.res * 2, vec + (size_t)i * VEC_DIM);
    }
}

void cs_render(void* h, int i, int W, int H, uint8_t* rgb) { renderRGB(((Sim*)h)->envs[i], W, H, rgb); }

void cs_get_inventory(void* h, int i, int32_t* out) {
    const Env& e = ((Sim*)h)->envs[i];
    for (int k = 0; k < NUM_ITEMS; k++) out[k] = e.inv[k];
}

void cs_set_inventory(void* h, int i, int item, int count) {
    if (item >= 0 && item < NUM_ITEMS) ((Sim*)h)->envs[i].inv[item] = count;
}

// px py pz vx vy vz yaw pitch onGround health t
void cs_get_state(void* h, int i, double* out) {
    const Env& e = ((Sim*)h)->envs[i];
    double v[11] = {e.px, e.py, e.pz, e.vx, e.vy, e.vz, e.yaw, e.pitch, (double)e.onGround, (double)e.health, (double)e.t};
    std::memcpy(out, v, sizeof(v));
}

void cs_set_pose(void* h, int i, float yaw, float pitch) {
    Env& e = ((Sim*)h)->envs[i];
    e.yaw = yaw;
    e.pitch = std::clamp(pitch, -90.f, 90.f);
}

int cs_get_block(void* h, int i, int x, int y, int z) { return get(((Sim*)h)->envs[i], x, y, z); }

} // extern "C"
