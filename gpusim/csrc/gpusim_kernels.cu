// gpusim: CUDA port of src/sim.cpp (VecEnv). One thread per env for physics/mining/crafting,
// one thread per (env, pixel) for the DDA observation, one block per env for pool resets.
// Rules mirror sim.cpp exactly (double precision physics, float DDA for the image rays).
// Worlds are generated on the CPU (sim.cpp generate()) and loaded from a pool; see gpusim.py.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace gs {

constexpr int WX = 64, WY = 64, WZ = 64;
constexpr int WORLD = WX * WY * WZ;
constexpr int SEA_LEVEL = 30;
constexpr int DEEPSLATE_Y = 8;
constexpr int NUM_BLOCKS = 18;
constexpr int NUM_ITEMS = 14;
constexpr int NUM_ACTIONS = 23;
constexpr int VEC_DIM = 39;
constexpr int GOAL_OFFSET = 25;
constexpr uint32_t GOAL_BIT = 1u << 31;
constexpr int MAX_ATTACK_TICKS = 80;

enum : uint8_t {
    AIR, GRASS, DIRT, STONE, COBBLE, LOG, LEAVES, PLANKS, TABLE, FURNACE,
    COAL_ORE, IRON_ORE, DIAMOND_ORE, BEDROCK, SAND, GRAVEL, WATER, DEEPSLATE
};
enum {
    I_LOG, I_PLANKS, I_STICK, I_TABLE, I_WOOD_PICK, I_COBBLE, I_STONE_PICK, I_FURNACE,
    I_COAL, I_IRON_ORE, I_IRON_INGOT, I_IRON_PICK, I_DIAMOND, I_DIRT
};
enum {
    A_NOOP, A_FWD, A_BACK, A_LEFT, A_RIGHT, A_JUMP, A_FWD_JUMP, A_TURN_L, A_TURN_R, A_LOOK_UP,
    A_LOOK_DOWN, A_ATTACK, A_PLACE_BLOCK, A_PLACE_TABLE, A_PLACE_FURNACE, A_CRAFT_PLANKS,
    A_CRAFT_STICKS, A_CRAFT_TABLE, A_CRAFT_WOOD_PICK, A_CRAFT_STONE_PICK, A_CRAFT_FURNACE,
    A_SMELT_IRON, A_CRAFT_IRON_PICK
};

constexpr double PI = 3.14159265358979323846;
constexpr double GRAVITY = 0.08;
constexpr double DRAG = 0.98;
constexpr double JUMP_VEL = 0.42;
constexpr double GROUND_FRICTION = 0.6 * 0.91;
constexpr double AIR_FRICTION = 0.91;
constexpr double GROUND_ACCEL = 0.1 * 0.98;
constexpr double AIR_ACCEL = 0.02 * 0.98;
constexpr double WATER_ACCEL = 0.02;
constexpr double WATER_DRAG = 0.8;
constexpr double WATER_JUMP = 0.04;
constexpr double HALF_W = 0.3, HEIGHT = 1.8, EYE = 1.62;
constexpr double REACH = 4.5;
constexpr float TURN_DEG = 15.f;
constexpr float VIEW_DIST = 24.f;
constexpr double FOV_DEG = 70.0;

// Block tables (sim.cpp BINFO): hardness, harvest tier, pickaxe flag, drop item (-1 none).
__device__ const float HARD[NUM_BLOCKS] = {
    0.f, 0.6f, 0.5f, 1.5f, 2.0f, 2.0f, 0.2f, 2.0f, 2.5f, 3.5f, 3.0f, 3.0f, 3.0f, -1.f, 0.5f, 0.6f, -1.f, 3.0f};
__device__ const int TIER[NUM_BLOCKS] = {0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 2, 3, 99, 0, 0, 99, 1};
__device__ const int PICK[NUM_BLOCKS] = {0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1};
__device__ const int DROP[NUM_BLOCKS] = {-1, I_DIRT, I_DIRT, I_COBBLE, I_COBBLE, I_LOG, -1, I_PLANKS,
                                         I_TABLE, I_FURNACE, I_COAL, I_IRON_ORE, I_DIAMOND, -1, -1, -1, -1, I_COBBLE};
__device__ const double TOOL_SPEED[4] = {1.0, 2.0, 4.0, 6.0};
__device__ const int ITEM_VALUE[NUM_ITEMS] = {1, 2, 4, 4, 8, 16, 32, 32, 16, 64, 128, 256, 1024, 0};

// Per-env state, one POD struct per env (AoS). Layout is read by name in get_env/set_env.
struct GEnv {
    double px, py, pz, vx, vy, vz;
    double fallDist;
    double prog;
    float yaw, pitch;
    int32_t health;
    int32_t onGround;
    int32_t inv[NUM_ITEMS];
    uint32_t got, kitGot;
    int32_t fuel, mx, my, mz;
    int32_t t, ticks, episode, lastAction;
    uint64_t seed;
    int32_t goal, kit, restart, bestDepthStep;
    uint32_t seenBlocks, seenObs;
    uint64_t visited[64];
    uint64_t rng;
    float lastNov;
    int32_t pad;
};

struct Hit {
    bool hit;
    int x, y, z, px, py, pz, face;
    double dist;
    uint8_t b;
};

__device__ __forceinline__ uint8_t wget(const uint8_t* w, int x, int y, int z) {
    if (y >= WY) return AIR;
    if (x < 0 || x >= WX || z < 0 || z >= WZ || y < 0) return BEDROCK;
    return w[(y * WZ + z) * WX + x];
}
__device__ __forceinline__ void wput(uint8_t* w, int x, int y, int z, uint8_t b) {
    if (x < 0 || x >= WX || z < 0 || z >= WZ || y < 0 || y >= WY) return;
    w[(y * WZ + z) * WX + x] = b;
}
__device__ __forceinline__ bool solid(uint8_t b) { return b != AIR && b != WATER; }

__device__ __forceinline__ uint64_t sm_next(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
__device__ __forceinline__ double sm_unit(uint64_t& s) {
    return (double)(sm_next(s) >> 11) * (1.0 / 9007199254740992.0);
}

__device__ __forceinline__ int depthStep(int fy) { return fy < SEA_LEVEL ? (SEA_LEVEL - fy) / 2 : 0; }
__device__ __forceinline__ int cellOf(int fx, int fy, int fz) {
    fx = min(max(fx, 0), WX - 1); fy = min(max(fy, 0), WY - 1); fz = min(max(fz, 0), WZ - 1);
    return ((fy >> 2) * 16 + (fz >> 2)) * 16 + (fx >> 2);
}

__device__ __forceinline__ float fl(float v) { return floorf(v); }
__device__ __forceinline__ double fl(double v) { return floor(v); }
__device__ __forceinline__ float ab(float v) { return fabsf(v); }
__device__ __forceinline__ double ab(double v) { return fabs(v); }

// DDA raycast (mirrors sim.cpp raycast). T = double for physics, float for image rays.
template <typename T>
__device__ Hit raycast(const uint8_t* w, T ox, T oy, T oz, T dx, T dy, T dz, T maxd, bool passWater) {
    Hit h;
    h.hit = false; h.x = h.y = h.z = h.px = h.py = h.pz = 0; h.face = -1; h.dist = 0; h.b = AIR;
    int x = (int)fl(ox), y = (int)fl(oy), z = (int)fl(oz);
    int sx = dx > (T)0 ? 1 : -1, sy = dy > (T)0 ? 1 : -1, sz = dz > (T)0 ? 1 : -1;
    const T BIG = (T)1e30;
    T tdx = dx != (T)0 ? ab((T)1 / dx) : BIG;
    T tdy = dy != (T)0 ? ab((T)1 / dy) : BIG;
    T tdz = dz != (T)0 ? ab((T)1 / dz) : BIG;
    T tmx = dx != (T)0 ? (dx > (T)0 ? (T)(x + 1) - ox : ox - (T)x) * tdx : BIG;
    T tmy = dy != (T)0 ? (dy > (T)0 ? (T)(y + 1) - oy : oy - (T)y) * tdy : BIG;
    T tmz = dz != (T)0 ? (dz > (T)0 ? (T)(z + 1) - oz : oz - (T)z) * tdz : BIG;
    int lx = x, ly = y, lz = z, face = -1;
    T t = (T)0;
    for (;;) {
        uint8_t b = wget(w, x, y, z);
        if (b != AIR && !(passWater && b == WATER)) {
            h.hit = true;
            h.x = x; h.y = y; h.z = z;
            h.px = lx; h.py = ly; h.pz = lz;
            h.face = face; h.dist = (double)t; h.b = b;
            return h;
        }
        if (y >= WY && dy >= (T)0) return h;
        lx = x; ly = y; lz = z;
        if (tmx < tmy && tmx < tmz)      { t = tmx; x += sx; tmx += tdx; face = 0; }
        else if (tmy < tmz)              { t = tmy; y += sy; tmy += tdy; face = 1; }
        else                             { t = tmz; z += sz; tmz += tdz; face = 2; }
        if (t > maxd) return h;
    }
}

__device__ __forceinline__ void lookDir(float yaw, float pitch, double& dx, double& dy, double& dz) {
    double y = (double)yaw * PI / 180.0, p = (double)pitch * PI / 180.0;
    dx = -sin(y) * cos(p);
    dy = -sin(p);
    dz = cos(y) * cos(p);
}

__device__ __forceinline__ bool eyeInWater(const uint8_t* w, const GEnv* e) {
    return wget(w, (int)floor(e->px), (int)floor(e->py + EYE), (int)floor(e->pz)) == WATER;
}

__device__ __forceinline__ Hit target(const uint8_t* w, const GEnv* e) {
    double dx, dy, dz;
    lookDir(e->yaw, e->pitch, dx, dy, dz);
    return raycast<double>(w, e->px, e->py + EYE, e->pz, dx, dy, dz, REACH, eyeInWater(w, e));
}

__device__ double sweep(const uint8_t* w, const double mn[3], const double mx[3], int a, double d) {
    if (d == 0) return 0;
    int o1 = (a + 1) % 3, o2 = (a + 2) % 3;
    int lo1 = (int)floor(mn[o1] + 1e-9), hi1 = (int)floor(mx[o1] - 1e-9);
    int lo2 = (int)floor(mn[o2] + 1e-9), hi2 = (int)floor(mx[o2] - 1e-9);
    if (d > 0) {
        int c0 = (int)floor(mx[a]), c1 = (int)floor(mx[a] + d);
        for (int c = c0; c <= c1; c++) {
            bool s = false;
            for (int i = lo1; i <= hi1 && !s; i++)
                for (int j = lo2; j <= hi2 && !s; j++) {
                    int p[3];
                    p[a] = c; p[o1] = i; p[o2] = j;
                    if (solid(wget(w, p[0], p[1], p[2]))) s = true;
                }
            if (s) return fmax(0.0, c - mx[a] - 1e-7);
        }
    } else {
        int c0 = (int)floor(mn[a]), c1 = (int)floor(mn[a] + d);
        for (int c = c0; c >= c1; c--) {
            bool s = false;
            for (int i = lo1; i <= hi1 && !s; i++)
                for (int j = lo2; j <= hi2 && !s; j++) {
                    int p[3];
                    p[a] = c; p[o1] = i; p[o2] = j;
                    if (solid(wget(w, p[0], p[1], p[2]))) s = true;
                }
            if (s) return fmin(0.0, c + 1 - mn[a] + 1e-7);
        }
    }
    return d;
}

__device__ bool freeAbove(const uint8_t* w, const GEnv* e, double dy) {
    int x0 = (int)floor(e->px - HALF_W + 1e-9), x1 = (int)floor(e->px + HALF_W - 1e-9);
    int z0 = (int)floor(e->pz - HALF_W + 1e-9), z1 = (int)floor(e->pz + HALF_W - 1e-9);
    int y0 = (int)floor(e->py + dy + 1e-9), y1 = (int)floor(e->py + dy + HEIGHT - 1e-9);
    for (int y = y0; y <= y1; y++)
        for (int z = z0; z <= z1; z++)
            for (int x = x0; x <= x1; x++)
                if (solid(wget(w, x, y, z))) return false;
    return true;
}

__device__ bool inWater(const uint8_t* w, const GEnv* e) {
    int x0 = (int)floor(e->px - HALF_W + 1e-9), x1 = (int)floor(e->px + HALF_W - 1e-9);
    int z0 = (int)floor(e->pz - HALF_W + 1e-9), z1 = (int)floor(e->pz + HALF_W - 1e-9);
    int y0 = (int)floor(e->py + 1e-9), y1 = (int)floor(e->py + HEIGHT - 1e-9);
    for (int y = y0; y <= y1; y++)
        for (int z = z0; z <= z1; z++)
            for (int x = x0; x <= x1; x++)
                if (wget(w, x, y, z) == WATER) return true;
    return false;
}

__device__ bool moveEntity(const uint8_t* w, GEnv* e, double dx, double dy, double dz) {
    double mn[3] = {e->px - HALF_W, e->py, e->pz - HALF_W};
    double mx[3] = {e->px + HALF_W, e->py + HEIGHT, e->pz + HALF_W};
    double ay = sweep(w, mn, mx, 1, dy); mn[1] += ay; mx[1] += ay;
    double ax = sweep(w, mn, mx, 0, dx); mn[0] += ax; mx[0] += ax;
    double az = sweep(w, mn, mx, 2, dz);
    e->px += ax; e->py += ay; e->pz += az;

    bool hitY = ay != dy;
    e->onGround = (hitY && dy < 0) ? 1 : 0;
    if (hitY) e->vy = 0;
    if (ax != dx) e->vx = 0;
    if (az != dz) e->vz = 0;

    if (e->onGround) {
        if (e->fallDist > 3.0) e->health -= (int)ceil(e->fallDist - 3.0);
        e->fallDist = 0;
    } else if (ay < 0) {
        e->fallDist -= ay;
    }
    return ax != dx || az != dz;
}

__device__ int pickTier(const GEnv* e) {
    if (e->inv[I_IRON_PICK] > 0) return 3;
    if (e->inv[I_STONE_PICK] > 0) return 2;
    if (e->inv[I_WOOD_PICK] > 0) return 1;
    return 0;
}

__device__ void addItem(GEnv* e, int item, int n, float& r) {
    e->inv[item] += n;
    const uint32_t bit = 1u << item;
    if (!((e->got | e->kitGot) & bit) && ITEM_VALUE[item] > 0) {
        e->got |= bit;
        r += 0.1f * (1.0f + log2f((float)ITEM_VALUE[item]));
    }
}

__device__ void attackTick(const uint8_t* w, GEnv* e, float& r) {
    Hit h = target(w, e);
    if (!h.hit) { e->prog = 0; e->mx = -1; return; }
    if (h.x != e->mx || h.y != e->my || h.z != e->mz) {
        e->prog = 0;
        e->mx = h.x; e->my = h.y; e->mz = h.z;
    }
    const float hard = HARD[h.b];
    if (hard < 0) return;
    int tier = pickTier(e);
    bool can = tier >= TIER[h.b];
    double speed = (PICK[h.b] && tier > 0) ? TOOL_SPEED[tier] : 1.0;
    if (!e->onGround) speed /= 5.0;
    if (eyeInWater(w, e)) speed /= 5.0;
    e->prog += hard == 0.f ? 1.0 : speed / (double)hard / (can ? 30.0 : 100.0);
    if (e->prog >= 1.0 - 1e-6) {
        wput((uint8_t*)w, h.x, h.y, h.z, AIR);
        if (can && DROP[h.b] >= 0) addItem(e, DROP[h.b], 1, r);
        e->prog = 0;
        e->mx = -1;
    }
}

__device__ void tick(uint8_t* w, GEnv* e, double fwd, double strafe, bool jump, bool attack, float& r) {
    double len = sqrt(fwd * fwd + strafe * strafe);
    if (len > 1) { fwd /= len; strafe /= len; }
    double yr = (double)e->yaw * PI / 180.0;
    double fx = -sin(yr), fz = cos(yr);
    double rx = -cos(yr), rz = -sin(yr);

    if (inWater(w, e)) {
        if (jump) e->vy += WATER_JUMP;
        e->vx += (fwd * fx + strafe * rx) * WATER_ACCEL;
        e->vz += (fwd * fz + strafe * rz) * WATER_ACCEL;
        bool blocked = moveEntity(w, e, e->vx, e->vy, e->vz);
        e->vx *= WATER_DRAG;
        e->vy *= WATER_DRAG;
        e->vz *= WATER_DRAG;
        e->vy -= GRAVITY / 16.0;
        if (blocked && freeAbove(w, e, 0.6)) e->vy = 0.3;
        e->fallDist = 0;
    } else {
        if (jump && e->onGround) e->vy = JUMP_VEL;
        double accel = e->onGround ? GROUND_ACCEL : AIR_ACCEL;
        e->vx += (fwd * fx + strafe * rx) * accel;
        e->vz += (fwd * fz + strafe * rz) * accel;
        double fric = e->onGround ? GROUND_FRICTION : AIR_FRICTION;
        moveEntity(w, e, e->vx, e->vy, e->vz);
        e->vy = (e->vy - GRAVITY) * DRAG;
        e->vx *= fric;
        e->vz *= fric;
    }

    if (attack) attackTick(w, e, r);
    else { e->prog = 0; e->mx = -1; }
}

__device__ bool nearBlock(const uint8_t* w, const GEnv* e, uint8_t b) {
    int cx = (int)floor(e->px), cy = (int)floor(e->py), cz = (int)floor(e->pz);
    for (int dy = -3; dy <= 4; dy++)
        for (int dz = -3; dz <= 3; dz++)
            for (int dx = -3; dx <= 3; dx++)
                if (wget(w, cx + dx, cy + dy, cz + dz) == b) return true;
    return false;
}

__device__ bool placeBlock(uint8_t* w, GEnv* e, uint8_t b) {
    Hit h = target(w, e);
    if (!h.hit || h.face < 0) return false;
    int x = h.px, y = h.py, z = h.pz;
    if (x < 0 || x >= WX || z < 0 || z >= WZ || y < 1 || y >= WY) return false;
    if (wget(w, x, y, z) != AIR) return false;
    if (x < e->px + HALF_W && x + 1 > e->px - HALF_W && y < e->py + HEIGHT && y + 1 > e->py &&
        z < e->pz + HALF_W && z + 1 > e->pz - HALF_W)
        return false;
    wput(w, x, y, z, b);
    e->prog = 0;
    e->mx = -1;
    return true;
}

__device__ void craftPick(const uint8_t* w, GEnv* e, int head, int out, float& r) {
    if (nearBlock(w, e, TABLE) && e->inv[head] >= 3 && e->inv[I_STICK] >= 2) {
        e->inv[head] -= 3;
        e->inv[I_STICK] -= 2;
        addItem(e, out, 1, r);
    }
}

__device__ void interact(uint8_t* w, GEnv* e, int a, float& r) {
    switch (a) {
    case A_PLACE_BLOCK: {
        int item = e->inv[I_COBBLE] ? I_COBBLE : e->inv[I_DIRT] ? I_DIRT : e->inv[I_PLANKS] ? I_PLANKS : -1;
        uint8_t b = item == I_COBBLE ? COBBLE : item == I_DIRT ? DIRT : PLANKS;
        if (item >= 0 && placeBlock(w, e, b)) e->inv[item]--;
        break;
    }
    case A_PLACE_TABLE:
        if (e->inv[I_TABLE] && placeBlock(w, e, TABLE)) e->inv[I_TABLE]--;
        break;
    case A_PLACE_FURNACE:
        if (e->inv[I_FURNACE] && placeBlock(w, e, FURNACE)) e->inv[I_FURNACE]--;
        break;
    case A_CRAFT_PLANKS:
        if (e->inv[I_LOG] >= 1) { e->inv[I_LOG]--; addItem(e, I_PLANKS, 4, r); }
        break;
    case A_CRAFT_STICKS:
        if (e->inv[I_PLANKS] >= 2) { e->inv[I_PLANKS] -= 2; addItem(e, I_STICK, 4, r); }
        break;
    case A_CRAFT_TABLE:
        if (e->inv[I_PLANKS] >= 4) { e->inv[I_PLANKS] -= 4; addItem(e, I_TABLE, 1, r); }
        break;
    case A_CRAFT_WOOD_PICK:  craftPick(w, e, I_PLANKS, I_WOOD_PICK, r); break;
    case A_CRAFT_STONE_PICK: craftPick(w, e, I_COBBLE, I_STONE_PICK, r); break;
    case A_CRAFT_IRON_PICK:  craftPick(w, e, I_IRON_INGOT, I_IRON_PICK, r); break;
    case A_CRAFT_FURNACE:
        if (nearBlock(w, e, TABLE) && e->inv[I_COBBLE] >= 8) { e->inv[I_COBBLE] -= 8; addItem(e, I_FURNACE, 1, r); }
        break;
    case A_SMELT_IRON:
        if (nearBlock(w, e, FURNACE) && e->inv[I_IRON_ORE] >= 1) {
            if (e->fuel == 0 && e->inv[I_COAL]) { e->inv[I_COAL]--; e->fuel = 8; }
            else if (e->fuel == 0 && e->inv[I_PLANKS]) { e->inv[I_PLANKS]--; e->fuel = 1; }
            if (e->fuel > 0) {
                e->fuel--;
                e->inv[I_IRON_ORE]--;
                addItem(e, I_IRON_INGOT, 1, r);
            }
        }
        break;
    default: break;
    }
}

// One agent step: action, then `ticks` physics ticks (longer while mining). Mirrors stepEnv.
__device__ float stepEnv(uint8_t* w, GEnv* e, int a, int ticks, int maxSteps, bool& done) {
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
    case A_TURN_L: e->yaw = fmodf(e->yaw - TURN_DEG + 360.f, 360.f); break;
    case A_TURN_R: e->yaw = fmodf(e->yaw + TURN_DEG, 360.f); break;
    case A_LOOK_UP: e->pitch = fmaxf(-90.f, e->pitch - TURN_DEG); break;
    case A_LOOK_DOWN: e->pitch = fminf(90.f, e->pitch + TURN_DEG); break;
    case A_ATTACK: attack = true; break;
    default: interact(w, e, a, r); break;
    }
    for (int k = 0; e->health > 0 && (k < ticks || (attack && e->mx >= 0 && k < MAX_ATTACK_TICKS)); k++) {
        tick(w, e, fwd, strafe, jump, attack, r);
        e->ticks++;
    }
    e->t++;
    done = e->health <= 0 || e->t >= maxSteps;
    return r;
}

// Movement novelty (raw): +1 first visit of a 4x4x4 cell, +3 per new depth step.
__device__ float noveltyMove(const GEnv* ce, GEnv* e) {
    int fx = (int)floor(e->px), fy = (int)floor(e->py), fz = (int)floor(e->pz);
    int c = cellOf(fx, fy, fz);
    float n = 0;
    uint64_t bit = 1ull << (c & 63);
    if (!(e->visited[c >> 6] & bit)) { e->visited[c >> 6] |= bit; n += 1.f; }
    int ds = depthStep(fy);
    if (ds > e->bestDepthStep) { n += 3.f * (float)(ds - e->bestDepthStep); e->bestDepthStep = ds; }
    (void)ce;
    return n;
}

// ---------------------------------------------------------------- kernels

// Physics + mining + crafting + goal check for every env. Writes rew (step part), nov (movement
// novelty), done, ach. Done envs are reset later by k_reset; their ach keeps the finished episode's mask.
__global__ void k_step(GEnv* envs, uint8_t* worlds, const int32_t* actions, float* rew, float* nov,
                       uint8_t* done, int64_t* ach, int N, int ticks, int maxSteps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    GEnv* e = envs + i;
    uint8_t* w = worlds + (size_t)i * WORLD;
    e->seenObs = 0;
    int a = min(max((int)actions[i], 0), NUM_ACTIONS - 1);
    e->lastAction = a;
    bool d = false;
    float r = stepEnv(w, e, a, ticks, maxSteps, d);
    float nv = noveltyMove(e, e);
    if (!(e->got & GOAL_BIT) && e->inv[e->goal] > 0) {
        r += 10.f;
        e->got |= GOAL_BIT;
        d = true;
    }
    done[i] = d ? 1 : 0;
    ach[i] = (int64_t)e->got;
    rew[i] = r;
    nov[i] = nv;
}

// Seeds per-env RNG state.
__global__ void k_seed(GEnv* envs, const int64_t* seeds, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    envs[i].rng = (uint64_t)seeds[i];
}

// Reset envs (done[i] or force): choose a pool world, sample goal, apply start kit, copy world.
// One block per env; block-uniform early exit. Mirrors initEnv (goal + kit + visited/markVisited).
__global__ void k_reset(GEnv* envs, uint8_t* worlds, const uint8_t* poolW, const double* poolSpawn,
                        const int64_t* poolSeed, int P, const float* goalW, const int32_t* kitTable,
                        const float* kitProb, const uint8_t* done, int force, int N) {
    int i = blockIdx.x;
    if (i >= N) return;
    if (!force && !done[i]) return;
    __shared__ int pSh;
    GEnv* e = envs + i;
    if (threadIdx.x == 0) {
        uint64_t s = e->rng;
        int p = (int)(sm_next(s) % (uint64_t)P);
        pSh = p;

        // goal ~ goalW (cumulative), like sampleGoal()
        double total = 0;
        for (int k = 0; k < NUM_ITEMS; k++) total += goalW[k];
        double u = sm_unit(s) * total, acc = 0;
        int g = -1;
        for (int k = 0; k < NUM_ITEMS && g < 0; k++) {
            acc += goalW[k];
            if (u < acc && goalW[k] > 0) g = k;
        }
        if (g < 0) for (int k = NUM_ITEMS - 1; k >= 0 && g < 0; k--) if (goalW[k] > 0) g = k;
        if (g < 0) g = 0;

        e->px = poolSpawn[p * 4 + 0];
        e->py = poolSpawn[p * 4 + 1];
        e->pz = poolSpawn[p * 4 + 2];
        e->yaw = (float)poolSpawn[p * 4 + 3];
        e->seed = (uint64_t)poolSeed[p];
        e->vx = e->vy = e->vz = 0;
        e->pitch = 0;
        e->onGround = 0;
        e->fallDist = 0;
        e->health = 20;
        for (int k = 0; k < NUM_ITEMS; k++) e->inv[k] = 0;
        e->got = 0;
        e->fuel = 0;
        e->mx = e->my = e->mz = -1;
        e->prog = 0;
        e->t = 0;
        e->ticks = 0;
        e->lastAction = 0;
        e->goal = g;
        e->kit = 0;
        e->kitGot = 0;
        e->restart = 0;
        const float pk = kitProb[g];
        if (pk > 0.f) {
            bool apply = pk >= 1.f;
            if (!apply) apply = (float)sm_unit(s) < pk;
            if (apply)
                for (int k = 0; k < NUM_ITEMS; k++) {
                    int c = kitTable[g * NUM_ITEMS + k];
                    if (k == g || c <= 0) continue;
                    e->inv[k] = c;
                    e->kitGot |= 1u << k;
                    e->kit = 1;
                }
        }
        for (int k = 0; k < 64; k++) e->visited[k] = 0;
        e->seenBlocks = 0;
        e->seenObs = 0;
        // markVisited at the spawn feet block
        int fx = (int)floor(e->px), fy = (int)floor(e->py), fz = (int)floor(e->pz);
        int c = cellOf(fx, fy, fz);
        e->visited[c >> 6] |= 1ull << (c & 63);
        e->bestDepthStep = depthStep(fy);
        if (force) { e->episode = 0; e->lastNov = 0.f; }
        else e->episode += 1;
        e->rng = s;
    }
    __syncthreads();
    const uint4* src = reinterpret_cast<const uint4*>(poolW + (size_t)pSh * WORLD);
    uint4* dst = reinterpret_cast<uint4*>(worlds + (size_t)i * WORLD);
    for (int k = threadIdx.x; k < WORLD / 16; k += blockDim.x) dst[k] = src[k];
}

// Observation vector (mirrors writeObs vector part).
__global__ void k_vec(const GEnv* envs, const uint8_t* worlds, float* vec, int N, int maxSteps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const GEnv* e = envs + i;
    const uint8_t* w = worlds + (size_t)i * WORLD;
    float* v = vec + (size_t)i * VEC_DIM;
    for (int k = 0; k < NUM_ITEMS; k++) v[k] = log1pf((float)e->inv[k]) / log1pf(64.f);
    int k = NUM_ITEMS;
    double yr = (double)e->yaw * PI / 180.0;
    v[k++] = e->health / 20.f;
    v[k++] = e->onGround ? 1.f : 0.f;
    v[k++] = e->pitch / 90.f;
    v[k++] = (float)sin(yr);
    v[k++] = (float)cos(yr);
    v[k++] = (float)e->prog;
    v[k++] = (float)((e->py - SEA_LEVEL) / 32.0);
    v[k++] = nearBlock(w, e, TABLE) ? 1.f : 0.f;
    v[k++] = nearBlock(w, e, FURNACE) ? 1.f : 0.f;
    v[k++] = min(e->fuel, 8) / 8.f;
    v[k++] = (float)e->t / (float)maxSteps;
    for (int g = 0; g < NUM_ITEMS; g++) v[GOAL_OFFSET + g] = g == e->goal ? 1.f : 0.f;
}

// One thread per (env, pixel): DDA ray, writes img[..,0] = block id (0 sky), img[..,1] = depth.
// Sets bits of seenObs (block ids hit) for the novelty bonus.
__global__ void k_obs(GEnv* envs, const uint8_t* worlds, uint8_t* img, int N, int R) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * R * R;
    if (idx >= total) return;
    int i = (int)(idx / (R * R));
    int p = (int)(idx % (R * R));
    int j = p / R, ii = p % R;
    GEnv* e = envs + i;
    const uint8_t* w = worlds + (size_t)i * WORLD;

    double y = (double)e->yaw * PI / 180.0, pp = (double)e->pitch * PI / 180.0;
    double f0 = -sin(y) * cos(pp), f1 = -sin(pp), f2 = cos(y) * cos(pp);
    double r0 = -cos(y), r1 = 0.0, r2 = -sin(y);
    double u0 = r1 * f2 - r2 * f1, u1 = r2 * f0 - r0 * f2, u2 = r0 * f1 - r1 * f0;
    double tanH = tan(FOV_DEG * 0.5 * PI / 180.0);
    double uu = ((ii + 0.5) / R * 2 - 1) * tanH;
    double vv = (1 - (j + 0.5) / R * 2) * tanH;
    double d0 = f0 + r0 * uu + u0 * vv, d1 = f1 + r1 * uu + u1 * vv, d2 = f2 + r2 * uu + u2 * vv;
    double n = sqrt(d0 * d0 + d1 * d1 + d2 * d2);
    d0 /= n; d1 /= n; d2 /= n;
    bool pass = eyeInWater(w, e);
    double ox = e->px, oy = e->py + EYE, oz = e->pz;
    Hit h = raycast<float>(w, (float)ox, (float)oy, (float)oz, (float)d0, (float)d1, (float)d2, VIEW_DIST, pass);
    uint8_t* q = img + (size_t)idx * 2;
    q[0] = h.hit ? h.b : 0;
    q[1] = h.hit ? (uint8_t)fminf(255.f, (float)h.dist / VIEW_DIST * 255.f) : 255;
    if (h.hit) atomicOr(&e->seenObs, 1u << h.b);
}

// Novelty bookkeeping + reward finish. Done envs: seenBlocks from the new episode, no block pay.
__global__ void k_finalize(GEnv* envs, float* rew, const float* nov, const uint8_t* done, float scale, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    GEnv* e = envs + i;
    float nv = nov[i];
    uint32_t seen = e->seenObs;
    if (done[i]) {
        e->seenBlocks = seen;
    } else {
        uint32_t fresh = seen & ~e->seenBlocks;
        e->seenBlocks |= seen;
        nv += 2.f * (float)__popc(fresh);
    }
    float rn = scale * nv;
    e->lastNov = rn;
    rew[i] = rew[i] + rn;
}

// Ground-truth raycast under the crosshair: out[i] = {hit, x, y, z, block}.
__global__ void k_target(const GEnv* envs, const uint8_t* worlds, int32_t* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    Hit h = target(worlds + (size_t)i * WORLD, envs + i);
    out[i * 5 + 0] = h.hit ? 1 : 0;
    out[i * 5 + 1] = h.x; out[i * 5 + 2] = h.y; out[i * 5 + 3] = h.z; out[i * 5 + 4] = h.b;
}

// ---------------------------------------------------------------- host side

cudaStream_t cur() { return at::cuda::getCurrentCUDAStream(); }

void check(const char* what) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err));
}

GEnv* envPtr(const torch::Tensor& t) { return reinterpret_cast<GEnv*>(t.data_ptr()); }

void h_step(torch::Tensor envs, torch::Tensor worlds, torch::Tensor actions, torch::Tensor rew,
            torch::Tensor nov, torch::Tensor done, torch::Tensor ach, int ticks, int maxSteps) {
    const int N = (int)actions.numel();
    if (N == 0) return;
    k_step<<<(N + 127) / 128, 128, 0, cur()>>>(envPtr(envs), worlds.data_ptr<uint8_t>(), actions.data_ptr<int32_t>(),
                                              rew.data_ptr<float>(), nov.data_ptr<float>(),
                                              done.data_ptr<uint8_t>(), ach.data_ptr<int64_t>(), N, ticks, maxSteps);
    check("k_step");
}

void h_seed(torch::Tensor envs, torch::Tensor seeds) {
    const int N = (int)seeds.numel();
    k_seed<<<(N + 127) / 128, 128, 0, cur()>>>(envPtr(envs), seeds.data_ptr<int64_t>(), N);
    check("k_seed");
}

void h_reset(torch::Tensor envs, torch::Tensor worlds, torch::Tensor poolW, torch::Tensor poolSpawn,
             torch::Tensor poolSeed, torch::Tensor goalW, torch::Tensor kitTable, torch::Tensor kitProb,
             torch::Tensor done, bool force) {
    const int N = (int)(envs.numel() / (int64_t)sizeof(GEnv));
    const int P = (int)poolSeed.numel();
    k_reset<<<N, 256, 0, cur()>>>(envPtr(envs), worlds.data_ptr<uint8_t>(), poolW.data_ptr<uint8_t>(),
                                  poolSpawn.data_ptr<double>(), poolSeed.data_ptr<int64_t>(), P,
                                  goalW.data_ptr<float>(), kitTable.data_ptr<int32_t>(), kitProb.data_ptr<float>(),
                                  done.data_ptr<uint8_t>(), force ? 1 : 0, N);
    check("k_reset");
}

void h_obs(torch::Tensor envs, torch::Tensor worlds, torch::Tensor img, torch::Tensor vec, int maxSteps, int R) {
    const int N = (int)(envs.numel() / (int64_t)sizeof(GEnv));
    const long long total = (long long)N * R * R;
    k_vec<<<(N + 127) / 128, 128, 0, cur()>>>(envPtr(envs), worlds.data_ptr<uint8_t>(), vec.data_ptr<float>(), N, maxSteps);
    check("k_vec");
    k_obs<<<(unsigned)((total + 255) / 256), 256, 0, cur()>>>(envPtr(envs), worlds.data_ptr<uint8_t>(), img.data_ptr<uint8_t>(), N, R);
    check("k_obs");
}

void h_finalize(torch::Tensor envs, torch::Tensor rew, torch::Tensor nov, torch::Tensor done, double scale) {
    const int N = (int)rew.numel();
    k_finalize<<<(N + 127) / 128, 128, 0, cur()>>>(envPtr(envs), rew.data_ptr<float>(), nov.data_ptr<float>(),
                                                  done.data_ptr<uint8_t>(), (float)scale, N);
    check("k_finalize");
}

void h_target(torch::Tensor envs, torch::Tensor worlds, torch::Tensor out) {
    const int N = (int)(envs.numel() / (int64_t)sizeof(GEnv));
    k_target<<<(N + 127) / 128, 128, 0, cur()>>>(envPtr(envs), worlds.data_ptr<uint8_t>(), out.data_ptr<int32_t>(), N);
    check("k_target");
}

// Field table for get_env/set_env (names match the Python state dict).
#define ENV_SCALARS_D(X) X(px) X(py) X(pz) X(vx) X(vy) X(vz) X(fallDist) X(prog)
#define ENV_SCALARS_I(X) X(health) X(onGround) X(fuel) X(mx) X(my) X(mz) X(t) X(ticks) X(episode) \
    X(lastAction) X(goal) X(kit) X(restart) X(bestDepthStep)
#define ENV_SCALARS_U(X) X(got) X(kitGot) X(seenBlocks) X(seenObs)

pybind11::dict getEnv(torch::Tensor envs, int i) {
    const int N = (int)(envs.numel() / (int64_t)sizeof(GEnv));
    TORCH_CHECK(i >= 0 && i < N, "env index out of range");
    GEnv h;
    cudaMemcpy(&h, envPtr(envs) + i, sizeof(GEnv), cudaMemcpyDeviceToHost);
    pybind11::dict d;
#define X_D(n) d[#n] = h.n;
#define X_I(n) d[#n] = (int64_t)h.n;
#define X_U(n) d[#n] = (int64_t)h.n;
    ENV_SCALARS_D(X_D) ENV_SCALARS_I(X_I) ENV_SCALARS_U(X_U)
#undef X_D
#undef X_I
#undef X_U
    d["yaw"] = h.yaw;
    d["pitch"] = h.pitch;
    d["lastNov"] = h.lastNov;
    d["seed"] = (int64_t)h.seed;
    d["rng"] = (int64_t)h.rng;
    pybind11::list inv, vis;
    for (int k = 0; k < NUM_ITEMS; k++) inv.append(h.inv[k]);
    for (int k = 0; k < 64; k++) vis.append((int64_t)h.visited[k]);
    d["inv"] = inv;
    d["visited"] = vis;
    return d;
}

void setEnv(torch::Tensor envs, int i, pybind11::dict d) {
    const int N = (int)(envs.numel() / (int64_t)sizeof(GEnv));
    TORCH_CHECK(i >= 0 && i < N, "env index out of range");
    GEnv h;
    cudaMemcpy(&h, envPtr(envs) + i, sizeof(GEnv), cudaMemcpyDeviceToHost);
#define X_D(n) if (d.contains(#n)) h.n = pybind11::cast<double>(d[#n]);
#define X_I(n) if (d.contains(#n)) h.n = pybind11::cast<int32_t>(d[#n]);
#define X_U(n) if (d.contains(#n)) h.n = pybind11::cast<uint32_t>(d[#n]);
    ENV_SCALARS_D(X_D) ENV_SCALARS_I(X_I) ENV_SCALARS_U(X_U)
#undef X_D
#undef X_I
#undef X_U
    if (d.contains("yaw")) h.yaw = pybind11::cast<float>(d["yaw"]);
    if (d.contains("pitch")) h.pitch = pybind11::cast<float>(d["pitch"]);
    if (d.contains("lastNov")) h.lastNov = pybind11::cast<float>(d["lastNov"]);
    if (d.contains("seed")) h.seed = pybind11::cast<uint64_t>(d["seed"]);
    if (d.contains("rng")) h.rng = pybind11::cast<uint64_t>(d["rng"]);
    if (d.contains("inv")) {
        pybind11::list inv = pybind11::cast<pybind11::list>(d["inv"]);
        for (int k = 0; k < NUM_ITEMS; k++) h.inv[k] = pybind11::cast<int32_t>(inv[k]);
    }
    if (d.contains("visited")) {
        pybind11::list vis = pybind11::cast<pybind11::list>(d["visited"]);
        for (int k = 0; k < 64; k++) h.visited[k] = pybind11::cast<uint64_t>(vis[k]);
    }
    cudaMemcpy(envPtr(envs) + i, &h, sizeof(GEnv), cudaMemcpyHostToDevice);
}

}  // namespace gs

PYBIND11_MODULE(gpusim_ext, m) {
    m.attr("ENV_BYTES") = (int64_t)sizeof(gs::GEnv);
    m.def("step", &gs::h_step);
    m.def("seed", &gs::h_seed);
    m.def("reset", &gs::h_reset);
    m.def("obs", &gs::h_obs);
    m.def("finalize", &gs::h_finalize);
    m.def("target", &gs::h_target);
    m.def("get_env", &gs::getEnv);
    m.def("set_env", &gs::setEnv);
}
