// Mobs (PROTOCOL 12.4, SIM-1): spawning, simple AI, physics, combat, drops, explosions. See mobs.h.
// Data comes from reg::MOBS (data/registry/mobs.json). Everything is deterministic given W.rng.
#include "mobs.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "sim.h"
#include "survival.h"

namespace craftsim::mobs {
namespace {

constexpr double PI = 3.14159265358979323846;
constexpr int TICKS = 4;                  // mob physics ticks per world step (CraftSim default ticks = 4)
constexpr double GRAVITY = 0.08, DRAG_Y = 0.98, JUMP_VEL = 0.42;
constexpr double EYE = 1.62;
constexpr double VIEW_DIST = 24.0;        // image ray length (sim.cpp VIEW_DIST)
// Image frustum: 70 degree FOV, square, so the corners are ~44.4 degrees off the view axis; 46 with margin.
constexpr double VIEW_CONE_DEG = 46.0;
const double VIEW_CONE_COS = std::cos(VIEW_CONE_DEG * PI / 180.0);
const double VIEW_CONE_SIN = std::sin(VIEW_CONE_DEG * PI / 180.0);
constexpr double SIGHT = 16.0;            // chase / shoot range (blocks)
constexpr double DESPAWN_DIST = 48.0;     // mobs farther than this from every agent are removed
constexpr double REACH = 3.0;             // attack() reach
constexpr int PASSIVE_CAP = 10, HOSTILE_CAP = 12;
constexpr int PASSIVE_PERIOD = 100;       // decisions between passive spawn attempts (20 s)
constexpr int HOSTILE_PERIOD = 20;        // decisions between hostile spawn attempts (4 s)
constexpr double MOB_SPEED_SCALE = 0.5;   // movement_speed attribute -> blocks per tick (see think())
constexpr int INITIAL_GROUPS = 3;         // passive groups per agent at the first step of a world
constexpr int MELEE_CD = 20, SHOT_CD = 30, FUSE_TICKS = 30, FLEE_TICKS = 60, BURN_TICKS = 160;
constexpr int BURN_PERIOD = 20;           // ticks between sun damage ticks (1 hp)
constexpr double EXPLODE_BLOCK_R = 2.6;   // blocks removed within this radius
constexpr double EXPLODE_DMG_R = 6.0;     // agents / mobs within this radius take damage
constexpr double EXPLODE_DMG = 20.0;      // damage at the centre (linear falloff)
constexpr int MIN_PASSIVE_LIGHT = 9;      // passive spawn needs light level >= 9 (0..15) on grass

// Passive drop tables, mirroring data/registry/mobs.json (names resolved to item ids once).
struct DropDef { const char* item; int lo, hi; float chance; };
struct MobDrops { const char* mob; DropDef drops[2]; int n; };
constexpr MobDrops DROP_TABLE[] = {
    {"zombie", {{"rotten_flesh", 0, 2, 1.f}, {"iron_ingot", 1, 1, 0.025f}}, 2},
    {"skeleton", {{"arrow", 0, 2, 1.f}, {"bone", 0, 2, 1.f}}, 2},
    {"spider", {{"string", 0, 2, 1.f}, {nullptr, 0, 0, 0.f}}, 1},
    {"creeper", {{"gunpowder", 0, 2, 1.f}, {nullptr, 0, 0, 0.f}}, 1},
    {"cow", {{"leather", 0, 2, 1.f}, {"beef", 1, 3, 1.f}}, 2},
    {"pig", {{"porkchop", 1, 3, 1.f}, {nullptr, 0, 0, 0.f}}, 1},
    {"sheep", {{"mutton", 1, 2, 1.f}, {"white_wool", 1, 1, 1.f}}, 2},
    {"chicken", {{"feather", 0, 2, 1.f}, {"chicken", 1, 1, 1.f}}, 2},
    {"zombified_piglin", {{"rotten_flesh", 0, 1, 1.f}, {"gold_ingot", 1, 1, 0.025f}}, 2},
};

// Item id of each drop entry, per mob type (-1 = item not in the registry). Resolved once (thread-safe init).
struct ResolvedDrops { int item[NUM_MOBS][2]; };
const ResolvedDrops& resolvedDrops() {
    static const ResolvedDrops R = [] {
        ResolvedDrops r{};
        for (int m = 0; m < NUM_MOBS; m++)
            for (int j = 0; j < 2; j++) r.item[m][j] = -1;
        for (const MobDrops& md : DROP_TABLE)
            for (int m = 0; m < NUM_MOBS; m++) {
                if (std::strcmp(reg::MOBS[m].name, md.mob) != 0) continue;
                for (int j = 0; j < md.n; j++)
                    for (int i = 0; i < NUM_ITEMS; i++)
                        if (std::strcmp(reg::ITEMS[i].name, md.drops[j].item) == 0) r.item[m][j] = i;
            }
        return r;
    }();
    return R;
}
const DropDef& dropDef(int mob, int j) {
    for (const MobDrops& md : DROP_TABLE)
        if (std::strcmp(reg::MOBS[mob].name, md.mob) == 0) return md.drops[j];
    return DROP_TABLE[0].drops[0];
}

enum : int8_t { ST_IDLE, ST_WANDER, ST_FLEE, ST_CHASE, ST_SHOOT, ST_FUSE };

using Rng = std::mt19937_64;

inline double u01(Rng& r) { return (double)(r() >> 11) * (1.0 / 9007199254740992.0); }
inline int randInt(Rng& r, int lo, int hi) {
    if (hi <= lo) return lo;
    return lo + (int)(r() % (uint64_t)(hi - lo + 1));
}
inline double clampd(double v, double lo, double hi) { return v < lo ? lo : v > hi ? hi : v; }

// ---- block access (same rules as sim.cpp get(): above the world is air, outside x/z or below is bedrock)
inline int cellOf(int x, int y, int z) { return (y * WZ + z) * WX + x; }
inline uint8_t blockAt(const World& W, int x, int y, int z) {
    if (y >= WY) return AIR;
    if (x < 0 || x >= WX || z < 0 || z >= WZ || y < 0) return BEDROCK;
    return W.w[cellOf(x, y, z)];
}
inline bool solidB(uint8_t b) { return b != AIR && b != WATER; }
inline bool solidAt(const World& W, int x, int y, int z) { return solidB(blockAt(W, x, y, z)); }

// Sweep of a box [mn, mx] along axis a by d against solid blocks (same as sim.cpp sweep()).
double sweep(const World& W, const double mn[3], const double mx[3], int a, double d) {
    if (d == 0) return 0;
    const int o1 = (a + 1) % 3, o2 = (a + 2) % 3;
    const int lo1 = (int)std::floor(mn[o1] + 1e-9), hi1 = (int)std::floor(mx[o1] - 1e-9);
    const int lo2 = (int)std::floor(mn[o2] + 1e-9), hi2 = (int)std::floor(mx[o2] - 1e-9);
    auto slabSolid = [&](int c) {
        int p[3];
        p[a] = c;
        for (int i = lo1; i <= hi1; i++)
            for (int j = lo2; j <= hi2; j++) {
                p[o1] = i;
                p[o2] = j;
                if (solidAt(W, p[0], p[1], p[2])) return true;
            }
        return false;
    };
    if (d > 0) {
        const int c0 = (int)std::floor(mx[a]), c1 = (int)std::floor(mx[a] + d);
        for (int c = c0; c <= c1; c++)
            if (slabSolid(c)) return std::max(0.0, c - mx[a] - 1e-7);
    } else {
        const int c0 = (int)std::floor(mn[a]), c1 = (int)std::floor(mn[a] + d);
        for (int c = c0; c >= c1; c--)
            if (slabSolid(c)) return std::min(0.0, c + 1 - mn[a] + 1e-7);
    }
    return d;
}

struct MoveRes { bool ground, ceiling, blocked; };

// Moves a box (feet at x,y,z; half width hw; height h) by (dx,dy,dz) with block collision.
MoveRes moveBox(const World& W, double& x, double& y, double& z, double hw, double h,
                double dx, double dy, double dz) {
    double mn[3] = {x - hw, y, z - hw}, mx[3] = {x + hw, y + h, z + hw};
    const double ay = sweep(W, mn, mx, 1, dy);
    mn[1] += ay; mx[1] += ay;
    const double ax = sweep(W, mn, mx, 0, dx);
    mn[0] += ax; mx[0] += ax;
    const double az = sweep(W, mn, mx, 2, dz);
    x += ax; y += ay; z += az;
    MoveRes r;
    r.ground = dy < 0 && ay != dy;
    r.ceiling = dy > 0 && ay != dy;
    r.blocked = ax != dx || az != dz;
    return r;
}

// True when the mob's head column has no block above it (sun reaches it).
bool skyExposed(const World& W, const Mob& m) {
    const int x = (int)std::floor(m.x), z = (int)std::floor(m.z);
    for (int y = (int)std::floor(m.y) + 2; y < WY; y++)
        if (solidAt(W, x, y, z)) return false;
    return true;
}

bool inWaterAt(const World& W, const Mob& m) {
    return blockAt(W, (int)std::floor(m.x), (int)std::floor(m.y + 0.5), (int)std::floor(m.z)) == WATER;
}

// Light level 0..15 at a block (survival::lightAt returns 0..1, 1 = daylight).
int lightLevel(const World& W, int x, int y, int z) {
    const float l = survival::lightAt(W, x, y, z);
    return (int)std::lround(clampd((double)l, 0.0, 1.0) * 15.0);
}

// Spawn cell check on column (bx, bz): the top solid block at or near hintY with two air blocks above.
// feetY = the y of the first air block above the ground.
bool groundFeet(const World& W, int bx, int bz, int hintY, bool grassOnly, double& feetY) {
    for (int y = std::min(WY - 3, hintY + 6); y >= std::max(1, hintY - 12); y--) {
        const uint8_t g = blockAt(W, bx, y, bz);
        if (!solidB(g)) continue;
        if (grassOnly && g != GRASS) return false;
        if (blockAt(W, bx, y + 1, bz) != AIR || blockAt(W, bx, y + 2, bz) != AIR) return false;
        feetY = y + 1;
        return true;
    }
    return false;
}

bool isPassive(const reg::MobDef& d) { return !d.hostile && d.attack == reg::A_NONE && d.dimension == 0; }
bool isHostile(const reg::MobDef& d) { return d.hostile && d.dimension == 0; }

int countKind(const State& S, bool hostile) {
    int n = 0;
    for (const Mob& m : S.mobs)
        if (m.type >= 0 && reg::MOBS[m.type].hostile == (hostile ? 1 : 0)) n++;
    return n;
}

bool placeMob(State& S, int type, double x, double y, double z) {
    for (Mob& m : S.mobs) {
        if (m.type >= 0) continue;
        m = Mob();
        m.type = (int8_t)type;
        m.x = x; m.y = y; m.z = z;
        m.hp = reg::MOBS[type].health;
        return true;
    }
    return false;
}

// Random alive agent index, -1 when none is alive.
int pickAgent(const Agent* ag, int K, Rng& rng) {
    int alive = 0;
    for (int k = 0; k < K; k++) alive += ag[k].health > 0;
    if (alive == 0) return -1;
    int pick = randInt(rng, 0, alive - 1);
    for (int k = 0; k < K; k++) {
        if (ag[k].health <= 0) continue;
        if (pick-- == 0) return k;
    }
    return -1;
}

// Light rule for spawning: passive mobs need daylight on grass; hostile mobs need night or a light level
// at or below their spawn light_max.
bool lightAllows(const World& W, const reg::MobDef& d, int x, int y, int z, bool isDay, bool isNight) {
    const int lvl = lightLevel(W, x, y, z);
    if (isPassive(d)) return isDay && lvl >= MIN_PASSIVE_LIGHT;
    return isNight || lvl <= d.lightMax;
}

// Spawns up to `count` mobs of `type` around (cx, cz), scattered within 3 blocks.
void spawnGroup(World& W, int type, int cx, int cz, int hintY, int count, bool isDay, bool isNight, Rng& rng) {
    const reg::MobDef& d = reg::MOBS[type];
    const bool grassOnly = isPassive(d);
    int placed = 0;
    for (int tries = 0; tries < count * 4 && placed < count; tries++) {
        const int bx = cx + randInt(rng, -3, 3), bz = cz + randInt(rng, -3, 3);
        double feet = 0;
        if (!groundFeet(W, bx, bz, hintY, grassOnly, feet)) continue;
        if (!lightAllows(W, d, bx, (int)feet, bz, isDay, isNight)) continue;
        if (!placeMob(W.mobs, type, bx + 0.5, feet, bz + 0.5)) return;
        placed++;
    }
}

// Passive group near agent k at distance [dMin, dMax].
void spawnPassiveNear(World& W, const Agent* ag, int k, int dMin, int dMax, bool isDay, bool isNight, Rng& rng) {
    if (!isDay || k < 0) return;
    if (countKind(W.mobs, false) >= PASSIVE_CAP) return;
    int cand[NUM_MOBS], n = 0;
    for (int m = 0; m < NUM_MOBS; m++)
        if (isPassive(reg::MOBS[m])) cand[n++] = m;
    if (n == 0) return;
    const int type = cand[randInt(rng, 0, n - 1)];
    const reg::MobDef& d = reg::MOBS[type];
    const int group = randInt(rng, d.groupMin, d.groupMax);
    const double ang = u01(rng) * 2.0 * PI;
    const double dist = randInt(rng, dMin, dMax);
    const int cx = (int)std::floor(ag[k].px + std::cos(ang) * dist);
    const int cz = (int)std::floor(ag[k].pz + std::sin(ang) * dist);
    spawnGroup(W, type, cx, cz, (int)std::floor(ag[k].py), group, isDay, isNight, rng);
}

void spawnHostileNear(World& W, const Agent* ag, int K, bool isDay, bool isNight, Rng& rng) {
    if (countKind(W.mobs, true) >= HOSTILE_CAP) return;
    const int k = pickAgent(ag, K, rng);
    if (k < 0) return;
    int cand[NUM_MOBS], n = 0;
    for (int m = 0; m < NUM_MOBS; m++)
        if (isHostile(reg::MOBS[m])) cand[n++] = m;
    if (n == 0) return;
    const int type = cand[randInt(rng, 0, n - 1)];
    const reg::MobDef& d = reg::MOBS[type];
    const int group = randInt(rng, d.groupMin, d.groupMax);
    const double ang = u01(rng) * 2.0 * PI;
    const double dist = randInt(rng, 16, 24);
    const int cx = (int)std::floor(ag[k].px + std::cos(ang) * dist);
    const int cz = (int)std::floor(ag[k].pz + std::sin(ang) * dist);
    spawnGroup(W, type, cx, cz, (int)std::floor(ag[k].py), group, isDay, isNight, rng);
}

// Explosion centred at (cx, cy, cz): removes blocks within EXPLODE_BLOCK_R (not bedrock, not water),
// damages agents and other mobs with linear falloff.
void explode(World& W, Agent* ag, int K, double cx, double cy, double cz) {
    const int R = (int)std::ceil(EXPLODE_BLOCK_R);
    for (int y = (int)std::floor(cy) - R; y <= (int)std::floor(cy) + R; y++) {
        if (y < 0 || y >= WY) continue;
        for (int z = (int)std::floor(cz) - R; z <= (int)std::floor(cz) + R; z++) {
            if (z < 0 || z >= WZ) continue;
            for (int x = (int)std::floor(cx) - R; x <= (int)std::floor(cx) + R; x++) {
                if (x < 0 || x >= WX) continue;
                const double bx = x + 0.5 - cx, by = y + 0.5 - cy, bz = z + 0.5 - cz;
                if (bx * bx + by * by + bz * bz > EXPLODE_BLOCK_R * EXPLODE_BLOCK_R) continue;
                const int c = cellOf(x, y, z);
                const uint8_t b = W.w[c];
                if (b == AIR || b == BEDROCK || b == WATER) continue;
                if (b == CHEST) W.chests.erase(c);
                W.w[c] = AIR;
            }
        }
    }
    for (int k = 0; k < K; k++) {
        if (ag[k].health <= 0) continue;
        const double dx = ag[k].px - cx, dy = ag[k].py + 0.9 - cy, dz = ag[k].pz - cz;
        const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (d >= EXPLODE_DMG_R) continue;
        ag[k].health -= (int)std::ceil(EXPLODE_DMG * (1.0 - d / EXPLODE_DMG_R));
    }
    for (Mob& m : W.mobs.mobs) {
        if (m.type < 0) continue;
        const double dx = m.x - cx, dy = m.y + 0.5 - cy, dz = m.z - cz;
        const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (d >= EXPLODE_DMG_R) continue;
        m.hp -= (float)std::ceil(EXPLODE_DMG * (1.0 - d / EXPLODE_DMG_R));
        if (m.hp <= 0) m.type = -1;
    }
}

// Line of sight from the mob's head to the agent's eyes (solid blocks block it).
bool lineOfSight(const World& W, const Mob& m, const Agent& a) {
    const double ox = m.x, oy = m.y + 1.5, oz = m.z;
    const double tx = a.px, ty = a.py + EYE, tz = a.pz;
    const double dist = std::sqrt((tx - ox) * (tx - ox) + (ty - oy) * (ty - oy) + (tz - oz) * (tz - oz));
    const int n = (int)(dist / 0.25) + 1;
    for (int i = 1; i < n; i++) {
        const double f = (double)i / n;
        if (solidAt(W, (int)std::floor(ox + (tx - ox) * f), (int)std::floor(oy + (ty - oy) * f),
                    (int)std::floor(oz + (tz - oz) * f)))
            return false;
    }
    return true;
}

void walkDir(Mob& m, double ux, double uz, double speed) {
    m.vx = ux * speed;
    m.vz = uz * speed;
}

void wander(Mob& m, double speed, Rng& rng) {
    if (m.wander <= 0) {
        if (u01(rng) < 0.4) {
            m.wdx = m.wdz = 0;
            m.wander = (int16_t)randInt(rng, 60, 120);
        } else {
            const double a = u01(rng) * 2.0 * PI;
            m.wdx = std::cos(a);
            m.wdz = std::sin(a);
            m.wander = (int16_t)randInt(rng, 40, 100);
        }
        m.state = m.wdx == 0 && m.wdz == 0 ? ST_IDLE : ST_WANDER;
    }
    m.wander = (int16_t)(m.wander - TICKS);
    walkDir(m, m.wdx, m.wdz, speed);
}

inline void countDown(int16_t& v) {
    if (v > 0) v = (int16_t)std::max(0, v - TICKS);
}

// One AI decision of mob m: sets its velocity for the coming ticks. Returns true when the mob is removed
// (it exploded, or it burned to death).
bool think(World& W, Mob& m, Agent* ag, int K, bool isDay, Rng& rng) {
    const reg::MobDef& d = reg::MOBS[m.type];
    // The registry holds vanilla's movement_speed attribute, not blocks per tick: vanilla mob AI walks at
    // about half of it per tick (adult zombie 0.23 -> ~2.3 blocks/s, slower than a walking player's 4.3).
    const double spd = d.speed * MOB_SPEED_SCALE;
    m.vx = m.vz = 0;
    m.age++;
    m.stuck = m.blocked ? (int8_t)std::min(100, m.stuck + 1) : (int8_t)0;
    countDown(m.cd);
    countDown(m.flee);
    countDown(m.burnT);
    countDown(m.burn);

    // Sun: zombies / skeletons burn while exposed to daylight (water puts the fire out).
    if (d.burnsInSun && isDay && skyExposed(W, m) && !inWaterAt(W, m)) m.burn = BURN_TICKS;
    if (inWaterAt(W, m)) m.burn = 0;
    if (m.burn > 0 && m.burnT <= 0) {
        m.burnT = BURN_PERIOD;
        m.hp -= 1.f;
        if (m.hp <= 0) return true;
    }

    // Nearest alive agent.
    int ti = -1;
    double td2 = 1e30;
    for (int k = 0; k < K; k++) {
        if (ag[k].health <= 0) continue;
        const double dx = ag[k].px - m.x, dy = ag[k].py - m.y, dz = ag[k].pz - m.z;
        const double d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < td2) { td2 = d2; ti = k; }
    }
    const bool near = ti >= 0 && td2 <= SIGHT * SIGHT;
    const double tdist = std::sqrt(td2);
    double dx = 0, dz = 0, hd = 0;
    if (ti >= 0) {
        dx = ag[ti].px - m.x;
        dz = ag[ti].pz - m.z;
        hd = std::sqrt(dx * dx + dz * dz);
    }
    const double ux = hd > 1e-6 ? dx / hd : 1.0, uz = hd > 1e-6 ? dz / hd : 0.0;

    if (!d.hostile) {   // passive: flee after a hit, else wander
        if (m.flee > 0) {
            double fx = m.x - m.fx, fz = m.z - m.fz;
            const double L = std::sqrt(fx * fx + fz * fz);
            if (L > 1e-6) walkDir(m, fx / L, fz / L, spd * 1.3);
            else walkDir(m, 1.0, 0.0, spd * 1.3);
            m.state = ST_FLEE;
        } else {
            wander(m, spd * 0.5, rng);
        }
        return false;
    }

    // Stuck against terrain (no path finding): random walk for a while.
    // Blocked while chasing: slide sideways (random side) for a short time, then chase again.
    if (near && m.stuck >= 6 && m.wander <= 0 && hd > 1e-6) {
        m.stuck = 0;
        m.wander = (int16_t)randInt(rng, 10, 20);
        const double side = u01(rng) < 0.5 ? -1.0 : 1.0;
        m.wdx = -uz * side;
        m.wdz = ux * side;
    }

    if (d.attack == reg::A_MELEE) {   // zombie, spider, zombified piglin
        if (!near) {
            wander(m, spd * 0.5, rng);
            return false;
        }
        if (m.wander > 0) {
            wander(m, spd * 0.7, rng);
            return false;
        }
        m.state = ST_CHASE;
        if (hd > 1.0) walkDir(m, ux, uz, spd);
        if (hd < 1.6 && std::fabs(ag[ti].py - m.y) < 2.0 && m.cd <= 0) {
            ag[ti].health -= (int)std::ceil(d.damage);
            m.cd = MELEE_CD;
        }
        return false;
    }

    if (d.attack == reg::A_RANGED) {   // skeleton: keeps its distance and shoots
        if (!near) {
            wander(m, spd * 0.5, rng);
            return false;
        }
        m.state = ST_SHOOT;
        if (m.wander > 0) {
            wander(m, spd * 0.7, rng);
        } else if (hd < 4.0) walkDir(m, -ux, -uz, spd * 0.8);
        else if (hd > 10.0) walkDir(m, ux, uz, spd);
        else walkDir(m, -uz, ux, spd * 0.5);   // strafe
        if (m.cd <= 0 && lineOfSight(W, m, ag[ti])) {
            // Hitscan with accuracy falloff with distance.
            const double p = clampd(0.9 - 0.03 * tdist, 0.2, 0.9);
            if (u01(rng) < p) ag[ti].health -= (int)std::ceil(d.damage);
            m.cd = SHOT_CD;
        }
        return false;
    }

    // A_EXPLODE: creeper. Approaches, fuses within 3 blocks, explodes when the fuse runs out.
    if (!near) {
        m.fuse = (int16_t)std::max(0, m.fuse - TICKS);
        wander(m, spd * 0.5, rng);
        return false;
    }
    m.state = ST_FUSE;
    if (m.wander > 0) {
        wander(m, spd * 0.7, rng);
    } else if (tdist > 7.0) {
        m.fuse = 0;
        walkDir(m, ux, uz, spd);
    } else if (tdist > 3.0) {
        m.fuse = (int16_t)std::max(0, m.fuse - TICKS);
        walkDir(m, ux, uz, spd);
    } else {
        m.fuse = (int16_t)(m.fuse + TICKS);
        if (m.fuse >= FUSE_TICKS) {
            explode(W, ag, K, m.x, m.y + 0.85, m.z);
            return true;
        }
    }
    return false;
}

// Physics for TICKS ticks: gravity, drag, collision; jumps when blocked on the ground.
void physics(const World& W, Mob& m) {
    const reg::MobDef& d = reg::MOBS[m.type];
    const double hw = d.width * 0.5, h = d.height;
    for (int t = 0; t < TICKS; t++) {
        m.vy = (m.vy - GRAVITY) * DRAG_Y;
        const MoveRes r = moveBox(W, m.x, m.y, m.z, hw, h, m.vx, m.vy, m.vz);
        if (r.ground || r.ceiling) m.vy = 0;
        m.ground = r.ground;
        m.blocked = r.blocked && (m.vx != 0 || m.vz != 0);
        if (r.blocked && r.ground) m.vy = JUMP_VEL;
    }
}

// Writes the drops of mob `type` into the attacker's inventory.
void giveDrops(World& W, Agent& a, int type) {
    const ResolvedDrops& R = resolvedDrops();
    for (int j = 0; j < 2; j++) {
        const int item = R.item[type][j];
        if (item < 0) continue;
        const DropDef& dd = dropDef(type, j);
        if (u01(W.rng) >= dd.chance) continue;
        const int n = randInt(W.rng, dd.lo, dd.hi);
        if (n > 0) a.inv[item] += n;
    }
}

// Best melee damage of the attacker: swords 4/5/6/7 (tier 1-4), axes 7/9/9/9; else the fist (1).
float weaponDamage(const Agent& a) {
    float best = 1.f;
    for (int i = 0; i < NUM_ITEMS; i++) {
        if (a.inv[i] <= 0) continue;
        const reg::ItemDef& it = reg::ITEMS[i];
        if (it.toolType == reg::T_SWORD && it.toolTier >= 1 && it.toolTier <= 4)
            best = std::max(best, (float)(3 + it.toolTier));
        else if (it.toolType == reg::T_AXE && it.toolTier >= 1 && it.toolTier <= 4)
            best = std::max(best, it.toolTier == 1 ? 7.f : 9.f);
    }
    return best;
}

// Ray / mob-box slab test; tEnter in [0, tmax].
bool slabHit(const Mob& m, const reg::MobDef& d, const double o[3], const double dv[3], const double inv[3],
             double tmax, double& tEnter) {
    const double hw = d.width * 0.5;
    const double lo[3] = {m.x - hw, m.y, m.z - hw}, hi[3] = {m.x + hw, m.y + d.height, m.z + hw};
    double t0 = 0.0, t1 = tmax;
    for (int a = 0; a < 3; a++) {
        if (std::fabs(dv[a]) < 1e-12) {
            if (o[a] < lo[a] || o[a] > hi[a]) return false;
            continue;
        }
        double ta = (lo[a] - o[a]) * inv[a], tb = (hi[a] - o[a]) * inv[a];
        if (ta > tb) std::swap(ta, tb);
        t0 = std::max(t0, ta);
        t1 = std::min(t1, tb);
        if (t0 > t1) return false;
    }
    tEnter = t0;
    return true;
}

// Mobs that can be hit from origin o within maxDist (bounding sphere test). Memoised on the origin: all image
// rays of one agent share its eye position, so the candidate set is built once per agent per step. With a
// view cone f (beginView) only the mobs whose bounding sphere meets the cone are kept.
void buildMemo(const State& S, const double o[3], double maxDist, const double* f) {
    S.memoGen = S.gen;
    S.memoO[0] = o[0]; S.memoO[1] = o[1]; S.memoO[2] = o[2];
    S.memoMax = maxDist;
    S.memoCone = f != nullptr;
    if (f) { S.memoF[0] = f[0]; S.memoF[1] = f[1]; S.memoF[2] = f[2]; }
    S.memoN = 0;
    for (int i = 0; i < MAX_MOBS; i++) {
        const Mob& m = S.mobs[i];
        if (m.type < 0) continue;
        const reg::MobDef& d = reg::MOBS[m.type];
        const double rad = std::sqrt(d.width * d.width * 0.25 + d.height * d.height * 0.25) + 1e-6;
        const double cx = m.x - o[0], cy = m.y + d.height * 0.5 - o[1], cz = m.z - o[2];
        const double vv = cx * cx + cy * cy + cz * cz;
        if (std::sqrt(vv) > maxDist + rad) continue;
        if (f) {
            // Sphere vs cone (axis f, half-angle VIEW_CONE_DEG): keep when the sphere meets the cone.
            const double p = cx * f[0] + cy * f[1] + cz * f[2];
            const double q = std::sqrt(std::max(0.0, vv - p * p));
            if (p + rad < 0 || q * VIEW_CONE_COS - p * VIEW_CONE_SIN > rad) continue;
        }
        const int n = S.memoN++;
        S.memoIdx[n] = i;
        S.memoC[n][0] = cx; S.memoC[n][1] = cy; S.memoC[n][2] = cz;
        S.memoV[n] = vv;
        S.memoRad[n] = rad;
        S.memoR2[n] = rad * rad;
    }
}

// Index of the nearest mob hit by the ray before maxDist, -1 if none (d is a unit vector).
int findMob(const World& W, const double o[3], const double dv[3], double maxDist, double& tOut) {
    const State& S = W.mobs;
    const bool sameOrigin = S.memoO[0] == o[0] && S.memoO[1] == o[1] && S.memoO[2] == o[2];
    if (!sameOrigin || S.memoMax != maxDist || S.memoGen != S.gen) {
        buildMemo(S, o, maxDist, nullptr);
    } else if (S.memoCone) {
        const double dc = dv[0] * S.memoF[0] + dv[1] * S.memoF[1] + dv[2] * S.memoF[2];
        if (dc < VIEW_CONE_COS) buildMemo(S, o, maxDist, nullptr);   // ray outside the pre-selected cone
    }
    double best = maxDist;
    int bi = -1;
    double inv[3] = {0, 0, 0};
    bool haveInv = false;
    for (int n = 0; n < S.memoN; n++) {
        // Sphere reject without divisions: the ray passes within rad of the centre at parameter t.
        const double t = S.memoC[n][0] * dv[0] + S.memoC[n][1] * dv[1] + S.memoC[n][2] * dv[2];
        if (t + S.memoRad[n] < 0 || t - S.memoRad[n] > best) continue;
        if (S.memoV[n] - t * t > S.memoR2[n]) continue;
        const int i = S.memoIdx[n];
        const Mob& m = S.mobs[i];
        if (!haveInv) {
            for (int a = 0; a < 3; a++) inv[a] = std::fabs(dv[a]) < 1e-12 ? 0.0 : 1.0 / dv[a];
            haveInv = true;
        }
        double tt;
        if (slabHit(m, reg::MOBS[m.type], o, dv, inv, best, tt) && tt < best) {
            best = tt;
            bi = i;
        }
    }
    tOut = best;
    return bi;
}

}  // namespace

void beginView(const World& W, const Agent& a) {
    const double yr = a.yaw * PI / 180.0, pr = a.pitch * PI / 180.0;
    const double f[3] = {-std::sin(yr) * std::cos(pr), -std::sin(pr), std::cos(yr) * std::cos(pr)};
    const double o[3] = {a.px, a.py + EYE, a.pz};
    buildMemo(W.mobs, o, VIEW_DIST, f);
}

void init(World& W) {
    W.mobs = State();
    W.mobs.passiveT = PASSIVE_PERIOD;
    W.mobs.hostileT = HOSTILE_PERIOD;
}

void tick(World& W, Agent* ag, int K) {
    State& S = W.mobs;
    Rng& rng = W.rng;
    if (!ag) K = 0;
    const bool isDay = survival::timeOfDay(W) < 0.5f;
    const bool isNight = !isDay;

    if (K > 0 && !S.started) {
        S.started = true;
        for (int k = 0; k < K; k++) {
            if (ag[k].health <= 0) continue;
            for (int g = 0; g < INITIAL_GROUPS; g++) spawnPassiveNear(W, ag, k, 6, 14, isDay, isNight, rng);
        }
    }
    if (K > 0 && --S.passiveT <= 0) {
        S.passiveT = PASSIVE_PERIOD;
        spawnPassiveNear(W, ag, pickAgent(ag, K, rng), 8, 20, isDay, isNight, rng);
    }
    if (K > 0 && --S.hostileT <= 0) {
        S.hostileT = HOSTILE_PERIOD;
        spawnHostileNear(W, ag, K, isDay, isNight, rng);
    }

    for (Mob& m : S.mobs) {
        if (m.type < 0) continue;
        if (think(W, m, ag, K, isDay, rng)) {
            m.type = -1;
            continue;
        }
        if (m.hp <= 0) {
            m.type = -1;
            continue;
        }
        if (K > 0) {   // despawn when far from every agent
            double best = 1e30;
            for (int k = 0; k < K; k++) {
                if (ag[k].health <= 0) continue;
                const double dx = ag[k].px - m.x, dy = ag[k].py - m.y, dz = ag[k].pz - m.z;
                best = std::min(best, dx * dx + dy * dy + dz * dz);
            }
            if (best > DESPAWN_DIST * DESPAWN_DIST) {
                m.type = -1;
                continue;
            }
        }
        physics(W, m);
    }
    S.gen++;
}

bool rayHit(const World& W, const double o[3], const double d[3], double maxDist, float& tHit, uint8_t& blockId) {
    double t;
    const int i = findMob(W, o, d, maxDist, t);
    if (i < 0) return false;
    tHit = (float)t;
    blockId = reg::MOBS[W.mobs.mobs[i].type].block;
    return true;
}

bool attack(World& W, Agent& a, float damage) {
    if (a.health <= 0) return false;
    const double yr = a.yaw * PI / 180.0, pr = a.pitch * PI / 180.0;
    const double dv[3] = {-std::sin(yr) * std::cos(pr), -std::sin(pr), std::cos(yr) * std::cos(pr)};
    const double o[3] = {a.px, a.py + EYE, a.pz};
    // Nearest block along the view ray caps the reach (a block in front of the mob is mined instead).
    double lim = REACH;
    for (double t = 0.0; t < REACH; t += 0.02) {
        if (solidAt(W, (int)std::floor(o[0] + dv[0] * t), (int)std::floor(o[1] + dv[1] * t),
                    (int)std::floor(o[2] + dv[2] * t))) {
            lim = t;
            break;
        }
    }
    double t;
    const int i = findMob(W, o, dv, lim, t);
    if (i < 0) return false;
    Mob& m = W.mobs.mobs[i];
    W.mobs.gen++;
    m.hp -= damage < 0 ? weaponDamage(a) : damage;
    if (m.hp <= 0) {
        const int type = m.type;
        giveDrops(W, a, type);
        a.st.killed[type]++;
        m.type = -1;
        return true;
    }
    // Hit: passive mobs flee from the attacker, all mobs get knocked back.
    m.flee = FLEE_TICKS;
    m.fx = a.px;
    m.fz = a.pz;
    double kx = m.x - a.px, kz = m.z - a.pz;
    const double L = std::sqrt(kx * kx + kz * kz);
    if (L > 1e-6) {
        m.vx = kx / L * 0.3;
        m.vz = kz / L * 0.3;
    }
    if (m.ground) m.vy = 0.4;
    return true;
}

void obs(const World& W, const Agent& a, float* out) {
    for (int i = 0; i < 18; i++) out[i] = 0.f;
    // Three nearest mobs (feet-to-body-centre distance from the agent's body centre).
    int best[3] = {-1, -1, -1};
    double bd[3] = {1e30, 1e30, 1e30};
    for (int i = 0; i < MAX_MOBS; i++) {
        const Mob& m = W.mobs.mobs[i];
        if (m.type < 0) continue;
        const double dx = m.x - a.px, dy = m.y + reg::MOBS[m.type].height * 0.5 - (a.py + 0.9),
                     dz = m.z - a.pz;
        const double d2 = dx * dx + dy * dy + dz * dz;
        for (int j = 0; j < 3; j++) {
            if (d2 >= bd[j]) continue;
            for (int l = 2; l > j; l--) {
                bd[l] = bd[l - 1];
                best[l] = best[l - 1];
            }
            bd[j] = d2;
            best[j] = i;
            break;
        }
    }
    // Yaw frame (same as the teammate slots): xl = right, zl = forward.
    const double yr = a.yaw * PI / 180.0;
    const double fx = -std::sin(yr), fz = std::cos(yr);
    const double rx = -std::cos(yr), rz = -std::sin(yr);
    for (int j = 0; j < 3; j++) {
        if (best[j] < 0) continue;
        const Mob& m = W.mobs.mobs[best[j]];
        const reg::MobDef& d = reg::MOBS[m.type];
        const double dx = m.x - a.px, dy = m.y + d.height * 0.5 - (a.py + 0.9), dz = m.z - a.pz;
        float* s = out + j * 6;
        s[0] = 1.f;
        s[1] = (float)((dx * rx + dz * rz) / 16.0);
        s[2] = (float)(dy / 16.0);
        s[3] = (float)((dx * fx + dz * fz) / 16.0);
        s[4] = d.hostile ? 1.f : 0.f;
        s[5] = d.health > 0 ? m.hp / d.health : 0.f;
    }
}

}  // namespace craftsim::mobs
