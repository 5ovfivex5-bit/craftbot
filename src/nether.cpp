// Nether + portal (PROTOCOL 12.4, SIM-3). See nether.h for the dimension model.
#include "nether.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>

#include "sim.h"

namespace craftsim::nether {
namespace {

constexpr uint8_t B_AIR = static_cast<uint8_t>(reg::BLK_AIR);
constexpr uint8_t B_STONE = static_cast<uint8_t>(reg::BLK_STONE);
constexpr uint8_t B_BEDROCK = static_cast<uint8_t>(reg::BLK_BEDROCK);
constexpr uint8_t B_WATER = static_cast<uint8_t>(reg::BLK_WATER);
constexpr uint8_t B_LAVA = static_cast<uint8_t>(reg::BLK_LAVA);
constexpr uint8_t B_OBSIDIAN = static_cast<uint8_t>(reg::BLK_OBSIDIAN);
constexpr uint8_t B_NETHERRACK = static_cast<uint8_t>(reg::BLK_NETHERRACK);
constexpr uint8_t B_QUARTZ_ORE = static_cast<uint8_t>(reg::BLK_NETHER_QUARTZ_ORE);
constexpr uint8_t B_SOUL_SAND = static_cast<uint8_t>(reg::BLK_SOUL_SAND);
constexpr uint8_t B_GLOWSTONE = static_cast<uint8_t>(reg::BLK_GLOWSTONE);

constexpr int LAVA_TOP = 10;             // lava sea fills y = 1..LAVA_TOP
constexpr int CAVE_MIN_Y = 11;           // caves are carved from here ...
constexpr int CAVE_MAX_Y = WY - 4;       // ... up to here (ceiling is WY-3..WY-1)
constexpr float CAVE_THRESHOLD = 0.63f;  // trilinear value noise above this is air
constexpr int NETHER_FLOOR_MIN = 12;     // portal floors stay above the lava sea
constexpr int CAVE_LATTICE = 4;          // cave noise lattice spacing (blocks)

inline int cellOf(int x, int y, int z) { return (y * WZ + z) * WX + x; }
inline bool inBounds(int x, int y, int z) { return x >= 0 && x < WX && y >= 0 && y < WY && z >= 0 && z < WZ; }

// Block of the live array. Outside the world: air above, bedrock on the sides and below (as the sim).
inline uint8_t at(const World& W, int x, int y, int z) {
    if (y >= WY) return B_AIR;
    if (x < 0 || x >= WX || z < 0 || z >= WZ || y < 0) return B_BEDROCK;
    return W.w[cellOf(x, y, z)];
}
inline void setAt(World& W, int x, int y, int z, uint8_t b) {
    if (inBounds(x, y, z)) W.w[cellOf(x, y, z)] = b;
}
// A block a player or mob can stand on.
inline bool isFloor(uint8_t b) { return b != B_AIR && b != B_WATER && b != B_LAVA; }

inline uint32_t hash3(int x, int y, int z, uint32_t s) {
    uint32_t h = s ^ (uint32_t)x * 0x85EBCA6Bu ^ (uint32_t)y * 0xC2B2AE35u ^ (uint32_t)z * 0x27D4EB2Fu;
    h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
    return h;
}
inline float hashF(int x, int y, int z, uint32_t s) { return hash3(x, y, z, s) * (1.0f / 4294967296.0f); }

// 2D value noise in [0, 1) on a unit lattice, smoothstep-interpolated.
float valueNoise2(float fx, float fz, uint32_t s) {
    const int x0 = (int)std::floor(fx), z0 = (int)std::floor(fz);
    float tx = fx - x0, tz = fz - z0;
    tx = tx * tx * (3 - 2 * tx);
    tz = tz * tz * (3 - 2 * tz);
    const float a = hashF(x0, 0, z0, s), b = hashF(x0 + 1, 0, z0, s);
    const float c = hashF(x0, 0, z0 + 1, s), d = hashF(x0 + 1, 0, z0 + 1, s);
    const float ab = a + (b - a) * tx, cd = c + (d - c) * tx;
    return ab + (cd - ab) * tz;
}

// The nether block array (WX*WY*WZ). Netherrack everywhere, bedrock floor and ceiling, a lava sea at
// y 1..LAVA_TOP, caves from 3D value noise, soul sand on cave floors, glowstone clusters on the ceiling
// and quartz ore veins. Deterministic in `seed`. No water.
std::vector<uint8_t> buildNether(uint64_t seed) {
    const uint32_t s = (uint32_t)seed ^ ((uint32_t)(seed >> 32) * 0x9E3779B1u);
    std::vector<uint8_t> v((size_t)WX * WY * WZ, B_NETHERRACK);
    for (int z = 0; z < WZ; z++)
        for (int x = 0; x < WX; x++) {
            v[cellOf(x, 0, z)] = B_BEDROCK;
            v[cellOf(x, WY - 1, z)] = B_BEDROCK;
        }
    for (int y = 1; y <= LAVA_TOP; y++)
        for (int z = 0; z < WZ; z++)
            for (int x = 0; x < WX; x++) v[cellOf(x, y, z)] = B_LAVA;

    // Caves: trilinear interpolation of a lattice of random values, one lattice cell per 4 blocks.
    constexpr int S = CAVE_LATTICE;
    const int gx = WX / S + 2, gy = WY / S + 2, gz = WZ / S + 2;
    std::vector<float> lat((size_t)gx * gy * gz);
    for (int k = 0; k < gz; k++)
        for (int j = 0; j < gy; j++)
            for (int i = 0; i < gx; i++) lat[((size_t)k * gy + j) * gx + i] = hashF(i, j + 1000, k, s);
    auto L = [&](int i, int j, int k) { return lat[((size_t)k * gy + j) * gx + i]; };
    int ix[WX], iz[WZ], iy[WY];
    float tx[WX], tz[WZ], ty[WY];
    auto prep = [&](int n, int* idx, float* t) {
        for (int i = 0; i < n; i++) {
            idx[i] = i / S;
            const float f = (float)(i % S) / S;
            t[i] = f * f * (3 - 2 * f);
        }
    };
    prep(WX, ix, tx);
    prep(WZ, iz, tz);
    prep(WY, iy, ty);
    for (int y = CAVE_MIN_Y; y <= CAVE_MAX_Y; y++) {
        const int j = iy[y];
        const float wy = ty[y];
        for (int z = 0; z < WZ; z++) {
            const int k = iz[z];
            const float wz = tz[z];
            for (int x = 0; x < WX; x++) {
                const int i = ix[x];
                const float wx = tx[x];
                const float c00 = L(i, j, k) + (L(i + 1, j, k) - L(i, j, k)) * wx;
                const float c10 = L(i, j + 1, k) + (L(i + 1, j + 1, k) - L(i, j + 1, k)) * wx;
                const float c01 = L(i, j, k + 1) + (L(i + 1, j, k + 1) - L(i, j, k + 1)) * wx;
                const float c11 = L(i, j + 1, k + 1) + (L(i + 1, j + 1, k + 1) - L(i, j + 1, k + 1)) * wx;
                const float c0 = c00 + (c10 - c00) * wy;
                const float c1 = c01 + (c11 - c01) * wy;
                const float n = c0 + (c1 - c0) * wz;
                const int idx = cellOf(x, y, z);
                if (n > CAVE_THRESHOLD && v[idx] == B_NETHERRACK) v[idx] = B_AIR;
            }
        }
    }

    // Soul sand on cave floors, in patches.
    for (int z = 0; z < WZ; z++)
        for (int x = 0; x < WX; x++) {
            if (valueNoise2(x / 6.0f, z / 6.0f, s ^ 0x5bd1e995u) < 0.7f) continue;
            for (int y = CAVE_MIN_Y; y <= CAVE_MAX_Y; y++)
                if (v[cellOf(x, y, z)] == B_NETHERRACK && v[cellOf(x, y + 1, z)] == B_AIR)
                    v[cellOf(x, y, z)] = B_SOUL_SAND;
        }

    // Glowstone clusters hanging from the ceiling.
    const uint32_t sg = s ^ 0x2545F491u;
    for (int z = 0; z < WZ; z++)
        for (int x = 0; x < WX; x++) {
            if (v[cellOf(x, CAVE_MAX_Y, z)] != B_AIR || v[cellOf(x, WY - 3, z)] != B_NETHERRACK) continue;
            if (hashF(x, 11, z, sg) >= 0.04f) continue;
            v[cellOf(x, CAVE_MAX_Y, z)] = B_GLOWSTONE;
            const int dxz[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int n = 0; n < 4; n++) {
                const int nx = x + dxz[n][0], nz = z + dxz[n][1];
                if (!inBounds(nx, CAVE_MAX_Y, nz) || v[cellOf(nx, CAVE_MAX_Y, nz)] != B_AIR) continue;
                if (hashF(nx, 12 + n, nz, sg) < 0.5f) v[cellOf(nx, CAVE_MAX_Y, nz)] = B_GLOWSTONE;
            }
        }

    // Nether quartz ore veins (random walks through netherrack).
    std::mt19937_64 rng(seed ^ 0xA5A5F00DC0FFEEull);
    std::uniform_int_distribution<int> ux(1, WX - 2), uz(1, WZ - 2), uy(CAVE_MIN_Y + 1, WY - 5), dir(0, 5);
    const int dd[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    for (int n = 0; n < 48; n++) {
        int x = ux(rng), y = uy(rng), z = uz(rng);
        for (int step = 0; step < 10; step++) {
            if (v[cellOf(x, y, z)] == B_NETHERRACK) v[cellOf(x, y, z)] = B_QUARTZ_ORE;
            const int* d = dd[dir(rng)];
            const int nx = x + d[0], ny = y + d[1], nz = z + d[2];
            if (nx < 1 || nx > WX - 2 || nz < 1 || nz > WZ - 2 || ny < CAVE_MIN_Y + 1 || ny > WY - 5) continue;
            x = nx; y = ny; z = nz;
        }
    }
    return v;
}

// ---------------------------------------------------------------- portals

// Interior cell (u, v) of portal p -> world coordinates.
inline void interiorXYZ(const Portal& p, int u, int v, int& x, int& y, int& z) {
    y = p.oy + v;
    if (p.axis == 0) { x = p.ox + u; z = p.oz; }
    else             { x = p.ox;     z = p.oz + u; }
}

int registerPortal(State& st, int dim, const Portal& p) {
    auto& list = st.portals[dim];
    const int idx = (int)list.size();
    list.push_back(p);
    for (int u = 0; u < p.w; u++)
        for (int v = 0; v < p.h; v++) {
            int x, y, z;
            interiorXYZ(p, u, v, x, y, z);
            st.portalAt[dim][cellOf(x, y, z)] = idx;
        }
    return idx;
}

// Finds a valid obsidian frame (interior 2..21 wide, 3..21 tall, air inside, obsidian ring) whose interior
// contains the air cell (cx, cy, cz). Tries both orientations.
bool findFrame(const World& W, int cx, int cy, int cz, Portal& out) {
    for (int axis = 0; axis < 2; axis++) {
        const int u0 = axis == 0 ? cx : cz, v0 = cy, fixed = axis == 0 ? cz : cx;
        auto cell = [&](int u, int v) { return axis == 0 ? at(W, u, v, fixed) : at(W, fixed, v, u); };
        if (cell(u0, v0) != B_AIR) continue;
        int uL = u0, uR = u0, vB = v0, vT = v0;
        while (u0 - uL < 21 && cell(uL - 1, v0) == B_AIR) uL--;
        while (uR - u0 < 21 && cell(uR + 1, v0) == B_AIR) uR++;
        while (v0 - vB < 21 && cell(u0, vB - 1) == B_AIR) vB--;
        while (vT - v0 < 21 && cell(u0, vT + 1) == B_AIR) vT++;
        const int w = uR - uL + 1, h = vT - vB + 1;
        if (w < 2 || w > 21 || h < 3 || h > 21) continue;
        bool ok = true;
        for (int u = uL; u <= uR && ok; u++) {
            if (cell(u, vB - 1) != B_OBSIDIAN || cell(u, vT + 1) != B_OBSIDIAN) ok = false;
            for (int v = vB; v <= vT && ok; v++)
                if (cell(u, v) != B_AIR) ok = false;
        }
        for (int v = vB; v <= vT && ok; v++)
            if (cell(uL - 1, v) != B_OBSIDIAN || cell(uR + 1, v) != B_OBSIDIAN) ok = false;
        if (!ok) continue;
        out = Portal{};
        out.axis = axis;
        out.w = w;
        out.h = h;
        if (axis == 0) { out.ox = uL; out.oy = vB; out.oz = fixed; }
        else           { out.ox = fixed; out.oy = vB; out.oz = uL; }
        return true;
    }
    return false;
}

// Builds a 2x3 (4x5 with the frame) axis-0 portal in the live array near (cx, cz): obsidian frame, air
// interior, floor pads and clearance on both sides so the team can arrive. Floor height: the highest solid
// floor with 3 air cells above, else the minimum height.
Portal createPortal(World& W, int dim, int cx, int cz) {
    const bool nether = dim == DIM_NETHER;
    const uint8_t floorB = nether ? B_NETHERRACK : B_STONE;
    const int lo = nether ? NETHER_FLOOR_MIN : 2;
    const int x0 = std::clamp(cx, 2, WX - 4), z0 = std::clamp(cz, 2, WZ - 3);
    int y0 = -1;
    for (int y = WY - 6; y >= lo + 1 && y0 < 0; y--) {
        if (!isFloor(at(W, x0, y - 1, z0))) continue;
        if (at(W, x0, y, z0) == B_AIR && at(W, x0, y + 1, z0) == B_AIR && at(W, x0, y + 2, z0) == B_AIR &&
            at(W, x0 + 1, y, z0) == B_AIR && at(W, x0 + 1, y + 1, z0) == B_AIR && at(W, x0 + 1, y + 2, z0) == B_AIR)
            y0 = y;
    }
    if (y0 < 0) y0 = lo + 1;

    for (int dx = -1; dx <= 2; dx++) {
        setAt(W, x0 + dx, y0 - 1, z0, B_OBSIDIAN);
        setAt(W, x0 + dx, y0 + 3, z0, B_OBSIDIAN);
    }
    for (int dy = 0; dy <= 2; dy++) {
        setAt(W, x0 - 1, y0 + dy, z0, B_OBSIDIAN);
        setAt(W, x0 + 2, y0 + dy, z0, B_OBSIDIAN);
    }
    for (int dx = 0; dx <= 1; dx++)
        for (int dy = 0; dy <= 2; dy++) setAt(W, x0 + dx, y0 + dy, z0, B_AIR);
    for (int dz : {-1, 1})
        for (int dx = -1; dx <= 2; dx++) {
            if (!isFloor(at(W, x0 + dx, y0 - 1, z0 + dz))) setAt(W, x0 + dx, y0 - 1, z0 + dz, floorB);
            for (int dy = 0; dy <= 2; dy++) {
                const uint8_t b = at(W, x0 + dx, y0 + dy, z0 + dz);
                if (b != B_AIR && b != B_BEDROCK) setAt(W, x0 + dx, y0 + dy, z0 + dz, B_AIR);
            }
        }
    Portal p;
    p.axis = 0;
    p.ox = x0;
    p.oy = y0;
    p.oz = z0;
    p.w = 2;
    p.h = 3;
    return p;
}

struct Pos { int x, y, z; };

// Standing cells next to a portal (feet cells): one block out from the frame plane, then further out, and
// along the frame. Ordered so the first ones are in front of the interior (team members spread along it).
std::vector<Pos> exitSpots(const World& W, const Portal& p, size_t n) {
    std::vector<Pos> out;
    const int alongX = p.axis == 0 ? 1 : 0, alongZ = p.axis == 0 ? 0 : 1;
    const int normX = p.axis == 0 ? 0 : 1, normZ = p.axis == 0 ? 1 : 0;
    std::vector<int> ts;
    for (int t = 0; t < p.w; t++) ts.push_back(t);
    ts.push_back(-1);
    ts.push_back(p.w);
    ts.push_back(-2);
    ts.push_back(p.w + 1);
    for (int d = 1; d <= 4 && out.size() < n; d++)
        for (int s : {1, -1})
            for (int t : ts) {
                if (out.size() >= n) break;
                const int x = p.ox + alongX * t + normX * s * d;
                const int z = p.oz + alongZ * t + normZ * s * d;
                const int y = p.oy;
                if (at(W, x, y, z) != B_AIR || at(W, x, y + 1, z) != B_AIR || !isFloor(at(W, x, y - 1, z))) continue;
                out.push_back({x, y, z});
            }
    if (out.empty()) out.push_back({p.ox, p.oy, p.oz});
    return out;
}

// Moves the whole team to the other dimension. `fromIdx` is the portal (in the current dimension) the
// trigger agent stands in. Links to the partner portal, creating it at the 1:8 scaled position if none.
void teamMove(World& W, int fromIdx) {
    State& st = W.nether;
    const int from = st.dim, to = 1 - from;
    if (st.otherW.empty() && to == DIM_OVERWORLD) return;   // the overworld is always stored; defensive

    // The incoming block array: built on the first trip to the nether.
    std::vector<uint8_t> incoming = st.otherW.empty() ? buildNether(st.netherSeed) : std::move(st.otherW);
    st.otherW = std::move(W.w);
    W.w = std::move(incoming);

    // Chests follow their dimension (the core's map is converted to the stash layout and back).
    std::map<int, StashedChest> mine;
    for (const auto& kv : W.chests) {
        StashedChest s{};
        for (int i = 0; i < reg::NUM_ITEMS; i++) s.count[i] = kv.second.count[i];
        s.depositMask = kv.second.depositMask;
        mine[kv.first] = s;
    }
    W.chests.clear();
    for (const auto& kv : st.otherChests) {
        Chest c{};
        for (int i = 0; i < reg::NUM_ITEMS; i++) c.count[i] = kv.second.count[i];
        c.depositMask = kv.second.depositMask;
        W.chests[kv.first] = c;
    }
    st.otherChests = std::move(mine);
    std::swap(W.mobs, st.otherMobs);
    st.dim = to;

    // Portal pairing: the source portal's partner, or a new portal at the scaled coordinates.
    int toIdx = st.portals[from][fromIdx].link;
    if (toIdx < 0 || toIdx >= (int)st.portals[to].size()) {
        const Portal src = st.portals[from][fromIdx];
        const int cx = to == DIM_NETHER ? src.ox / 8 : src.ox * 8;
        const int cz = to == DIM_NETHER ? src.oz / 8 : src.oz * 8;
        toIdx = registerPortal(st, to, createPortal(W, to, cx, cz));
        st.portals[from][fromIdx].link = toIdx;
        st.portals[to][toIdx].link = fromIdx;
    }

    // Everyone arrives at the portal's exit cells, in team order.
    const Portal dst = st.portals[to][toIdx];
    const std::vector<Pos> spots = exitSpots(W, dst, st.team.size());
    for (size_t i = 0; i < st.team.size(); i++) {
        Agent& a = *st.team[i];
        a.dim = to;
        if (a.health <= 0) continue;   // dead agents stay frozen where they fell
        const Pos p = spots[i % spots.size()];
        a.px = p.x + 0.5;
        a.py = p.y;
        a.pz = p.z + 0.5;
        a.vx = a.vy = a.vz = 0;
        a.onGround = false;
        a.fallDist = 0;
        a.mx = a.my = a.mz = -1;
        a.prog = 0;
    }
    std::fill(st.timers.begin(), st.timers.end(), 0);
}

}  // namespace

void generate(World& W, uint64_t seed) {
    W.nether = State{};
    W.nether.netherSeed = seed;
}

bool tryLightPortal(World& W, Agent&, int x, int y, int z) {
    State& st = W.nether;
    static const int D[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    for (const auto& d : D) {
        const int nx = x + d[0], ny = y + d[1], nz = z + d[2];
        if (!inBounds(nx, ny, nz) || at(W, nx, ny, nz) != B_AIR) continue;
        Portal p;
        if (!findFrame(W, nx, ny, nz, p)) continue;
        for (int u = 0; u < p.w; u++)
            for (int v = 0; v < p.h; v++) {
                int ix, iy, iz;
                interiorXYZ(p, u, v, ix, iy, iz);
                if (st.portalAt[st.dim].count(cellOf(ix, iy, iz))) return false;   // already lit
            }
        registerPortal(st, st.dim, p);
        return true;
    }
    return false;
}

void travel(World& W, Agent& a) {
    State& st = W.nether;
    size_t slot = 0;
    while (slot < st.team.size() && st.team[slot] != &a) slot++;
    if (slot == st.team.size()) {
        st.team.push_back(&a);
        st.timers.push_back(0);
    }
    if (a.health <= 0) {
        st.timers[slot] = 0;
        return;
    }
    const int fx = (int)std::floor(a.px), fy = (int)std::floor(a.py), fz = (int)std::floor(a.pz);
    int idx = -1;
    if (inBounds(fx, fy, fz)) {
        const auto it = st.portalAt[st.dim].find(cellOf(fx, fy, fz));
        if (it != st.portalAt[st.dim].end()) idx = it->second;
    }
    if (idx < 0) {
        st.timers[slot] = 0;
        return;
    }
    if (++st.timers[slot] < PORTAL_DECISIONS) return;
    teamMove(W, idx);
}

}  // namespace craftsim::nether
