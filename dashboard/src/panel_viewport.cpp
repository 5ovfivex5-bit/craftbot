// Viewport: renders the snapshot's voxel region into an FBO and shows it in ImGui.
// The face-culled cube mesh is rebuilt only when a new snapshot arrives.
// Palette, face shading and fog follow src/craftsim.cpp renderRGB().

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <imgui.h>

#include "app.h"
#include "gl_procs.h"
#include "net/protocol.h"
#include "panels.h"
#include "ui_util.h"

namespace {

constexpr float kEyeHeight = 1.62f;   // src/craftsim.cpp EYE
constexpr float kPlayerW = 0.6f;      // AABB 0.6 x 1.8
constexpr float kPlayerH = 1.8f;
constexpr float kFovDeg = 70.0f;      // display FOV (sim FOV is only used for its own renders)
constexpr float kViewDist = 24.0f;    // src/craftsim.cpp VIEW_DIST (fog range)
constexpr float kThirdDist = 4.0f;

const char* kVS = R"(#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aCol;
uniform mat4 uVP;
out vec3 vCol;
out vec3 vPos;
void main() {
    vPos = aPos;
    vCol = aCol;
    gl_Position = uVP * vec4(aPos, 1.0);
}
)";

const char* kFS = R"(#version 330 core
in vec3 vCol;
in vec3 vPos;
uniform vec3 uCam;
uniform vec3 uSky;
out vec4 frag;
void main() {
    float d = distance(vPos, uCam) / 24.0;  // VIEW_DIST of the sim
    float fog = clamp(d * d, 0.0, 1.0);
    frag = vec4(mix(vCol, uSky, fog), 1.0);
}
)";

struct Vtx {
    float p[3];
    float c[3];
};

struct ViewGL {
    bool ready = false;
    bool failed = false;
    unsigned prog = 0, vao = 0, vbo_mesh = 0, vbo_ov = 0;
    unsigned fbo = 0, rbo = 0, tex = 0;
    int fbo_w = 0, fbo_h = 0;
    int u_vp = -1, u_cam = -1, u_sky = -1;
    int mesh_count = 0;

    std::shared_ptr<const mm::Snapshot> mesh_src;  // snapshot the mesh was built from

    // Free camera state (world units, degrees).
    bool free_init = false;
    float fx = 0, fy = 0, fz = 0, fyaw = 0, fpitch = 0;
    bool free_from_snap = false;

    std::vector<Vtx> mesh_cpu;
    std::vector<Vtx> self_cpu;      // selected agent box (triangles; hidden in first person)
    std::vector<Vtx> tri_cpu;       // other team members and chests (triangles)
    std::vector<Vtx> outline_cpu;   // selected agent outline (lines; hidden in first person)
    std::vector<Vtx> line_cpu;      // target block outline (lines)
};

ViewGL g;

// ---------------------------------------------------------------- GL helpers

unsigned compile(unsigned type, const char* src, const char* name) {
    unsigned s = glp::CreateShader(type);
    glp::ShaderSource(s, 1, &src, nullptr);
    glp::CompileShader(s);
    int ok = 0;
    glp::GetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glp::GetShaderInfoLog(s, sizeof log, nullptr, log);
        std::fprintf(stderr, "%s shader compile failed: %s\n", name, log);
        glp::DeleteShader(s);
        return 0;
    }
    return s;
}

bool init_gl() {
    if (g.ready) return true;
    if (g.failed) return false;
    unsigned vs = compile(GL_VERTEX_SHADER, kVS, "viewport vertex");
    unsigned fs = compile(GL_FRAGMENT_SHADER, kFS, "viewport fragment");
    if (!vs || !fs) {
        g.failed = true;
        return false;
    }
    g.prog = glp::CreateProgram();
    glp::AttachShader(g.prog, vs);
    glp::AttachShader(g.prog, fs);
    glp::LinkProgram(g.prog);
    int ok = 0;
    glp::GetProgramiv(g.prog, GL_LINK_STATUS, &ok);
    glp::DeleteShader(vs);
    glp::DeleteShader(fs);
    if (!ok) {
        char log[1024];
        glp::GetProgramInfoLog(g.prog, sizeof log, nullptr, log);
        std::fprintf(stderr, "viewport program link failed: %s\n", log);
        g.failed = true;
        return false;
    }
    g.u_vp = glp::GetUniformLocation(g.prog, "uVP");
    g.u_cam = glp::GetUniformLocation(g.prog, "uCam");
    g.u_sky = glp::GetUniformLocation(g.prog, "uSky");

    glp::GenVertexArrays(1, &g.vao);
    glp::GenBuffers(1, &g.vbo_mesh);
    glp::GenBuffers(1, &g.vbo_ov);
    g.ready = true;
    return true;
}

void ensure_fbo(int w, int h) {
    if (g.fbo && g.fbo_w == w && g.fbo_h == h) return;
    if (!g.fbo) {
        glp::GenFramebuffers(1, &g.fbo);
        glp::GenRenderbuffers(1, &g.rbo);
        glGenTextures(1, &g.tex);
    }
    glBindTexture(GL_TEXTURE_2D, g.tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glp::BindRenderbuffer(GL_RENDERBUFFER, g.rbo);
    glp::RenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    glp::BindFramebuffer(GL_FRAMEBUFFER, g.fbo);
    glp::FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g.tex, 0);
    glp::FramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, g.rbo);
    glp::BindFramebuffer(GL_FRAMEBUFFER, 0);
    g.fbo_w = w;
    g.fbo_h = h;
}

// ---------------------------------------------------------------- geometry

void emit_face(std::vector<Vtx>& out, const float lo[3], const float hi[3], const float col[3]) {
    int a = (lo[0] == hi[0]) ? 0 : (lo[1] == hi[1]) ? 1 : 2;  // axis the face is flat on
    int b = (a + 1) % 3, c = (a + 2) % 3;
    static const float corner[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    Vtx v[4];
    for (int k = 0; k < 4; k++) {
        v[k].p[a] = lo[a];
        v[k].p[b] = lo[b] + corner[k][0] * (hi[b] - lo[b]);
        v[k].p[c] = lo[c] + corner[k][1] * (hi[c] - lo[c]);
        std::memcpy(v[k].c, col, sizeof v[k].c);
    }
    out.push_back(v[0]);
    out.push_back(v[1]);
    out.push_back(v[2]);
    out.push_back(v[0]);
    out.push_back(v[2]);
    out.push_back(v[3]);
}

void emit_box(std::vector<Vtx>& out, const float lo[3], const float hi[3], const float col[3]) {
    for (int a = 0; a < 3; a++)
        for (int s = 0; s < 2; s++) {
            float l[3] = {lo[0], lo[1], lo[2]}, h[3] = {hi[0], hi[1], hi[2]};
            float shade = a == 1 ? (s ? 1.0f : 0.5f) : (a == 0 ? 0.8f : 0.68f);
            float c[3] = {col[0] * shade, col[1] * shade, col[2] * shade};
            if (s) l[a] = h[a]; else h[a] = l[a];
            emit_face(out, l, h, c);
        }
}

float rnd01(int x, int y, int z, int face) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u + (uint32_t)z * 2246822519u +
                 (uint32_t)face * 3266489917u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return (h & 255) / 255.0f;
}

void build_mesh(const mm::Snapshot& s) {
    g.mesh_cpu.clear();
    const int sx = s.size[0], sy = s.size[1], sz = s.size[2];
    auto at = [&](int x, int y, int z) -> int {
        if (x < 0 || y < 0 || z < 0 || x >= sx || y >= sy || z >= sz) return 0;
        return s.blocks[((size_t)y * sz + z) * sx + x];
    };
    // Normal direction of the six faces: +X, -X, +Y, -Y, +Z, -Z.
    static const int nrm[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    if (s.blocks.empty()) return;
    for (int y = 0; y < sy; y++)
        for (int z = 0; z < sz; z++)
            for (int x = 0; x < sx; x++) {
                int b = at(x, y, z);
                if (b == 0) continue;
                const mm::BlockStyle& st = mm::block_style(b);
                if (st.pseudo) continue;  // agent / mob ids are entities, not voxels
                for (int f = 0; f < 6; f++) {
                    if (at(x + nrm[f][0], y + nrm[f][1], z + nrm[f][2]) != 0) continue;  // face culled
                    // Per-face texture jitter and speckle, mirroring the sim's per-texel hash at face scale.
                    float jitter = 0.88f + 0.24f * rnd01(x, y, z, f);
                    float base[3];
                    const uint8_t* rgb = st.rgb;
                    if (st.has_speckle && rnd01(z, x, y, f + 7) < 0.25f) rgb = st.speckle;
                    // Face shade follows renderRGB(): top 1.0, X-faces 0.8, Z-faces 0.68, bottom 0.5.
                    float shade = f == 2 ? 1.0f : f == 3 ? 0.5f : (f < 2 ? 0.8f : 0.68f);
                    for (int k = 0; k < 3; k++) base[k] = rgb[k] / 255.0f * shade * jitter;
                    float lo[3] = {(float)x, (float)y, (float)z};
                    float hi[3] = {x + 1.0f, y + 1.0f, z + 1.0f};
                    int axis = f / 2;
                    float plane = nrm[f][axis] > 0 ? (float)(lo[axis] + 1) : (float)lo[axis];
                    lo[axis] = hi[axis] = plane;
                    if (b == 1 && axis != 1) {
                        // Grass: side faces show dirt below 80% height, grass band on top (as in renderRGB).
                        float dl[3] = {lo[0], lo[1], lo[2]}, dh[3] = {hi[0], hi[1] * 0.8f + lo[1] * 0.2f, hi[2]};
                        float gl[3] = {lo[0], hi[1] * 0.8f + lo[1] * 0.2f, lo[2]}, gh[3] = {hi[0], hi[1], hi[2]};
                        dh[1] = y + 0.8f;
                        gl[1] = y + 0.8f;
                        float dirt[3], grass[3];
                        for (int k = 0; k < 3; k++) {
                            dirt[k] = mm::block_style(2).rgb[k] / 255.0f * shade * jitter;
                            grass[k] = base[k];
                        }
                        emit_face(g.mesh_cpu, dl, dh, dirt);
                        emit_face(g.mesh_cpu, gl, gh, grass);
                    } else {
                        emit_face(g.mesh_cpu, lo, hi, base);
                    }
                }
            }
    // Blocks are indexed relative to the snapshot origin; the player/camera are in world space.
    for (Vtx& v : g.mesh_cpu)
        for (int k = 0; k < 3; k++) v.p[k] += (float)s.origin[k];
    g.mesh_count = (int)g.mesh_cpu.size();
}

// Wireframe box (12 edges) as line-list vertices, optionally inflated by e.
void emit_wire_box(std::vector<Vtx>& out, const float lo0[3], const float hi0[3], const float col[3], float e) {
    float lo[3] = {lo0[0] - e, lo0[1] - e, lo0[2] - e};
    float hi[3] = {hi0[0] + e, hi0[1] + e, hi0[2] + e};
    for (int a = 0; a < 3; a++)
        for (int m = 0; m < 4; m++) {
            int b = (a + 1) % 3, c = (a + 2) % 3;
            float p0[3], p1[3];
            p0[a] = lo[a]; p1[a] = hi[a];
            p0[b] = p1[b] = (m & 1) ? hi[b] : lo[b];
            p0[c] = p1[c] = (m & 2) ? hi[c] : lo[c];
            Vtx v0{}, v1{};
            std::memcpy(v0.p, p0, sizeof p0);
            std::memcpy(v1.p, p1, sizeof p1);
            std::memcpy(v0.c, col, sizeof v0.c);
            std::memcpy(v1.c, col, sizeof v1.c);
            out.push_back(v0);
            out.push_back(v1);
        }
}

// Player-sized box with its feet at (x, y, z), as used for every team member.
void member_box(float x, float y, float z, float lo[3], float hi[3]) {
    lo[0] = x - kPlayerW * 0.5f; lo[1] = y;             lo[2] = z - kPlayerW * 0.5f;
    hi[0] = x + kPlayerW * 0.5f; hi[1] = y + kPlayerH;  hi[2] = z + kPlayerW * 0.5f;
}

void build_overlay(const mm::Snapshot& s) {
    g.self_cpu.clear();
    g.tri_cpu.clear();
    g.outline_cpu.clear();
    g.line_cpu.clear();

    // Selected agent (the snapshot's player). Pre-v6 trainers: the old red player box.
    float pl[3], ph[3];
    member_box((float)s.player.x, (float)s.player.y, (float)s.player.z, pl, ph);
    float self_col[3] = {0.95f, 0.45f, 0.30f};
    const float white[3] = {1.0f, 1.0f, 1.0f};
    if (s.has_team) {
        mm::Rgb c = mm::bot_color(s.agent_id);
        self_col[0] = c.r / 255.0f;
        self_col[1] = c.g / 255.0f;
        self_col[2] = c.b / 255.0f;
        emit_wire_box(g.outline_cpu, pl, ph, white, 0.03f);  // highlight
    }
    emit_box(g.self_cpu, pl, ph, self_col);

    if (s.has_team) {
        for (const mm::TeamMember& m : s.team) {
            if (m.agent == s.agent_id) continue;  // drawn above from the player state
            float lo[3], hi[3];
            member_box((float)m.x, (float)m.y, (float)m.z, lo, hi);
            mm::Rgb c = mm::bot_color(m.agent);
            float col[3] = {c.r / 255.0f, c.g / 255.0f, c.b / 255.0f};
            emit_box(g.tri_cpu, lo, hi, col);
        }
    }
    // Chests: brown boxes, slightly inflated so they sit over the block cube without z-fighting.
    const mm::BlockStyle& chest = mm::block_style(18);
    float chest_col[3] = {chest.rgb[0] / 255.0f, chest.rgb[1] / 255.0f, chest.rgb[2] / 255.0f};
    for (const mm::ChestInfo& c : s.chests) {
        float lo[3] = {(float)c.x, (float)c.y, (float)c.z};
        float hi[3] = {c.x + 1.0f, c.y + 1.0f, c.z + 1.0f};
        for (int k = 0; k < 3; k++) { lo[k] -= 0.03f; hi[k] += 0.03f; }
        emit_box(g.tri_cpu, lo, hi, chest_col);
    }

    // Entities (mobs) from the snapshot's optional list: coloured boxes sized from the registry.
    for (const mm::Entity& e : s.entities) {
        mm::MobLook ml = mm::mob_look(e.type);
        float lo[3] = {(float)e.x - ml.width * 0.5f, (float)e.y, (float)e.z - ml.width * 0.5f};
        float hi[3] = {(float)e.x + ml.width * 0.5f, (float)e.y + ml.height, (float)e.z + ml.width * 0.5f};
        float col[3] = {ml.color.r / 255.0f, ml.color.g / 255.0f, ml.color.b / 255.0f};
        emit_box(g.tri_cpu, lo, hi, col);
    }

    if (s.target) {
        float lo[3] = {(float)s.target->x, (float)s.target->y, (float)s.target->z};
        float hi[3] = {s.target->x + 1.0f, s.target->y + 1.0f, s.target->z + 1.0f};
        emit_wire_box(g.line_cpu, lo, hi, white, 0.01f);
    }
}

// Column-major 4x4 helpers.
void mat_perspective(float fovy, float aspect, float n, float f, float* m) {
    std::memset(m, 0, sizeof(float) * 16);
    float t = 1.0f / std::tan(fovy * 0.5f);
    m[0] = t / aspect;
    m[5] = t;
    m[10] = (f + n) / (n - f);
    m[11] = -1.0f;
    m[14] = 2.0f * f * n / (n - f);
}

void mat_lookat(const float* eye, const float* center, const float* up, float* m) {
    float f[3] = {center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]};
    float fl = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    for (float& v : f) v /= fl;
    float s[3] = {f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]};
    float sl = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    for (float& v : s) v /= sl;
    float u[3] = {s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2], s[0] * f[1] - s[1] * f[0]};
    m[0] = s[0]; m[4] = s[1]; m[8] = s[2];
    m[1] = u[0]; m[5] = u[1]; m[9] = u[2];
    m[2] = -f[0]; m[6] = -f[1]; m[10] = -f[2];
    m[12] = -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]);
    m[13] = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
    m[14] = f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2];
    m[3] = m[7] = m[11] = 0.0f;
    m[15] = 1.0f;
}

void mat_mul(const float* a, const float* b, float* out) {
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++) {
            float v = 0;
            for (int k = 0; k < 4; k++) v += a[k * 4 + r] * b[c * 4 + k];
            out[c * 4 + r] = v;
        }
}

// Sim convention (src/craftsim.cpp lookDir): forward from yaw/pitch in degrees.
void look_dir(float yaw_deg, float pitch_deg, float* f) {
    float y = yaw_deg * 3.14159265f / 180.0f, p = pitch_deg * 3.14159265f / 180.0f;
    f[0] = -std::sin(y) * std::cos(p);
    f[1] = -std::sin(p);
    f[2] = std::cos(y) * std::cos(p);
}

// Sky colour for the snapshot's time_of_day (PROTOCOL: [0,1), 0 dawn, 0.25 noon, 0.5 dusk, 0.75 midnight).
// Without it the sky stays at the day colour, as before.
void sky_colour(const std::optional<double>& tod, float* out) {
    const float day[3] = {135 / 255.0f, 190 / 255.0f, 255 / 255.0f};
    if (!tod) {
        std::memcpy(out, day, sizeof day);
        return;
    }
    const float night[3] = {10 / 255.0f, 14 / 255.0f, 36 / 255.0f};
    const float dusk[3] = {255 / 255.0f, 150 / 255.0f, 90 / 255.0f};
    const float elev = (float)std::sin(*tod * 2.0 * 3.14159265358979);
    auto smooth = [](float e0, float e1, float x) {
        float t = std::clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
        return t * t * (3.0f - 2.0f * t);
    };
    const float day_amt = smooth(-0.12f, 0.25f, elev);
    const float warm = std::max(0.0f, 1.0f - std::fabs(elev) / 0.3f) * 0.6f;  // horizon glow at dawn and dusk
    for (int k = 0; k < 3; k++) {
        float c = night[k] + (day[k] - night[k]) * day_amt;
        out[k] = c + (dusk[k] - c) * warm;
    }
}

// Uploads one vertex group into the overlay buffer and draws it (no-op when empty).
void draw_vtx_group(const std::vector<Vtx>& v, GLenum mode) {
    if (v.empty()) return;
    glp::BufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(v.size() * sizeof(Vtx)), v.data(), GL_STREAM_DRAW);
    glDrawArrays(mode, 0, (GLsizei)v.size());
}

// Ray-vs-chest picking helper: true when a solid block lies between the eye and distance t.
bool occluded(const mm::Snapshot& s, const float eye[3], const float dir[3], float t) {
    const int sx = s.size[0], sy = s.size[1], sz = s.size[2];
    if (s.blocks.empty()) return false;
    for (float d = 0.05f; d < t - 0.05f; d += 0.05f) {
        int x = (int)std::floor(eye[0] + dir[0] * d) - s.origin[0];
        int y = (int)std::floor(eye[1] + dir[1] * d) - s.origin[1];
        int z = (int)std::floor(eye[2] + dir[2] * d) - s.origin[2];
        if (x < 0 || y < 0 || z < 0 || x >= sx || y >= sy || z >= sz) continue;
        if (s.blocks[((size_t)y * sz + z) * sx + x] != 0) return true;
    }
    return false;
}

void draw_overlay_text(ImDrawList* dl, ImVec2 origin, const std::vector<std::string>& lines) {
    float lh = ImGui::GetTextLineHeight() + 2;
    float w = 0;
    for (const auto& l : lines) w = std::max(w, ImGui::CalcTextSize(l.c_str()).x);
    dl->AddRectFilled(ImVec2(origin.x + 6, origin.y + 6), ImVec2(origin.x + 14 + w, origin.y + 10 + lh * lines.size()),
                      IM_COL32(8, 10, 14, 170), 4.0f);
    float y = origin.y + 8;
    for (const auto& l : lines) {
        dl->AddText(ImVec2(origin.x + 10, y), IM_COL32(225, 232, 240, 255), l.c_str());
        y += lh;
    }
}

}  // namespace

void upload_texture(unsigned& tex, int w, int h, const unsigned char* rgba) {
    if (!tex) {
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    } else {
        glBindTexture(GL_TEXTURE_2D, tex);
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
}

void draw_viewport(App& app, bool* p_open) {
    app.client.set_snapshot_wanted(app.viz_enabled && *p_open);

    ImGui::SetNextWindowSize(ImVec2(960, 620), ImGuiCond_FirstUseEver);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4, 4));
    bool visible = ImGui::Begin(win::kViewport, p_open, ImGuiWindowFlags_NoScrollbar);
    ImGui::PopStyleVar();
    if (!visible) {
        ImGui::End();
        return;
    }

    // Toolbar.
    bool viz = app.viz_enabled;
    if (ImGui::Checkbox("Visualization enabled", &viz)) app.set_visualize(viz);
    ui::help_marker(
        "Streams the selected environment's voxels, player and observation to this viewport. "
        "Off means no snapshot traffic and no rendering cost on the trainer.");
    ImGui::SameLine(0, 16);
    int num_agents = trainer_connected(app) ? selectable_agents(app) : 0;
    ImGui::SetNextItemWidth(220);
    if (num_agents > 0) {
        if (ImGui::BeginCombo("##env", agent_label(app, app.selected_env).c_str())) {
            for (int i = 0; i < num_agents; i++) {
                bool sel = i == app.selected_env;
                if (ImGui::Selectable(agent_label(app, i).c_str(), sel) && !sel) app.set_selected_env(i);
                if (sel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
    } else {
        ImGui::TextDisabled("Env: no data");
    }
    ImGui::SameLine(0, 16);
    ImGui::TextUnformatted("Camera:");
    const char* cams[] = {"1 First person", "2 Third person", "3 Free"};
    for (int i = 0; i < 3; i++) {
        ImGui::SameLine();
        if (ImGui::RadioButton(cams[i], (int)app.cam_mode == i)) app.cam_mode = (CamMode)i;
    }
    if (app.cam_mode == CamMode::Free) {
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset free cam")) g.free_init = g.free_from_snap = false;
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    int w = std::max(64, (int)avail.x);
    int h = std::max(64, (int)avail.y);

    const mm::Frame* f = app.f.get();
    std::shared_ptr<const mm::Snapshot> snap = f ? f->snapshot : nullptr;
    bool connected = trainer_connected(app);

    if (!init_gl()) {
        ImGui::TextColored(ImVec4(1, 0.5f, 0.4f, 1), "OpenGL viewport failed to initialise (see stderr).");
        ImGui::End();
        return;
    }

    if (snap && snap != g.mesh_src) {
        build_mesh(*snap);
        build_overlay(*snap);
        g.mesh_src = snap;
    }

    // Camera.
    float eye[3] = {0, 0, 0}, center[3] = {0, 0, 1}, up[3] = {0, 1, 0};
    float fwd[3] = {0, 0, 1};
    if (snap) {
        const mm::Player& p = snap->player;
        float eye_fp[3] = {(float)p.x, (float)(p.y + kEyeHeight), (float)p.z};
        look_dir((float)p.yaw, (float)p.pitch, fwd);
        if (app.cam_mode == CamMode::FirstPerson) {
            for (int k = 0; k < 3; k++) {
                eye[k] = eye_fp[k];
                center[k] = eye_fp[k] + fwd[k];
            }
        } else if (app.cam_mode == CamMode::ThirdPerson) {
            for (int k = 0; k < 3; k++) {
                eye[k] = eye_fp[k] - fwd[k] * kThirdDist;
                center[k] = eye_fp[k];
            }
            eye[1] = std::max(eye[1], (float)p.y + 0.5f);
        }
    }
    if (app.cam_mode == CamMode::Free || !snap) {
        // Free camera, also used as the only camera before the first snapshot.
        // Initialise behind/above the player once a snapshot exists (re-init if it only had the fallback).
        if (!g.free_init || (snap && !g.free_from_snap)) {
            if (snap) {
                const mm::Player& p = snap->player;
                float yr = (float)p.yaw * 3.14159265f / 180.0f;
                g.fx = (float)p.x + std::sin(yr) * 10.0f;  // 10 blocks behind (forward is (-sin, cos))
                g.fy = (float)p.y + kEyeHeight + 6.0f;
                g.fz = (float)p.z - std::cos(yr) * 10.0f;
                g.fyaw = (float)p.yaw;
                g.fpitch = 25.0f;  // sim convention: positive pitch looks down
                g.free_from_snap = true;
            } else {
                g.fx = 0;
                g.fy = 40;
                g.fz = 0;
                g.fyaw = 0;
                g.fpitch = 30;
            }
            g.free_init = true;
        }
        ImGuiIO& io = ImGui::GetIO();
        bool hovering = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
        if (hovering && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            g.fyaw -= io.MouseDelta.x * 0.2f;
            g.fpitch = std::clamp(g.fpitch + io.MouseDelta.y * 0.2f, -89.0f, 89.0f);
        }
        if (hovering) {
            float dt = io.DeltaTime;
            float speed = 8.0f * (io.KeyShift ? 3.0f : 1.0f) * dt;
            float ff[3];
            look_dir(g.fyaw, g.fpitch, ff);
            float yr = g.fyaw * 3.14159265f / 180.0f;
            float rr[3] = {-std::cos(yr), 0.0f, -std::sin(yr)};
            auto down = [](ImGuiKey k) { return ImGui::IsKeyDown(k); };
            if (down(ImGuiKey_W)) { g.fx += ff[0] * speed; g.fy += ff[1] * speed; g.fz += ff[2] * speed; }
            if (down(ImGuiKey_S)) { g.fx -= ff[0] * speed; g.fy -= ff[1] * speed; g.fz -= ff[2] * speed; }
            if (down(ImGuiKey_D)) { g.fx += rr[0] * speed; g.fz += rr[2] * speed; }
            if (down(ImGuiKey_A)) { g.fx -= rr[0] * speed; g.fz -= rr[2] * speed; }
            if (down(ImGuiKey_E)) g.fy += speed;
            if (down(ImGuiKey_Q)) g.fy -= speed;
        }
        eye[0] = g.fx;
        eye[1] = g.fy;
        eye[2] = g.fz;
        look_dir(g.fyaw, g.fpitch, fwd);
        for (int k = 0; k < 3; k++) center[k] = eye[k] + fwd[k];
    }

    // Derive up from the horizontal right vector so looking straight up/down (pitch +-90)
    // doesn't make the look-at basis degenerate.
    {
        float yaw = app.cam_mode == CamMode::Free || !snap ? g.fyaw : (float)snap->player.yaw;
        float yr = yaw * 3.14159265f / 180.0f;
        float r[3] = {-std::cos(yr), 0.0f, -std::sin(yr)};
        float d[3] = {center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]};
        up[0] = r[1] * d[2] - r[2] * d[1];
        up[1] = r[2] * d[0] - r[0] * d[2];
        up[2] = r[0] * d[1] - r[1] * d[0];
    }

    float proj[16], view[16], vp[16];
    mat_perspective(kFovDeg * 3.14159265f / 180.0f, (float)w / (float)h, 0.05f, 300.0f, proj);
    mat_lookat(eye, center, up, view);
    mat_mul(proj, view, vp);

    // Render into the FBO.
    ensure_fbo(w, h);
    glp::BindFramebuffer(GL_FRAMEBUFFER, g.fbo);
    glViewport(0, 0, w, h);
    float sky[3];
    sky_colour(snap ? snap->time_of_day : std::nullopt, sky);
    glClearColor(sky[0], sky[1], sky[2], 1.0f);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    glDisable(GL_SCISSOR_TEST);

    if (snap && g.mesh_count > 0) {
        glp::UseProgram(g.prog);
        glp::UniformMatrix4fv(g.u_vp, 1, GL_FALSE, vp);
        glp::Uniform3f(g.u_cam, eye[0], eye[1], eye[2]);
        glp::Uniform3f(g.u_sky, sky[0], sky[1], sky[2]);
        glp::BindVertexArray(g.vao);
        glp::BindBuffer(GL_ARRAY_BUFFER, g.vbo_mesh);
        glp::EnableVertexAttribArray(0);
        glp::VertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)offsetof(Vtx, p));
        glp::EnableVertexAttribArray(1);
        glp::VertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)offsetof(Vtx, c));
        glp::BufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(g.mesh_cpu.size() * sizeof(Vtx)), g.mesh_cpu.data(), GL_STATIC_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, g.mesh_count);
    }
    // Overlay: team and chest boxes (depth tested), the selected agent's box and outline
    // (hidden in first person, where its box would enclose the eye), then line overlays.
    if (snap) {
        glp::UseProgram(g.prog);
        glp::UniformMatrix4fv(g.u_vp, 1, GL_FALSE, vp);
        glp::BindVertexArray(g.vao);
        glp::BindBuffer(GL_ARRAY_BUFFER, g.vbo_ov);
        glp::EnableVertexAttribArray(0);
        glp::VertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)offsetof(Vtx, p));
        glp::EnableVertexAttribArray(1);
        glp::VertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)offsetof(Vtx, c));
        const bool first = app.cam_mode == CamMode::FirstPerson;
        draw_vtx_group(g.tri_cpu, GL_TRIANGLES);
        if (!first) draw_vtx_group(g.self_cpu, GL_TRIANGLES);
        glDisable(GL_DEPTH_TEST);
        if (!first) draw_vtx_group(g.outline_cpu, GL_LINES);
        draw_vtx_group(g.line_cpu, GL_LINES);
    }
    glp::BindVertexArray(0);
    glp::UseProgram(0);
    glp::BindFramebuffer(GL_FRAMEBUFFER, 0);

    // Show the FBO. FBO rows are bottom-up, so flip V.
    ImVec2 img_pos = ImGui::GetCursorScreenPos();
    ImGui::Image((ImTextureID)(intptr_t)g.tex, ImVec2((float)w, (float)h), ImVec2(0, 1), ImVec2(1, 0));
    bool img_hovered = ImGui::IsItemHovered();
    (void)img_hovered;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 img_max(img_pos.x + w, img_pos.y + h);

    // Team labels: world -> screen with the same matrix as the scene.
    auto to_screen = [&](float x, float y, float z, ImVec2& out) -> bool {
        float cx = vp[0] * x + vp[4] * y + vp[8] * z + vp[12];
        float cy = vp[1] * x + vp[5] * y + vp[9] * z + vp[13];
        float cw = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
        if (cw <= 0.05f) return false;
        float nx = cx / cw, ny = cy / cw;
        if (nx < -1.1f || nx > 1.1f || ny < -1.1f || ny > 1.1f) return false;
        out = ImVec2(img_pos.x + (nx * 0.5f + 0.5f) * w, img_pos.y + (0.5f - ny * 0.5f) * h);
        return true;
    };
    if (snap && snap->has_team) {
        for (const mm::TeamMember& m : snap->team) {
            const bool is_sel = m.agent == snap->agent_id;
            if (is_sel && app.cam_mode == CamMode::FirstPerson) continue;
            double x = is_sel ? snap->player.x : m.x, y = is_sel ? snap->player.y : m.y, z = is_sel ? snap->player.z : m.z;
            ImVec2 p;
            if (!to_screen((float)x, (float)(y + kPlayerH + 0.3), (float)z, p)) continue;
            char txt[48];
            std::snprintf(txt, sizeof txt, is_sel ? "bot %d (selected)" : "bot %d", m.agent);
            mm::Rgb c = mm::bot_color(m.agent);
            ImVec2 ts = ImGui::CalcTextSize(txt);
            dl->AddRectFilled(ImVec2(p.x - ts.x * 0.5f - 4, p.y - ts.y * 0.5f - 2),
                              ImVec2(p.x + ts.x * 0.5f + 4, p.y + ts.y * 0.5f + 2), IM_COL32(8, 10, 14, 185), 3.0f);
            dl->AddText(ImVec2(p.x - ts.x * 0.5f, p.y - ts.y * 0.5f), IM_COL32(c.r, c.g, c.b, 255), txt);
        }
    }

    // Entity labels: type and health, placed above each box.
    if (snap) {
        for (const mm::Entity& e : snap->entities) {
            mm::MobLook ml = mm::mob_look(e.type);
            ImVec2 p;
            if (!to_screen((float)e.x, (float)(e.y + ml.height + 0.2), (float)e.z, p)) continue;
            std::string name = e.type.rfind("minecraft:", 0) == 0 ? e.type.substr(10) : e.type;
            char txt[96];
            if (e.health) std::snprintf(txt, sizeof txt, "%s  %.0f hp", name.c_str(), *e.health);
            else std::snprintf(txt, sizeof txt, "%s", name.c_str());
            ImVec2 ts = ImGui::CalcTextSize(txt);
            // Text colour: the mob colour lifted towards white so dark mobs stay readable.
            ImU32 tc = IM_COL32((ml.color.r + 255) / 2, (ml.color.g + 255) / 2, (ml.color.b + 255) / 2, 255);
            dl->AddRectFilled(ImVec2(p.x - ts.x * 0.5f - 4, p.y - ts.y * 0.5f - 2),
                              ImVec2(p.x + ts.x * 0.5f + 4, p.y + ts.y * 0.5f + 2), IM_COL32(8, 10, 14, 185), 3.0f);
            dl->AddText(ImVec2(p.x - ts.x * 0.5f, p.y - ts.y * 0.5f), tc, txt);
        }
    }

    // Chest tooltip: nearest chest under the mouse that no solid block hides.
    if (img_hovered && snap && !snap->chests.empty()) {
        ImVec2 mp = ImGui::GetIO().MousePos;
        float nx = (mp.x - img_pos.x) / w * 2.0f - 1.0f;
        float ny = 1.0f - (mp.y - img_pos.y) / h * 2.0f;
        float fc[3] = {center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]};
        float fl = std::sqrt(fc[0] * fc[0] + fc[1] * fc[1] + fc[2] * fc[2]);
        for (float& v : fc) v /= fl;
        float sr[3] = {fc[1] * up[2] - fc[2] * up[1], fc[2] * up[0] - fc[0] * up[2], fc[0] * up[1] - fc[1] * up[0]};
        float sl = std::sqrt(sr[0] * sr[0] + sr[1] * sr[1] + sr[2] * sr[2]);
        for (float& v : sr) v /= sl;
        float ur[3] = {sr[1] * fc[2] - sr[2] * fc[1], sr[2] * fc[0] - sr[0] * fc[2], sr[0] * fc[1] - sr[1] * fc[0]};
        float tn = std::tan(kFovDeg * 3.14159265f / 360.0f);
        float aspect = (float)w / (float)h;
        float dir[3];
        for (int k = 0; k < 3; k++) dir[k] = fc[k] + sr[k] * nx * aspect * tn + ur[k] * ny * tn;
        float dl2 = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
        for (float& v : dir) v /= dl2;

        const mm::ChestInfo* best = nullptr;
        float best_t = 1e30f;
        for (const mm::ChestInfo& c : snap->chests) {
            float lo[3] = {(float)c.x, (float)c.y, (float)c.z};
            float hi[3] = {c.x + 1.0f, c.y + 1.0f, c.z + 1.0f};
            float t0 = 0.0f, t1 = 1e30f;
            bool miss = false;
            for (int a = 0; a < 3 && !miss; a++) {
                if (std::fabs(dir[a]) < 1e-8f) {
                    if (eye[a] < lo[a] || eye[a] > hi[a]) miss = true;
                    continue;
                }
                float ta = (lo[a] - eye[a]) / dir[a], tb = (hi[a] - eye[a]) / dir[a];
                if (ta > tb) std::swap(ta, tb);
                t0 = std::max(t0, ta);
                t1 = std::min(t1, tb);
                if (t0 > t1) miss = true;
            }
            if (miss || t1 < 0.0f) continue;
            if (t0 < best_t && !occluded(*snap, eye, dir, t0)) {
                best_t = t0;
                best = &c;
            }
        }
        if (best) {
            ImGui::BeginTooltip();
            ImGui::Text("Chest at %d, %d, %d", best->x, best->y, best->z);
            if (best->items.empty()) {
                ImGui::TextDisabled("empty");
            } else {
                for (const auto& [item, n] : best->items) ImGui::Text("  %s  x%d", item.c_str(), n);
            }
            ImGui::EndTooltip();
        }
    }

    // Crosshair in first person.
    if (app.cam_mode == CamMode::FirstPerson && snap) {
        ImVec2 c((img_pos.x + img_max.x) * 0.5f, (img_pos.y + img_max.y) * 0.5f);
        dl->AddLine(ImVec2(c.x - 6, c.y), ImVec2(c.x + 6, c.y), IM_COL32(255, 255, 255, 200));
        dl->AddLine(ImVec2(c.x, c.y - 6), ImVec2(c.x, c.y + 6), IM_COL32(255, 255, 255, 200));
    }

    if (!connected) {
        ImVec2 tsz = ImGui::CalcTextSize("No data - trainer not connected");
        ImVec2 p((img_pos.x + img_max.x) * 0.5f - tsz.x * 0.5f, img_pos.y + 12);
        dl->AddRectFilled(ImVec2(p.x - 8, p.y - 4), ImVec2(p.x + tsz.x + 8, p.y + tsz.y + 4), IM_COL32(0, 0, 0, 160), 4);
        dl->AddText(p, IM_COL32(240, 180, 80, 255), "No data - trainer not connected");
    } else if (!app.viz_enabled) {
        ImVec2 tsz = ImGui::CalcTextSize("Visualization disabled - tick the checkbox above to stream");
        ImVec2 p((img_pos.x + img_max.x) * 0.5f - tsz.x * 0.5f, img_pos.y + 12);
        dl->AddRectFilled(ImVec2(p.x - 8, p.y - 4), ImVec2(p.x + tsz.x + 8, p.y + tsz.y + 4), IM_COL32(0, 0, 0, 160), 4);
        dl->AddText(p, IM_COL32(200, 210, 225, 255), "Visualization disabled - tick the checkbox above to stream");
    } else if (!snap) {
        ImVec2 tsz = ImGui::CalcTextSize("Waiting for the first snapshot...");
        ImVec2 p((img_pos.x + img_max.x) * 0.5f - tsz.x * 0.5f, img_pos.y + 12);
        dl->AddRectFilled(ImVec2(p.x - 8, p.y - 4), ImVec2(p.x + tsz.x + 8, p.y + tsz.y + 4), IM_COL32(0, 0, 0, 160), 4);
        dl->AddText(p, IM_COL32(200, 210, 225, 255), "Waiting for the first snapshot...");
    }

    if (snap) {
        const mm::Snapshot& s = *snap;
        const mm::Player& p = s.player;
        float speed = std::sqrt((float)(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz));
        char l1[160], l2[160], l3[160], l4[160], l5[160], l6[160];
        if (s.has_team)
            std::snprintf(l1, sizeof l1, "agent %d   world %d  bot %d   episode %lld   step %lld", app.selected_env,
                          s.world_id, s.agent_id, s.episode, s.step);
        else
            std::snprintf(l1, sizeof l1, "env %d   episode %lld   step %lld   seed %lld", app.selected_env, s.episode,
                          s.step, s.seed);
        std::snprintf(l2, sizeof l2, "action  %s", s.last_action.empty() ? "-" : s.last_action.c_str());
        std::snprintf(l3, sizeof l3, "pos     x %.2f  y %.2f  z %.2f%s", p.x, p.y, p.z, p.on_ground ? "  (ground)" : "");
        std::snprintf(l4, sizeof l4, "vel     %.2f m/s  (%.2f, %.2f, %.2f)", speed, p.vx, p.vy, p.vz);
        std::snprintf(l5, sizeof l5, "health  %.1f", p.health);
        // Task and family only when the trainer sends them (v7); older trainers get the bare health line.
        std::string l5s = l5;
        if (!s.task.empty()) l5s += "   task " + s.task;
        if (!s.family.empty()) l5s += "  [" + s.family + "]";
        std::snprintf(l6, sizeof l6, "reward  last %+.3f   episode %+.3f", s.last_reward, s.ep_reward);
        draw_overlay_text(dl, img_pos, {l1, l2, l3, l4, l5s, l6});
    }

    ImGui::End();
}
