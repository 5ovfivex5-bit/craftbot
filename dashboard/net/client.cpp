#include "client.h"

#include <array>
#include <cerrno>
#include <cstring>
#include <utility>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

using nlohmann::json;

namespace mm {
namespace {

double jnum(const json& j, const char* k, double def) {
    auto it = j.find(k);
    if (it == j.end() || !it->is_number()) return def;
    return it->get<double>();
}

long long jint(const json& j, const char* k, long long def) {
    auto it = j.find(k);
    if (it == j.end() || !it->is_number()) return def;
    return it->get<long long>();
}

std::string jstr(const json& j, const char* k, const std::string& def) {
    auto it = j.find(k);
    if (it == j.end() || !it->is_string()) return def;
    return it->get<std::string>();
}

std::optional<double> jonum(const json& j, const char* k) {
    auto it = j.find(k);
    if (it == j.end() || !it->is_number()) return std::nullopt;
    return it->get<double>();
}

std::optional<std::string> jostr(const json& j, const char* k) {
    auto it = j.find(k);
    if (it == j.end() || !it->is_string()) return std::nullopt;
    return it->get<std::string>();
}

json jobj(const json& j, const char* k) {
    auto it = j.find(k);
    if (it == j.end() || !it->is_object()) return json::object();
    return *it;
}

std::vector<int> jints(const json& j, const char* k) {
    std::vector<int> out;
    auto it = j.find(k);
    if (it == j.end() || !it->is_array()) return out;
    for (const auto& v : *it)
        if (v.is_number()) out.push_back(v.get<int>());
    return out;
}

std::vector<float> jfloats(const json& j, const char* k) {
    std::vector<float> out;
    auto it = j.find(k);
    if (it == j.end() || !it->is_array()) return out;
    for (const auto& v : *it)
        if (v.is_number()) out.push_back(v.get<float>());
    return out;
}

Stat4 jstat(const json& j) {
    Stat4 s;
    s.mean = (float)jnum(j, "mean", 0);
    s.std = (float)jnum(j, "std", 0);
    s.min = (float)jnum(j, "min", 0);
    s.max = (float)jnum(j, "max", 0);
    return s;
}

std::vector<uint8_t> b64decode(const std::string& in) {
    static const auto table = [] {
        std::array<int, 256> t{};
        t.fill(-1);
        const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) t[(unsigned char)alphabet[i]] = i;
        return t;
    }();
    std::vector<uint8_t> out;
    out.reserve(in.size() * 3 / 4);
    uint32_t acc = 0;
    int bits = 0;
    for (unsigned char c : in) {
        if (c == '=') break;
        int v = table[c];
        if (v < 0) continue;  // whitespace and anything unexpected
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((uint8_t)((acc >> bits) & 0xFF));
        }
    }
    return out;
}

Metrics parse_metrics(const json& j) {
    Metrics m;
    m.t = jnum(j, "t", 0);
    m.state = jstr(j, "state", "idle");
    m.mode = jstr(j, "mode", "fast");
    m.env_steps = jint(j, "env_steps", 0);
    m.target_steps = jint(j, "target_steps", 0);
    m.sps = jnum(j, "sps", 0);
    m.episodes = jint(j, "episodes", 0);
    m.avg_ep_len = jnum(j, "avg_ep_len", 0);
    m.avg_reward = jnum(j, "avg_reward", 0);
    m.updates = jint(j, "updates", 0);
    m.policy_loss = jnum(j, "policy_loss", 0);
    m.value_loss = jnum(j, "value_loss", 0);
    m.entropy = jnum(j, "entropy", 0);
    m.approx_kl = jnum(j, "approx_kl", 0);
    m.cpu_percent = jnum(j, "cpu_percent", 0);
    m.rss_mb = jnum(j, "rss_mb", 0);
    m.mem_warning = jostr(j, "mem_warning");
    m.elapsed_s = jnum(j, "elapsed_s", 0);
    m.eta_s = jonum(j, "eta_s");
    m.latest_checkpoint = jostr(j, "latest_checkpoint");
    m.curriculum_stage = jostr(j, "curriculum_stage");
    m.num_envs = (int)jint(j, "num_envs", 0);
    m.config = jobj(j, "config");
    if (auto it = j.find("success_rate"); it != j.end() && it->is_object())
        for (auto& [k, v] : it->items())
            if (v.is_number()) m.success_rate[k] = v.get<double>();
    // v6 team metrics: absent on older trainers.
    if (auto v = jonum(j, "agents")) m.agents = (int)*v;
    m.msg_entropy = jonum(j, "msg_entropy");
    m.msg_rate = jonum(j, "msg_rate");
    m.chest_deposits = jonum(j, "chest_deposits");
    m.chest_withdraws = jonum(j, "chest_withdraws");
    m.gives = jonum(j, "gives");
    // v7 success breakdowns (absent on older trainers -> empty maps).
    for (const char* key : {"family_success", "task_success"}) {
        auto& dst = std::string(key) == "family_success" ? m.family_success : m.task_success;
        if (auto it = j.find(key); it != j.end() && it->is_object())
            for (auto& [k, v] : it->items())
                if (v.is_number()) dst[k] = v.get<double>();
    }
    return m;
}

Snapshot parse_snapshot(const json& j) {
    Snapshot s;
    s.env_id = (int)jint(j, "env_id", -1);
    s.episode = jint(j, "episode", 0);
    s.seed = jint(j, "seed", 0);
    s.step = jint(j, "step", 0);
    json p = jobj(j, "player");
    s.player.x = jnum(p, "x", 0);
    s.player.y = jnum(p, "y", 0);
    s.player.z = jnum(p, "z", 0);
    s.player.vx = jnum(p, "vx", 0);
    s.player.vy = jnum(p, "vy", 0);
    s.player.vz = jnum(p, "vz", 0);
    s.player.yaw = jnum(p, "yaw", 0);
    s.player.pitch = jnum(p, "pitch", 0);
    s.player.health = jnum(p, "health", 0);
    s.player.on_ground = p.value("on_ground", false);
    s.last_action = jstr(j, "last_action", "");
    s.last_reward = jnum(j, "last_reward", 0);
    s.ep_reward = jnum(j, "ep_reward", 0);
    s.task = jstr(j, "task", "");
    s.family = jstr(j, "family", "");
    if (auto t = jonum(j, "time_of_day"); t && *t >= 0.0 && *t < 1.0) s.time_of_day = *t;
    if (auto it = j.find("target"); it != j.end() && it->is_object()) {
        Target t;
        t.x = (int)jint(*it, "x", 0);
        t.y = (int)jint(*it, "y", 0);
        t.z = (int)jint(*it, "z", 0);
        t.block = jstr(*it, "block", "");
        s.target = t;
    }
    json inv = jobj(j, "inventory");
    for (auto& [k, v] : inv.items())
        if (v.is_number()) s.inventory[k] = v.get<int>();
    if (auto it = j.find("action_history"); it != j.end() && it->is_array())
        for (const auto& v : *it)
            if (v.is_string()) s.action_history.push_back(v.get<std::string>());
    auto o = jints(j, "origin");
    auto z = jints(j, "size");
    for (int i = 0; i < 3; i++) {
        s.origin[i] = i < (int)o.size() ? o[i] : 0;
        s.size[i] = i < (int)z.size() ? z[i] : 0;
    }
    s.blocks = b64decode(jstr(j, "blocks", ""));
    if (s.blocks.size() != (size_t)s.size[0] * s.size[1] * s.size[2]) s.blocks.clear();
    s.obs_res = (int)jint(j, "obs_res", 0);
    s.obs = b64decode(jstr(j, "obs", ""));
    if (s.obs.size() != (size_t)s.obs_res * s.obs_res * 2) s.obs.clear();

    // v6 team fields. Presence of "team" marks a v6 trainer.
    auto counts_of = [](const json& o, const char* key) {
        std::map<std::string, int> inv;
        const json obj = jobj(o, key);  // named: items() must not outlive a temporary
        for (auto& [k, v] : obj.items())
            if (v.is_number()) inv[k] = v.get<int>();
        return inv;
    };
    if (auto it = j.find("team"); it != j.end() && it->is_array()) {
        s.has_team = true;
        for (const auto& e : *it) {
            if (!e.is_object()) continue;
            TeamMember t;
            t.agent = (int)jint(e, "agent", 0);
            t.x = jnum(e, "x", 0);
            t.y = jnum(e, "y", 0);
            t.z = jnum(e, "z", 0);
            t.yaw = jnum(e, "yaw", 0);
            t.pitch = jnum(e, "pitch", 0);
            t.health = jnum(e, "health", 0);
            t.last_action = jstr(e, "last_action", "");
            t.msg = (int)jint(e, "msg", 0);
            t.inventory = counts_of(e, "inventory");
            s.team.push_back(std::move(t));
        }
    }
    s.world_id = (int)jint(j, "world_id", -1);
    s.agent_id = (int)jint(j, "agent_id", -1);
    s.agents_per_world = (int)jint(j, "agents_per_world", 0);
    if (auto it = j.find("chests"); it != j.end() && it->is_array()) {
        for (const auto& c : *it) {
            if (!c.is_object()) continue;
            ChestInfo ch;
            ch.x = (int)jint(c, "x", 0);
            ch.y = (int)jint(c, "y", 0);
            ch.z = (int)jint(c, "z", 0);
            ch.items = counts_of(c, "items");
            s.chests.push_back(std::move(ch));
        }
    }
    if (auto it = j.find("entities"); it != j.end() && it->is_array()) {
        for (const auto& e : *it) {
            if (!e.is_object()) continue;
            Entity en;
            en.type = jstr(e, "type", "");
            en.x = jnum(e, "x", 0);
            en.y = jnum(e, "y", 0);
            en.z = jnum(e, "z", 0);
            en.health = jonum(e, "health");
            s.entities.push_back(std::move(en));
        }
    }
    if (auto it = j.find("chat"); it != j.end() && it->is_array()) {
        for (const auto& c : *it) {
            if (!c.is_object()) continue;
            ChatEntry e;
            e.step = jint(c, "step", 0);
            e.agent = (int)jint(c, "agent", 0);
            e.msg = (int)jint(c, "msg", 0);
            s.chat.push_back(e);
        }
    }
    return s;
}

Activations parse_acts(const json& j) {
    Activations a;
    a.env_id = (int)jint(j, "env_id", -1);
    a.step = jint(j, "step", 0);
    a.action_probs = jfloats(j, "action_probs");
    a.entropy = (float)jnum(j, "entropy", 0);
    a.value = (float)jnum(j, "value", 0);
    const json layers = jobj(j, "layers");
    for (auto& [name, l] : layers.items()) {
        LayerAct la;
        la.shape = jints(l, "shape");
        la.sample = jfloats(l, "sample");
        la.mean = (float)jnum(l, "mean", 0);
        la.std = (float)jnum(l, "std", 0);
        la.min = (float)jnum(l, "min", 0);
        la.max = (float)jnum(l, "max", 0);
        json h = jobj(l, "hist");
        la.hist_edges = jfloats(h, "edges");
        for (int c : jints(h, "counts")) la.hist_counts.push_back(c);
        a.layers[name] = std::move(la);
    }
    if (auto it = j.find("gru"); it != j.end() && it->is_object()) {
        a.has_gru = true;
        a.gru = jstat(*it);
        a.gru_norm = (float)jnum(*it, "norm", 0);
    }
    const json grads = jobj(j, "grad_norms");
    for (auto& [name, v] : grads.items())
        if (v.is_number()) a.grad_norms[name] = v.get<float>();
    const json wstats = jobj(j, "weight_stats");
    for (auto& [name, v] : wstats.items())
        if (v.is_object()) a.weight_stats[name] = jstat(v);
    a.nan_inf = j.value("nan_inf", false);
    return a;
}

ModelInfo parse_model(const json& j) {
    ModelInfo m;
    m.name = jstr(j, "name", "");
    m.params_total = jint(j, "params_total", 0);
    json in = jobj(j, "input");
    m.img_shape = jints(in, "img");
    auto vec = jints(in, "vec");
    m.vec_dim = vec.empty() ? 0 : vec[0];
    m.gru_hidden = (int)jint(j, "gru_hidden", 0);
    if (auto it = j.find("layers"); it != j.end() && it->is_array()) {
        for (const auto& l : *it) {
            LayerInfo li;
            li.name = jstr(l, "name", "");
            li.type = jstr(l, "type", "");
            li.params = jint(l, "params", 0);
            li.in_shape = jints(l, "in_shape");
            li.out_shape = jints(l, "out_shape");
            li.trainable = l.value("trainable", true);
            m.layers.push_back(std::move(li));
        }
    }
    if (auto it = j.find("edges"); it != j.end() && it->is_array()) {
        for (const auto& e : *it)
            if (e.is_array() && e.size() == 2 && e[0].is_string() && e[1].is_string())
                m.edges.emplace_back(e[0].get<std::string>(), e[1].get<std::string>());
    }
    return m;
}

Hello parse_hello(const json& j) {
    Hello h;
    h.schema = (int)jint(j, "schema", 0);
    h.run_id = jstr(j, "run_id", "");
    h.state = jstr(j, "state", "");
    h.config = jobj(j, "config");
    auto ws = jints(j, "world_size");
    for (int i = 0; i < 3; i++) h.world_size[i] = i < (int)ws.size() ? ws[i] : 0;
    auto strs = [&](const char* k) {
        std::vector<std::string> out;
        if (auto it = j.find(k); it != j.end() && it->is_array())
            for (const auto& v : *it)
                if (v.is_string()) out.push_back(v.get<std::string>());
        return out;
    };
    h.actions = strs("actions");
    h.items = strs("items");
    h.blocks = strs("blocks");
    return h;
}

}  // namespace

Client::Client(std::string host, int port)
    : host_(std::move(host)), port_(port), endpoint_(host_ + ":" + std::to_string(port_)),
      t0_(std::chrono::steady_clock::now()) {
    published_ = std::make_shared<const Frame>();
}

Client::~Client() { stop(); }

double Client::now() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
}

void Client::start() {
    if (thread_.joinable()) return;
    {
        std::lock_guard lk(mu_);
        w_.conn_text = "Connecting to " + endpoint_;
        publish_locked();
    }
    thread_ = std::thread([this] { run(); });
}

void Client::stop() {
    stop_ = true;
    if (thread_.joinable()) thread_.join();
}

std::shared_ptr<const Frame> Client::frame() const {
    std::lock_guard lk(mu_);
    return published_;
}

int Client::send(const std::string& cmd, json args) {
    std::lock_guard lk(mu_);
    int id = w_.next_id++;
    if (w_.conn != ConnState::Connected) {
        push_ack_locked(Ack{id, cmd, false, "not connected: command dropped", now()});
        publish_locked();
        return id;
    }
    if (!args.is_object()) args = json::object();
    args["cmd"] = cmd;
    args["id"] = id;
    w_.outbox.push_back(args.dump() + "\n");
    w_.pending[id] = cmd;
    publish_locked();
    return id;
}

void Client::set_snapshot_wanted(bool on) {
    std::lock_guard lk(mu_);
    if (w_.snapshot_wanted == on) return;
    w_.snapshot_wanted = on;
    if (w_.conn == ConnState::Connected)
        enqueue_locked("subscribe", json{{"snapshot", w_.snapshot_wanted}, {"nn", w_.nn_wanted}});
    publish_locked();
}

void Client::set_nn_wanted(bool on) {
    std::lock_guard lk(mu_);
    if (w_.nn_wanted == on) return;
    w_.nn_wanted = on;
    if (w_.conn == ConnState::Connected)
        enqueue_locked("subscribe", json{{"snapshot", w_.snapshot_wanted}, {"nn", w_.nn_wanted}});
    publish_locked();
}

// Caller holds mu_. Used for internal commands; ids are assigned here.
void Client::enqueue_locked(const std::string& cmd, json args) {
    int id = w_.next_id++;
    args["cmd"] = cmd;
    args["id"] = id;
    w_.outbox.push_back(args.dump() + "\n");
    w_.pending[id] = cmd;
}

void Client::push_ack_locked(Ack a) {
    w_.acks.push_back(std::move(a));
    while (w_.acks.size() > kAckCap) w_.acks.pop_front();
    w_.acks_dirty = true;
}

void Client::set_conn(ConnState s, std::string text) {
    std::lock_guard lk(mu_);
    w_.conn = s;
    w_.conn_text = std::move(text);
    publish_locked();
}

void Client::publish_locked() {
    if (w_.history_dirty) {
        w_.history_pub = std::make_shared<const std::vector<Metrics>>(w_.history.begin(), w_.history.end());
        w_.history_dirty = false;
    }
    if (w_.logs_dirty) {
        w_.logs_pub = std::make_shared<const std::vector<LogLine>>(w_.logs.begin(), w_.logs.end());
        w_.logs_dirty = false;
    }
    if (w_.acks_dirty) {
        w_.acks_pub = std::make_shared<const std::vector<Ack>>(w_.acks.begin(), w_.acks.end());
        w_.acks_dirty = false;
    }
    auto f = std::make_shared<Frame>();
    f->conn = w_.conn;
    f->conn_text = w_.conn_text;
    f->endpoint = endpoint_;
    f->since_rx_s = w_.last_rx < 0 ? -1 : now() - w_.last_rx;
    f->has_hello = w_.has_hello;
    f->hello = w_.hello;
    f->metrics = w_.metrics;
    f->history = w_.history_pub;
    f->snapshot = w_.snapshot;
    f->acts = w_.acts;
    f->model = w_.model;
    f->checkpoints = w_.checkpoints;
    f->logs = w_.logs_pub;
    f->acks = w_.acks_pub;
    f->pending = w_.pending;
    f->snapshot_wanted = w_.snapshot_wanted;
    f->nn_wanted = w_.nn_wanted;
    f->gen = published_ ? published_->gen + 1 : 1;
    published_ = std::move(f);
}

void Client::sleep_interruptible(int ms) {
    for (int waited = 0; waited < ms && !stop_; waited += 25)
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
}

void Client::run() {
    while (!stop_) {
        set_conn(ConnState::Connecting, "Connecting to " + endpoint_ + " ...");

        addrinfo hints{};
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        addrinfo* res = nullptr;
        std::string err;
        int fd = -1;
        if (getaddrinfo(host_.c_str(), std::to_string(port_).c_str(), &hints, &res) != 0 || !res) {
            err = "cannot resolve host";
        } else {
            fd = ::socket(res->ai_family, res->ai_socktype | SOCK_NONBLOCK | SOCK_CLOEXEC, res->ai_protocol);
            if (fd < 0) {
                err = std::strerror(errno);
            } else {
                int r = ::connect(fd, res->ai_addr, res->ai_addrlen);
                if (r < 0 && errno != EINPROGRESS) {
                    err = std::strerror(errno);
                    ::close(fd);
                    fd = -1;
                } else if (r < 0) {
                    pollfd pf{fd, POLLOUT, 0};
                    int pr = ::poll(&pf, 1, 1000);
                    int so_err = 0;
                    socklen_t len = sizeof so_err;
                    if (pr <= 0) {
                        err = pr == 0 ? "timed out" : std::strerror(errno);
                        ::close(fd);
                        fd = -1;
                    } else {
                        ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_err, &len);
                        if (so_err) {
                            err = std::strerror(so_err);
                            ::close(fd);
                            fd = -1;
                        }
                    }
                }
            }
        }
        if (res) freeaddrinfo(res);

        if (fd < 0) {
            set_conn(ConnState::Disconnected, "Not connected to " + endpoint_ + " (" + err + ") - retrying in 1 s");
            sleep_interruptible(1000);
            continue;
        }

        int one = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        bool clean = session(fd);
        ::close(fd);
        set_conn(ConnState::Disconnected,
                 std::string(clean ? "Disconnected from " : "Connection lost to ") + endpoint_ + " - retrying in 1 s");
        sleep_interruptible(1000);
    }
}

// Returns true when the session ended by a stop request.
bool Client::session(int fd) {
    {
        std::lock_guard lk(mu_);
        w_.conn = ConnState::Connected;
        w_.conn_text = "Connected to " + endpoint_;
        w_.last_rx = -1;
        // Ask for the state a fresh client needs; subscriptions are re-sent from the wanted flags.
        enqueue_locked("subscribe", json{{"snapshot", w_.snapshot_wanted}, {"nn", w_.nn_wanted}});
        enqueue_locked("list_checkpoints", json::object());
        enqueue_locked("get_history", json{{"last", 1000}});
        publish_locked();
    }

    std::string rbuf;
    std::string wbuf;
    double last_publish = now();
    while (!stop_) {
        {
            std::lock_guard lk(mu_);
            while (!w_.outbox.empty()) {
                wbuf += w_.outbox.front();
                w_.outbox.pop_front();
            }
        }
        pollfd pf{fd, static_cast<short>(POLLIN | (wbuf.empty() ? 0 : POLLOUT)), 0};
        int pr = ::poll(&pf, 1, 50);
        if (pr < 0 && errno != EINTR) break;

        if (pr > 0 && (pf.revents & POLLIN)) {
            char buf[65536];
            ssize_t n = ::recv(fd, buf, sizeof buf, 0);
            if (n == 0) break;  // peer closed
            if (n < 0) {
                if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) break;
            } else {
                rbuf.append(buf, (size_t)n);
                size_t nl;
                while ((nl = rbuf.find('\n')) != std::string::npos) {
                    std::string line = rbuf.substr(0, nl);
                    rbuf.erase(0, nl + 1);
                    if (!line.empty()) handle_line(line);
                }
                if (rbuf.size() > (64u << 20)) break;  // protocol violation: no newline for 64 MB
            }
        }
        if (pr > 0 && (pf.revents & (POLLERR | POLLHUP | POLLNVAL)) && !(pf.revents & POLLIN)) break;

        if (!wbuf.empty() && pr > 0 && (pf.revents & POLLOUT)) {
            ssize_t n = ::send(fd, wbuf.data(), wbuf.size(), MSG_NOSIGNAL);
            if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) break;
            if (n > 0) wbuf.erase(0, (size_t)n);
        }

        if (now() - last_publish > 0.5) {
            std::lock_guard lk(mu_);
            publish_locked();
            last_publish = now();
        }
    }

    // Connection gone: every command still waiting for an ack has failed.
    std::lock_guard lk(mu_);
    for (auto& [id, cmd] : w_.pending) push_ack_locked(Ack{id, cmd, false, "connection lost before ack", now()});
    w_.pending.clear();
    w_.outbox.clear();
    w_.conn = ConnState::Disconnected;
    publish_locked();
    return stop_;
}

void Client::handle_line(const std::string& line) {
    json j = json::parse(line, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        std::lock_guard lk(mu_);
        w_.logs.push_back(LogLine{now(), "warn", "dropped malformed line from trainer"});
        while (w_.logs.size() > kLogCap) w_.logs.pop_front();
        w_.logs_dirty = true;
        publish_locked();
        return;
    }
    handle(j);
}

void Client::handle(const json& j) {
    std::string type = jstr(j, "type", "");
    std::lock_guard lk(mu_);
    w_.last_rx = now();

    if (type == "hello") {
        w_.hello = parse_hello(j);
        w_.has_hello = true;
    } else if (type == "metrics") {
        Metrics m = parse_metrics(j);
        w_.metrics = std::make_shared<const Metrics>(m);
        w_.history.push_back(std::move(m));
        while (w_.history.size() > kMetricsCap) w_.history.pop_front();
        w_.history_dirty = true;
    } else if (type == "history") {
        // Backfill: rows older than what we already have go in front.
        std::vector<Metrics> rows;
        if (auto it = j.find("rows"); it != j.end() && it->is_array())
            for (const auto& r : *it)
                if (r.is_object()) rows.push_back(parse_metrics(r));
        double first_t = w_.history.empty() ? 1e300 : w_.history.front().t;
        std::deque<Metrics> merged;
        for (auto& r : rows)
            if (r.t < first_t) merged.push_back(std::move(r));
        for (auto& r : w_.history) merged.push_back(std::move(r));
        while (merged.size() > kMetricsCap) merged.pop_front();
        w_.history = std::move(merged);
        w_.history_dirty = true;
    } else if (type == "snapshot") {
        w_.snapshot = std::make_shared<const Snapshot>(parse_snapshot(j));
    } else if (type == "activations") {
        w_.acts = std::make_shared<const Activations>(parse_acts(j));
    } else if (type == "model") {
        w_.model = std::make_shared<const ModelInfo>(parse_model(j));
    } else if (type == "checkpoints") {
        auto v = std::make_shared<std::vector<Checkpoint>>();
        if (auto it = j.find("items"); it != j.end() && it->is_array()) {
            for (const auto& c : *it) {
                Checkpoint k;
                k.path = jstr(c, "path", "");
                k.env_steps = jint(c, "env_steps", 0);
                k.updates = jint(c, "updates", 0);
                k.time = jnum(c, "time", 0);
                k.size_bytes = jint(c, "size_bytes", 0);
                v->push_back(std::move(k));
            }
        }
        w_.checkpoints = v;
    } else if (type == "log") {
        w_.logs.push_back(LogLine{jnum(j, "t", now()), jstr(j, "level", "info"), jstr(j, "msg", "")});
        while (w_.logs.size() > kLogCap) w_.logs.pop_front();
        w_.logs_dirty = true;
    } else if (type == "ack") {
        int id = (int)jint(j, "id", -1);
        std::string cmd = "?";
        if (auto it = w_.pending.find(id); it != w_.pending.end()) {
            cmd = it->second;
            w_.pending.erase(it);
        }
        push_ack_locked(Ack{id, cmd, j.value("ok", false), jstr(j, "msg", ""), now()});
    } else {
        w_.logs.push_back(LogLine{now(), "warn", "unknown message type: " + type});
        while (w_.logs.size() > kLogCap) w_.logs.pop_front();
        w_.logs_dirty = true;
    }
    publish_locked();
}

}  // namespace mm
