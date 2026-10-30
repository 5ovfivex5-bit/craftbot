// Background TCP client for the MineMind trainer (PROTOCOL.md section 2).
// One worker thread owns the socket: non-blocking connect, auto-reconnect every
// second, JSON-lines framing, command outbox with ack tracking. The UI thread only
// calls frame() / send() / set_*_wanted(), which never block on the network.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "protocol.h"

namespace mm {

class Client {
public:
    static constexpr size_t kMetricsCap = 6000;  // ~50 min at 2 Hz
    static constexpr size_t kLogCap = 2000;
    static constexpr size_t kAckCap = 300;

    Client(std::string host, int port);
    ~Client();
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    void start();
    void stop();

    // Latest immutable state. Cheap (copies shared_ptrs only).
    std::shared_ptr<const Frame> frame() const;

    // Queues a command with a fresh id. When not connected the command is
    // rejected immediately with a failed ack (never silently dropped).
    int send(const std::string& cmd, nlohmann::json args = nlohmann::json::object());

    // Subscriptions are per-client; they are re-sent automatically on reconnect.
    void set_snapshot_wanted(bool on);
    void set_nn_wanted(bool on);

    double now() const;  // seconds since the client was constructed

private:
    struct Work {
        ConnState conn = ConnState::Disconnected;
        std::string conn_text = "Starting";
        double last_rx = -1;
        bool has_hello = false;
        Hello hello;
        std::shared_ptr<const Metrics> metrics;
        std::deque<Metrics> history;
        std::shared_ptr<const std::vector<Metrics>> history_pub;
        std::shared_ptr<const Snapshot> snapshot;
        std::shared_ptr<const Activations> acts;
        std::shared_ptr<const ModelInfo> model;
        std::shared_ptr<const std::vector<Checkpoint>> checkpoints;
        std::deque<LogLine> logs;
        std::shared_ptr<const std::vector<LogLine>> logs_pub;
        std::deque<Ack> acks;
        std::shared_ptr<const std::vector<Ack>> acks_pub;
        std::map<int, std::string> pending;
        std::deque<std::string> outbox;  // serialized JSON lines
        bool snapshot_wanted = false;
        bool nn_wanted = false;
        bool history_dirty = false, logs_dirty = false, acks_dirty = false;
        int next_id = 1;
    };

    void run();
    bool session(int fd);
    void handle_line(const std::string& line);
    void handle(const nlohmann::json& j);
    void set_conn(ConnState s, std::string text);  // takes lock
    void publish_locked();
    void push_ack_locked(Ack a);
    void enqueue_locked(const std::string& cmd, nlohmann::json args);
    void sleep_interruptible(int ms);

    std::string host_;
    int port_;
    std::string endpoint_;
    std::chrono::steady_clock::time_point t0_;

    mutable std::mutex mu_;
    Work w_;
    std::shared_ptr<const Frame> published_;

    std::atomic<bool> stop_{false};
    std::thread thread_;
};

}  // namespace mm
