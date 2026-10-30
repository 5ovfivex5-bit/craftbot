"""Threaded JSON-lines TCP server for the dashboard.

The training loop never blocks on sockets: broadcast()/send_to() only append bytes to per-client
outboxes under a short lock. Each client has a writer thread (blocking sendall lives there) and a
reader thread that parses commands into one thread-safe queue consumed by the training loop.

Delivery policy per client:
  - snapshot / activations: latest-wins slot (only the newest message is kept)
  - everything else (metrics, log, ack, hello, model, ...): bounded deque, oldest dropped on overflow
"""
import json
import queue
import socket
import threading
from collections import deque

LATEST_ONLY = ("snapshot", "activations")
MAX_LINE = 1 << 20


class _Client:
    def __init__(self, cid, sock, addr, outbox_max):
        self.id = cid
        self.sock = sock
        self.addr = addr
        self.outbox = deque(maxlen=outbox_max)
        self.slots = {}
        self.cv = threading.Condition(threading.Lock())
        self.snapshot = False
        self.nn = False
        self.closed = False
        self.dropped = 0
        self.sent = 0


class TelemetryServer:
    def __init__(self, host="127.0.0.1", port=7777, outbox_max=512):
        self.host = host
        self._outbox_max = outbox_max
        self._lock = threading.Lock()
        self._clients = {}
        self._next_id = 1
        self._cmds = queue.Queue()
        self._closing = False
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._accept_thread = threading.Thread(target=self._accept_loop, name="telemetry-accept",
                                               daemon=True)
        self._accept_thread.start()

    # ----- accept / read / write threads -----
    def _accept_loop(self):
        while not self._closing:
            try:
                conn, addr = self._sock.accept()
            except OSError:
                break
            conn.settimeout(None)
            with self._lock:
                cid = self._next_id
                self._next_id += 1
                client = _Client(cid, conn, addr, self._outbox_max)
                self._clients[cid] = client
            threading.Thread(target=self._reader, args=(client,), daemon=True,
                             name=f"telemetry-r{cid}").start()
            threading.Thread(target=self._writer, args=(client,), daemon=True,
                             name=f"telemetry-w{cid}").start()
            self._cmds.put({"cmd": "_connect", "_client": cid, "_addr": f"{addr[0]}:{addr[1]}"})

    def _reader(self, client):
        buf = b""
        try:
            while not client.closed:
                data = client.sock.recv(65536)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line.decode("utf-8"))
                    except (ValueError, UnicodeDecodeError) as e:
                        self.send_to(client.id, "ack", {"id": None, "ok": False,
                                                        "msg": f"invalid JSON line: {e}"})
                        continue
                    if not isinstance(msg, dict):
                        continue
                    msg["_client"] = client.id
                    self._cmds.put(msg)
                if len(buf) > MAX_LINE:
                    break
        except OSError:
            pass
        finally:
            self._drop(client)

    def _writer(self, client):
        while True:
            with client.cv:
                while not client.outbox and not client.slots and not client.closed:
                    client.cv.wait(1.0)
                if client.closed and not client.outbox and not client.slots:
                    return
                if client.outbox:
                    data = client.outbox.popleft()
                else:
                    _, data = client.slots.popitem()
            try:
                client.sock.sendall(data)
                client.sent += 1
            except OSError:
                self._drop(client)
                return

    def _drop(self, client):
        with self._lock:
            if client.closed:
                return
            client.closed = True
            self._clients.pop(client.id, None)
        with client.cv:
            client.cv.notify_all()
        try:
            client.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            client.sock.close()
        except OSError:
            pass
        self._cmds.put({"cmd": "_disconnect", "_client": client.id})

    # ----- public API (thread-safe) -----
    @staticmethod
    def _encode(msg_type, payload):
        d = {"type": msg_type}
        d.update(payload)
        return (json.dumps(d, separators=(",", ":")) + "\n").encode("utf-8")

    def _push(self, client, msg_type, data):
        with client.cv:
            if client.closed:
                return
            if msg_type in LATEST_ONLY:
                client.slots[msg_type] = data
            else:
                if len(client.outbox) == client.outbox.maxlen:
                    client.dropped += 1
                client.outbox.append(data)
            client.cv.notify()

    def broadcast(self, msg_type, payload=None, need=None):
        """Send to every client (or only clients with subscription flag `need` = 'snapshot'|'nn')."""
        with self._lock:
            targets = [c for c in self._clients.values() if need is None or getattr(c, need)]
        if not targets:
            return 0
        data = self._encode(msg_type, payload or {})
        for c in targets:
            self._push(c, msg_type, data)
        return len(targets)

    def send_to(self, client_id, msg_type, payload=None):
        with self._lock:
            c = self._clients.get(client_id)
        if c is None:
            return False
        self._push(c, msg_type, self._encode(msg_type, payload or {}))
        return True

    def set_subscription(self, client_id, snapshot=None, nn=None):
        with self._lock:
            c = self._clients.get(client_id)
            if c is None:
                return False
            if snapshot is not None:
                c.snapshot = bool(snapshot)
            if nn is not None:
                c.nn = bool(nn)
            return True

    def any_subscribed(self, kind):
        with self._lock:
            return any(getattr(c, kind) for c in self._clients.values())

    def client_count(self):
        with self._lock:
            return len(self._clients)

    def poll(self, limit=256):
        out = []
        while len(out) < limit:
            try:
                out.append(self._cmds.get_nowait())
            except queue.Empty:
                break
        return out

    def stats(self):
        with self._lock:
            return {c.id: {"addr": f"{c.addr[0]}:{c.addr[1]}", "sent": c.sent, "dropped": c.dropped,
                           "snapshot": c.snapshot, "nn": c.nn} for c in self._clients.values()}

    def close(self):
        self._closing = True
        try:
            self._sock.close()
        except OSError:
            pass
        with self._lock:
            clients = list(self._clients.values())
        for c in clients:
            self._drop(c)
