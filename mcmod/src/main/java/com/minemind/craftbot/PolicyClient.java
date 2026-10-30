package com.minemind.craftbot;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Talks to the Python policy bridge (mcbridge/bridge.py) over localhost TCP, one JSON object per line.
 * All socket IO happens on a single background thread so the server tick never blocks.
 *
 * Requests: {"cmd":"info"} -> {"res","vec_dim","num_blocks","num_actions","version"} (once, to pick the layout);
 * {"bot","reset","res","img","vec"} -> {"action","msg","prob","value"} or {"error": "..."}.
 */
public final class PolicyClient {
	public static final String HOST = "127.0.0.1";
	public static final int PORT = 7788;

	private final ExecutorService io = Executors.newSingleThreadExecutor(r -> {
		Thread t = new Thread(r, "craftbot-policy-io");
		t.setDaemon(true);
		return t;
	});
	private Socket socket;
	private BufferedReader in;
	private OutputStream out;
	private volatile Obs.Layout layout;
	private volatile String version = "?";

	public record Decision(int action, int msg, float prob, float value) {}

	/** The checkpoint's observation layout, or null until {@link #probe()} succeeded. */
	public Obs.Layout layout() {
		return layout;
	}

	public String version() {
		return version;
	}

	/** Asks the bridge for its checkpoint layout; completes exceptionally if unreachable or unsupported. */
	public CompletableFuture<Obs.Layout> probe() {
		return CompletableFuture.supplyAsync(() -> {
			try {
				JsonObject req = new JsonObject();
				req.addProperty("cmd", "info");
				JsonObject r = call(req);
				Obs.Layout L = Obs.Layout.forVecDim(r.get("vec_dim").getAsInt());
				if (L == null) throw new IllegalStateException("bridge checkpoint VEC_DIM " + r.get("vec_dim") + " unsupported");
				if (L.res != r.get("res").getAsInt())
					throw new IllegalStateException("bridge res " + r.get("res") + " != layout res " + L.res);
				version = r.has("version") ? r.get("version").getAsString() : "?";
				layout = L;
				return L;
			} catch (Exception e) {
				throw new RuntimeException(e.getMessage(), e);
			}
		}, io);
	}

	/** Ask the policy for the next action. Completes exceptionally if the bridge is unreachable or refuses. */
	public CompletableFuture<Decision> decide(String bot, Obs.Layout L, byte[] img, float[] vec, boolean reset) {
		JsonObject msg = new JsonObject();
		msg.addProperty("bot", bot);
		msg.addProperty("reset", reset);
		msg.addProperty("res", L.res);
		msg.addProperty("img", Base64.getEncoder().encodeToString(img));
		JsonArray v = new JsonArray();
		for (float f : vec) v.add(f);
		msg.add("vec", v);
		return CompletableFuture.supplyAsync(() -> {
			try {
				JsonObject r = call(msg);
				return new Decision(r.get("action").getAsInt(), r.has("msg") ? r.get("msg").getAsInt() : 0,
					r.get("prob").getAsFloat(), r.get("value").getAsFloat());
			} catch (Exception e) {
				throw new RuntimeException(e.getMessage(), e);
			}
		}, io);
	}

	/**
	 * Raw request/reply on the IO thread (training: {"cmd":"train_step", ...}). The reply is handed back
	 * as a future so the server thread never waits on the socket. Fails if unreachable or the bridge errors.
	 */
	public CompletableFuture<JsonObject> send(JsonObject req) {
		return CompletableFuture.supplyAsync(() -> {
			try {
				return call(req);
			} catch (Exception e) {
				throw new RuntimeException(e.getMessage(), e);
			}
		}, io);
	}

	/** Blocking round trip (runs on the IO thread). Throws if the bridge answers with an error. */
	private JsonObject call(JsonObject req) throws Exception {
		try {
			ensureConnected();
			out.write((req + "\n").getBytes(StandardCharsets.UTF_8));
			out.flush();
			String reply = in.readLine();
			if (reply == null) throw new IllegalStateException("bridge closed the connection");
			JsonObject r = JsonParser.parseString(reply).getAsJsonObject();
			if (r.has("error")) throw new IllegalStateException(r.get("error").getAsString());
			return r;
		} catch (Exception e) {
			close();
			throw e;
		}
	}

	private void ensureConnected() throws Exception {
		if (socket != null && socket.isConnected() && !socket.isClosed()) return;
		Socket s = new Socket();
		s.connect(new InetSocketAddress(HOST, PORT), 1000);
		s.setTcpNoDelay(true);
		s.setSoTimeout(5000);
		socket = s;
		in = new BufferedReader(new InputStreamReader(s.getInputStream(), StandardCharsets.UTF_8));
		out = s.getOutputStream();
	}

	private void close() {
		try {
			if (socket != null) socket.close();
		} catch (Exception ignored) {
		}
		socket = null;
	}
}
