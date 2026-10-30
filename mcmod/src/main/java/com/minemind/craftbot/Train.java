package com.minemind.craftbot;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.mojang.authlib.GameProfile;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.core.BlockPos;
import net.minecraft.core.UUIDUtil;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ClientInformation;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.network.CommonListenerCookie;
import net.minecraft.tags.BiomeTags;
import net.minecraft.world.Container;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.chunk.status.ChunkStatus;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraft.world.phys.Vec3;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.CompletableFuture;

/**
 * Online fine-tuning inside Minecraft (PROTOCOL section 11). {@code /aibot train start|stop|status|speed|goals}.
 *
 * Teams of fake players (same {@link Bot} machinery as /aibot spawn) play one episode at a time in their own
 * far-away spot. Every decision tick ALL training bots are sent in ONE request to the bridge
 * ({@code {"cmd":"train_step"}}); the reply is applied on a later server tick, so the server thread never waits
 * on the socket.
 *
 * FT-1 pipeline (one batch, the bridge protocol is unchanged):
 *  1. capture (server thread, when every bot is ready and no batch is in flight): episode bookkeeping, rewards,
 *     episode reset, stuck check, then one {@link Obs.Snapshot} per team and one {@link Obs.Pose} per bot.
 *  2. observations are built on {@link ObsWorker} threads from those immutable copies (raycasts never touch
 *     the server thread).
 *  3. when all observations are done (a later tick), the batch is sent to the bridge in one request.
 *  4. the reply's actions are applied on a later tick (bots start their next decision window).
 * Rewards mirror the sim (src/sim.cpp obtain() and the goal rule): team-first item achievements
 * 0.1*(1+log2(value)) split by team_share 0.5, goal +10 split the same way, episode ends when a team member holds
 * the goal item or after 3000 decisions. Death respawns the bot on the team spot and keeps the episode going.
 *
 * Perf: every 60 s the server tick cost of training and the server mspt/TPS are logged and shown in
 * {@code /aibot train status}. Adaptive tick rate: lowered when mspt > 45, raised back when mspt < 25, never
 * above the user's {@code /aibot train speed}. Stuck bots (no movement and no reward for 400 decisions) are
 * moved to a safe spot within 20 blocks.
 *
 * Everything runs inside try/catch in {@link CraftBotMod}: a failure is logged and training is stopped after
 * repeated errors; the server keeps running.
 */
public final class Train {
	private Train() {}

	static final double TEAM_SHARE = 0.5, GOAL_REWARD = 10.0;
	static final double MIN_SPOT_DIST = 300.0, SEARCH_RADIUS = 3000.0;
	static final int MAX_STEPS = Obs.MAX_STEPS;
	static final int DEFAULT_TEAMS = 2, DEFAULT_AGENTS = 4, MAX_TEAMS = 8, MAX_AGENTS = 5;
	static final int FORCE_RADIUS = 2;             // force-loaded chunks: (2R+1)^2 around each team spot
	static final int DEFAULT_TPS = 40;
	static final int RETRY_TICKS = 40, REGION_RADIUS = 16, REGION_EVERY = 20, MAX_ERRORS = 20;
	/** Block box around each bot for the team snapshot: rays reach 24 blocks, chest scan 16, near box 4. */
	static final int SNAP_MARGIN = 26;
	/** Stuck detection (FT-1): no movement (>= 1 block from the anchor) and no reward for this many decisions. */
	static final int STUCK_DECISIONS = 400;
	static final double STUCK_MOVE = 1.0, STUCK_RADIUS = 20.0;
	/** Adaptive tick rate (FT-1): mspt thresholds, checked every ADAPT_EVERY ticks. */
	static final double ADAPT_HIGH_MSPT = 45.0, ADAPT_LOW_MSPT = 25.0;
	static final float ADAPT_MIN_TPS = 10f;
	static final int ADAPT_EVERY = 20, ADAPT_HOLD_TICKS = 200;
	static final long PERF_WINDOW_NS = 60_000_000_000L;
	/** MineRL item values (sim ITEM_VALUE): index = sim item id; dirt (13) has none. */
	static final int[] ITEM_VALUE = {1, 2, 4, 4, 8, 16, 32, 32, 16, 64, 128, 256, 1024, 0, 8};
	static final int[] DEFAULT_GOALS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

	/** One team: a spot in the world, its bots (slot = index), the goal and the team achievement mask. */
	static final class Team {
		final int index;
		final List<Bot> bots = new ArrayList<>();
		final double[] pend;           // reward accumulated per slot since the last batch
		double spotX, spotY, spotZ;
		int cx, cz;                    // chunk of the spot (center of the force-loaded square)
		int goal;
		long ach;                      // bit i = item i obtained by the team this episode; bit 31 = goal
		boolean goalDone;
		int steps;                     // completed decisions in this episode (sim W.t)
		boolean fresh = true;          // first batch of the session: no action has completed yet
		int episode = 1;

		Team(int index, int agents) {
			this.index = index;
			this.pend = new double[agents];
		}

		/** Item obtained by bot b (mined, crafted or smelted): sim obtain(). */
		void gain(Bot b, int idx) {
			int v = ITEM_VALUE[idx];
			long bit = 1L << idx;
			if (v <= 0 || (ach & bit) != 0) return;
			ach |= bit;
			double r = 0.1 * (1.0 + Math.log(v) / Math.log(2.0));
			for (int k = 0; k < pend.length; k++) pend[k] += TEAM_SHARE * r;
			pend[b.slot] += (1.0 - TEAM_SHARE) * r;
		}
	}

	/** Stuck detection state per bot (keyed by bot name). */
	static final class Track {
		boolean init;
		double ax, az;                 // anchor: where the bot was when it last moved or was rewarded
		int idle;                      // decisions since the anchor moved
	}

	/** A batch whose observations are being built on worker threads; sent once all are done. */
	static final class PendingBatch {
		final List<CompletableFuture<JsonObject>> elems;
		final long createdNs;

		PendingBatch(List<CompletableFuture<JsonObject>> elems) {
			this.elems = elems;
			this.createdNs = System.nanoTime();
		}

		boolean allDone() {
			for (CompletableFuture<JsonObject> f : elems) if (!f.isDone()) return false;
			return true;
		}
	}

	static boolean running;
	static final List<Team> teams = new ArrayList<>();
	static final Map<String, Bot> bots = new LinkedHashMap<>();
	static final List<double[]> usedSpots = new ArrayList<>();
	static final Map<String, Track> tracks = new HashMap<>();
	static double[] refPos = {0, 0};
	static int agentCount = DEFAULT_AGENTS;
	static int[] goals = DEFAULT_GOALS.clone();
	static float userTps = DEFAULT_TPS;  // ceiling set by /aibot train speed
	static float tps = DEFAULT_TPS;      // effective tick rate: adaptive, never above userTps
	static long adaptHoldUntil;          // no raise before this server tick (after a lowering)
	static double savedTpsValue = 20.0;
	static boolean sprinting;
	static CompletableFuture<JsonObject> inflight;
	static PendingBatch pending;
	static CompletableFuture<Obs.Layout> probe;
	static int retryAt;
	static boolean needNewSession;
	static String select;                 // dashboard-selected bot (name), set by the bridge
	static int lastRegionTick = -1000;
	static final Random RNG = new Random();

	static long startedMs, decisions, episodes;
	static int stuckTeleports;
	static final ArrayDeque<Double> returns = new ArrayDeque<>();
	static JsonObject bridgeStats = new JsonObject();
	static String lastError = "";
	static long lastErrorTick = -1000;
	static int consecutiveErrors;

	// perf window (FT-1): reset every PERF_WINDOW_NS, last line kept for /aibot train status
	static long perfStartNs, perfStartTick, perfTicks, perfTrainNs, perfTrainMaxNs;
	static long perfCaptureNs, perfCaptureMaxNs, perfCaptures, perfObsNs, perfObsMaxNs, perfBatches;
	static double perfMsptSum, perfMsptMax;
	static String perfLast = "perf: no 60 s window yet";

	// ---------------------------------------------------------------- commands

	static void start(CommandSourceStack src, int teamCount, int agents) {
		MinecraftServer server = src.getServer();
		if (running) {
			src.sendFailure(Component.literal("training already running - /aibot train stop first"));
			return;
		}
		if (teamCount < 1 || teamCount > MAX_TEAMS || agents < 1 || agents > MAX_AGENTS) {
			src.sendFailure(Component.literal("teams 1..8, agents 1..5"));
			return;
		}
		for (int t = 1; t <= teamCount; t++)
			for (int k = 0; k < agents; k++) {
				String name = name(t, k);
				if (bots.containsKey(name) || server.getPlayerList().getPlayerByName(name) != null) {
					src.sendFailure(Component.literal("a player named " + name + " already exists"));
					return;
				}
			}
		ServerLevel level = server.overworld();
		double rx, rz;
		if (src.getEntity() instanceof ServerPlayer p) {
			rx = p.getX();
			rz = p.getZ();
		} else {
			rx = 0;                              // no player ran the command: measure from the world origin
			rz = 0;
		}
		refPos = new double[] {rx, rz};
		usedSpots.clear();
		List<double[]> spots = new ArrayList<>();
		for (int t = 1; t <= teamCount; t++) {
			double[] spot = pickSpot(level);
			if (spot == null) {
				usedSpots.clear();
				src.sendFailure(Component.literal("could not find a dry grass spot 300+ blocks away - try again"));
				return;
			}
			usedSpots.add(spot);
			spots.add(spot);
		}
		agentCount = agents;
		teams.clear();
		bots.clear();
		tracks.clear();
		for (int t = 1; t <= teamCount; t++) {
			Team team = new Team(t, agents);
			double[] spot = spots.get(t - 1);
			team.spotX = spot[0];
			team.spotY = spot[1];
			team.spotZ = spot[2];
			team.cx = (int) Math.floor(spot[0]) >> 4;
			team.cz = (int) Math.floor(spot[2]) >> 4;
			team.goal = pickGoal();
			forceArea(level, team, true);
			teams.add(team);
			for (int k = 0; k < agents; k++) {
				String name = name(t, k);
				double[] p = slotPos(team, k, agents);
				ServerPlayer player = fakePlayer(server, level, name, p[0], p[1], p[2], (float) p[3]);
				Bot b = new Bot(name, player, CraftBotMod.POLICY);
				b.team = team;
				b.slot = k;
				team.bots.add(b);
				bots.put(name, b);
				placeBot(b, team, agents);
			}
		}
		running = true;
		startedMs = System.currentTimeMillis();
		decisions = 0;
		episodes = 0;
		stuckTeleports = 0;
		returns.clear();
		bridgeStats = new JsonObject();
		lastError = "";
		consecutiveErrors = 0;
		needNewSession = true;
		inflight = null;
		pending = null;
		probe = null;
		retryAt = 0;
		select = null;
		tps = userTps;
		adaptHoldUntil = 0;
		savedTpsValue = server.tickRateManager().tickrate();
		applyTps(server);
		perfReset(server);
		String msg = String.format("training started: %d team(s) x %d bot(s) at %s, tick rate %.0f (cap %.0f)",
			teamCount, agents, describeSpots(), tps, userTps);
		src.sendSuccess(() -> Component.literal(msg), true);
		System.out.println("[craftbot-train] " + msg);
	}

	static void stop(MinecraftServer server, String why) {
		if (!running) return;
		running = false;
		JsonObject end = new JsonObject();
		end.addProperty("cmd", "train_end");
		end.addProperty("reason", why);
		CraftBotMod.POLICY.send(end).exceptionally(e -> null);  // bridge saves a checkpoint on this
		ServerLevel level = server.overworld();
		for (Team t : teams) forceArea(level, t, false);
		for (Bot b : new ArrayList<>(bots.values())) drop(server, b);
		bots.clear();
		teams.clear();
		usedSpots.clear();
		tracks.clear();
		inflight = null;
		pending = null;
		if (sprinting) server.tickRateManager().stopSprinting();
		sprinting = false;
		server.tickRateManager().setTickRate((float) savedTpsValue);
		server.onTickRateChanged();
		System.out.println("[craftbot-train] stopped (" + why + "), " + decisions + " decisions, " + episodes + " episodes");
	}

	static void speed(CommandSourceStack src, MinecraftServer server, float value) {
		userTps = value;
		tps = value;
		sprinting = false;
		applyTps(server);
		src.sendSuccess(() -> Component.literal(String.format("tick rate %.0f (cap; adaptive lowers it when mspt > %.0f)%s",
			userTps, ADAPT_HIGH_MSPT, running ? "" : " (applies when training starts)")), true);
	}

	static void sprint(CommandSourceStack src, MinecraftServer server) {
		sprinting = true;
		if (running) applyTps(server);
		src.sendSuccess(() -> Component.literal("training: vanilla tick sprint"
			+ (running ? "" : " (applies when training starts)")), true);
	}

	static void goals(CommandSourceStack src, String csv) {
		if (csv.equals("default")) {
			goals = DEFAULT_GOALS.clone();
		} else {
			List<Integer> out = new ArrayList<>();
			for (String raw : csv.split(",")) {
				String name = raw.trim();
				int idx = Arrays.asList(Obs.ITEM_NAMES).indexOf(name);
				if (idx < 0 || ITEM_VALUE[idx] == 0) {
					src.sendFailure(Component.literal("bad goal '" + name + "'. Options: "
						+ String.join(", ", Obs.ITEM_NAMES) + " (dirt is not a goal)"));
					return;
				}
				if (!out.contains(idx)) out.add(idx);
			}
			goals = out.stream().mapToInt(Integer::intValue).toArray();
		}
		StringBuilder sb = new StringBuilder();
		for (int g : goals) sb.append(sb.length() > 0 ? ", " : "").append(Obs.ITEM_NAMES[g]);
		String text = "training goals: " + sb + " (applies to each team's next episode)";
		src.sendSuccess(() -> Component.literal(text), true);
	}

	static void status(CommandSourceStack src) {
		StringBuilder sb = new StringBuilder();
		if (!running) {
			sb.append("training: stopped");
		} else {
			MinecraftServer server = src.getServer();
			double secs = Math.max(1e-3, (System.currentTimeMillis() - startedMs) / 1000.0);
			double meanRet = 0;
			for (double r : returns) meanRet += r;
			if (!returns.isEmpty()) meanRet /= returns.size();
			sb.append(String.format("training: running, %d team(s) x %d, tick rate %.0f (cap %.0f%s)",
				teams.size(), agentCount, tps, userTps, sprinting ? ", sprint" : ""));
			sb.append(String.format("%n  decisions %d (%.1f/s), episodes %d, mean return(last %d) %.3f",
				decisions, decisions / secs, episodes, returns.size(), meanRet));
			sb.append(String.format("%n  server now: mspt %.1f (rolling avg), TPS %.1f",
				server.getAverageTickTimeNanos() / 1e6, tpsNow(server)));
			sb.append(String.format("%n  %s", perfLast));
			sb.append(String.format("%n  stuck teleports %d", stuckTeleports));
			for (Team t : teams)
				sb.append(String.format("%n  team %d: goal=%s episode=%d steps=%d ach=%s", t.index,
					Obs.ITEM_NAMES[t.goal], t.episode, t.steps, t.goalDone ? "goal" : "-"));
			sb.append(String.format("%n  bridge: %s", inflight != null ? "request in flight"
				: pending != null ? "observations building" : (CraftBotMod.POLICY.layout() == null ? "not connected" : "idle")));
			if (bridgeStats.size() > 0) sb.append("\n  bridge stats: ").append(bridgeStats);
			if (!lastError.isEmpty()) sb.append("\n  last error: ").append(lastError);
		}
		String text = sb.toString();
		src.sendSuccess(() -> Component.literal(text), false);
	}

	// ---------------------------------------------------------------- per tick

	/** Called from END_SERVER_TICK. Never throws (CraftBotMod wraps it too). */
	static void tick(MinecraftServer server) {
		if (!running) return;
		long t0 = System.nanoTime();
		try {
			tickTraining(server);
		} finally {
			long dt = System.nanoTime() - t0;
			perfTrainNs += dt;
			perfTrainMaxNs = Math.max(perfTrainMaxNs, dt);
			if (running) monitor(server);
		}
	}

	private static void tickTraining(MinecraftServer server) {
		if (CraftBotMod.POLICY.layout() == null) {
			connect(server);
			return;
		}
		ServerLevel level = server.overworld();
		for (Bot b : new ArrayList<>(bots.values())) {
			if (b.player.isRemoved() || b.player.isDeadOrDying()) respawn(server, b);
			b.tick(server);
		}
		if (inflight != null && inflight.isDone()) {
			CompletableFuture<JsonObject> f = inflight;
			inflight = null;
			try {
				apply(server, f.join());
				consecutiveErrors = 0;
			} catch (Exception e) {
				fail(server, e);
			}
		}
		if (!running) return;   // a failure may have stopped training
		if (pending == null && inflight == null && server.getTickCount() >= retryAt && allReady(level)) {
			try {
				capture(server, level);
			} catch (Exception e) {
				fail(server, e);
			}
		}
		if (running && pending != null && pending.allDone()) {
			try {
				send(server);
			} catch (Exception e) {
				fail(server, e);
			}
		}
		if (running && server.getTickCount() % ADAPT_EVERY == 0) adapt(server);
	}

	/** An exception in the training tick: logged, training stops after repeated errors. Never rethrown. */
	static void onTickError(MinecraftServer server, Throwable t) {
		System.err.println("[craftbot-train] tick error: " + t);
		t.printStackTrace();
		if (!running) return;
		lastError = t.toString();
		if (++consecutiveErrors >= MAX_ERRORS) stop(server, "too many errors: " + t);
	}

	/** First contact: the bridge's checkpoint layout (same probe as /aibot spawn). Bots wait meanwhile. */
	private static void connect(MinecraftServer server) {
		if (server.getTickCount() < retryAt) return;
		if (probe == null) {
			probe = CraftBotMod.POLICY.probe();
			return;
		}
		if (!probe.isDone()) return;
		if (probe.isCompletedExceptionally()) fail(server, new IllegalStateException("policy bridge unreachable - is mcbridge --train running?"));
		probe = null;
	}

	private static boolean allReady(ServerLevel level) {
		for (Bot b : bots.values()) {
			if (!b.ready()) return false;
			if (!level.isLoaded(b.player.blockPosition())) {
				warn(level.getServer(), "bot " + b.name + " is in an unloaded chunk; holding the batch");
				return false;
			}
		}
		return true;
	}

	/**
	 * Server thread, step 1 of a batch: episode bookkeeping and rewards (as before), stuck check, then the block
	 * snapshot per team and the body pose per bot. Observations are then built on worker threads.
	 */
	private static void capture(MinecraftServer server, ServerLevel level) {
		long t0 = System.nanoTime();
		Obs.Layout L = CraftBotMod.POLICY.layout();
		List<CompletableFuture<JsonObject>> elems = new ArrayList<>();
		for (Team t : teams) {
			if (t.fresh) t.fresh = false;
			else t.steps++;                       // the window that just ended belongs to this episode
			int n = t.bots.size();
			if (!t.goalDone) {
				for (Bot b : t.bots)
					if (Obs.count(b.player.getInventory(), Obs.ITEMS[t.goal]) > 0) {
						t.goalDone = true;
						t.ach |= 1L << 31;
						for (int k = 0; k < n; k++) t.pend[k] += TEAM_SHARE * GOAL_REWARD;
						t.pend[b.slot] += (1.0 - TEAM_SHARE) * GOAL_REWARD;
						break;
					}
			}
			boolean over = t.goalDone || t.steps >= MAX_STEPS;
			double[] reward = new double[n];
			double[] epRet = new double[n];
			for (int k = 0; k < n; k++) {
				Bot b = t.bots.get(k);
				reward[k] = t.pend[k];
				t.pend[k] = 0;
				b.epRet += reward[k];
				epRet[k] = b.epRet;
			}
			boolean succ = t.goalDone;
			int endedLen = t.steps;
			if (over) {
				double mean = 0;
				for (double r : epRet) mean += r;
				pushReturn(mean / Math.max(1, n));
				episodes++;
				resetTeam(server, level, t);
			}
			// Stuck check after any reset (a reset counts as movement) and before the snapshot, so the
			// observation reflects the teleported position.
			for (int k = 0; k < n; k++) checkStuck(level, t.bots.get(k), reward[k], over);

			Obs.Snapshot snap = snapshotTeam(level, t);
			for (int k = 0; k < n; k++) {
				Bot b = t.bots.get(k);
				Obs.Pose pose = Obs.Pose.capture(level, b.player.getX(), b.player.getY(), b.player.getZ(),
					b.player.getYRot(), b.player.getXRot(), b.player.onGround(), b.player.getHealth(),
					b.mineProgress, b.fuel, t.goal, t.steps, b.player.getInventory(), teammates(t, b), bodies(t, b));
				JsonObject info = new JsonObject();
				info.addProperty("team", t.index);
				info.addProperty("slot", b.slot);
				info.addProperty("goal", t.goal);
				info.addProperty("episode", t.episode);
				info.addProperty("step", t.steps);
				info.addProperty("ach", t.ach);
				info.addProperty("success", over && succ);
				info.addProperty("ep_ret", over ? epRet[k] : 0.0);
				info.addProperty("ep_len", over ? endedLen : 0);
				info.addProperty("died", b.diedFlag);
				info.addProperty("msg", b.msg);
				info.addProperty("hp", b.player.getHealth());
				info.addProperty("yaw", b.player.getYRot());
				info.addProperty("pitch", b.player.getXRot());
				JsonArray posJson = new JsonArray();
				posJson.add(b.player.getX());
				posJson.add(b.player.getY());
				posJson.add(b.player.getZ());
				info.add("pos", posJson);
				info.add("inv", inventoryJson(b.player.getInventory()));
				boolean wantRegion = select != null && select.equals(b.name)
					&& server.getTickCount() - lastRegionTick >= REGION_EVERY;
				if (wantRegion) lastRegionTick = server.getTickCount();
				b.diedFlag = false;
				final String name = b.name;
				final int team = t.index, slot = b.slot;
				final double rew = reward[k];
				final boolean done = over;
				elems.add(ObsWorker.submit(() -> element(L, snap, pose, info, name, team, slot, rew, done, wantRegion)));
			}
		}
		pending = new PendingBatch(elems);
		decisions += elems.size();
		long dt = System.nanoTime() - t0;
		perfCaptures++;
		perfCaptureNs += dt;
		perfCaptureMaxNs = Math.max(perfCaptureMaxNs, dt);
	}

	/** Server thread, step 3: all observations are ready; send the batch (PROTOCOL 11.3). */
	private static void send(MinecraftServer server) {
		PendingBatch pb = pending;
		pending = null;
		JsonArray arr = new JsonArray();
		for (CompletableFuture<JsonObject> f : pb.elems) arr.add(f.join());
		long obsNs = System.nanoTime() - pb.createdNs;
		perfObsNs += obsNs;
		perfObsMaxNs = Math.max(perfObsMaxNs, obsNs);
		perfBatches++;
		JsonObject req = new JsonObject();
		req.addProperty("cmd", "train_step");
		if (needNewSession) req.addProperty("new_session", true);
		req.add("bots", arr);
		inflight = CraftBotMod.POLICY.send(req);
	}

	/** Worker thread: one bot's element of the batch. Reads only the immutable snapshot and pose. */
	private static JsonObject element(Obs.Layout L, Obs.Snapshot snap, Obs.Pose pose, JsonObject info, String name,
	                                  int team, int slot, double reward, boolean over, boolean wantRegion) {
		byte[] img = L.newImage();
		float[] vec = L.newVec();
		Obs.buildFrom(snap, L, pose, img, vec);
		if (wantRegion) info.add("region", region(snap, pose.x, pose.y, pose.z));
		JsonObject e = new JsonObject();
		e.addProperty("id", name);
		e.addProperty("team", team);
		e.addProperty("slot", slot);
		e.addProperty("img", Base64.getEncoder().encodeToString(img));
		JsonArray vj = new JsonArray();
		for (float f : vec) vj.add(f);
		e.add("vec", vj);
		e.addProperty("reward", reward);
		e.addProperty("done", over);
		e.add("info", info);
		return e;
	}

	/** Applies the bridge's actions: each bot starts a new decision window. */
	private static void apply(MinecraftServer server, JsonObject reply) {
		JsonArray acts = reply.getAsJsonArray("actions");
		for (JsonElement el : acts) {
			JsonObject a = el.getAsJsonObject();
			Bot b = bots.get(a.get("id").getAsString());
			if (b == null) continue;
			b.setMsg(server, a.has("msg") ? a.get("msg").getAsInt() : 0);
			b.start(a.get("action").getAsInt());
		}
		select = reply.has("select") && !reply.get("select").isJsonNull() ? reply.get("select").getAsString() : null;
		if (reply.has("stats") && reply.get("stats").isJsonObject()) bridgeStats = reply.getAsJsonObject("stats");
		needNewSession = false;
	}

	/** Bridge down, reply lost or error: bots idle, next batch starts a fresh session after a short pause. */
	private static void fail(MinecraftServer server, Exception e) {
		fail(server, e.getMessage() != null ? e.getMessage() : e.toString());
	}

	private static void fail(MinecraftServer server, String text) {
		needNewSession = true;
		pending = null;
		retryAt = server.getTickCount() + RETRY_TICKS;
		for (Bot b : bots.values()) b.resetState();
		lastError = text;
		if (++consecutiveErrors >= MAX_ERRORS) {
			stop(server, "too many bridge errors: " + text);
			return;
		}
		warn(server, "bridge error (" + text + "), bots idle, retrying");
	}

	private static void warn(MinecraftServer server, String text) {
		long now = server.getTickCount();
		if (now - lastErrorTick > 200) {
			lastErrorTick = now;
			System.out.println("[craftbot-train] " + text);
			server.getPlayerList().broadcastSystemMessage(Component.literal("[train] " + text), false);
		}
	}

	// ---------------------------------------------------------------- perf (FT-1)

	private static void perfReset(MinecraftServer server) {
		perfStartNs = System.nanoTime();
		perfStartTick = server.getTickCount();
		perfTicks = 0;
		perfTrainNs = perfTrainMaxNs = 0;
		perfCaptureNs = perfCaptureMaxNs = perfCaptures = 0;
		perfObsNs = perfObsMaxNs = perfBatches = 0;
		perfMsptSum = perfMsptMax = 0;
	}

	/** Samples mspt every tick; every 60 s logs the window and keeps it for /aibot train status. */
	private static void monitor(MinecraftServer server) {
		perfTicks++;
		double mspt = server.getAverageTickTimeNanos() / 1e6;
		perfMsptSum += mspt;
		perfMsptMax = Math.max(perfMsptMax, mspt);
		long now = System.nanoTime();
		if (now - perfStartNs < PERF_WINDOW_NS) return;
		double secs = (now - perfStartNs) / 1e9;
		double measuredTps = (server.getTickCount() - perfStartTick) / secs;
		double trainAvg = perfTicks > 0 ? perfTrainNs / 1e6 / perfTicks : 0;
		double captureAvg = perfCaptures > 0 ? perfCaptureNs / 1e6 / perfCaptures : 0;
		double obsAvg = perfBatches > 0 ? perfObsNs / 1e6 / perfBatches : 0;
		perfLast = String.format("perf (60 s): mspt avg %.2f max %.2f, TPS %.1f, train cost/tick avg %.2f ms max %.2f, "
				+ "capture avg %.2f ms max %.2f, obs latency avg %.1f ms max %.1f, batches %d",
			perfTicks > 0 ? perfMsptSum / perfTicks : 0, perfMsptMax, measuredTps, trainAvg, perfTrainMaxNs / 1e6,
			captureAvg, perfCaptureMaxNs / 1e6, obsAvg, perfObsMaxNs / 1e6, perfBatches);
		System.out.println("[craftbot-train] " + perfLast + ", tick rate " + tps + " (cap " + userTps + ")");
		perfReset(server);
	}

	private static double tpsNow(MinecraftServer server) {
		double mspt = server.getAverageTickTimeNanos() / 1e6;
		return mspt <= 0 ? server.tickRateManager().tickrate() : Math.min(server.tickRateManager().tickrate(), 1000.0 / mspt);
	}

	/**
	 * Adaptive tick rate: lower when mspt > 45, raise back (10% steps) when mspt < 25, never above the user's
	 * cap. Skipped during vanilla sprint, which sets its own rate.
	 */
	private static void adapt(MinecraftServer server) {
		if (sprinting) return;
		double mspt = server.getAverageTickTimeNanos() / 1e6;
		long now = server.getTickCount();
		if (mspt > ADAPT_HIGH_MSPT && tps > ADAPT_MIN_TPS) {
			tps = Math.max(ADAPT_MIN_TPS, tps * 0.8f);
			adaptHoldUntil = now + ADAPT_HOLD_TICKS;
			applyTps(server);
			System.out.println(String.format("[craftbot-train] mspt %.1f > %.0f: tick rate lowered to %.0f", mspt, ADAPT_HIGH_MSPT, tps));
		} else if (mspt < ADAPT_LOW_MSPT && tps < userTps && now >= adaptHoldUntil) {
			tps = Math.min(userTps, tps * 1.1f);
			applyTps(server);
		}
	}

	// ---------------------------------------------------------------- episodes, spots, bots

	private static void pushReturn(double r) {
		returns.addLast(r);
		while (returns.size() > 200) returns.removeFirst();
	}

	private static int pickGoal() {
		return goals[RNG.nextInt(goals.length)];
	}

	/** Team reset: new far spot, new goal, cleared inventories, healed and fed, bots placed on the spot. */
	private static void resetTeam(MinecraftServer server, ServerLevel level, Team t) {
		forceArea(level, t, false);
		double[] spot = pickSpot(level);
		if (spot != null) {
			usedSpots.add(spot);
			t.spotX = spot[0];
			t.spotY = spot[1];
			t.spotZ = spot[2];
			t.cx = (int) Math.floor(spot[0]) >> 4;
			t.cz = (int) Math.floor(spot[2]) >> 4;
		} else {
			warn(server, "team " + t.index + ": no new spot found, staying at the old one");
		}
		forceArea(level, t, true);
		t.goal = pickGoal();
		t.ach = 0;
		t.goalDone = false;
		t.steps = 0;
		t.episode++;
		java.util.Arrays.fill(t.pend, 0.0);
		for (Bot b : t.bots) {
			b.epRet = 0;
			b.resetState();
			ServerPlayer p = b.player;
			p.getInventory().clearContent();
			p.setHealth(p.getMaxHealth());
			p.getFoodData().setFoodLevel(20);
			p.getFoodData().setSaturation(5f);
			p.removeAllEffects();
			placeBot(b, t, agentCount);
		}
	}

	/** Respawn a dead training bot on its team spot (inventory is lost, the episode goes on). */
	private static void respawn(MinecraftServer server, Bot b) {
		ServerPlayer old = b.player;
		if (server.getPlayerList().getPlayer(old.getUUID()) != null) server.getPlayerList().remove(old);
		ServerLevel level = server.overworld();
		double[] p = slotPos(b.team, b.slot, agentCount);
		ServerPlayer np = fakePlayer(server, level, b.name, p[0], p[1], p[2], (float) p[3]);
		b.replacePlayer(np);
		placeBot(b, b.team, agentCount);
		b.diedFlag = true;
		System.out.println("[craftbot-train] " + b.name + " died, respawned on team " + b.team.index + " spot");
	}

	/** Slot k stands 2 blocks from the spot centre, facing it. Returns x, y, z, yaw. */
	private static double[] slotPos(Team t, int k, int agents) {
		double a = 2.0 * Math.PI * k / Math.max(1, agents);
		double cx = Math.cos(a), sz = Math.sin(a);
		return new double[] {t.spotX + 2.0 * cx, t.spotY, t.spotZ + 2.0 * sz,
			Math.toDegrees(Math.atan2(cx, sz))};
	}

	private static void placeBot(Bot b, Team t, int agents) {
		double[] p = slotPos(t, b.slot, agents);
		b.player.snapTo(p[0], p[1], p[2], (float) p[3], 0f);
		b.player.setDeltaMovement(Vec3.ZERO);
	}

	/** Random safe spot 300+ blocks from the start point and from every used spot (see spotY). */
	private static double[] pickSpot(ServerLevel level) {
		for (int attempt = 0; attempt < 80; attempt++) {
			double ang = RNG.nextDouble() * 2.0 * Math.PI;
			double dist = MIN_SPOT_DIST + RNG.nextDouble() * SEARCH_RADIUS;
			double x = refPos[0] + Math.cos(ang) * dist, z = refPos[1] + Math.sin(ang) * dist;
			if (!farEnough(x, z)) continue;
			int bx = (int) Math.floor(x), bz = (int) Math.floor(z);
			ChunkAccess chunk = level.getChunk(bx >> 4, bz >> 4, ChunkStatus.FULL, true);  // generates if needed
			if (chunk == null) continue;
			int h = spotY(level, bx, bz);
			if (h < 0) continue;
			return new double[] {bx + 0.5, h, bz + 0.5};
		}
		return null;
	}

	/**
	 * Feet y of a good spawn or teleport column, or -1. Good = dry grass or dirt on flat ground, with no water
	 * on or beside it, not ocean/river, and open sky above the feet (no cave ceiling, no leaves over it).
	 * The chunk must already be loaded.
	 */
	static int spotY(ServerLevel level, int bx, int bz) {
		if (level.getChunkSource().getChunkNow(bx >> 4, bz >> 4) == null) return -1;
		int h = level.getHeight(Heightmap.Types.MOTION_BLOCKING_NO_LEAVES, bx, bz);   // feet: first air above ground
		if (h <= level.getSeaLevel() + 1) return -1;                                  // ocean floor, beach, underwater
		if (level.getHeight(Heightmap.Types.WORLD_SURFACE, bx, bz) > h) return -1;     // leaves, overhang, cave ceiling
		if (!level.getFluidState(new BlockPos(bx, h, bz)).isEmpty()) return -1;
		BlockState top = level.getBlockState(new BlockPos(bx, h - 1, bz));
		if (!(top.is(Blocks.GRASS_BLOCK) || top.is(Blocks.DIRT))) return -1;
		var biome = level.getBiome(new BlockPos(bx, h, bz));
		if (biome.is(BiomeTags.IS_OCEAN) || biome.is(BiomeTags.IS_DEEP_OCEAN) || biome.is(BiomeTags.IS_RIVER)) return -1;
		for (int dz = -2; dz <= 2; dz++)
			for (int dx = -2; dx <= 2; dx++) {
				int nx = bx + dx, nz = bz + dz;
				int nh = level.getHeight(Heightmap.Types.MOTION_BLOCKING_NO_LEAVES, nx, nz);
				if (Math.abs(nh - h) > 1) return -1;                                  // not flat
				if (!level.getFluidState(new BlockPos(nx, nh, nz)).isEmpty()
					|| !level.getFluidState(new BlockPos(nx, nh - 1, nz)).isEmpty()) return -1;  // water beside
			}
		return h;
	}

	/** A safe column within STUCK_RADIUS of (cx, cz), at least 4 blocks away. Returns x, y, z or null. */
	private static double[] nearSpot(ServerLevel level, double cx, double cz) {
		for (int attempt = 0; attempt < 60; attempt++) {
			double ang = RNG.nextDouble() * 2.0 * Math.PI;
			double d = 4.0 + RNG.nextDouble() * (STUCK_RADIUS - 4.0);
			int bx = (int) Math.floor(cx + Math.cos(ang) * d), bz = (int) Math.floor(cz + Math.sin(ang) * d);
			int h = spotY(level, bx, bz);
			if (h >= 0) return new double[] {bx + 0.5, h, bz + 0.5};
		}
		return null;
	}

	/**
	 * Stuck detection (FT-1): no movement (>= 1 block from the anchor) and no reward for STUCK_DECISIONS decisions
	 * -> teleport within 20 blocks to a safe spot. A reward or an episode reset counts as progress.
	 */
	private static void checkStuck(ServerLevel level, Bot b, double reward, boolean reset) {
		Track tr = tracks.computeIfAbsent(b.name, k -> new Track());
		double x = b.player.getX(), z = b.player.getZ();
		if (Bot.needsRescue(b.player)) {   // lava / void / suffocating: move to a safe column right away
			double[] p = nearSpot(level, x, z);
			if (p != null) {
				b.player.snapTo(p[0], p[1], p[2], b.player.getYRot(), 0f);
				b.player.setDeltaMovement(Vec3.ZERO);
				b.player.clearFire();
				b.player.resetFallDistance();
				b.resetState();
				tr.ax = p[0];
				tr.az = p[2];
				tr.idle = 0;
				tr.init = true;
				stuckTeleports++;
				System.out.println(String.format("[craftbot-train] %s rescued (lava/void/wall) to (%.0f, %.0f, %.0f)",
					b.name, p[0], p[1], p[2]));
				return;
			}
		}
		double dx = x - tr.ax, dz = z - tr.az;
		if (!tr.init || reset || reward != 0 || dx * dx + dz * dz >= STUCK_MOVE * STUCK_MOVE) {
			tr.init = true;
			tr.ax = x;
			tr.az = z;
			tr.idle = 0;
			return;
		}
		if (++tr.idle < STUCK_DECISIONS) return;
		double[] p = nearSpot(level, x, z);
		if (p == null) {
			tr.idle = STUCK_DECISIONS - 25;   // no safe spot nearby: try again in 25 decisions
			return;
		}
		b.player.snapTo(p[0], p[1], p[2], b.player.getYRot(), 0f);
		b.player.setDeltaMovement(Vec3.ZERO);
		b.resetState();
		tr.ax = p[0];
		tr.az = p[2];
		tr.idle = 0;
		stuckTeleports++;
		System.out.println(String.format("[craftbot-train] %s stuck for %d decisions, teleported to (%.0f, %.0f, %.0f)",
			b.name, STUCK_DECISIONS, p[0], p[1], p[2]));
	}

	private static boolean farEnough(double x, double z) {
		double dx = x - refPos[0], dz = z - refPos[1];
		if (dx * dx + dz * dz < MIN_SPOT_DIST * MIN_SPOT_DIST) return false;
		for (double[] s : usedSpots) {
			dx = x - s[0];
			dz = z - s[2];
			if (dx * dx + dz * dz < MIN_SPOT_DIST * MIN_SPOT_DIST) return false;
		}
		return true;
	}

	private static void forceArea(ServerLevel level, Team t, boolean on) {
		for (int dx = -FORCE_RADIUS; dx <= FORCE_RADIUS; dx++)
			for (int dz = -FORCE_RADIUS; dz <= FORCE_RADIUS; dz++)
				level.setChunkForced(t.cx + dx, t.cz + dz, on);
	}

	private static String describeSpots() {
		StringBuilder sb = new StringBuilder();
		for (Team t : teams) {
			if (sb.length() > 0) sb.append(", ");
			sb.append(String.format("team %d (%d, %d)", t.index, (int) t.spotX, (int) t.spotZ));
		}
		return sb.toString();
	}

	static String name(int team, int slot) {
		return "ft" + team + "_" + slot;
	}

	private static ServerPlayer fakePlayer(MinecraftServer server, ServerLevel level, String name,
	                                       double x, double y, double z, float yaw) {
		GameProfile profile = new GameProfile(UUIDUtil.createOfflinePlayerUUID(name), name);
		ServerPlayer player = new ServerPlayer(server, level, profile, ClientInformation.createDefault());
		player.snapTo(x, y, z, yaw, 0f);
		server.getPlayerList().placeNewPlayer(new FakeConnection(), player,
			new CommonListenerCookie(profile, 0, player.clientInformation(), false));
		player.setGameMode(GameType.SURVIVAL);
		player.setHealth(player.getMaxHealth());
		return player;
	}

	private static void drop(MinecraftServer server, Bot b) {
		bots.remove(b.name);
		if (server.getPlayerList().getPlayer(b.player.getUUID()) != null) server.getPlayerList().remove(b.player);
	}

	private static void applyTps(MinecraftServer server) {
		var rm = server.tickRateManager();
		rm.setTickRate(tps);
		if (sprinting) rm.requestGameToSprint(1_000_000);
		server.onTickRateChanged();
	}

	// ---------------------------------------------------------------- observations for the batch

	/** Other slots of the team in slot order (absent = far away, zero in the obs); self excluded. */
	private static List<Obs.Mate> teammates(Team t, Bot self) {
		List<Obs.Mate> out = new ArrayList<>();
		for (Bot o : t.bots) {
			if (o == self) continue;
			out.add(new Obs.Mate(o.player.getX(), o.player.getY(), o.player.getZ(), o.msg));
		}
		return out;
	}

	/** Teammates as rays see them (AGENT pseudo-block). Only present members. */
	private static List<Obs.Mate> bodies(Team t, Bot self) {
		List<Obs.Mate> out = new ArrayList<>();
		for (Bot o : t.bots)
			if (o != self) out.add(new Obs.Mate(o.player.getX(), o.player.getY(), o.player.getZ(), o.msg));
		return out;
	}

	private static JsonObject inventoryJson(Inventory inv) {
		JsonObject o = new JsonObject();
		for (int i = 0; i < Obs.ITEM_NAMES.length; i++) {
			int n = Obs.count(inv, Obs.ITEMS[i]);
			if (n > 0) o.addProperty(Obs.ITEM_NAMES[i], n);
		}
		return o;
	}

	/**
	 * One block box covering every bot of the team: each bot's eye +- 24 (rays), chest scan and near box. Copied
	 * on the server thread with section-level reads (no per-block chunk lookups).
	 */
	private static Obs.Snapshot snapshotTeam(ServerLevel level, Team t) {
		int x0 = Integer.MAX_VALUE, y0 = Integer.MAX_VALUE, z0 = Integer.MAX_VALUE;
		int x1 = Integer.MIN_VALUE, y1 = Integer.MIN_VALUE, z1 = Integer.MIN_VALUE;
		for (Bot b : t.bots) {
			int px = (int) Math.floor(b.player.getX()), py = (int) Math.floor(b.player.getY());
			int pz = (int) Math.floor(b.player.getZ());
			x0 = Math.min(x0, px - SNAP_MARGIN);
			x1 = Math.max(x1, px + SNAP_MARGIN);
			y0 = Math.min(y0, py - SNAP_MARGIN);
			y1 = Math.max(y1, py + SNAP_MARGIN + 1);
			z0 = Math.min(z0, pz - SNAP_MARGIN);
			z1 = Math.max(z1, pz + SNAP_MARGIN);
		}
		return Obs.Snapshot.capture(level, x0, y0, z0, x1, y1, z1);
	}

	/** Block region (2R+1)^2 x (2R+1) around the bot, index (y*sz+z)*sx+x, sim block ids (base64). */
	private static JsonObject region(Obs.BlockSource src, double px, double py, double pz) {
		int R = REGION_RADIUS, n = 2 * R + 1;
		int cx = (int) Math.floor(px), cy = (int) Math.floor(py), cz = (int) Math.floor(pz);
		byte[] arr = new byte[n * n * n];
		for (int y = 0; y < n; y++)
			for (int z = 0; z < n; z++)
				for (int x = 0; x < n; x++)
					arr[(y * n + z) * n + x] = (byte) Obs.blockId(src.at(cx - R + x, cy - R + y, cz - R + z));
		JsonObject o = new JsonObject();
		JsonArray origin = new JsonArray();
		origin.add(cx - R);
		origin.add(cy - R);
		origin.add(cz - R);
		o.add("origin", origin);
		JsonArray size = new JsonArray();
		size.add(n);
		size.add(n);
		size.add(n);
		o.add("size", size);
		o.addProperty("blocks", Base64.getEncoder().encodeToString(arr));
		return o;
	}

	/** Counts per sim item (index = sim item id), used to find items a bot obtained by mining. */
	static int[] counts(Container c) {
		int[] out = new int[Obs.ITEMS.length];
		for (int i = 0; i < out.length; i++) out[i] = Obs.count(c, Obs.ITEMS[i]);
		return out;
	}

	/** Sim item id of a crafted or smelted item, or -1. */
	static int itemIndex(Item item) {
		ItemStack s = new ItemStack(item);
		for (int i = 0; i < Obs.ITEMS.length; i++) if (Obs.ITEMS[i].test(s)) return i;
		return -1;
	}

	/** Server shutdown or /aibot train stop from CraftBotMod. */
	static void shutdown(MinecraftServer server) {
		stop(server, "server stopping");
		ObsWorker.shutdown();
	}
}
