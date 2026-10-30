package com.minemind.recorder;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.minemind.recorder.FrameRenderer.Pose;
import com.minemind.recorder.FrameRenderer.Region;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.core.BlockPos;
import net.minecraft.core.component.DataComponents;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import net.minecraft.resources.ResourceKey;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.stats.Stat;
import net.minecraft.stats.StatType;
import net.minecraft.stats.Stats;
import net.minecraft.world.Container;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.inventory.AbstractContainerMenu;
import net.minecraft.world.inventory.AbstractFurnaceMenu;
import net.minecraft.world.inventory.ChestMenu;
import net.minecraft.world.inventory.CraftingMenu;
import net.minecraft.world.inventory.InventoryMenu;
import net.minecraft.world.damagesource.DamageSource;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;

import java.io.IOException;
import java.nio.file.Path;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Server side of the recorder (PROTOCOL 9). One session per play session (auto-start on singleplayer join, F8 or
 * /aibot record start). Every server tick (server thread) it diffs the player's statistics and inventory into exact
 * events, snapshots the blocks around the player, and hands one frame to the writer thread, which raycasts the
 * observation and writes chunks / input / events. Callbacks run on the server thread too. Entry points catch and log.
 */
public final class Recorder {
	private Recorder() {}

	public static final String FORMAT = "minemind-rec-2";
	public static final String[] ACTIONS = {"noop", "forward", "back", "left", "right", "jump", "forward_jump",
		"turn_left", "turn_right", "look_up", "look_down", "attack", "place_block", "place_table", "place_furnace",
		"craft_planks", "craft_sticks", "craft_table", "craft_wooden_pickaxe", "craft_stone_pickaxe", "craft_furnace",
		"smelt_iron", "craft_iron_pickaxe", "craft_chest", "place_chest", "deposit", "withdraw", "give"};
	/** Per-tick turn threshold: half the sim turn step (7.5 deg) spread over the 4 ticks of one decision. */
	public static final float TURN_TICK = 7.5f / 4f;
	static final String[] BLOCK_NAMES = {"air", "grass", "dirt", "stone", "cobblestone", "log", "leaves",
		"planks", "crafting_table", "furnace", "coal_ore", "iron_ore", "diamond_ore", "bedrock", "sand", "gravel",
		"water", "deepslate", "chest", "agent"};
	static final int STEP_TICKS = 4;
	static final int ERROR_LIMIT = 50;
	static final int FLUSH_FRAMES = 100;     // ~5 s
	static final int META_FRAMES = 1200;     // 1 min
	static final int CAMERA_MAX_DEG = 45;    // camera bins saturate at this per-tick turn
	static final double TICKS_PER_SECOND = 20.0;
	private static final int JOIN_TRIES = 200;

	public static volatile boolean active = false;
	public static volatile boolean paused = false;
	public static volatile int frameCount = 0;

	private static volatile Session session;
	private static Config config = new Config();
	private static UUID pendingJoin;
	private static int joinTries;

	public static void init() {
		config = Config.load(FabricLoader.getInstance().getConfigDir().resolve("minemind_recorder.json"));
	}

	// ----- commands / keybind (server thread) -----

	public static String start(MinecraftServer server, ServerPlayer player, String goalArg) {
		if (session != null) return "already recording into " + session.dir;
		try {
			session = new Session(server, player, goalArg == null ? "none" : goalArg);
		} catch (Throwable t) {
			CraftLog.error("cannot start recording", t);
			session = null;
			return "cannot create recording: " + t.getMessage();
		}
		active = true;
		paused = false;
		frameCount = 0;
		InputSamples.clearAndEnable(true);
		return "recording to " + session.dir;
	}

	public static String stop(String reason) {
		Session s = session;
		if (s == null) return "not recording";
		session = null;
		String dir = s.dir.toString();
		try {
			s.close(reason);
		} catch (Throwable t) {
			CraftLog.error("finalize failed", t);
		}
		active = false;
		paused = false;
		InputSamples.clearAndEnable(false);
		return "stopped (" + reason + "): " + dir + " frames=" + s.frames;
	}

	public static String pause() {
		Session s = session;
		if (s == null) return "not recording";
		if (s.paused) return "already paused";
		s.pauseNow();
		paused = true;
		return "paused";
	}

	public static String resume() {
		Session s = session;
		if (s == null) return "not recording";
		if (!s.paused) return "not paused";
		s.paused = false;
		paused = false;
		return "resumed";
	}

	public static String status() {
		Session s = session;
		if (s == null) return "not recording";
		return String.format("%s %s, %d frames, context %s -> %s", paused ? "paused" : "recording",
			clock(s.frames), s.frames, s.ctxText == null ? "none" : s.ctxText, s.dir);
	}

	/** /aibot context <text>: starts a context (ends the open one). */
	public static String contextSet(String text) {
		Session s = session;
		if (s == null) return "not recording";
		if (s.paused) return "recording is paused; resume it to set a context";
		String t = text == null ? "" : text.trim();
		if (t.isEmpty()) return contextShow();
		if (t.equals("finish")) return contextFinish();
		if (s.ctxText != null) s.endContext("replaced");
		s.startContext(t);
		return "Context: " + t;
	}

	public static String contextFinish() {
		Session s = session;
		if (s == null) return "not recording";
		if (s.ctxText == null) return "no context open";
		String t = s.ctxText;
		s.endContext("finish");
		return "Context finished: " + t;
	}

	public static String contextShow() {
		Session s = session;
		if (s == null) return "not recording";
		return s.ctxText == null ? "no context open" : "Context: " + s.ctxText;
	}

	/** F8: start when not recording, otherwise pause / resume. */
	public static void toggleFromKey(MinecraftServer server) {
		if (session == null) {
			List<ServerPlayer> players = server.getPlayerList().getPlayers();
			if (players.isEmpty()) return;
			CraftLog.info(start(server, players.get(0), "none"));
		} else if (session.paused) {
			CraftLog.info(resume());
		} else {
			CraftLog.info(pause());
		}
	}

	// ----- server hooks (server thread) -----

	public static void tick(MinecraftServer server) {
		try {
			Session s = session;
			if (s == null) {
				maybeAutoStart(server);
				return;
			}
			long t0 = System.nanoTime();
			boolean keep;
			try {
				keep = s.tick(server);
				s.errors = 0;   // the limit counts consecutive failures, not a whole session's
			} catch (Throwable t) {
				CraftLog.error("recorder tick failed", t);
				s.errors++;
				keep = s.errors <= ERROR_LIMIT;
				if (!keep) s.stopReason = "errors: " + t;
			}
			s.perf.add(System.nanoTime() - t0);
			if (s.perf.windowFull(config.perfLogSeconds * (int) TICKS_PER_SECOND)) CraftLog.info("perf " + s.perf.takeWindow());
			if (!keep) CraftLog.info(stop(s.stopReason != null ? s.stopReason : "stopped"));
		} catch (Throwable t) {
			CraftLog.error("recorder tick crashed", t);
		}
	}

	public static void onJoin(UUID uuid) {
		pendingJoin = uuid;
		joinTries = 0;
	}

	public static void onDisconnect(UUID uuid) {
		Session s = session;
		if (s != null && s.playerId.equals(uuid)) CraftLog.info(stop("disconnect"));
	}

	public static void onBreak(Level level, Player player, BlockPos pos, BlockState state) {
		Session s = session;
		if (s == null || s.paused) return;
		try {
			if (!(player instanceof ServerPlayer sp) || !sp.getUUID().equals(s.playerId)) return;
			List<Session.Drop> drops = new ArrayList<>();
			if (level instanceof ServerLevel sl) {
				for (ItemStack st : Block.getDrops(state, sl, pos, null, player, player.getMainHandItem())) {
					if (!st.isEmpty()) drops.add(new Session.Drop(key(st.getItem()), mmName(st.getItem()), st.getCount()));
				}
			}
			s.pendingBreaks.add(new Session.Brk(key(state.getBlock()), pos.getX(), pos.getY(), pos.getZ(), drops));
			s.snap.markDirty(pos.getX(), pos.getY(), pos.getZ());
		} catch (Throwable t) {
			CraftLog.error("break hook failed", t);
		}
	}

	public static void onAttack(Player player, Entity entity) {
		Session s = session;
		if (s == null || s.paused) return;
		try {
			if (!player.getUUID().equals(s.playerId)) return;
			s.pendingAttacks.add(new Session.Attack(player.getMainHandItem().getItem(), key(entity.getType()), entity.getId()));
		} catch (Throwable t) {
			CraftLog.error("attack hook failed", t);
		}
	}

	public static void onAllowDamage(LivingEntity entity, DamageSource source) {
		Session s = session;
		if (s == null || s.paused) return;
		try {
			if (entity instanceof ServerPlayer sp && sp.getUUID().equals(s.playerId)) s.lastDamageSource = source.getMsgId();
		} catch (Throwable t) {
			CraftLog.error("damage hook failed", t);
		}
	}

	public static void onAfterDeath(LivingEntity entity, DamageSource source) {
		Session s = session;
		if (s == null || s.paused) return;
		try {
			if (entity instanceof ServerPlayer sp && sp.getUUID().equals(s.playerId)) {
				Session.Ev e = s.event("death");
				e.source = source.getMsgId();
				e.entity = source.getEntity() == null ? null : key(source.getEntity().getType());
				s.pending.add(e);
			}
		} catch (Throwable t) {
			CraftLog.error("death hook failed", t);
		}
	}

	public static void onRespawn(ServerPlayer newPlayer) {
		Session s = session;
		if (s == null || s.paused) return;
		try {
			if (newPlayer.getUUID().equals(s.playerId)) s.pending.add(s.event("respawn"));
		} catch (Throwable t) {
			CraftLog.error("respawn hook failed", t);
		}
	}

	public static void onSleep(LivingEntity entity, BlockPos pos, boolean start) {
		Session s = session;
		if (s == null || s.paused) return;
		try {
			if (entity instanceof ServerPlayer sp && sp.getUUID().equals(s.playerId)) {
				Session.Ev e = s.event(start ? "sleep" : "wake");
				e.x = pos.getX(); e.y = pos.getY(); e.z = pos.getZ();
				s.pending.add(e);
			}
		} catch (Throwable t) {
			CraftLog.error("sleep hook failed", t);
		}
	}

	public static void onChat(ServerPlayer sender, String text) {
		Session s = session;
		if (s == null || s.paused) return;
		try {
			if (sender.getUUID().equals(s.playerId)) {
				Session.Ev e = s.event("chat");
				e.text = text;
				s.pending.add(e);
			}
		} catch (Throwable t) {
			CraftLog.error("chat hook failed", t);
		}
	}

	public static void serverStopping() {
		if (session != null) CraftLog.info(stop("server stopping"));
	}

	/** HUD line 1 (null when not recording). */
	public static String hudStatus() {
		Session s = session;
		if (s == null) return null;
		return (s.paused ? "PAUSED " : "REC ") + clock(s.frames) + "  " + s.frames + " frames";
	}

	/** HUD line 2: the open context, or null. */
	public static String hudContext() {
		Session s = session;
		return s == null ? null : s.ctxText;
	}

	private static void maybeAutoStart(MinecraftServer server) {
		if (pendingJoin == null) return;
		if (!config.autoStart || !server.isSingleplayer()) {
			pendingJoin = null;
			return;
		}
		ServerPlayer p = server.getPlayerList().getPlayer(pendingJoin);
		if (p == null) {
			if (++joinTries > JOIN_TRIES) pendingJoin = null;
			return;
		}
		pendingJoin = null;
		CraftLog.info(start(server, p, "none"));
	}

	static String clock(long frames) {
		long s = (long) (frames / TICKS_PER_SECOND);
		return String.format("%02d:%02d:%02d", s / 3600, s / 60 % 60, s % 60);
	}

	// ----- helpers -----

	static String key(Item item) { return BuiltInRegistries.ITEM.getKey(item).toString(); }

	static String key(Block block) { return BuiltInRegistries.BLOCK.getKey(block).toString(); }

	static String key(EntityType<?> type) { return BuiltInRegistries.ENTITY_TYPE.getKey(type).toString(); }

	/** MineMind item index of a stack (Obs.ITEMS predicates), or -1. */
	static int itemIndex(ItemStack s) {
		if (s.isEmpty()) return -1;
		for (int i = 0; i < Obs.ITEMS.length; i++) if (Obs.ITEMS[i].test(s)) return i;
		return -1;
	}

	static String mmName(Item item) {
		int i = itemIndex(new ItemStack(item));
		return i < 0 ? null : Obs.ITEM_NAMES[i];
	}

	static int indexOf(String[] arr, String v) {
		for (int i = 0; i < arr.length; i++) if (arr[i].equals(v)) return i;
		return -1;
	}

	static int actionIndex(String name) { return indexOf(ACTIONS, name); }

	static String placeAction(String block) {
		return switch (block == null ? "" : block) {
			case "minecraft:crafting_table" -> "place_table";
			case "minecraft:furnace" -> "place_furnace";
			case "minecraft:chest" -> "place_chest";
			default -> "place_block";
		};
	}

	static String craftAction(String name) {
		return switch (name) {
			case "planks" -> "craft_planks";
			case "stick" -> "craft_sticks";
			case "crafting_table" -> "craft_table";
			case "wooden_pickaxe" -> "craft_wooden_pickaxe";
			case "stone_pickaxe" -> "craft_stone_pickaxe";
			case "furnace" -> "craft_furnace";
			case "iron_pickaxe" -> "craft_iron_pickaxe";
			case "chest" -> "craft_chest";
			default -> null;
		};
	}

	static String menuName(AbstractContainerMenu m) {
		if (m == null) return null;
		if (m instanceof InventoryMenu) return "inventory";
		if (m instanceof CraftingMenu) return "crafting_table";
		if (m instanceof AbstractFurnaceMenu) return snake(m.getClass().getSimpleName().replace("Menu", ""));
		if (m instanceof ChestMenu) return "chest";
		return snake(m.getClass().getSimpleName());
	}

	private static String snake(String camel) {
		return camel.replaceAll("([a-z0-9])([A-Z])", "$1_$2").toLowerCase();
	}

	private static final Map<Item, List<String>> TAGS = new HashMap<>();

	/** Item tags (minecraft: namespace): the classification input for pretrain/timeline. */
	static List<String> tagsOf(Item item) {
		return TAGS.computeIfAbsent(item, it -> {
			List<String> out = new ArrayList<>();
			new ItemStack(it).typeHolder().tags().forEach(tk -> {
				String l = tk.location().toString();
				if (l.startsWith("minecraft:")) out.add(l);
			});
			return out;
		});
	}

	static boolean isFood(Item item) {
		return new ItemStack(item).get(DataComponents.FOOD) != null;
	}

	static String heldId(ServerPlayer p) {
		ItemStack h = p.getMainHandItem();
		return h.isEmpty() ? "minecraft:air" : key(h.getItem());
	}

	static Map<Item, Integer> invCounts(ServerPlayer p) {
		Map<Item, Integer> m = new LinkedHashMap<>();
		var inv = p.getInventory();
		for (int i = 0; i < inv.getContainerSize(); i++) {
			ItemStack st = inv.getItem(i);
			if (!st.isEmpty()) m.merge(st.getItem(), st.getCount(), Integer::sum);
		}
		return m;
	}

	/** Camera bin 0..10, 5 = none: mu-law (mu = 10) of the per-tick angle, saturating at CAMERA_MAX_DEG. */
	static int cameraBin(float deg) {
		double n = Math.max(-1, Math.min(1, deg / CAMERA_MAX_DEG));
		double v = Math.signum(n) * Math.log(1 + 10 * Math.abs(n)) / Math.log(11);
		return Math.max(0, Math.min(10, 5 + (int) Math.round(v * 5)));
	}

	// ----- one session -----

	static final class Session {
		final Path dir;
		final UUID playerId;
		final String goalArg;
		final long startMillis = System.currentTimeMillis();
		final String startStamp;
		final String worldSeed;
		final String playerName;
		final SessionWriter writer;
		final FrameRenderer renderer = new FrameRenderer();
		final Snapshot snap = new Snapshot();
		final StatTracker stats = new StatTracker();
		final Perf perf = new Perf();
		final ThreadPoolExecutor exec;
		final Queue<byte[]> pool = new ConcurrentLinkedQueue<>();
		final AtomicLong workerNs = new AtomicLong();
		final AtomicLong workerFrames = new AtomicLong();
		final AtomicLong writeErrors = new AtomicLong();
		final byte[] img = new byte[ChunkWriter.IMG_BYTES];
		final float[] vec = new float[ChunkWriter.VEC_DIM];
		final List<Ev> pending = new ArrayList<>();
		final List<Brk> pendingBreaks = new ArrayList<>();
		final List<Attack> pendingAttacks = new ArrayList<>();
		final List<CtxRec> contexts = new ArrayList<>();
		final Set<String> advDone = new HashSet<>();
		String lastDamageSource;
		String stopReason;
		long frames = 0;
		int errors = 0;
		boolean paused = false;
		ServerPlayer lastPlayer;
		Map<Item, Integer> prevInv = new LinkedHashMap<>();
		int prevXp;
		int prevFood;
		int prevSelected = -1;
		ResourceKey<Level> prevDim;
		InputSamples.Sample lastSample;
		String ctxText;
		int ctxId;
		int nextCtxId = 1;

		record Drop(String item, String name, int count) {}
		record Brk(String block, int x, int y, int z, List<Drop> drops) {}
		record Attack(Item weapon, String entity, int entityNum) {}

		static final class CtxRec {
			final int id; final String text; final long start; long end = -1; String reason;
			CtxRec(int id, String text, long start) { this.id = id; this.text = text; this.start = start; }
		}

		/** One event before serialization; field names are the canonical PROTOCOL 9.7 names. */
		static final class Ev {
			final String type;
			String item, name, block, entity, weapon, damageSource, via, held, gui, text, reason, id;
			String source;
			Integer count;
			Double damage;
			Integer ctxId;
			List<Drop> drops;
			List<String> tags;
			boolean food;
			double x, y, z;
			Ev(String type) { this.type = type; }
		}

		Session(MinecraftServer server, ServerPlayer p, String goalArg) throws IOException {
			this.playerId = p.getUUID();
			this.playerName = p.getName().getString();
			this.goalArg = goalArg;
			String seed;
			try {
				seed = String.valueOf(((ServerLevel) p.level()).getSeed());
			} catch (Throwable t) {
				seed = null;
			}
			this.worldSeed = seed;
			this.startStamp = new SimpleDateFormat("yyyyMMdd-HHmmss").format(new Date(startMillis));
			Path root = FabricLoader.getInstance().getGameDir().resolve("minemind_recordings");
			this.dir = root.resolve(startStamp + "-session");
			this.writer = new SessionWriter(dir);
			this.exec = new ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<>(128),
				r -> { Thread t = new Thread(r, "minemind-rec-writer"); t.setDaemon(true); return t; },
				(r, ex) -> {
					try {
						ex.getQueue().put(r);
					} catch (InterruptedException e) {
						Thread.currentThread().interrupt();
					}
				});
			this.lastPlayer = p;
			baseline(p);
			ServerLevel level = (ServerLevel) p.level();
			advancementPoll(server, p, false);
			snap.update(level, p.getX(), p.getY(), p.getZ());
			snap.refresh(level);
			writer.writeMeta(meta(false, null));
		}

		/** Sets every diff baseline to the current state (no events). */
		void baseline(ServerPlayer p) {
			stats.rebaseline(p);
			prevInv = invCounts(p);
			prevXp = p.experienceLevel;
			prevFood = p.getFoodData().getFoodLevel();
			prevDim = ((ServerLevel) p.level()).dimension();
			prevSelected = p.getInventory().getSelectedSlot();
		}

		Ev event(String type) {
			Ev e = new Ev(type);
			ServerPlayer p = lastPlayer;
			if (p != null) {
				e.x = Math.floor(p.getX()); e.y = Math.floor(p.getY()); e.z = Math.floor(p.getZ());
				e.held = heldId(p);
				e.gui = menuName(p.containerMenu);
			}
			return e;
		}

		boolean tick(MinecraftServer server) {
			ServerPlayer p = server.getPlayerList().getPlayer(playerId);
			if (p == null || p.isRemoved()) {
				stopReason = "player left the world";
				return false;
			}
			if (paused) {
				baseline(p);
				lastPlayer = p;
				InputSamples.take();
				pending.clear();
				pendingBreaks.clear();
				pendingAttacks.clear();
				lastDamageSource = null;
				return true;
			}
			if (p != lastPlayer) {
				baseline(p);
				lastPlayer = p;
			}
			ServerLevel level = (ServerLevel) p.level();
			List<Ev> evs = new ArrayList<>(pending);
			pending.clear();

			ResourceKey<Level> dim = level.dimension();
			if (!dim.equals(prevDim)) {
				Ev e = event("dimension");
				e.id = dim.identifier().toString();
				evs.add(e);
				prevDim = dim;
			}

			List<Attack> attacks = new ArrayList<>(pendingAttacks);
			pendingAttacks.clear();
			List<StatTracker.Delta> deltas = stats.diff(p);
			Map<Item, Integer> inv = invCounts(p);
			Map<Item, Integer> crafted = new HashMap<>(), picked = new HashMap<>(), dropped = new HashMap<>(), used = new HashMap<>();
			int mined = 0;
			double dealt = 0;
			Item heldItem = p.getMainHandItem().getItem();
			Map<Item, Integer> attackWeapons = new HashMap<>();
			for (Attack a : attacks) attackWeapons.merge(a.weapon(), 1, Integer::sum);

			List<Ev> statEvs = new ArrayList<>();
			for (StatTracker.Delta d : deltas) {
				Stat<?> s = d.stat();
				StatType<?> t = s.getType();
				Object v = s.getValue();
				int n = d.delta();
				if (t == Stats.BLOCK_MINED && v instanceof Block b) {
					mined += n;
					String bk = key(b);
					Ev e = event("mine");
					e.block = bk;
					e.count = n;
					e.name = BLOCK_NAMES[Obs.blockId(b.defaultBlockState())];
					e.drops = new ArrayList<>();
					for (Brk br : pendingBreaks) {
						if (br.block().equals(bk)) {
							e.x = br.x(); e.y = br.y(); e.z = br.z();
							e.drops = br.drops();
							break;
						}
					}
					statEvs.add(e);
				} else if (t == Stats.ITEM_CRAFTED && v instanceof Item it) {
					crafted.merge(it, n, Integer::sum);
					boolean smelt = p.containerMenu instanceof AbstractFurnaceMenu;
					Ev e = event(smelt ? "smelt" : "craft");
					fillItem(e, it, n);
					e.via = menuName(p.containerMenu);
					statEvs.add(e);
				} else if (t == Stats.ITEM_PICKED_UP && v instanceof Item it) {
					picked.merge(it, n, Integer::sum);
					Ev e = event("pickup");
					fillItem(e, it, n);
					statEvs.add(e);
				} else if (t == Stats.ITEM_DROPPED && v instanceof Item it) {
					dropped.merge(it, n, Integer::sum);
					Ev e = event("drop");
					fillItem(e, it, n);
					statEvs.add(e);
				} else if (t == Stats.ITEM_BROKEN && v instanceof Item it) {
					Ev e = event("break_item");
					fillItem(e, it, n);
					statEvs.add(e);
				} else if (t == Stats.ITEM_USED && v instanceof Item it) {
					used.merge(it, n, Integer::sum);
				} else if (t == Stats.ENTITY_KILLED && v instanceof EntityType<?> et) {
					String ek = key(et);
					Attack hit = null;
					for (Attack a : attacks) if (a.entity().equals(ek)) { hit = a; break; }
					Ev e = event("kill");
					e.entity = ek;
					e.count = n;
					e.weapon = hit != null ? key(hit.weapon()) : heldId(p);
					statEvs.add(e);
				} else if (t == Stats.ENTITY_KILLED_BY && v instanceof EntityType<?> et) {
					Ev e = event("killed_by");
					e.entity = key(et);
					e.count = n;
					statEvs.add(e);
				} else if (t == Stats.CUSTOM && v instanceof Identifier id) {
					if (id.getPath().equals("damage_dealt")) dealt += n / 10.0;
					customStat(statEvs, id, n);
				}
			}

			// Attack events (damage dealt this tick goes to the first attack)
			for (Attack a : attacks) {
				Ev e = event("attack");
				e.entity = a.entity();
				e.weapon = key(a.weapon());
				e.count = 1;
				if (dealt > 0) e.damage = dealt;
				dealt = 0;
				evs.add(e);
			}
			if (dealt > 0) {
				Ev e = event("damage_dealt");
				e.damage = dealt;
				evs.add(e);
			}
			evs.addAll(statEvs);

			// Item use: explained by mining (held tool) and attacks (weapon); the rest is eat / place / use.
			for (Map.Entry<Item, Integer> en : used.entrySet()) {
				Item it = en.getKey();
				int raw = en.getValue();
				int explained = (it == heldItem ? mined : 0) + attackWeapons.getOrDefault(it, 0);
				int r = raw - Math.min(raw, explained);
				if (r <= 0) continue;
				if (isFood(it)) {
					Ev e = event("eat");
					fillItem(e, it, r);
					e.food = true;
					evs.add(e);
				} else if (it instanceof BlockItem bi) {
					Ev e = event("place");
					fillItem(e, it, r);
					e.block = key(bi.getBlock());
					Obs.Hit th = Obs.target(level, p.getX(), p.getY(), p.getZ(), p.getYRot(), p.getXRot());
					if (th.hit) { e.x = th.px; e.y = th.py; e.z = th.pz; }
					evs.add(e);
				} else {
					Ev e = event("use");
					fillItem(e, it, r);
					evs.add(e);
				}
			}

			// Chest / furnace / other container transfers: inventory diff minus stat-explained changes.
			AbstractContainerMenu menu = p.containerMenu;
			if (!(menu instanceof InventoryMenu) && !(menu instanceof CraftingMenu)) {
				Set<Item> keys = new HashSet<>(prevInv.keySet());
				keys.addAll(inv.keySet());
				for (Item it : keys) {
					int delta = inv.getOrDefault(it, 0) - prevInv.getOrDefault(it, 0);
					int transfer = delta - picked.getOrDefault(it, 0) - crafted.getOrDefault(it, 0)
						+ dropped.getOrDefault(it, 0) + used.getOrDefault(it, 0);
					if (transfer == 0) continue;
					boolean take = transfer > 0;
					Ev e;
					if (menu instanceof ChestMenu) e = event(take ? "chest_take" : "chest_put");
					else {
						e = event(take ? "container_take" : "container_put");
						e.via = menuName(menu);
					}
					fillItem(e, it, Math.abs(transfer));
					evs.add(e);
				}
			}
			prevInv = inv;

			if (p.experienceLevel != prevXp) {
				Ev e = event("level_up");
				e.count = p.experienceLevel;
				prevXp = p.experienceLevel;
				evs.add(e);
			}
			if (frames % 20 == 0) advancementPoll(server, p, true);
			evs.addAll(pending);
			pending.clear();

			// Input sample for this tick; a missing sample repeats the previous one with repeat=true.
			InputSamples.Sample in = InputSamples.take();
			boolean repeat = false;
			if (in == null) {
				repeat = true;
				in = lastSample != null ? repeatOf(lastSample) : new InputSamples.Sample();
			} else {
				lastSample = in;
			}
			int sel = p.getInventory().getSelectedSlot();
			int ext = encodeExt(in, sel, repeat);
			prevSelected = sel;
			prevFood = p.getFoodData().getFoodLevel();
			lastDamageSource = null;
			pendingBreaks.clear();

			snap.update(level, p.getX(), p.getY(), p.getZ());
			snap.refresh(level);
			if (in.use || in.attack) snap.markLook(p.getX(), p.getY(), p.getZ(), p.getYRot(), p.getXRot());
			Pose pose = new Pose();
			pose.x = p.getX(); pose.y = p.getY(); pose.z = p.getZ();
			pose.yaw = p.getYRot(); pose.pitch = p.getXRot();
			pose.onGround = p.onGround(); pose.health = p.getHealth();
			pose.items = new int[15];
			for (int i = 0; i < 15; i++) pose.items[i] = Obs.count(p.getInventory(), Obs.ITEMS[i]);
			pose.fuel = pose.items[8];
			pose.step = (int) ((frames / STEP_TICKS) % Obs.MAX_STEPS);
			pose.bodies = bodies(server, p, level);
			Obs.Hit th = Obs.target(level, p.getX(), p.getY(), p.getZ(), p.getYRot(), p.getXRot());
			if (th.hit && th.id == Obs.CHEST) {
				Container c = Obs.chestAt(level, new BlockPos(th.x, th.y, th.z));
				if (c != null) {
					pose.xhair = new float[9];
					for (int i = 0; i < 9; i++)
						pose.xhair[i] = (float) (Math.log(1.0 + Obs.count(c, Obs.ITEMS[i])) / Math.log(2.0) / 6.0);
				}
			}

			int action = label(evs, in, nearOther(server, p, 4.0));
			long gameTick = level.getGameTime();
			String inputLine = inputLine(p, in, repeat, ext, sel, gameTick);
			List<String> lines = new ArrayList<>(evs.size());
			double t = frames / TICKS_PER_SECOND;
			for (Ev e : evs) lines.add(toJson(e, frames, t, gameTick));

			byte[] reg = pool.poll();
			if (reg == null) reg = new byte[Snapshot.SX * Snapshot.SY * Snapshot.SZ];
			snap.copy(reg);
			final byte[] fr = reg;
			final int ox = snap.ox(), oy = snap.oy(), oz = snap.oz(), minY = snap.minY(), maxY = snap.maxY();
			final long tick = frames;
			final int act = action, extF = ext;
			exec.execute(() -> render(fr, ox, oy, oz, minY, maxY, pose, tick, act, extF, inputLine, lines));

			frames++;
			frameCount = (int) frames;
			if (frames % META_FRAMES == 0) {
				String m = meta(false, null);
				exec.execute(() -> writeMetaSafe(m));
			}
			return true;
		}

		private void fillItem(Ev e, Item it, int n) {
			e.item = key(it);
			e.name = mmName(it);
			e.count = n;
			e.tags = tagsOf(it);
		}

		/** Writer thread: render the observation and write the frame. */
		private void render(byte[] fr, int ox, int oy, int oz, int minY, int maxY, Pose pose, long tick, int action,
		                    int ext, String inputLine, List<String> lines) {
			long t0 = System.nanoTime();
			try {
				renderer.render(new Region(fr, ox, oy, oz, Snapshot.SX, Snapshot.SY, Snapshot.SZ, minY, maxY), pose, img, vec);
				writer.frame(tick, img, vec, action, ext, inputLine, lines);
				if ((tick + 1) % FLUSH_FRAMES == 0) writer.flush();
			} catch (Throwable t) {
				writeErrors.incrementAndGet();
				CraftLog.error("frame write failed at tick " + tick, t);
			} finally {
				if (pool.size() < 64) pool.offer(fr);
				workerNs.addAndGet(System.nanoTime() - t0);
				workerFrames.incrementAndGet();
			}
		}

		private void writeMetaSafe(String json) {
			try {
				writer.writeMeta(json);
			} catch (Throwable t) {
				CraftLog.error("meta write failed", t);
			}
		}

		private void customStat(List<Ev> out, Identifier id, int n) {
			String path = id.getPath();
			if (path.equals("damage_taken")) {
				Ev e = event("damage_taken");
				e.damage = n / 10.0;
				e.source = lastDamageSource;
				out.add(e);
			} else if (path.equals("traded_with_villager")) {
				Ev e = event("trade");
				e.count = n;
				out.add(e);
			} else if (path.equals("enchant_item")) {
				Ev e = event("enchant");
				e.count = n;
				out.add(e);
			} else if (!skipCustom(path)) {
				Ev e = event("stat");
				e.id = id.toString();
				e.count = n;
				out.add(e);
			}
		}

		private static boolean skipCustom(String path) {
			return path.endsWith("_one_cm") || path.endsWith("_time") || path.startsWith("time_since")
				|| path.equals("jump") || path.equals("drop") || path.equals("leave_game")
				|| path.equals("sleep_in_bed") || path.equals("deaths") || path.equals("damage_taken")
				|| path.equals("damage_dealt");
		}

		private List<Obs.Mate> bodies(MinecraftServer server, ServerPlayer p, ServerLevel level) {
			List<Obs.Mate> out = new ArrayList<>();
			for (ServerPlayer o : server.getPlayerList().getPlayers()) {
				if (o != p && o.level() == level && o.distanceToSqr(p) <= 40.0 * 40.0)
					out.add(new Obs.Mate(o.getX(), o.getY(), o.getZ(), 0));
			}
			return out;
		}

		private boolean nearOther(MinecraftServer server, ServerPlayer p, double r) {
			for (ServerPlayer o : server.getPlayerList().getPlayers())
				if (o != p && o.distanceToSqr(p) <= r * r) return true;
			return false;
		}

		private void advancementPoll(MinecraftServer server, ServerPlayer p, boolean emit) {
			var pa = p.getAdvancements();
			for (var h : server.getAdvancements().getAllAdvancements()) {
				String id = h.id().toString();
				if (advDone.contains(id)) continue;
				if (pa.getOrStartProgress(h).isDone()) {
					advDone.add(id);
					if (emit) {
						Ev e = event("advancement");
						e.id = id;
						pending.add(e);
					}
				}
			}
		}

		private InputSamples.Sample repeatOf(InputSamples.Sample s) {
			InputSamples.Sample r = new InputSamples.Sample();
			r.gui = s.gui;
			return r;
		}

		private int encodeExt(InputSamples.Sample in, int sel, boolean repeat) {
			int e = 0;
			if (in.forward) e |= 1;
			if (in.back) e |= 1 << 1;
			if (in.left) e |= 1 << 2;
			if (in.right) e |= 1 << 3;
			if (in.jump) e |= 1 << 4;
			if (in.sneak) e |= 1 << 5;
			if (in.sprint) e |= 1 << 6;
			if (in.attack) e |= 1 << 7;
			if (in.use) e |= 1 << 8;
			if (in.drop) e |= 1 << 9;
			if (in.inventory) e |= 1 << 10;
			if (in.swapOffhand) e |= 1 << 11;
			if (in.pick) e |= 1 << 12;
			int hot = 0;
			if (!repeat) for (int i = 0; i < 9 && hot == 0; i++) if (in.hotbar[i]) hot = i + 1;
			if (hot == 0 && prevSelected >= 0 && sel != prevSelected) hot = sel + 1;
			e |= hot << 13;
			if (repeat) e |= (5 << 17) | (5 << 22);
			else e |= (cameraBin(in.dyaw) << 17) | (cameraBin(in.dpitch) << 22);
			return e;
		}

		/** Turn / movement / craft / chest labels (PROTOCOL 8.3 priority, extended by 9.4). Never -1. */
		private int label(List<Ev> evs, InputSamples.Sample in, boolean nearOther) {
			for (Ev e : evs) if ("craft".equals(e.type) && e.name != null) {
				String a = craftAction(e.name);
				if (a != null) return actionIndex(a);
			}
			for (Ev e : evs) if ("smelt".equals(e.type) && "iron_ingot".equals(e.name)) return actionIndex("smelt_iron");
			for (Ev e : evs) if ("chest_put".equals(e.type)) return actionIndex("deposit");
			for (Ev e : evs) if ("chest_take".equals(e.type)) return actionIndex("withdraw");
			for (Ev e : evs) if ("place".equals(e.type)) return actionIndex(placeAction(e.block));
			boolean attack = in.attack && in.attackBlock;
			for (Ev e : evs) if ("attack".equals(e.type)) attack = true;
			if (attack) return actionIndex("attack");
			if (Math.abs(in.dyaw) >= TURN_TICK) return actionIndex(in.dyaw > 0 ? "turn_right" : "turn_left");
			if (Math.abs(in.dpitch) >= TURN_TICK) return actionIndex(in.dpitch < 0 ? "look_up" : "look_down");
			if (nearOther) for (Ev e : evs) if ("drop".equals(e.type)) return actionIndex("give");
			if (in.forward && in.jump) return actionIndex("forward_jump");
			if (in.jump) return actionIndex("jump");
			if (in.forward) return actionIndex("forward");
			if (in.back) return actionIndex("back");
			if (in.left) return actionIndex("left");
			if (in.right) return actionIndex("right");
			return actionIndex("noop");
		}

		/** One input.jsonl line, always a complete JSON object (Gson). */
		private String inputLine(ServerPlayer p, InputSamples.Sample in, boolean repeat, int ext, int sel, long gameTick) {
			JsonObject o = new JsonObject();
			o.addProperty("tick", frames);
			o.addProperty("game_tick", gameTick);
			o.addProperty("repeat", repeat);
			o.addProperty("ext_action", ext);
			o.addProperty("context", ctxText);
			JsonObject keys = new JsonObject();
			keys.addProperty("forward", in.forward);
			keys.addProperty("back", in.back);
			keys.addProperty("left", in.left);
			keys.addProperty("right", in.right);
			keys.addProperty("jump", in.jump);
			keys.addProperty("sneak", in.sneak);
			keys.addProperty("sprint", in.sprint);
			keys.addProperty("attack", in.attack);
			keys.addProperty("use", in.use);
			keys.addProperty("drop", in.drop);
			keys.addProperty("inventory", in.inventory);
			keys.addProperty("swap_offhand", in.swapOffhand);
			keys.addProperty("pick", in.pick);
			JsonObject hot = new JsonObject();
			for (int i = 0; i < 9; i++) hot.addProperty(String.valueOf(i + 1), !repeat && in.hotbar[i]);
			keys.add("hotbar", hot);
			o.add("keys", keys);
			o.addProperty("dyaw", repeat ? 0f : in.dyaw);
			o.addProperty("dpitch", repeat ? 0f : in.dpitch);
			o.addProperty("yaw", p.getYRot());
			o.addProperty("pitch", p.getXRot());
			o.addProperty("x", p.getX());
			o.addProperty("y", p.getY());
			o.addProperty("z", p.getZ());
			o.addProperty("on_ground", p.onGround());
			o.addProperty("health", p.getHealth());
			o.addProperty("selected_slot", sel);
			ItemStack held = p.getMainHandItem();
			o.addProperty("held_item", held.isEmpty() ? null : key(held.getItem()));
			o.addProperty("gui", in.gui);
			Map<String, Integer> named = new LinkedHashMap<>();
			var invc = p.getInventory();
			for (int i = 0; i < invc.getContainerSize(); i++) {
				ItemStack st = invc.getItem(i);
				if (st.isEmpty()) continue;
				int mm = itemIndex(st);
				named.merge(mm >= 0 ? Obs.ITEM_NAMES[mm] : "other", st.getCount(), Integer::sum);
			}
			JsonObject inv = new JsonObject();
			for (Map.Entry<String, Integer> e : named.entrySet()) inv.addProperty(e.getKey(), e.getValue());
			o.add("inventory", inv);
			JsonObject all = new JsonObject();
			for (Map.Entry<Item, Integer> e : invCounts(p).entrySet()) all.addProperty(key(e.getKey()), e.getValue());
			o.add("inventory_all", all);
			return Json.GSON.toJson(o) + "\n";
		}

		/** One events.jsonl line (canonical 9.7 field names). */
		private String toJson(Ev e, long tick, double t, long gameTick) {
			JsonObject o = new JsonObject();
			o.addProperty("tick", tick);
			o.addProperty("t", Json.round3(t));
			o.addProperty("type", e.type);
			o.addProperty("item", e.item);
			o.addProperty("name", e.name);
			if (e.count != null) o.addProperty("count", e.count);
			if (e.block != null) o.addProperty("block", e.block);
			if (e.drops != null) {
				JsonArray arr = new JsonArray();
				for (Drop d : e.drops) {
					JsonObject dj = new JsonObject();
					dj.addProperty("item", d.item());
					dj.addProperty("name", d.name());
					dj.addProperty("count", d.count());
					arr.add(dj);
				}
				o.add("drops", arr);
			}
			if (e.entity != null) o.addProperty("entity", e.entity);
			if (e.weapon != null) o.addProperty("weapon", e.weapon);
			if (e.damage != null) o.addProperty("damage", e.damage);
			if (e.source != null) o.addProperty("source", e.source);
			if (e.via != null) o.addProperty("via", e.via);
			o.addProperty("x", (long) e.x);
			o.addProperty("y", (long) e.y);
			o.addProperty("z", (long) e.z);
			o.addProperty("held", e.held);
			o.addProperty("gui", e.gui);
			if (e.text != null) o.addProperty("text", e.text);
			if (e.id != null) o.addProperty("id", e.id);
			if (e.ctxId != null) o.addProperty("id", e.ctxId);
			if (e.reason != null) o.addProperty("reason", e.reason);
			if (e.food) o.addProperty("food", true);
			if (e.tags != null) {
				JsonArray arr = new JsonArray();
				for (String s : e.tags) arr.add(s);
				o.add("item_tags", arr);
			}
			o.addProperty("game_tick", gameTick);
			return Json.GSON.toJson(o) + "\n";
		}

		void startContext(String text) {
			ctxId = nextCtxId++;
			ctxText = text;
			contexts.add(new CtxRec(ctxId, text, frames));
			Ev e = event("context_start");
			e.text = text;
			e.ctxId = ctxId;
			pending.add(e);
		}

		void endContext(String reason) {
			if (ctxText == null) return;
			for (CtxRec c : contexts) if (c.id == ctxId) { c.end = frames; c.reason = reason; }
			Ev e = event("context_end");
			e.ctxId = ctxId;
			e.reason = reason;
			pending.add(e);
			ctxText = null;
		}

		void pauseNow() {
			endContext("stop");
			paused = true;
			flushPendingNow();
		}

		/** Writes pending events without a frame (pause). */
		private void flushPendingNow() {
			List<String> lines = new ArrayList<>();
			double t = frames / TICKS_PER_SECOND;
			long gt = lastPlayer == null ? 0 : lastPlayer.level().getGameTime();
			for (Ev e : pending) lines.add(toJson(e, frames, t, gt));
			pending.clear();
			if (lines.isEmpty()) return;
			exec.execute(() -> {
				try {
					writer.eventLines(lines);
				} catch (Throwable ex) {
					CraftLog.error("event write failed", ex);
				}
			});
		}

		void close(String reason) throws IOException {
			endContext("stop");
			List<String> tail = new ArrayList<>();
			double t = frames / TICKS_PER_SECOND;
			long gt = lastPlayer == null ? 0 : lastPlayer.level().getGameTime();
			for (Ev e : pending) tail.add(toJson(e, frames, t, gt));
			pending.clear();
			Future<?> f = exec.submit(() -> {
				try {
					writer.eventLines(tail);
					writer.closeData();
				} catch (IOException e) {
					throw new IllegalStateException(e);
				}
			});
			try {
				f.get(60, TimeUnit.SECONDS);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			} catch (ExecutionException | java.util.concurrent.TimeoutException e) {
				CraftLog.error("writer did not finish cleanly", e);
			}
			exec.shutdown();
			writer.writeMeta(meta(true, reason));
			CraftLog.info(String.format("session perf: %s, worker %.3f ms/frame", perf.takeWindow(),
				workerFrames.get() == 0 ? 0 : workerNs.get() / 1e6 / workerFrames.get()));
		}

		String meta(boolean finished, String reason) {
			JsonObject o = new JsonObject();
			o.addProperty("format", FORMAT);
			o.addProperty("mc_version", net.minecraft.SharedConstants.getCurrentVersion().name());
			o.addProperty("player", playerName);
			o.addProperty("player_uuid", playerId.toString());
			o.addProperty("world_seed", worldSeed == null ? null : worldSeed);
			o.addProperty("goal", goalArg);
			o.addProperty("start_time", startStamp);
			o.addProperty("ticks", frames);
			o.addProperty("frames", frames);
			o.addProperty("chunks", writer.chunks.chunksWritten());
			o.addProperty("chunk_frames", ChunkWriter.CHUNK_FRAMES);
			o.addProperty("frame_bytes", ChunkWriter.FRAME_BYTES);
			o.addProperty("res", ChunkWriter.RES);
			o.addProperty("vec_dim", ChunkWriter.VEC_DIM);
			o.add("vec_names", strArr(vecNames()));
			o.add("actions", strArr(ACTIONS));
			o.add("items", strArr(Obs.ITEM_NAMES));
			o.add("blocks", strArr(BLOCK_NAMES));
			o.addProperty("frame_every_ticks", 1);
			o.addProperty("sim_steps_ticks_per_step", STEP_TICKS);
			o.addProperty("ext_action_bits", "0-12 forward,back,left,right,jump,sneak,sprint,attack,use,drop,inventory,"
				+ "swap_offhand,pick; 13-16 hotbar (slot+1, 0 none); 17-21 yaw bin; 22-26 pitch bin; bins 0-10 mu-law, 5 none");
			JsonArray ctxs = new JsonArray();
			for (CtxRec c : contexts) {
				JsonObject cj = new JsonObject();
				cj.addProperty("id", c.id);
				cj.addProperty("text", c.text);
				cj.addProperty("start_tick", c.start);
				cj.addProperty("end_tick", c.end < 0 ? null : c.end);
				cj.addProperty("end_reason", c.reason);
				ctxs.add(cj);
			}
			o.add("contexts", ctxs);
			o.addProperty("finished", finished);
			o.addProperty("stop_reason", reason);
			o.addProperty("elapsed_s", (System.currentTimeMillis() - startMillis) / 1000.0);
			o.addProperty("recorded_s", frames / TICKS_PER_SECOND);
			o.addProperty("server_tick_avg_ms", perf.sessionAvgMs());
			o.addProperty("worker_ms_per_frame", workerFrames.get() == 0 ? 0 : workerNs.get() / 1e6 / workerFrames.get());
			o.addProperty("snapshot_fills", snap.fills);
			o.addProperty("snapshot_refreshes", snap.refreshes);
			o.addProperty("write_errors", writeErrors.get());
			return Json.GSON.toJson(o) + "\n";
		}

		private static JsonArray strArr(String[] xs) {
			JsonArray a = new JsonArray();
			for (String s : xs) a.add(s);
			return a;
		}

		static String[] vecNames() {
			List<String> n = new ArrayList<>();
			for (String it : Obs.ITEM_NAMES) n.add("inv_" + it);
			for (String s : new String[] {"health", "on_ground", "pitch", "sin_yaw", "cos_yaw", "mining_progress", "height",
				"near_table", "near_furnace", "fuel", "time_frac"}) n.add(s);
			for (String it : Obs.ITEM_NAMES) n.add("goal_" + it);
			for (int j = 0; j < 4; j++) {
				n.add("tm" + j + "_present"); n.add("tm" + j + "_dx"); n.add("tm" + j + "_dy"); n.add("tm" + j + "_dz");
				for (int m = 0; m < Obs.NUM_MSGS; m++) n.add("tm" + j + "_msg" + m);
			}
			n.add("chest_present"); n.add("chest_dx"); n.add("chest_dy"); n.add("chest_dz");
			for (int i = 0; i < 9; i++) n.add("xhair_chest_" + Obs.ITEM_NAMES[i]);
			return n.toArray(new String[0]);
		}
	}
}
