// Copied verbatim from mcmod/src/main/java/com/minemind/craftbot/Obs.java (bot mod; marker mcmod/V6_OBS_READY).
// Only the package line was changed (com.minemind.craftbot -> com.minemind.recorder).
package com.minemind.recorder;

import net.minecraft.core.BlockPos;
import net.minecraft.tags.BlockTags;
import net.minecraft.tags.FluidTags;
import net.minecraft.tags.ItemTags;
import net.minecraft.world.Container;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;

import java.util.List;
import java.util.function.Predicate;

/**
 * Self-contained MineMind observation builder (copy this single file into any mod).
 *
 * Mirrors src/sim.cpp writeObs() for the v6 layout (res 64, VEC 102, 20 blocks, 15 items) and for the
 * legacy v5 layout (res 32, VEC 39, 18 blocks, 14 items). Only Minecraft and JDK imports; no dependency on
 * any other mod class. Everything is static: callers pass the level, the body state and the teammates.
 *
 * Coordinates: x/z are block units, y is the feet height, yaw/pitch in degrees (MC convention, same as the sim).
 * Height scalar: (y - 63) / 32 (MC sea level 63 = sim SEA_LEVEL 30), as in v2..v5.
 */
public final class Obs {
	private Obs() {}

	public static final int MAX_STEPS = 3000;
	public static final double VIEW_DIST = 24.0, FOV_DEG = 70.0, REACH = 4.5, EYE = 1.62;
	public static final double HALF_W = 0.3, HEIGHT = 1.8, TEAM_RANGE = 32.0, CHEST_RANGE = 16.0;
	public static final int MAX_TEAM_SLOTS = 4, NUM_MSGS = 8;

	// Sim block ids (src/sim.h Block enum).
	public static final int AIR = 0, GRASS = 1, DIRT = 2, STONE = 3, COBBLE = 4, LOG = 5, LEAVES = 6, PLANKS = 7,
		TABLE = 8, FURNACE = 9, COAL_ORE = 10, IRON_ORE = 11, DIAMOND_ORE = 12, BEDROCK = 13,
		SAND = 14, GRAVEL = 15, WATER = 16, DEEPSLATE = 17, CHEST = 18, AGENT = 19;

	/** Sim item ids (src/sim.h Item enum). */
	public static final String[] ITEM_NAMES = {"log", "planks", "stick", "crafting_table", "wooden_pickaxe",
		"cobblestone", "stone_pickaxe", "furnace", "coal", "iron_ore", "iron_ingot", "iron_pickaxe", "diamond", "dirt",
		"chest"};

	/** Observation layout. v6 = current policy; v5 = legacy checkpoints (chosen at runtime from VEC_DIM). */
	public static final class Layout {
		public final boolean v6;
		public final int res, vecDim, numBlocks, numItems, goalOffset;

		private Layout(boolean v6, int res, int vecDim, int numBlocks, int numItems, int goalOffset) {
			this.v6 = v6; this.res = res; this.vecDim = vecDim; this.numBlocks = numBlocks;
			this.numItems = numItems; this.goalOffset = goalOffset;
		}

		public static final Layout V6 = new Layout(true, 64, 102, 20, 15, 26);
		public static final Layout V5 = new Layout(false, 32, 39, 18, 14, 25);

		/** The layout for a policy's VEC_DIM, or null if unknown. */
		public static Layout forVecDim(int vecDim) {
			return vecDim == V6.vecDim ? V6 : vecDim == V5.vecDim ? V5 : null;
		}

		public byte[] newImage() { return new byte[res * res * 2]; }
		public float[] newVec() { return new float[vecDim]; }
	}

	/** A body other entities see: a teammate slot or a ray target (feet position). */
	public record Mate(double x, double y, double z, int msg) {}

	/** Item predicates, index = sim item id (15 entries; v5 uses the first 14). */
	public static final Predicate<ItemStack>[] ITEMS = items();

	@SuppressWarnings("unchecked")
	private static Predicate<ItemStack>[] items() {
		return new Predicate[] {
			(Predicate<ItemStack>) s -> s.is(ItemTags.LOGS),
			(Predicate<ItemStack>) s -> s.is(ItemTags.PLANKS),
			(Predicate<ItemStack>) s -> s.is(Items.STICK),
			(Predicate<ItemStack>) s -> s.is(Items.CRAFTING_TABLE),
			(Predicate<ItemStack>) s -> s.is(Items.WOODEN_PICKAXE),
			(Predicate<ItemStack>) s -> s.is(ItemTags.STONE_TOOL_MATERIALS),
			(Predicate<ItemStack>) s -> s.is(Items.STONE_PICKAXE),
			(Predicate<ItemStack>) s -> s.is(Items.FURNACE),
			(Predicate<ItemStack>) s -> s.is(ItemTags.COALS),
			(Predicate<ItemStack>) s -> s.is(Items.RAW_IRON),
			(Predicate<ItemStack>) s -> s.is(Items.IRON_INGOT),
			(Predicate<ItemStack>) s -> s.is(Items.IRON_PICKAXE),
			(Predicate<ItemStack>) s -> s.is(Items.DIAMOND),
			(Predicate<ItemStack>) s -> s.is(Items.DIRT),
			(Predicate<ItemStack>) s -> s.is(Items.CHEST),
		};
	}

	/** Pickaxes are never moved by deposit or give (sim: "all non-tool items"). */
	public static boolean isTool(ItemStack s) {
		return s.is(Items.WOODEN_PICKAXE) || s.is(Items.STONE_PICKAXE) || s.is(Items.IRON_PICKAXE);
	}

	private static boolean isSandstone(Block b) {
		return b == Blocks.SANDSTONE || b == Blocks.CHISELED_SANDSTONE || b == Blocks.CUT_SANDSTONE
			|| b == Blocks.SMOOTH_SANDSTONE || b == Blocks.RED_SANDSTONE || b == Blocks.CHISELED_RED_SANDSTONE
			|| b == Blocks.CUT_RED_SANDSTONE || b == Blocks.SMOOTH_RED_SANDSTONE;
	}

	/** Sim block id of a Minecraft block state (v6 ids; chests -> CHEST). */
	public static int blockId(BlockState s) {
		if (s.isAir()) return AIR;
		Block b = s.getBlock();
		// Fluids first: any water (incl. waterlogged) stops rays and reports WATER; lava is an obstacle.
		var fluid = s.getFluidState().getType();
		if (fluid.is(FluidTags.WATER)) return WATER;
		if (fluid.is(FluidTags.LAVA)) return STONE;
		if (b == Blocks.SAND || b == Blocks.RED_SAND || isSandstone(b)) return SAND;
		if (b == Blocks.GRAVEL) return GRAVEL;
		if (b == Blocks.DEEPSLATE || b == Blocks.TUFF) return DEEPSLATE;
		if (b == Blocks.GRASS_BLOCK || b == Blocks.MYCELIUM || b == Blocks.PODZOL) return GRASS;
		if (b == Blocks.DIRT || b == Blocks.COARSE_DIRT || b == Blocks.ROOTED_DIRT || b == Blocks.DIRT_PATH
			|| b == Blocks.FARMLAND || b == Blocks.MUD || b == Blocks.CLAY) return DIRT;
		if (s.is(BlockTags.LOGS)) return LOG;
		if (s.is(BlockTags.LEAVES)) return LEAVES;
		if (s.is(BlockTags.PLANKS)) return PLANKS;
		if (b == Blocks.CRAFTING_TABLE) return TABLE;
		if (b == Blocks.FURNACE) return FURNACE;
		if (b == Blocks.CHEST || b == Blocks.TRAPPED_CHEST) return CHEST;
		if (b == Blocks.COAL_ORE || b == Blocks.DEEPSLATE_COAL_ORE) return COAL_ORE;
		if (s.is(BlockTags.IRON_ORES)) return IRON_ORE;
		if (b == Blocks.DIAMOND_ORE || b == Blocks.DEEPSLATE_DIAMOND_ORE) return DIAMOND_ORE;
		if (b == Blocks.BEDROCK) return BEDROCK;
		if (b == Blocks.COBBLESTONE || b == Blocks.MOSSY_COBBLESTONE || b == Blocks.COBBLED_DEEPSLATE) return COBBLE;
		// Plants, flowers, snow layers, torches... are see-through for the policy.
		if (!s.blocksMotion()) return AIR;
		return STONE;
	}

	/** Result of one ray. Reused between rays (no per-ray allocation). */
	public static final class Hit {
		public boolean hit;
		public int x, y, z, px, py, pz, face = -1, id, agent = -1;
		public double dist;
	}

	public static double[] lookDir(float yaw, float pitch) {
		double y = Math.toRadians(yaw), p = Math.toRadians(pitch);
		return new double[] {-Math.sin(y) * Math.cos(p), -Math.sin(p), Math.cos(y) * Math.cos(p)};
	}

	/** Port of sim.cpp boxHit(): ray vs a 0.6 x 1.8 x 0.6 box at feet (m). Returns entry t, or NaN on miss. */
	private static double boxHit(Mate m, double ox, double oy, double oz, double dx, double dy, double dz,
	                             double maxd, int[] faceOut) {
		double[] mn = {m.x() - HALF_W, m.y(), m.z() - HALF_W};
		double[] mx = {m.x() + HALF_W, m.y() + HEIGHT, m.z() + HALF_W};
		double[] org = {ox, oy, oz}, dir = {dx, dy, dz};
		double t0 = -1e30, t1 = 1e30;
		int f = -1;
		for (int a = 0; a < 3; a++) {
			if (Math.abs(dir[a]) < 1e-12) {
				if (org[a] < mn[a] || org[a] > mx[a]) return Double.NaN;
				continue;
			}
			double inv = 1.0 / dir[a];
			double ta = (mn[a] - org[a]) * inv, tb = (mx[a] - org[a]) * inv;
			if (ta > tb) { double tmp = ta; ta = tb; tb = tmp; }
			if (ta > t0) { t0 = ta; f = a; }
			if (tb < t1) t1 = tb;
			if (t0 > t1) return Double.NaN;
		}
		if (t1 < 0 || t0 > maxd) return Double.NaN;
		faceOut[0] = f < 0 ? 1 : f;
		return Math.max(t0, 0.0);
	}

	/**
	 * Voxel DDA, same algorithm as sim.cpp raycast(). Water stops the ray unless passWater. Bodies (other
	 * players/bots, only when `agents`) are hit before any block they precede.
	 */
	static void cast(Level level, BlockPos.MutableBlockPos mp, double ox, double oy, double oz,
	                 double dx, double dy, double dz, double maxd, boolean passWater,
	                 List<Mate> bodies, boolean agents, Hit h) {
		h.hit = false; h.face = -1; h.agent = -1; h.id = AIR; h.dist = 0; h.x = h.y = h.z = 0;
		double aT = 1e30;
		int aK = -1, aFace = -1;
		int[] faceOut = new int[1];
		if (agents && bodies != null)
			for (int k = 0; k < bodies.size(); k++) {
				double te = boxHit(bodies.get(k), ox, oy, oz, dx, dy, dz, maxd, faceOut);
				if (!Double.isNaN(te) && te < aT) { aT = te; aK = k; aFace = faceOut[0]; }
			}
		int x = (int) Math.floor(ox), y = (int) Math.floor(oy), z = (int) Math.floor(oz);
		int sx = dx > 0 ? 1 : -1, sy = dy > 0 ? 1 : -1, sz = dz > 0 ? 1 : -1;
		double tdx = dx != 0 ? Math.abs(1.0 / dx) : 1e30, tdy = dy != 0 ? Math.abs(1.0 / dy) : 1e30;
		double tdz = dz != 0 ? Math.abs(1.0 / dz) : 1e30;
		double tmx = dx != 0 ? (dx > 0 ? x + 1 - ox : ox - x) * tdx : 1e30;
		double tmy = dy != 0 ? (dy > 0 ? y + 1 - oy : oy - y) * tdy : 1e30;
		double tmz = dz != 0 ? (dz > 0 ? z + 1 - oz : oz - z) * tdz : 1e30;
		int lx = x, ly = y, lz = z, face = -1;
		double t = 0;
		int minY = level.getMinY(), maxY = level.getMaxY();
		for (;;) {
			if (aK >= 0 && aT <= t) { agentHit(bodies, aK, aT, aFace, h); return; }
			int id;
			if (y < minY) id = BEDROCK;
			else if (y > maxY) id = AIR;
			else id = blockId(level.getBlockState(mp.set(x, y, z)));
			if (id != AIR && !(passWater && id == WATER)) {
				h.hit = true;
				h.x = x; h.y = y; h.z = z;
				h.px = lx; h.py = ly; h.pz = lz;
				h.face = face; h.dist = t; h.id = id;
				return;
			}
			if (y > maxY && dy >= 0) {
				if (aK >= 0 && aT <= maxd) agentHit(bodies, aK, aT, aFace, h);
				return;
			}
			lx = x; ly = y; lz = z;
			if (tmx < tmy && tmx < tmz) { t = tmx; x += sx; tmx += tdx; face = 0; }
			else if (tmy < tmz) { t = tmy; y += sy; tmy += tdy; face = 1; }
			else { t = tmz; z += sz; tmz += tdz; face = 2; }
			if (t > maxd) {
				if (aK >= 0 && aT <= maxd) agentHit(bodies, aK, aT, aFace, h);
				return;
			}
		}
	}

	private static void agentHit(List<Mate> bodies, int k, double t, int face, Hit h) {
		Mate m = bodies.get(k);
		h.hit = true;
		h.id = AGENT;
		h.agent = k;
		h.dist = t;
		h.face = face;
		h.x = (int) Math.floor(m.x()); h.y = (int) Math.floor(m.y()); h.z = (int) Math.floor(m.z());
	}

	/** The block under the crosshair as the policy sees it (blocks only, water stops unless submerged). */
	public static Hit target(Level level, double x, double y, double z, float yaw, float pitch) {
		double[] d = lookDir(yaw, pitch);
		BlockPos.MutableBlockPos mp = new BlockPos.MutableBlockPos();
		Hit h = new Hit();
		boolean pass = eyeInWater(level, mp, x, y, z);
		cast(level, mp, x, y + EYE, z, d[0], d[1], d[2], REACH, pass, null, false, h);
		return h;
	}

	private static boolean eyeInWater(Level level, BlockPos.MutableBlockPos mp, double x, double y, double z) {
		return blockId(level.getBlockState(mp.set(Math.floor(x), Math.floor(y + EYE), Math.floor(z)))) == WATER;
	}

	/** Number of stacks/items in a container matching a predicate. */
	public static int count(Container c, Predicate<ItemStack> pred) {
		int n = 0;
		for (int i = 0; i < c.getContainerSize(); i++) {
			ItemStack s = c.getItem(i);
			if (!s.isEmpty() && pred.test(s)) n += s.getCount();
		}
		return n;
	}

	/** Sim near-block test: box dy -3..4, dx/dz -3..3 around the feet block. */
	public static boolean near(Level level, double x, double y, double z, Block b) {
		int cx = (int) Math.floor(x), cy = (int) Math.floor(y), cz = (int) Math.floor(z);
		BlockPos.MutableBlockPos mp = new BlockPos.MutableBlockPos();
		for (int dy = -3; dy <= 4; dy++)
			for (int dz = -3; dz <= 3; dz++)
				for (int dx = -3; dx <= 3; dx++)
					if (level.getBlockState(mp.set(cx + dx, cy + dy, cz + dz)).is(b)) return true;
		return false;
	}

	/** The chest container at a block, or null if the block has no container. */
	public static Container chestAt(Level level, BlockPos pos) {
		return level.getBlockEntity(pos) instanceof Container c ? c : null;
	}

	/**
	 * Fills img (layout.res * res * 2 bytes) and vec (layout.vecDim floats) exactly like sim.cpp writeObs().
	 * teammates: the other craftbot bots in spawn order (slot = position in the list, absent = zeros; only
	 * v6 uses them). bodies: every other player/bot in the level (rays hit them as AGENT, v6 only).
	 * step: the decision count this episode (sim W.t). goal: sim item id of the goal.
	 */
	public static void build(Level level, Layout L, double x, double y, double z, float yaw, float pitch,
	                         boolean onGround, float health, float mineProgress, int fuel, int goal, int step,
	                         Container inventory, List<Mate> teammates, List<Mate> bodies,
	                         byte[] img, float[] vec) {
		java.util.Arrays.fill(vec, 0f);
		BlockPos.MutableBlockPos mp = new BlockPos.MutableBlockPos();
		boolean pass = eyeInWater(level, mp, x, y, z);
		double[] f = lookDir(yaw, pitch);
		double yr = Math.toRadians(yaw);
		double[] r = {-Math.cos(yr), 0, -Math.sin(yr)};
		double[] u = {r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]};
		double tanH = Math.tan(Math.toRadians(FOV_DEG * 0.5));
		double ox = x, oy = y + EYE, oz = z;
		int R = L.res;
		List<Mate> rays = L.v6 ? (bodies == null ? List.of() : bodies) : List.of();
		Hit h = new Hit();
		for (int j = 0; j < R; j++)
			for (int i = 0; i < R; i++) {
				double a = ((i + 0.5) / R * 2 - 1) * tanH, b = (1 - (j + 0.5) / R * 2) * tanH;
				double dx = f[0] + r[0] * a + u[0] * b, dy = f[1] + r[1] * a + u[1] * b, dz = f[2] + r[2] * a + u[2] * b;
				double n = Math.sqrt(dx * dx + dy * dy + dz * dz);
				cast(level, mp, ox, oy, oz, dx / n, dy / n, dz / n, VIEW_DIST, pass, rays, L.v6, h);
				int id = h.hit ? h.id : AIR;
				if (!L.v6 && id >= 18) id = id == CHEST ? PLANKS : AIR;  // v5 ids: chest -> planks, agent -> air
				int k = (j * R + i) * 2;
				img[k] = (byte) id;
				img[k + 1] = (byte) (h.hit ? (int) Math.min(255.0, h.dist / VIEW_DIST * 255.0) : 255);
			}

		double l64 = Math.log1p(64);
		for (int i = 0; i < L.numItems; i++) vec[i] = (float) (Math.log1p(count(inventory, ITEMS[i])) / l64);
		int k = L.numItems;
		vec[k++] = health / 20f;
		vec[k++] = onGround ? 1f : 0f;
		vec[k++] = pitch / 90f;
		vec[k++] = (float) Math.sin(yr);
		vec[k++] = (float) Math.cos(yr);
		vec[k++] = mineProgress;
		vec[k++] = (float) ((y - 63.0) / 32.0);
		vec[k++] = near(level, x, y, z, Blocks.CRAFTING_TABLE) ? 1f : 0f;
		vec[k++] = near(level, x, y, z, Blocks.FURNACE) ? 1f : 0f;
		vec[k++] = Math.min(fuel, 8) / 8f;
		vec[k++] = (float) step / MAX_STEPS;
		for (int i = 0; i < L.numItems; i++) vec[L.goalOffset + i] = i == goal ? 1f : 0f;
		if (!L.v6) return;

		// Teammate slots (6.5): 4 slots x [present, dx/32, dy/32, dz/32, msg one-hot(8)] in the yaw frame.
		double yawR = yr, fx = -Math.sin(yawR), fz = Math.cos(yawR), rx = -Math.cos(yawR), rz = -Math.sin(yawR);
		if (teammates != null)
			for (int s = 0; s < teammates.size() && s < MAX_TEAM_SLOTS; s++) {
				Mate m = teammates.get(s);
				double dx = m.x() - x, dy = m.y() - y, dz = m.z() - z;
				if (dx * dx + dy * dy + dz * dz > TEAM_RANGE * TEAM_RANGE) continue;
				int o = 41 + s * 12;
				double xl = dx * rx + dz * rz, zl = dx * fx + dz * fz;
				vec[o] = 1f;
				vec[o + 1] = (float) (xl / 32.0);
				vec[o + 2] = (float) (dy / 32.0);
				vec[o + 3] = (float) (zl / 32.0);
				vec[o + 4 + Math.min(Math.max(m.msg(), 0), NUM_MSGS - 1)] = 1f;
			}

		// Nearest chest within 16 blocks (feet to chest centre). Scan order y, z, x ascending, ties to the
		// later cell, like the sim's key-ordered map.
		double bestD = 256.0;
		int bx = 0, by = 0, bz = 0;
		boolean found = false;
		int cx0 = (int) Math.floor(x), cy0 = (int) Math.floor(y), cz0 = (int) Math.floor(z);
		int R16 = (int) CHEST_RANGE;
		for (int cy = cy0 - R16; cy <= cy0 + R16; cy++)
			for (int cz = cz0 - R16; cz <= cz0 + R16; cz++)
				for (int cx = cx0 - R16; cx <= cx0 + R16; cx++) {
					if (blockId(level.getBlockState(mp.set(cx, cy, cz))) != CHEST) continue;
					double dx = cx + 0.5 - x, dy = cy + 0.5 - y, dz = cz + 0.5 - z;
					double d2 = dx * dx + dy * dy + dz * dz;
					if (d2 <= bestD) { bestD = d2; bx = cx; by = cy; bz = cz; found = true; }
				}
		if (found) {
			double dx = bx + 0.5 - x, dz = bz + 0.5 - z;
			vec[89] = 1f;
			vec[90] = (float) ((dx * rx + dz * rz) / 16.0);
			vec[91] = (float) ((by + 0.5 - y) / 16.0);
			vec[92] = (float) ((dx * fx + dz * fz) / 16.0);
		}

		// Crosshair chest contents (within reach only): log2(1 + count) / 6 for items 0..8.
		Hit th = target(level, x, y, z, yaw, pitch);
		if (th.hit && th.id == CHEST) {
			Container c = chestAt(level, new BlockPos(th.x, th.y, th.z));
			if (c != null)
				for (int i = 0; i < 9; i++)
					vec[93 + i] = (float) (Math.log(1.0 + count(c, ITEMS[i])) / Math.log(2.0) / 6.0);
		}

	}
}
