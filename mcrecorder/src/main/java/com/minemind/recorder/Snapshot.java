package com.minemind.recorder;

import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.block.state.BlockState;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Server-thread block-id cache (16^3 sections, Obs.blockId per cell) around the player, so the worker can raycast
 * without touching the live level. Sections the window needs are filled synchronously when first needed; cached
 * sections are refreshed round-robin (REFRESH_PER_TICK per tick) and immediately when a block break or a
 * use/attack on a block marks them dirty. Blocks changed by other causes (fluids, fire, mobs) can be stale for up
 * to one full refresh cycle (about 6 s with 125 sections).
 */
final class Snapshot {
	static final int SEC = 16;
	static final int HX = 26, YB = 26, YA = 28;           // window: x,z in [p-26, p+26], y in [p-26, p+28]
	static final int SX = 2 * HX + 1, SZ = SX, SY = YB + YA + 1;
	static final int REFRESH_PER_TICK = 1, DIRTY_PER_TICK = 4, MAX_CACHED = 600;

	record Key(int x, int y, int z) {}

	private final Map<Key, byte[]> secs = new HashMap<>();
	private final Set<Key> dirty = new HashSet<>();
	private final BlockPos.MutableBlockPos mp = new BlockPos.MutableBlockPos();
	private List<Key> window = new ArrayList<>();
	private byte[][][][] grid;
	private int sxBase, syBase, szBase, nx, ny, nz;
	private int ox, oy, oz, minY, maxY, rr;
	long fills, refreshes;

	int ox() { return ox; }
	int oy() { return oy; }
	int oz() { return oz; }
	int minY() { return minY; }
	int maxY() { return maxY; }

	/** Moves the window to the player and fills missing sections. */
	void update(ServerLevel level, double px, double py, double pz) {
		minY = level.getMinY();
		maxY = level.getMaxY();
		int fx = (int) Math.floor(px), fy = (int) Math.floor(py), fz = (int) Math.floor(pz);
		ox = fx - HX; oy = fy - YB; oz = fz - HX;
		sxBase = Math.floorDiv(ox, SEC); nx = Math.floorDiv(ox + SX - 1, SEC) - sxBase + 1;
		syBase = Math.floorDiv(oy, SEC); ny = Math.floorDiv(oy + SY - 1, SEC) - syBase + 1;
		szBase = Math.floorDiv(oz, SEC); nz = Math.floorDiv(oz + SZ - 1, SEC) - szBase + 1;
		List<Key> w = new ArrayList<>(nx * ny * nz);
		grid = new byte[nx][ny][nz][];
		for (int a = 0; a < nx; a++)
			for (int b = 0; b < ny; b++)
				for (int c = 0; c < nz; c++) {
					Key k = new Key(sxBase + a, syBase + b, szBase + c);
					byte[] arr = secs.get(k);
					if (arr == null) {
						arr = new byte[SEC * SEC * SEC];
						fill(level, k, arr);
						secs.put(k, arr);
						fills++;
					}
					grid[a][b][c] = arr;
					w.add(k);
				}
		window = w;
		if (secs.size() > MAX_CACHED) {
			Set<Key> keep = new HashSet<>(w);
			secs.keySet().removeIf(k -> !keep.contains(k));
			dirty.removeIf(k -> !keep.contains(k));
		}
	}

	/** Marks the section holding a block as needing a refresh on the next tick. */
	void markDirty(int x, int y, int z) {
		dirty.add(new Key(Math.floorDiv(x, SEC), Math.floorDiv(y, SEC), Math.floorDiv(z, SEC)));
	}

	/** Marks the sections along the look ray (up to 6 blocks) dirty. */
	void markLook(double x, double y, double z, double yaw, double pitch) {
		double[] d = Obs.lookDir((float) yaw, (float) pitch);
		for (int s = 0; s <= 6; s++)
			markDirty((int) Math.floor(x + d[0] * s), (int) Math.floor(y + Obs.EYE + d[1] * s),
				(int) Math.floor(z + d[2] * s));
	}

	/** Refreshes dirty sections, then REFRESH_PER_TICK window sections round-robin. */
	void refresh(ServerLevel level) {
		int n = 0;
		Iterator<Key> it = dirty.iterator();
		while (it.hasNext() && n < DIRTY_PER_TICK) {
			Key k = it.next();
			it.remove();
			byte[] a = secs.get(k);
			if (a != null) { fill(level, k, a); refreshes++; }
			n++;
		}
		for (int i = 0; i < REFRESH_PER_TICK && !window.isEmpty(); i++) {
			if (rr >= window.size()) rr = 0;   // the window shrinks when the player moves
			Key k = window.get(rr);
			rr = (rr + 1) % window.size();
			fill(level, k, secs.get(k));
			refreshes++;
		}
	}

	/** Copies the window into dst, layout index = (iy * SZ + iz) * SX + ix, cells relative to ox/oy/oz. */
	void copy(byte[] dst) {
		for (int iy = 0; iy < SY; iy++) {
			int wy = oy + iy;
			int gy = Math.floorDiv(wy, SEC) - syBase, ly = Math.floorMod(wy, SEC);
			for (int iz = 0; iz < SZ; iz++) {
				int wz = oz + iz;
				int gz = Math.floorDiv(wz, SEC) - szBase, lz = Math.floorMod(wz, SEC);
				int base = (iy * SZ + iz) * SX;
				int ix = 0;
				while (ix < SX) {
					int wx = ox + ix;
					int gx = Math.floorDiv(wx, SEC) - sxBase, lx = Math.floorMod(wx, SEC);
					int run = Math.min(SEC - lx, SX - ix);
					byte[] a = grid[gx][gy][gz];
					if (a == null) Arrays.fill(dst, base + ix, base + ix + run, (byte) 0);
					else System.arraycopy(a, (ly * SEC + lz) * SEC + lx, dst, base + ix, run);
					ix += run;
				}
			}
		}
	}

	private void fill(ServerLevel level, Key k, byte[] a) {
		int bx = k.x() * SEC, by = k.y() * SEC, bz = k.z() * SEC;
		for (int ly = 0; ly < SEC; ly++) {
			int y = by + ly;
			if (y < minY || y > maxY) continue;
			for (int lz = 0; lz < SEC; lz++)
				for (int lx = 0; lx < SEC; lx++) {
					BlockState s = level.getBlockState(mp.set(bx + lx, y, bz + lz));
					a[(ly * SEC + lz) * SEC + lx] = (byte) Obs.blockId(s);
				}
		}
	}
}
