package com.minemind.recorder;

import java.util.Arrays;
import java.util.List;

/**
 * Worker-thread observation renderer. Same math as Obs.build (v6 layout, teammates zero, goal zero) but reads a
 * block-id snapshot (Snapshot) instead of a live Level, so the 64x64 raycast never runs on the server thread.
 * Obs.java itself is untouched; the DDA and box test are copies of its private helpers.
 */
final class FrameRenderer {
	static final class Region {
		final byte[] ids;
		final int ox, oy, oz, sx, sy, sz, minY, maxY;

		Region(byte[] ids, int ox, int oy, int oz, int sx, int sy, int sz, int minY, int maxY) {
			this.ids = ids; this.ox = ox; this.oy = oy; this.oz = oz;
			this.sx = sx; this.sy = sy; this.sz = sz; this.minY = minY; this.maxY = maxY;
		}
	}

	/** Player state for one tick, captured on the server thread. */
	static final class Pose {
		double x, y, z;
		float yaw, pitch, health;
		boolean onGround;
		int[] items;              // Obs.ITEMS counts (15)
		int fuel;                 // coal count
		int step;                 // (frames / 4) % MAX_STEPS
		float[] xhair;            // null, or 9 values for vec[93..101]
		List<Obs.Mate> bodies;    // other players (rays hit them)
	}

	private final int[] faceOut = new int[1];
	private Region r;
	int outsideLookups;

	void render(Region region, Pose p, byte[] img, float[] vec) {
		this.r = region;
		Arrays.fill(vec, 0f);
		final int R = Obs.Layout.V6.res;
		boolean pass = idAt(floor(p.x), floor(p.y + Obs.EYE), floor(p.z)) == Obs.WATER;
		double[] f = Obs.lookDir(p.yaw, p.pitch);
		double yr = Math.toRadians(p.yaw);
		double[] rv = {-Math.cos(yr), 0, -Math.sin(yr)};
		double[] u = {rv[1] * f[2] - rv[2] * f[1], rv[2] * f[0] - rv[0] * f[2], rv[0] * f[1] - rv[1] * f[0]};
		double tanH = Math.tan(Math.toRadians(Obs.FOV_DEG * 0.5));
		double ox = p.x, oy = p.y + Obs.EYE, oz = p.z;
		Obs.Hit h = new Obs.Hit();
		for (int j = 0; j < R; j++)
			for (int i = 0; i < R; i++) {
				double a = ((i + 0.5) / R * 2 - 1) * tanH, b = (1 - (j + 0.5) / R * 2) * tanH;
				double dx = f[0] + rv[0] * a + u[0] * b, dy = f[1] + rv[1] * a + u[1] * b, dz = f[2] + rv[2] * a + u[2] * b;
				double n = Math.sqrt(dx * dx + dy * dy + dz * dz);
				cast(ox, oy, oz, dx / n, dy / n, dz / n, Obs.VIEW_DIST, pass, p.bodies, h);
				int id = h.hit ? h.id : Obs.AIR;
				int k = (j * R + i) * 2;
				img[k] = (byte) id;
				img[k + 1] = (byte) (h.hit ? (int) Math.min(255.0, h.dist / Obs.VIEW_DIST * 255.0) : 255);
			}

		double l64 = Math.log1p(64);
		for (int i = 0; i < 15; i++) vec[i] = (float) (Math.log1p(p.items[i]) / l64);
		int k = 15;
		double sinY = Math.sin(yr), cosY = Math.cos(yr);
		vec[k++] = p.health / 20f;
		vec[k++] = p.onGround ? 1f : 0f;
		vec[k++] = p.pitch / 90f;
		vec[k++] = (float) sinY;
		vec[k++] = (float) cosY;
		vec[k++] = 0f;                                   // mining progress: not observable from the server
		vec[k++] = (float) ((p.y - 63.0) / 32.0);
		vec[k++] = near(p.x, p.y, p.z, Obs.TABLE) ? 1f : 0f;
		vec[k++] = near(p.x, p.y, p.z, Obs.FURNACE) ? 1f : 0f;
		vec[k++] = Math.min(p.fuel, 8) / 8f;
		vec[k++] = (float) p.step / Obs.MAX_STEPS;
		// 26..40 goal one-hot: zeros (hindsight goals are assigned in Python). 41..88 teammates: zeros.

		// Nearest chest within 16 blocks, same scan order as Obs.build.
		double bestD = 256.0;
		int bx = 0, by = 0, bz = 0;
		boolean found = false;
		int cx0 = floor(p.x), cy0 = floor(p.y), cz0 = floor(p.z);
		int R16 = (int) Obs.CHEST_RANGE;
		for (int cy = cy0 - R16; cy <= cy0 + R16; cy++)
			for (int cz = cz0 - R16; cz <= cz0 + R16; cz++)
				for (int cx = cx0 - R16; cx <= cx0 + R16; cx++) {
					if (idAt(cx, cy, cz) != Obs.CHEST) continue;
					double dx = cx + 0.5 - p.x, dy = cy + 0.5 - p.y, dz = cz + 0.5 - p.z;
					double d2 = dx * dx + dy * dy + dz * dz;
					if (d2 <= bestD) { bestD = d2; bx = cx; by = cy; bz = cz; found = true; }
				}
		if (found) {
			double fx = -sinY, fz = cosY, rx = -cosY, rz = -sinY;
			double dx = bx + 0.5 - p.x, dz = bz + 0.5 - p.z;
			vec[89] = 1f;
			vec[90] = (float) ((dx * rx + dz * rz) / 16.0);
			vec[91] = (float) ((by + 0.5 - p.y) / 16.0);
			vec[92] = (float) ((dx * fx + dz * fz) / 16.0);
		}
		if (p.xhair != null) System.arraycopy(p.xhair, 0, vec, 93, 9);
		this.r = null;
	}

	private static int floor(double v) { return (int) Math.floor(v); }

	private boolean near(double x, double y, double z, int block) {
		int cx = floor(x), cy = floor(y), cz = floor(z);
		for (int dy = -3; dy <= 4; dy++)
			for (int dz = -3; dz <= 3; dz++)
				for (int dx = -3; dx <= 3; dx++)
					if (idAt(cx + dx, cy + dy, cz + dz) == block) return true;
		return false;
	}

	private int idAt(int x, int y, int z) {
		if (y < r.minY) return Obs.BEDROCK;
		if (y > r.maxY) return Obs.AIR;
		int ix = x - r.ox, iy = y - r.oy, iz = z - r.oz;
		if (ix < 0 || iy < 0 || iz < 0 || ix >= r.sx || iy >= r.sy || iz >= r.sz) {
			outsideLookups++;
			return Obs.AIR;
		}
		return r.ids[(iy * r.sz + iz) * r.sx + ix];
	}

	/** Copy of Obs.cast (voxel DDA, water rule, bodies) over the snapshot. */
	private void cast(double ox, double oy, double oz, double dx, double dy, double dz, double maxd,
	                  boolean passWater, List<Obs.Mate> bodies, Obs.Hit h) {
		h.hit = false; h.face = -1; h.agent = -1; h.id = Obs.AIR; h.dist = 0; h.x = h.y = h.z = 0;
		double aT = 1e30;
		int aK = -1, aFace = -1;
		if (bodies != null)
			for (int k = 0; k < bodies.size(); k++) {
				double te = boxHit(bodies.get(k), ox, oy, oz, dx, dy, dz, maxd);
				if (!Double.isNaN(te) && te < aT) { aT = te; aK = k; aFace = faceOut[0]; }
			}
		int x = floor(ox), y = floor(oy), z = floor(oz);
		int sx = dx > 0 ? 1 : -1, sy = dy > 0 ? 1 : -1, sz = dz > 0 ? 1 : -1;
		double tdx = dx != 0 ? Math.abs(1.0 / dx) : 1e30, tdy = dy != 0 ? Math.abs(1.0 / dy) : 1e30;
		double tdz = dz != 0 ? Math.abs(1.0 / dz) : 1e30;
		double tmx = dx != 0 ? (dx > 0 ? x + 1 - ox : ox - x) * tdx : 1e30;
		double tmy = dy != 0 ? (dy > 0 ? y + 1 - oy : oy - y) * tdy : 1e30;
		double tmz = dz != 0 ? (dz > 0 ? z + 1 - oz : oz - z) * tdz : 1e30;
		int lx = x, ly = y, lz = z, face = -1;
		double t = 0;
		for (;;) {
			if (aK >= 0 && aT <= t) { agentHit(bodies, aK, aT, aFace, h); return; }
			int id = idAt(x, y, z);
			if (id != Obs.AIR && !(passWater && id == Obs.WATER)) {
				h.hit = true;
				h.x = x; h.y = y; h.z = z;
				h.px = lx; h.py = ly; h.pz = lz;
				h.face = face; h.dist = t; h.id = id;
				return;
			}
			if (y > r.maxY && dy >= 0) {
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

	private static void agentHit(List<Obs.Mate> bodies, int k, double t, int face, Obs.Hit h) {
		Obs.Mate m = bodies.get(k);
		h.hit = true;
		h.id = Obs.AGENT;
		h.agent = k;
		h.dist = t;
		h.face = face;
		h.x = floor(m.x()); h.y = floor(m.y()); h.z = floor(m.z());
	}

	/** Copy of Obs.boxHit (private there). */
	private double boxHit(Obs.Mate m, double ox, double oy, double oz, double dx, double dy, double dz, double maxd) {
		double[] mn = {m.x() - Obs.HALF_W, m.y(), m.z() - Obs.HALF_W};
		double[] mx = {m.x() + Obs.HALF_W, m.y() + Obs.HEIGHT, m.z() + Obs.HALF_W};
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
}
