package com.minemind.recorder;

import it.unimi.dsi.fastutil.objects.Object2IntMap;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.stats.Stat;
import net.minecraft.stats.StatsCounter;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

/**
 * Per-tick diff of the server player's statistics (StatsCounter). Vanilla statistics are the exact source for
 * crafts, mining, pickups, drops, item uses, kills, damage. The counter's entry map is read by reflection because
 * StatsCounter exposes no iterator (its `stats` field is protected).
 */
final class StatTracker {
	record Delta(Stat<?> stat, int delta) {}

	private static final Field STATS;

	static {
		Field f = null;
		try {
			f = StatsCounter.class.getDeclaredField("stats");
			f.setAccessible(true);
		} catch (ReflectiveOperationException | RuntimeException e) {
			CraftLog.error("StatsCounter.stats not accessible; stat events disabled", e);
		}
		STATS = f;
	}

	private final Object2IntOpenHashMap<Stat<?>> prev = new Object2IntOpenHashMap<>();
	private Object counter;

	/** Sets the current counter values as the baseline (no deltas reported). */
	void rebaseline(ServerPlayer p) {
		counter = p.getStats();
		prev.clear();
		for (Object2IntMap.Entry<Stat<?>> e : map(counter).object2IntEntrySet()) prev.put(e.getKey(), e.getIntValue());
	}

	/** Changes since the previous call. A new counter object (respawn, reload) re-baselines silently. */
	List<Delta> diff(ServerPlayer p) {
		Object c = p.getStats();
		if (c != counter) {
			rebaseline(p);
			return List.of();
		}
		List<Delta> out = new ArrayList<>();
		for (Object2IntMap.Entry<Stat<?>> e : map(c).object2IntEntrySet()) {
			Stat<?> s = e.getKey();
			int cur = e.getIntValue();
			int old = prev.getInt(s);
			if (cur != old) {
				if (cur > old) out.add(new Delta(s, cur - old));
				prev.put(s, cur);
			}
		}
		return out;
	}

	@SuppressWarnings("unchecked")
	private static Object2IntMap<Stat<?>> map(Object c) {
		if (STATS == null) return new Object2IntOpenHashMap<>();
		try {
			return (Object2IntMap<Stat<?>>) STATS.get(c);
		} catch (IllegalAccessException e) {
			throw new IllegalStateException(e);
		}
	}
}
