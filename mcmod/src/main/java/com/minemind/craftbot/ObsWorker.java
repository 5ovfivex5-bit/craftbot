package com.minemind.craftbot;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

/**
 * Background pool for training observations (FT-1). Train.java captures a block snapshot and a body pose on the
 * server thread, submits {@link Obs#buildFrom} here, and applies the finished JSON on a later server tick.
 *
 * Only JDK imports, so it can be exercised without Minecraft. Jobs must only read immutable data (a Snapshot,
 * a Pose, Mate records, the Layout) and must never touch Level, entities, containers or Bot state.
 * Worker threads are daemons; {@link #shutdown} drops the pool and a later {@link #submit} creates a new one.
 */
public final class ObsWorker {
	private ObsWorker() {}

	private static final AtomicInteger IDS = new AtomicInteger();
	private static ExecutorService pool;

	/** Worker count: leave one core for the server thread, at least 2, at most 6. */
	public static int threads() {
		return Math.max(2, Math.min(6, Runtime.getRuntime().availableProcessors() - 1));
	}

	private static synchronized ExecutorService pool() {
		if (pool == null)
			pool = Executors.newFixedThreadPool(threads(), r -> {
				Thread t = new Thread(r, "craftbot-obs-" + IDS.incrementAndGet());
				t.setDaemon(true);
				return t;
			});
		return pool;
	}

	/** Runs job on a worker thread. The future completes exceptionally if the job throws. */
	public static <T> CompletableFuture<T> submit(Supplier<T> job) {
		return CompletableFuture.supplyAsync(job, pool());
	}

	/** Drops the pool (server stopping). Running jobs are interrupted; their results are discarded by callers. */
	public static synchronized void shutdown() {
		if (pool != null) {
			pool.shutdownNow();
			pool = null;
		}
	}
}
