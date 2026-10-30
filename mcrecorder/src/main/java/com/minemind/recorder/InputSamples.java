package com.minemind.recorder;

import java.util.concurrent.ConcurrentLinkedDeque;

/**
 * Hand-off between the client thread (raw input, one sample per client tick) and the server thread (one
 * frame per server tick). Deliberately free of client classes so the server side loads on any side.
 * Samples are consumed in tick order, one per server tick, which joins them to frames tick by tick.
 */
public final class InputSamples {
	private InputSamples() {}

	public static volatile boolean enabled = false;
	private static final ConcurrentLinkedDeque<Sample> QUEUE = new ConcurrentLinkedDeque<>();
	private static final int MAX_LAG = 4;

	public static final class Sample {
		public boolean forward, back, left, right, jump, sneak, sprint, attack, use, drop, inventory, swapOffhand, pick, attackBlock;
		public final boolean[] hotbar = new boolean[9];
		public float dyaw, dpitch, yaw, pitch;
		public int selectedSlot;
		public String gui;   // null | "inventory" | "crafting" | "furnace" | "chest" | other screen class
	}

	public static void clearAndEnable(boolean on) {
		QUEUE.clear();
		enabled = on;
	}

	/** Bounded: while paused nobody takes samples, so the oldest are dropped. */
	public static void put(Sample s) {
		QUEUE.addLast(s);
		while (QUEUE.size() > MAX_LAG * 4) QUEUE.pollFirst();
	}

	/** Oldest pending sample for this server tick, or null if the client produced none. Drops a backlog. */
	public static Sample take() {
		while (QUEUE.size() > MAX_LAG) QUEUE.pollFirst();
		return QUEUE.pollFirst();
	}
}
