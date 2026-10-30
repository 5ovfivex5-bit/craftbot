package com.minemind.recorder;

/** Server-thread cost of Recorder.tick, averaged per window and overall. */
final class Perf {
	private long winNs, winTicks, winMax, totNs, totTicks, totMax;

	void add(long ns) {
		winNs += ns; winTicks++; winMax = Math.max(winMax, ns);
		totNs += ns; totTicks++; totMax = Math.max(totMax, ns);
	}

	boolean windowFull(int ticks) { return winTicks >= ticks; }

	/** Report for the window, then resets the window. */
	String takeWindow() {
		String s = String.format("window %d ticks avg %.3f ms max %.3f ms | session %d ticks avg %.3f ms max %.3f ms",
			winTicks, avgMs(winNs, winTicks), winMax / 1e6, totTicks, avgMs(totNs, totTicks), totMax / 1e6);
		winNs = 0; winTicks = 0; winMax = 0;
		return s;
	}

	double sessionAvgMs() { return avgMs(totNs, totTicks); }

	long ticks() { return totTicks; }

	private static double avgMs(long ns, long n) { return n == 0 ? 0 : ns / 1e6 / n; }
}
