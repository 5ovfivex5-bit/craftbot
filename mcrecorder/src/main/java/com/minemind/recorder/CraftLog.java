package com.minemind.recorder;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Logging with the mod's logger; never throws. */
final class CraftLog {
	private static final Logger LOG = LoggerFactory.getLogger("minemind_recorder");

	private CraftLog() {}

	static void info(String msg) {
		try { LOG.info(msg); } catch (Throwable ignored) { }
	}

	static void error(String msg, Throwable t) {
		try { LOG.error(msg, t); } catch (Throwable ignored) { }
	}
}
