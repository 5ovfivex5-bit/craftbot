package com.minemind.recorder;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

/** JSON helpers: every line the recorder writes goes through Gson (valid escaping, one object per line). */
final class Json {
	private Json() {}

	static final Gson GSON = new GsonBuilder().disableHtmlEscaping().create();

	/** Quoted, escaped JSON string. */
	static String q(String s) {
		return GSON.toJson(s);
	}

	static double round3(double v) {
		return Math.round(v * 1000.0) / 1000.0;
	}
}
