package com.minemind.recorder;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.nio.file.Files;
import java.nio.file.Path;

/** config/minemind_recorder.json: created with defaults on first start. */
final class Config {
	boolean autoStart = true;
	int perfLogSeconds = 300;

	static Config load(Path path) {
		Config c = new Config();
		try {
			if (Files.exists(path)) {
				JsonObject o = JsonParser.parseString(Files.readString(path)).getAsJsonObject();
				if (o.has("autoStart")) c.autoStart = o.get("autoStart").getAsBoolean();
				if (o.has("perfLogSeconds")) c.perfLogSeconds = Math.max(10, o.get("perfLogSeconds").getAsInt());
			} else {
				Files.createDirectories(path.getParent());
				Files.writeString(path, "{\n  \"autoStart\": true,\n  \"perfLogSeconds\": 300\n}\n");
			}
		} catch (Exception e) {
			CraftLog.error("config " + path + " unreadable; using defaults", e);
		}
		return c;
	}
}
