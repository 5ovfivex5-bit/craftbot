package com.minemind.recorder;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.zip.GZIPOutputStream;

/**
 * Everything a session writes to disk: chunks (ChunkWriter), input.jsonl.gz, events.jsonl.gz (both gzip with
 * sync flush), meta.json (rewritten atomically). Owned by the writer thread; pure Java.
 */
final class SessionWriter {
	final Path dir;
	final ChunkWriter chunks;
	private final BufferedWriter input, events;

	SessionWriter(Path dir) throws IOException {
		this.dir = dir;
		Files.createDirectories(dir);
		this.chunks = new ChunkWriter(dir, ChunkWriter.CHUNK_FRAMES);
		this.input = gz("input.jsonl.gz");
		this.events = gz("events.jsonl.gz");
	}

	private BufferedWriter gz(String name) throws IOException {
		return new BufferedWriter(new OutputStreamWriter(
			new GZIPOutputStream(new FileOutputStream(dir.resolve(name).toFile()), 1 << 16, true), StandardCharsets.UTF_8));
	}

	/** One recorded tick: binary record, input line and zero or more event lines. */
	void frame(long tick, byte[] img, float[] vec, int action, int extAction, String inputLine, List<String> eventLines)
			throws IOException {
		chunks.append(tick, img, vec, action, extAction);
		input.write(inputLine);
		for (String e : eventLines) events.write(e);
	}

	/** Event lines without a frame (pause, stop). */
	void eventLines(List<String> lines) throws IOException {
		for (String e : lines) events.write(e);
		events.flush();
	}

	/** Sync-flushes every stream (called about every 5 s of frames). */
	void flush() throws IOException {
		chunks.flush();
		input.flush();
		events.flush();
	}

	/** Completes the partial chunk and closes the gzip streams. */
	void closeData() throws IOException {
		try {
			chunks.finish();
		} finally {
			try {
				input.close();
			} finally {
				events.close();
			}
		}
	}

	/** Atomic rewrite of meta.json. */
	void writeMeta(String json) throws IOException {
		Path tmp = dir.resolve("meta.json.tmp");
		Files.writeString(tmp, json, StandardCharsets.UTF_8);
		Files.move(tmp, dir.resolve("meta.json"), StandardCopyOption.REPLACE_EXISTING);
	}
}
