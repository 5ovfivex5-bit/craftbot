package com.minemind.recorder;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.zip.GZIPInputStream;
import java.util.zip.Inflater;

/**
 * Format self-check (no Minecraft needed): writes a 2500-frame session through SessionWriter (3 chunks), then
 * re-reads chunks/index.jsonl, every chunk record, input.jsonl.gz and events.jsonl.gz and checks that every JSON
 * line parses. Run: java -cp build/classes/java/main:build/classes/java/test:<gson.jar> com.minemind.recorder.FormatSelfCheck
 */
public final class FormatSelfCheck {
	public static void main(String[] args) throws Exception {
		Path dir = Files.createTempDirectory("minemind-rec-check");
		int frames = 2500;
		Random rnd = new Random(7);
		byte[] img = new byte[ChunkWriter.IMG_BYTES];
		float[] vec = new float[ChunkWriter.VEC_DIM];
		SessionWriter w = new SessionWriter(dir);
		List<byte[]> imgs = new ArrayList<>();
		for (int t = 0; t < frames; t++) {
			rnd.nextBytes(img);
			for (int i = 0; i < vec.length; i++) vec[i] = rnd.nextFloat();
			int action = t % 28, ext = t * 7;
			JsonObject in = new JsonObject();
			in.addProperty("tick", t);
			in.addProperty("repeat", t % 50 == 0);
			in.addProperty("context", t % 100 == 0 ? "build \"house\"\n" : null);
			String inputLine = Json.GSON.toJson(in) + "\n";
			JsonObject ev = new JsonObject();
			ev.addProperty("tick", t);
			ev.addProperty("type", "mine");
			ev.addProperty("item", "minecraft:oak_log");
			String evLine = Json.GSON.toJson(ev) + "\n";
			List<String> evs = t % 10 == 0 ? List.of(evLine) : List.of();
			w.frame(t, img.clone(), vec.clone(), action, ext, inputLine, evs);
			imgs.add(img.clone());
		}
		w.closeData();
		w.writeMeta(Json.GSON.toJson(new JsonObject()) + "\n");

		// index
		List<String> index = Files.readAllLines(dir.resolve("chunks/index.jsonl"), StandardCharsets.UTF_8);
		check(index.size() == 3, "index has 3 chunks, got " + index.size());
		int[] expectFrames = {1200, 1200, 100};
		for (int c = 0; c < 3; c++) {
			JsonObject j = JsonParser.parseString(index.get(c)).getAsJsonObject();
			check(j.get("chunk").getAsInt() == c, "chunk number");
			check(j.get("frames").getAsInt() == expectFrames[c], "frames in chunk " + c);
			check(j.get("first_tick").getAsInt() == c * 1200, "first_tick in chunk " + c);
			check(j.get("bytes").getAsLong() == Files.size(dir.resolve(String.format("chunks/%06d.bin", c))), "bytes");
		}
		// chunk records
		int tickNo = 0;
		for (int c = 0; c < 3; c++) {
			byte[] raw = inflateFile(dir.resolve(String.format("chunks/%06d.bin", c)));
			check(raw.length == expectFrames[c] * ChunkWriter.FRAME_BYTES, "raw size chunk " + c);
			ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
			for (int f = 0; f < expectFrames[c]; f++) {
				int base = f * ChunkWriter.FRAME_BYTES;
				byte[] got = new byte[ChunkWriter.IMG_BYTES];
				System.arraycopy(raw, base, got, 0, got.length);
				check(java.util.Arrays.equals(got, imgs.get(tickNo)), "img at tick " + tickNo);
				int vb = base + ChunkWriter.IMG_BYTES;
				int action = bb.getInt(vb + ChunkWriter.VEC_DIM * 4);
				int ext = bb.getInt(vb + ChunkWriter.VEC_DIM * 4 + 4);
				check(action == tickNo % 28 && ext == tickNo * 7, "action/ext at tick " + tickNo);
				tickNo++;
			}
		}
		check(tickNo == frames, "all frames read back");

		// every line of input and events is one valid JSON object
		int inLines = parseLines(dir.resolve("input.jsonl.gz"));
		check(inLines == frames, "input lines " + inLines);
		int evLines = parseLines(dir.resolve("events.jsonl.gz"));
		check(evLines == frames / 10, "event lines " + evLines);
		System.out.println("format self-check OK: " + frames + " frames, " + index.size() + " chunks, "
			+ inLines + " input lines, " + evLines + " event lines in " + dir);
	}

	private static int parseLines(Path gz) throws Exception {
		int n = 0;
		try (BufferedReader r = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(gz.toFile())), StandardCharsets.UTF_8))) {
			String line;
			while ((line = r.readLine()) != null) {
				JsonObject o = JsonParser.parseString(line).getAsJsonObject();
				check(o.has("tick"), "line has tick: " + line);
				n++;
			}
		}
		return n;
	}

	private static byte[] inflateFile(Path p) throws Exception {
		byte[] in = Files.readAllBytes(p);
		Inflater inf = new Inflater();
		inf.setInput(in);
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		byte[] buf = new byte[1 << 16];
		while (!inf.finished()) {
			int k = inf.inflate(buf);
			if (k == 0 && (inf.needsInput() || inf.needsDictionary())) break;
			out.write(buf, 0, k);
		}
		inf.end();
		return out.toByteArray();
	}

	private static void check(boolean ok, String what) {
		if (!ok) throw new AssertionError("FAILED: " + what);
	}
}
