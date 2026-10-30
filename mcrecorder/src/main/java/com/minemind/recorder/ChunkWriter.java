package com.minemind.recorder;

import java.io.BufferedOutputStream;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.zip.Deflater;
import java.util.zip.DeflaterOutputStream;

/**
 * PROTOCOL 9.2 chunk storage. Frames are grouped in chunks of CHUNK_FRAMES; a chunk is one zlib (Deflater) stream
 * of fixed-size little-endian records:
 * img u8[64*64*2], vec f32[102], action i32, ext_action i32  (FRAME_BYTES = 8608 uncompressed).
 * The chunk being written is chunks/NNNNNN.bin.tmp (sync-flushed, so completed deflate blocks are readable after a
 * crash); on completion it is finished and renamed to chunks/NNNNNN.bin and one line is appended to
 * chunks/index.jsonl: {"chunk", "first_tick", "frames", "bytes"}.
 * Not thread-safe: owned by the writer thread.
 */
final class ChunkWriter {
	static final int RES = 64;
	static final int VEC_DIM = 102;
	static final int IMG_BYTES = RES * RES * 2;
	static final int FRAME_BYTES = IMG_BYTES + VEC_DIM * 4 + 8;
	static final int CHUNK_FRAMES = 1200;

	private final Path chunkDir;
	private final int chunkFrames;
	private final BufferedWriter index;
	private final ByteBuffer rec = ByteBuffer.allocate(FRAME_BYTES).order(ByteOrder.LITTLE_ENDIAN);
	private int nextChunk = 0;
	private long framesTotal = 0;
	private int framesInChunk = 0;
	private long chunkFirstTick = -1;
	private Path tmp;
	private Deflater deflater;
	private DeflaterOutputStream out;

	ChunkWriter(Path sessionDir, int chunkFrames) throws IOException {
		this.chunkDir = sessionDir.resolve("chunks");
		this.chunkFrames = chunkFrames;
		Files.createDirectories(chunkDir);
		this.index = Files.newBufferedWriter(chunkDir.resolve("index.jsonl"), StandardCharsets.UTF_8,
			StandardOpenOption.CREATE, StandardOpenOption.APPEND);
	}

	long framesTotal() { return framesTotal; }

	int chunksWritten() { return nextChunk; }

	void append(long tick, byte[] img, float[] vec, int action, int extAction) throws IOException {
		if (out == null) open(tick);
		rec.clear();
		rec.put(img, 0, IMG_BYTES);
		for (int i = 0; i < VEC_DIM; i++) rec.putFloat(vec[i]);
		rec.putInt(action);
		rec.putInt(extAction);
		out.write(rec.array(), 0, FRAME_BYTES);
		framesInChunk++;
		framesTotal++;
		if (framesInChunk >= chunkFrames) closeChunk();
	}

	/** Sync-flushes the open chunk and the index so everything written so far is readable. */
	void flush() throws IOException {
		if (out != null) out.flush();
		index.flush();
	}

	/** Completes the open chunk (partial chunks included) and closes the index. */
	void finish() throws IOException {
		try {
			closeChunk();
		} finally {
			index.close();
		}
	}

	private void open(long tick) throws IOException {
		tmp = chunkDir.resolve(String.format("%06d.bin.tmp", nextChunk));
		deflater = new Deflater(Deflater.DEFAULT_COMPRESSION, false);
		out = new DeflaterOutputStream(new BufferedOutputStream(new FileOutputStream(tmp.toFile()), 1 << 16),
			deflater, 1 << 16, true);
		chunkFirstTick = tick;
		framesInChunk = 0;
	}

	private void closeChunk() throws IOException {
		if (out == null) return;
		try {
			out.finish();
			out.close();
		} finally {
			deflater.end();
			out = null;
			deflater = null;
		}
		Path fin = chunkDir.resolve(String.format("%06d.bin", nextChunk));
		Files.move(tmp, fin, StandardCopyOption.REPLACE_EXISTING);
		long bytes = Files.size(fin);
		index.write("{\"chunk\":" + nextChunk + ",\"first_tick\":" + chunkFirstTick + ",\"frames\":" + framesInChunk
			+ ",\"bytes\":" + bytes + "}\n");
		index.flush();
		nextChunk++;
	}
}
