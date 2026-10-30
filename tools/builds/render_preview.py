"""Writes one PNG per blueprint to data/builds/previews/<name>.png (layer grid).

Each Y layer is drawn as a top-down z-by-x grid, bottom layer on the left, so a house reads as a
stack of floors. Pure stdlib (zlib + struct); PIL is not required.
Usage: .venv/bin/python tools/builds/render_preview.py [name ...]
"""
import json
import os
import struct
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BUILDS = os.path.join(ROOT, "data", "builds", "blueprints.json")
OUTDIR = os.path.join(ROOT, "data", "builds", "previews")

BG = (30, 32, 36)
GRID = (20, 20, 22)
COLORS = {
    "oak_planks": (176, 132, 78),
    "oak_log": (98, 70, 40),
    "cobblestone": (125, 125, 128),
    "dirt": (134, 96, 60),
    "sand": (222, 208, 150),
    "gravel": (140, 134, 130),
    "glass": (150, 210, 230),
    "obsidian": (30, 18, 46),
    "furnace": (90, 90, 96),
    "crafting_table": (150, 100, 60),
    "chest": (170, 120, 50),
}


def color_of(item):
    if item in COLORS:
        return COLORS[item]
    h = 0
    for ch in item:
        h = (h * 31 + ord(ch)) & 0xFFFFFF
    return (60 + h % 160, 60 + (h >> 8) % 160, 60 + (h >> 16) % 160)


def write_png(path, w, h, pixels):
    """pixels: bytearray of w*h*3 RGB."""
    raw = bytearray()
    stride = w * 3
    for y in range(h):
        raw.append(0)
        raw += pixels[y * stride:(y + 1) * stride]

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def render(bp, cell=None):
    sx, sy, sz = bp["size"]
    layers = bp["layers"]
    pal = bp["palette"]
    gap = 8
    if cell is None:
        total = sy * sx + (sy - 1) * 2
        cell = 14 if total <= 60 else (10 if total <= 120 else 6)
    width = sy * sx * cell + (sy - 1) * gap + 2 * gap
    height = sz * cell + 2 * gap
    px = bytearray(BG * (width * height))

    def put(x0, y0, w, h, rgb):
        for yy in range(y0, y0 + h):
            row = (yy * width + x0) * 3
            for xx in range(w):
                px[row + xx * 3:row + xx * 3 + 3] = bytes(rgb)

    for y in range(sy):
        ox = gap + y * (sx * cell + gap)
        oy = gap
        for z in range(sz):
            for x in range(sx):
                c = layers[y][z][x]
                if c == ".":
                    continue
                rgb = color_of(pal[c])
                # inner cell with a 1px grid line on the right and bottom edge
                put(ox + x * cell, oy + z * cell, cell - 1, cell - 1, rgb)
                put(ox + x * cell + cell - 1, oy + z * cell, 1, cell, GRID)
                put(ox + x * cell, oy + z * cell + cell - 1, cell, 1, GRID)
        # faint frame around each layer so empty air is visible
        put(ox, oy, sx * cell, 1, (70, 72, 78))
        put(ox, oy + sz * cell - 1, sx * cell, 1, (70, 72, 78))
        put(ox, oy, 1, sz * cell, (70, 72, 78))
        put(ox + sx * cell - 1, oy, 1, sz * cell, (70, 72, 78))
    return width, height, px


def main(argv):
    with open(BUILDS) as f:
        bps = json.load(f)
    want = set(argv)
    os.makedirs(OUTDIR, exist_ok=True)
    n = 0
    for bp in bps:
        if want and bp["name"] not in want:
            continue
        w, h, px = render(bp)
        write_png(os.path.join(OUTDIR, bp["name"] + ".png"), w, h, px)
        n += 1
    print("wrote %d previews to %s" % (n, OUTDIR))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
