"""Minimal PNG writer: 8-bit RGB, adaptive filtering, zlib for deflate."""

import struct
import zlib

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, payload: bytes) -> bytes:
    """length + type + data + crc32(type + data)"""
    crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def _filter_scanlines(rgb: np.ndarray) -> bytes:
    """Per-scanline adaptive filtering (None/Sub/Up/Average/Paeth).

    Picks the filter with the smallest sum of abs signed bytes per row,
    the usual libpng heuristic. Filters read unfiltered neighbours, so all
    five candidates for a row are computed at once.
    """
    height, width, _ = rgb.shape
    stride = width * 3  # bytes per scanline
    rows = rgb.reshape(height, stride).astype(np.int16)
    prior = np.zeros(stride, dtype=np.int16)
    out = bytearray()

    for y in range(height):
        cur = rows[y]
        left = np.concatenate((np.zeros(3, np.int16), cur[:-3]))       # a
        up = prior                                                      # b
        upleft = np.concatenate((np.zeros(3, np.int16), prior[:-3]))    # c

        # paeth: whichever of a, b, c is closest to a + b - c
        p = left + up - upleft
        pa, pb, pc = np.abs(p - left), np.abs(p - up), np.abs(p - upleft)
        paeth = np.where(
            (pa <= pb) & (pa <= pc), left, np.where(pb <= pc, up, upleft)
        )

        candidates = (
            cur,                                # 0 None
            cur - left,                         # 1 Sub
            cur - up,                           # 2 Up
            cur - ((left + up) >> 1),           # 3 Average
            cur - paeth,                        # 4 Paeth
        )

        best, best_cost = 0, None
        for idx, cand in enumerate(candidates):
            band = cand.astype(np.uint8).astype(np.int16)
            # treat bytes as signed, so 255 counts as -1
            cost = int(np.minimum(band, 256 - band).sum())
            if best_cost is None or cost < best_cost:
                best, best_cost = idx, cost

        out.append(best)
        out += candidates[best].astype(np.uint8).tobytes()
        prior = cur

    return bytes(out)


def write_png(path: str, rgb: np.ndarray) -> int:
    """Write an (H, W, 3) uint8 array as an 8-bit RGB PNG. Returns file size."""
    height, width, _ = rgb.shape
    ihdr = struct.pack(
        ">IIBBBBB",
        width, height,
        8,   # bit depth
        2,   # colour type 2 = truecolour RGB
        0,   # compression method (deflate, only option)
        0,   # filter method 0
        0,   # no interlacing
    )

    compressor = zlib.compressobj(level=9, strategy=zlib.Z_FILTERED)
    idat = compressor.compress(_filter_scanlines(rgb)) + compressor.flush()

    blob = (
        PNG_SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"gAMA", struct.pack(">I", 45455))   # gamma 1/2.2
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )
    with open(path, "wb") as fh:
        fh.write(blob)
    return len(blob)
