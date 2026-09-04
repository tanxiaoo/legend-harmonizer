"""Fast truncation check for downloaded GeoTIFFs.

Answers "is this file complete?" in milliseconds, by comparing what the TIFF's
own tile/strip directory says the file contains against how many bytes are
actually on disk. A truncated download declares tile N at byte offset X with
length L where X+L runs past the end of the file.

**Why this exists rather than just using ``indexer --check``.** That probe reads
a grid of windows through GDAL and takes ~15 minutes for a large dataset,
because it decompresses real pixels. This reads only the header and the offset
tables -- no pixel data -- so it scans hundreds of files in under a second. Use
this first; use ``--check`` when you need to know *which regions* are unreadable
rather than *whether* the file is whole.

It also reports the giveaway a truncated batch leaves behind: several files of
**identical byte length**. Independently compressed rasters of different regions
essentially never land on the same size, so a repeated size is a download that
was cut off at a common point (a disk that filled, a session that expired, a
transfer killed mid-flight), not a coincidence.

Run::

    python scripts/check_downloads.py                    # every folder in data/
    python scripts/check_downloads.py data/GLC_FCS30D_2019
    python scripts/check_downloads.py --delete           # remove truncated files
"""

from __future__ import annotations

import argparse
import collections
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Tags naming where pixel data lives and how long it is.
_TILE_OFFSETS, _STRIP_OFFSETS = 324, 273
_TILE_BYTES, _STRIP_BYTES = 325, 279
_TYPE_SIZE = {1: 1, 3: 2, 4: 4, 16: 8, 17: 8, 18: 8}
_TYPE_FMT = {1: "B", 3: "H", 4: "I", 16: "Q", 17: "q", 18: "Q"}


def declared_extent(path: Path) -> int:
    """The highest byte offset the TIFF's own directories reference.

    Walks every IFD (so overviews count too) and returns ``max(offset + length)``
    over all tile/strip entries. Handles both classic TIFF and BigTIFF, and
    either byte order.
    """
    with open(path, "rb") as fh:
        header = fh.read(16)
        if len(header) < 8:
            raise ValueError("file too short to be a TIFF")
        byte_order = "<" if header[:2] == b"II" else ">"
        version = struct.unpack(byte_order + "H", header[2:4])[0]
        big = version == 43
        if big:
            offset = struct.unpack(byte_order + "Q", header[8:16])[0]
        else:
            offset = struct.unpack(byte_order + "I", header[4:8])[0]

        furthest = 0
        visited: set[int] = set()
        while offset and offset not in visited:
            visited.add(offset)
            fh.seek(offset)
            if big:
                count = struct.unpack(byte_order + "Q", fh.read(8))[0]
                entry_size = 20
            else:
                count = struct.unpack(byte_order + "H", fh.read(2))[0]
                entry_size = 12
            entries = fh.read(count * entry_size)

            offsets = lengths = None
            for i in range(count):
                entry = entries[i * entry_size : (i + 1) * entry_size]
                tag = struct.unpack(byte_order + "H", entry[:2])[0]
                if tag not in (_TILE_OFFSETS, _STRIP_OFFSETS, _TILE_BYTES, _STRIP_BYTES):
                    continue
                typ = struct.unpack(byte_order + "H", entry[2:4])[0]
                if big:
                    n = struct.unpack(byte_order + "Q", entry[4:12])[0]
                    inline = entry[12:20]
                else:
                    n = struct.unpack(byte_order + "I", entry[4:8])[0]
                    inline = entry[8:12]

                total = n * _TYPE_SIZE.get(typ, 4)
                if total <= len(inline):
                    raw = inline[:total]
                else:
                    ptr = struct.unpack(
                        byte_order + ("Q" if big else "I"),
                        inline[: 8 if big else 4],
                    )[0]
                    here = fh.tell()
                    fh.seek(ptr)
                    raw = fh.read(total)
                    fh.seek(here)
                values = struct.unpack(byte_order + _TYPE_FMT.get(typ, "I") * n, raw)
                if tag in (_TILE_OFFSETS, _STRIP_OFFSETS):
                    offsets = values
                else:
                    lengths = values

            if offsets and lengths:
                for off, length in zip(offsets, lengths):
                    if off:
                        furthest = max(furthest, off + length)

            fh.seek(offset + (8 if big else 2) + count * entry_size)
            nxt = fh.read(8 if big else 4)
            if len(nxt) < (8 if big else 4):
                break
            offset = struct.unpack(byte_order + ("Q" if big else "I"), nxt)[0]
    return furthest


def check_folder(folder: Path, delete: bool = False) -> tuple[int, int]:
    """Report truncated files in one folder. Returns ``(n_bad, n_total)``."""
    tifs = sorted(p for p in folder.glob("*.tif") if not p.name.endswith(".aux.xml"))
    if not tifs:
        return 0, 0

    bad: list[tuple[Path, int, int]] = []
    sizes: collections.Counter[int] = collections.Counter()
    for path in tifs:
        size = path.stat().st_size
        sizes[size] += 1
        try:
            need = declared_extent(path)
        except Exception as exc:
            print(f"    {path.name}: unreadable header ({type(exc).__name__}: {exc})")
            bad.append((path, size, -1))
            continue
        if need > size:
            bad.append((path, size, need))

    print(f"\n{folder.name}: {len(tifs)} file(s)")
    for path, size, need in bad:
        if need < 0:
            continue
        print(
            f"    TRUNCATED {path.name}\n"
            f"              has {size:,} bytes, needs {need:,} "
            f"(missing {need - size:,})"
        )

    # The batch signature: several files cut to the same length.
    repeated = [(s, c) for s, c in sizes.items() if c > 1]
    for size, count in sorted(repeated, key=lambda x: -x[1]):
        print(
            f"    !! {count} files share the exact size {size:,} bytes -- "
            f"independently compressed rasters do not do that. This is the "
            f"signature of a download batch cut off at a common point."
        )

    if bad:
        print(f"    => {len(bad)} of {len(tifs)} file(s) incomplete; re-download these.")
        if delete:
            for path, _size, _need in bad:
                try:
                    path.unlink()
                    print(f"    deleted {path.name}")
                except OSError as exc:
                    print(f"    could not delete {path.name}: {exc}")
    else:
        print("    all complete")
    return len(bad), len(tifs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folders", nargs="*", help="folders to check (default: all of data/)")
    ap.add_argument(
        "--delete",
        action="store_true",
        help="delete truncated files so a re-download replaces them",
    )
    args = ap.parse_args()

    from harmonizer.config import CONFIG

    if args.folders:
        folders = [Path(f) for f in args.folders]
    else:
        folders = sorted(
            p for p in CONFIG.data_dir.iterdir() if p.is_dir() and p.name != "legend"
        )

    total_bad = total_files = 0
    for folder in folders:
        if not folder.is_dir():
            print(f"{folder}: not a directory")
            continue
        bad, n = check_folder(folder, delete=args.delete)
        total_bad += bad
        total_files += n

    print(f"\n{'=' * 70}")
    if total_bad:
        print(f"{total_bad} of {total_files} file(s) are incomplete downloads.")
        return 1
    print(f"All {total_files} file(s) are complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
