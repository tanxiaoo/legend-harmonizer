"""HRLC local adapter (Stage 1).

Reads the CCI HRLC GeoTIFF from ``data/`` with rasterio. CRS alignment is this
adapter's responsibility: sample coordinates arrive in EPSG:4326 and must be
transformed into the raster's own CRS before reading the pixel value. Fill/
no-data values map to "no label". Getting the transform wrong silently returns
neighbouring-pixel labels, so it is done explicitly here and exercised by the
Stage 1 verification.

See docs/PIPELINE.md, Stage 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from pyproj import Transformer

from harmonizer.config import CONFIG
from harmonizer.registry.products import Coord, LabelAdapter, LabelResult


class HRLCLocalAdapter(LabelAdapter):
    """Reads HRLC class labels from a local GeoTIFF via rasterio.

    The GeoTIFF is discovered in the configured data directory. Coordinates are
    transformed from EPSG:4326 into the raster CRS before pixel lookup. Points
    outside the raster bounds, or reading a nodata value, return a masked
    :class:`LabelResult`.
    """

    name = "hrlc-local"

    def __init__(self, tif_path: str | Path | None = None) -> None:
        self._tif_path = Path(tif_path) if tif_path is not None else self._find_tif()
        # Opened lazily on first sample so constructing the adapter is cheap and
        # never fails just because a file has not been dropped in yet.
        self._dataset = None
        self._transformer: Transformer | None = None

    # ------------------------------------------------------------------ #
    # Discovery / setup
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_tif() -> Path:
        """Locate the single HRLC GeoTIFF in the configured data directory."""
        data_dir: Path = CONFIG.maps.hrlc_data_dir
        candidates = sorted(
            p
            for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
            for p in data_dir.glob(ext)
        )
        if not candidates:
            raise FileNotFoundError(
                f"No HRLC GeoTIFF (*.tif/*.tiff) found in {data_dir}. "
                "Place the CCI HRLC raster there (see docs/PIPELINE.md, section 4)."
            )
        if len(candidates) > 1:
            # Deterministic choice, but warn via the exception message contract:
            # downstream expects exactly one reference raster in the MVP.
            names = ", ".join(p.name for p in candidates)
            raise ValueError(
                f"Multiple GeoTIFFs found in {data_dir} ({names}); "
                "keep exactly one HRLC raster, or pass tif_path explicitly."
            )
        return candidates[0]

    def _ensure_open(self) -> None:
        if self._dataset is not None:
            return
        self._dataset = rasterio.open(self._tif_path)
        # Transform EPSG:4326 -> raster CRS. always_xy keeps (lon, lat) order so
        # inputs and outputs are unambiguously (x=lon, y=lat).
        self._transformer = Transformer.from_crs(
            CONFIG.maps.target_crs,
            self._dataset.crs,
            always_xy=True,
        )

    @property
    def tif_path(self) -> Path:
        return self._tif_path

    @property
    def crs(self):
        self._ensure_open()
        return self._dataset.crs

    @property
    def nodata(self):
        self._ensure_open()
        return self._dataset.nodata

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #

    def sample_labels(self, coords: Sequence[Coord]) -> list[LabelResult]:
        self._ensure_open()
        ds = self._dataset
        nodata = ds.nodata
        results: list[LabelResult] = []

        # Transform all coordinates (lon, lat) -> raster CRS (x, y) up front.
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        xs, ys = self._transformer.transform(lons, lats)

        for x, y in zip(np.atleast_1d(xs), np.atleast_1d(ys)):
            if not np.isfinite(x) or not np.isfinite(y):
                results.append(LabelResult.no_data())
                continue
            # Bounds check in raster CRS before reading.
            left, bottom, right, top = ds.bounds
            if not (left <= x <= right and bottom <= y <= top):
                results.append(LabelResult.no_data())
                continue

            row, col = ds.index(x, y)
            if not (0 <= row < ds.height and 0 <= col < ds.width):
                results.append(LabelResult.no_data())
                continue

            # Read the single pixel from band 1 via a 1x1 window.
            window = rasterio.windows.Window(col_off=col, row_off=row, width=1, height=1)
            value = ds.read(1, window=window)[0, 0]

            if nodata is not None and value == nodata:
                results.append(LabelResult.no_data())
            else:
                results.append(LabelResult.of(int(value)))

        return results

    def close(self) -> None:
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    def __enter__(self) -> "HRLCLocalAdapter":
        self._ensure_open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
