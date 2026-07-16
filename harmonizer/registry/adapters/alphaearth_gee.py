"""AlphaEarth adapter (Stage 1).

Reads from GEE, sampling the 64-dim AlphaEarth embedding at the given points for
the working year. Points whose embedding is masked/no-data are flagged so they
can be dropped later. Only coordinates and small point tables cross the network;
never rasters.

See docs/PIPELINE.md, Stage 1.
"""

from __future__ import annotations

from typing import Sequence

from harmonizer.config import CONFIG
from harmonizer.registry.adapters import _gee
from harmonizer.registry.legends import spec as _product_spec
from harmonizer.registry.products import Coord, EmbeddingAdapter, EmbeddingResult

# The 64 embedding bands are named A00..A63.
_BAND_NAMES = tuple(f"A{i:02d}" for i in range(64))


class AlphaEarthAdapter(EmbeddingAdapter):
    """Samples the 64-dim AlphaEarth embedding at coordinates via GEE.

    Asset id and sampling scale come from the product registry
    (``alphaearth.yaml``, the single source of truth), not from config.
    """

    name = "alphaearth-gee"
    product_id = "alphaearth"

    def __init__(self) -> None:
        self._image = None  # built lazily so import/construction need no network
        self._spec = _product_spec(self.product_id)

    @property
    def _scale_m(self) -> float:
        return float(self._spec.resolution_m or 10.0)

    def _annual_image(self):
        """Select the working-year AlphaEarth embedding image (64 bands)."""
        import ee

        year = CONFIG.maps.working_year
        start = f"{year}-01-01"
        end = f"{year + 1}-01-01"

        collection = ee.ImageCollection(self._spec.access.asset_id).filterDate(
            start, end
        )
        # The annual embedding is one image per year; mosaic guards against the
        # AOI straddling tile boundaries. Band order is fixed as A00..A63.
        return collection.mosaic().select(list(_BAND_NAMES))

    def _ensure_image(self):
        if self._image is None:
            _gee.ensure_initialized()
            self._image = self._annual_image()
        return self._image

    def sample_embeddings(self, coords: Sequence[Coord]) -> list[EmbeddingResult]:
        image = self._ensure_image()
        rows = _gee.sample_image(image, coords, scale=self._scale_m)

        results: list[EmbeddingResult] = []
        for props in rows:
            if props is None:
                results.append(EmbeddingResult.no_data())
                continue
            # Require all 64 bands present and non-null; otherwise treat as masked.
            vector = [props.get(b) for b in _BAND_NAMES]
            if any(v is None for v in vector):
                results.append(EmbeddingResult.no_data())
            else:
                results.append(EmbeddingResult.of(vector))
        return results
