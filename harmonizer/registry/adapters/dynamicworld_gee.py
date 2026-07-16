"""Dynamic World adapter (Stage 1).

Reads from GEE. Dynamic World is per-scene, so this adapter first composites the
working-year collection to a single annual label by taking the modal class per
pixel over the year, then samples the label at the given points. Only
coordinates and small point tables cross the network; never rasters.

See docs/PIPELINE.md, Stage 1.
"""

from __future__ import annotations

from typing import Sequence

from harmonizer.config import CONFIG
from harmonizer.registry.adapters import _gee
from harmonizer.registry.legends import spec as _product_spec
from harmonizer.registry.products import Coord, LabelAdapter, LabelResult


class DynamicWorldAdapter(LabelAdapter):
    """Samples the annual modal Dynamic World class at coordinates via GEE.

    Asset id, label band, and sampling scale come from the product registry
    (``dynamicworld.yaml``, the single source of truth), not from config. The
    working year is a run-level setting and stays in config.
    """

    name = "dynamicworld-gee"
    product_id = "dynamicworld"

    def __init__(self) -> None:
        self._image = None  # built lazily so import/construction need no network
        self._spec = _product_spec(self.product_id)

    @property
    def _band(self) -> str:
        return str(self._spec.band)

    @property
    def _scale_m(self) -> float:
        return float(self._spec.resolution_m or 10.0)

    def _annual_modal_image(self):
        """Composite the working-year DW collection to a per-pixel modal label."""
        import ee

        year = CONFIG.maps.working_year
        start = f"{year}-01-01"
        end = f"{year + 1}-01-01"
        band = self._band

        collection = (
            ee.ImageCollection(self._spec.access.asset_id)
            .filterDate(start, end)
            .select(band)
        )
        # Per-pixel modal class over the year -> a single annual label image.
        return collection.reduce(ee.Reducer.mode()).rename(band)

    def _ensure_image(self):
        if self._image is None:
            _gee.ensure_initialized()
            self._image = self._annual_modal_image()
        return self._image

    def sample_labels(self, coords: Sequence[Coord]) -> list[LabelResult]:
        image = self._ensure_image()
        rows = _gee.sample_image(image, coords, scale=self._scale_m)

        band = self._band
        results: list[LabelResult] = []
        for props in rows:
            if props is None or band not in props or props[band] is None:
                results.append(LabelResult.no_data())
            else:
                results.append(LabelResult.of(int(props[band])))
        return results
