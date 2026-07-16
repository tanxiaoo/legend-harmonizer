"""ESA WorldCover adapter (Stage 1, test-swap reference).

Reads from GEE. WorldCover v200 is a single global static map for 2021, exposed
as an ImageCollection with one image; this adapter takes ``.first()`` and samples
the discrete ``Map`` band at the given points. It stands in for the local HRLC
reference so testing needs no download (see docs/PIPELINE.md, section 2, the
test-swap note). Only coordinates and small point tables cross the network;
never rasters.

See docs/PIPELINE.md, Stage 1.
"""

from __future__ import annotations

from typing import Sequence

from harmonizer.registry.adapters import _gee
from harmonizer.registry.legends import spec as _product_spec
from harmonizer.registry.products import Coord, LabelAdapter, LabelResult


class WorldCoverAdapter(LabelAdapter):
    """Samples the ESA WorldCover v200 class at coordinates via GEE.

    Asset id, band, and sampling scale come from the product registry
    (``worldcover.yaml``, the single source of truth), not from config.
    """

    name = "worldcover-gee"
    product_id = "worldcover"

    def __init__(self) -> None:
        self._image = None  # built lazily so import/construction need no network
        self._spec = _product_spec(self.product_id)

    @property
    def _band(self) -> str:
        return str(self._spec.band)

    @property
    def _scale_m(self) -> float:
        return float(self._spec.resolution_m or 10.0)

    def _static_image(self):
        """Select the single global WorldCover v200 image and its Map band."""
        import ee

        return (
            ee.ImageCollection(self._spec.access.asset_id)
            .first()
            .select(self._band)
        )

    def _ensure_image(self):
        if self._image is None:
            _gee.ensure_initialized()
            self._image = self._static_image()
        return self._image

    def sample_labels(self, coords: Sequence[Coord]) -> list[LabelResult]:
        image = self._ensure_image()
        rows = _gee.sample_image(image, coords, scale=self._scale_m)

        band = self._band
        results: list[LabelResult] = []
        for props in rows:
            if props is None or props.get(band) is None:
                results.append(LabelResult.no_data())
            else:
                results.append(LabelResult.of(int(props[band])))
        return results
