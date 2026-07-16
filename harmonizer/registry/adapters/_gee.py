"""Shared Earth Engine helpers for the GEE adapters (Stage 1).

Keeps the initialisation and the "sample an image at a list of points" plumbing
in one place so the Dynamic World and AlphaEarth adapters stay focused on what
each one composites. Only coordinates and small point tables cross the network;
never rasters.

See docs/PIPELINE.md, Stage 1.
"""

from __future__ import annotations

import os
from typing import Sequence

from harmonizer.config import CONFIG
from harmonizer.registry.products import Coord

# Property name carrying the original input index, so results can be restored to
# input order regardless of how GEE returns the feature collection.
_IDX_PROP = "__idx__"

# ee.Initialize is idempotent but not free (an HTTP handshake); once per process
# is enough, so repeated interactive calls (evidence explorer) skip it.
_INITIALIZED = False


def ensure_initialized() -> None:
    """Initialise Earth Engine, authenticating under the user's own account.

    GEE runs under the user's credentials established once with
    ``earthengine authenticate`` (see docs/PIPELINE.md, section 4). We only call
    ``ee.Initialize`` here; we never pass credentials through code.

    Current GEE accounts require a Google Cloud project. It is read from the
    ``EE_PROJECT`` environment variable if set, otherwise from
    ``CONFIG.maps.gee_project``.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return
    import ee

    project = os.environ.get("EE_PROJECT") or CONFIG.maps.gee_project
    ee.Initialize(project=project)
    _INITIALIZED = True


def points_fc(coords: Sequence[Coord]):
    """Build an ``ee.FeatureCollection`` of points, tagging each with its index."""
    import ee

    features = [
        ee.Feature(
            ee.Geometry.Point([float(lon), float(lat)]),
            {_IDX_PROP: i},
        )
        for i, (lon, lat) in enumerate(coords)
    ]
    return ee.FeatureCollection(features)


def sample_image(image, coords: Sequence[Coord], scale: float) -> list[dict]:
    """Sample ``image`` at ``coords`` and return per-point property dicts.

    Returns a list aligned to the input order. Each element is the ``properties``
    dict of the sampled feature (band values plus ``__idx__``). Points that fall
    on masked/no-data pixels are simply absent from the sampled collection; the
    caller reconstructs a full-length, ordered list and marks the gaps as masked.
    """
    fc = points_fc(coords)
    sampled = image.sampleRegions(
        collection=fc,
        scale=scale,
        geometries=False,
        tileScale=4,
    )
    info = sampled.getInfo()
    props_by_idx: dict[int, dict] = {}
    for feat in info.get("features", []):
        props = feat.get("properties", {})
        idx = props.get(_IDX_PROP)
        if idx is not None:
            props_by_idx[int(idx)] = props

    # Preserve None gaps for masked points so the caller can mask them in order.
    return [props_by_idx.get(i) for i in range(len(coords))]
