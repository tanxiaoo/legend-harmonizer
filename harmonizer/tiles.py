"""Label-map tiles and class legends for the comparison map (Stage 5.2).

Stage 5.2 draws the two selected label maps side by side on a Leaflet split map,
each with per-class show/hide toggles. Leaflet needs *raster tiles*, so this
module asks Earth Engine to render the annual label image as tiles and hands the
frontend a standard XYZ tile-URL template (``getMapId``). It also exposes each
product's published legend -- the class values, human-readable names, and the
official palette colours -- so the frontend can build the toggle list and colour
swatches, and so a tile request can restrict rendering to a chosen subset of
classes.

This is a *display-only* layer. It reuses the same annual label images the Stage 1
adapters sample (WorldCover ``.first()``; Dynamic World per-pixel modal over the
working year), so what the user sees matches what was sampled. Only tile URLs and
small legend tables cross the network here; the rasters stay on Earth Engine and
are streamed as tiles straight to the browser, never through this process.

The published palettes and class names are properties of each product's legend,
read from the product registry (the per-map YAML files, docs/PIPELINE.md section
2.5) -- the single source of truth. They are legend metadata, not tunable pipeline
constants, so they live with the product (its YAML) rather than in ``config`` or
hardcoded here.

See docs/PIPELINE.md, Stage 5.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from harmonizer.config import CONFIG
from harmonizer.registry.adapters import _gee
from harmonizer.registry.legends import legend_classes, spec as _product_spec

# Products the tile renderer knows how to draw (the two MVP label maps). Their
# metadata and legends come from the registry; this set only bounds what is
# drawable (AlphaEarth has no legend, HRLC is local and not tiled here).
_DRAWABLE = ("worldcover", "dynamicworld")


@dataclass(frozen=True)
class LegendEntry:
    value: int
    name: str
    color: str  # 6-digit hex, no leading '#'
    description: str | None = None  # human-readable class definition (registry)


def _label_band(product_id: str) -> str:
    """The name of the discrete class band the product renders on (from registry)."""
    s = _product_spec(product_id)
    if s is None or product_id not in _DRAWABLE or s.band is None:
        raise KeyError(f"no tile support for product: {product_id}")
    return str(s.band)


def legend(product_id: str) -> list[LegendEntry]:
    """The product's class legend: value, readable name, and palette colour.

    Read from the product registry (per-map YAML), in declared order. Raises
    ``KeyError`` for a product with no drawable label legend (only the two MVP
    label maps are drawable). Colours are stored ``#RRGGBB`` in the registry and
    returned here as 6-digit hex without the leading ``#`` (Earth Engine's
    palette format).
    """
    if product_id not in _DRAWABLE:
        raise KeyError(f"no tile support for product: {product_id}")
    classes = legend_classes(product_id)
    if not classes:
        raise KeyError(f"no tile support for product: {product_id}")
    return [
        LegendEntry(
            value=c.code,
            name=c.name,
            color=c.color.lstrip("#"),
            description=c.description,
        )
        for c in classes
    ]


def _annual_label_image(product_id: str):
    """The same annual label image the Stage 1 adapters sample, for rendering.

    WorldCover is the single static image (``.first()``); Dynamic World is the
    per-pixel modal label over the working year. Built lazily so importing this
    module never requires Earth Engine.
    """
    import ee

    s = _product_spec(product_id)
    if s is None or product_id not in _DRAWABLE:
        raise KeyError(f"no tile support for product: {product_id}")
    asset = s.access.asset_id
    band = str(s.band)

    if product_id == "worldcover":
        return ee.ImageCollection(asset).first().select(band)
    if product_id == "dynamicworld":
        year = CONFIG.maps.working_year
        collection = (
            ee.ImageCollection(asset)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .select(band)
        )
        return collection.reduce("mode").rename(band)
    raise KeyError(f"no tile support for product: {product_id}")


def tile_template(
    product_id: str, visible_values: Sequence[int] | None = None
) -> str:
    """An XYZ tile-URL template for the product's label map, for Leaflet.

    ``visible_values`` optionally restricts rendering to a subset of class values
    (the per-class show/hide toggles): pixels of any other class are masked out so
    only the requested classes are painted, letting the base map show through.
    ``None`` renders every class. The returned string is a standard
    ``.../{z}/{x}/{y}`` template Leaflet's ``L.tileLayer`` consumes directly; the
    rasters never pass through this process.
    """
    _gee.ensure_initialized()
    import ee

    image = _annual_label_image(product_id)
    entries = legend(product_id)

    if visible_values is not None:
        keep = set(int(v) for v in visible_values)
        if not keep:
            # Nothing selected: mask the whole layer so the map shows only the base.
            image = image.updateMask(ee.Image(0))
        else:
            # Keep only pixels whose class is in the selected set.
            band = _label_band(product_id)
            mask = image.remap(
                list(keep), [1] * len(keep), 0, band
            )
            image = image.updateMask(mask)

    values = [e.value for e in entries]
    palette = [e.color for e in entries]
    # A palette maps a continuous [min,max] ramp, but label values are sparse and
    # unevenly spaced, so remap to dense 0..N-1 indices before applying the palette
    # -- otherwise WorldCover's 10..100 codes would smear colours across the ramp.
    band = _label_band(product_id)
    dense = image.remap(values, list(range(len(values))), bandName=band)
    vis = {"min": 0, "max": len(values) - 1, "palette": palette}

    map_id = ee.Image(dense).getMapId(vis)
    return map_id["tile_fetcher"].url_format
