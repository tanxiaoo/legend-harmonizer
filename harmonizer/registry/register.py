"""Registration flow and reconciliation check for the product registry.

Two operations over the per-map YAML registry (docs/PIPELINE.md, section 2.5):

* **Registration** (:func:`build_registration_yaml`) inspects a source and
  pre-fills a YAML skeleton: everything derivable from the source is filled in
  automatically (CRS, footprint bounds, resolution, the class codes actually
  present, and names/colours from a GeoTIFF colour table or GEE asset properties
  where they exist), and the rest is left as clearly-marked ``TODO`` placeholders
  for a human to complete or confirm. Automatic where the source allows, prompted
  where it does not.

* **Reconciliation** (:func:`reconcile`) compares a map's declared legend against
  the class codes actually observed in the source over an AOI, and reports codes
  present in the raster but missing from the legend (**undeclared**) and legend
  classes not observed anywhere in the AOI (the **absent-in-AOI** signal).

Registration touches the source (rasterio / GEE) and so imports those lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from harmonizer.registry.schema import ProductSpec, load_product_file


# --------------------------------------------------------------------------- #
# Reconciliation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Reconciliation:
    """The result of comparing a declared legend against observed class codes.

    ``undeclared`` are codes seen in the raster/label band but missing from the
    legend (the YAML is incomplete). ``absent_in_aoi`` are legend classes declared
    in the YAML but not observed anywhere in the AOI -- the registry-level
    "absent in AOI" signal that a class is legitimately part of the legend but has
    no pixels in this run's area. ``matched`` are codes present in both.
    """

    product_id: str
    declared_codes: tuple[int, ...]
    observed_codes: tuple[int, ...]
    undeclared: tuple[int, ...]
    absent_in_aoi: tuple[int, ...]
    matched: tuple[int, ...]

    @property
    def ok(self) -> bool:
        """True when the declared legend covers every observed code."""
        return not self.undeclared


def reconcile_codes(
    spec: ProductSpec, observed_codes) -> Reconciliation:
    """Reconcile a spec's declared legend against an already-observed code set."""
    declared = set(spec.class_codes)
    observed = set(int(c) for c in observed_codes)
    return Reconciliation(
        product_id=spec.id,
        declared_codes=tuple(sorted(declared)),
        observed_codes=tuple(sorted(observed)),
        undeclared=tuple(sorted(observed - declared)),
        absent_in_aoi=tuple(sorted(declared - observed)),
        matched=tuple(sorted(declared & observed)),
    )


def observe_codes(spec: ProductSpec, aoi=None) -> list[int]:
    """The distinct class codes actually present in the source over ``aoi``.

    ``aoi`` is a ``(min_lon, min_lat, max_lon, max_lat)`` box in EPSG:4326, or
    ``None`` (whole source). Dispatches on the access method: a GEE map is reduced
    server-side (reusing Stage 2's ``present_classes``); a local raster is scanned
    with rasterio.
    """
    if spec.access.method == "gee":
        return _observe_codes_gee(spec, aoi)
    if spec.access.method == "local_raster":
        return _observe_codes_local(spec, aoi)
    raise ValueError(f"unknown access method for {spec.id!r}: {spec.access.method!r}")


def _observe_codes_gee(spec: ProductSpec, aoi) -> list[int]:
    import ee

    from harmonizer.registry.adapters import _gee
    from harmonizer.sampling import _label_image_for, present_classes

    _gee.ensure_initialized()
    image = _label_image_for(spec.id)
    if aoi is not None:
        region = ee.Geometry.Rectangle(list(aoi))
    else:
        region = image.geometry()
    return present_classes(image.clip(region), region)


def _observe_codes_local(spec: ProductSpec, aoi) -> list[int]:
    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds
    from pyproj import Transformer

    path = _resolve_local_path(spec)
    with rasterio.open(path) as ds:
        band = int(spec.band) if spec.band is not None else 1
        if aoi is not None:
            # Transform the EPSG:4326 AOI into the raster CRS and window-read it.
            transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
            xs, ys = transformer.transform(
                [aoi[0], aoi[2]], [aoi[1], aoi[3]]
            )
            window = from_bounds(
                min(xs), min(ys), max(xs), max(ys), transform=ds.transform
            )
            data = ds.read(band, window=window)
        else:
            data = ds.read(band)
        values = np.unique(data)
        if ds.nodata is not None:
            values = values[values != ds.nodata]
        return sorted(int(v) for v in values)


def reconcile(spec: ProductSpec, aoi=None) -> Reconciliation:
    """Observe the source's codes over ``aoi`` and reconcile against the legend."""
    return reconcile_codes(spec, observe_codes(spec, aoi))


def reconcile_file(path: str | Path, aoi=None) -> Reconciliation:
    """Load a registry YAML file and reconcile it against its source."""
    return reconcile(load_product_file(Path(path)), aoi)


# --------------------------------------------------------------------------- #
# Registration (auto-detect, then confirm by hand)
# --------------------------------------------------------------------------- #

# Quoted so the generated skeleton is valid YAML even though the text contains a
# colon; the value round-trips to the string below for a human to replace.
_TODO = '"TODO: complete or confirm by hand"'


@dataclass
class Detected:
    """What the registration flow could derive automatically from the source.

    Fields left ``None`` are the ones the source did not provide -- those become
    ``TODO`` placeholders in the generated YAML for a human to fill in.
    """

    crs: str | None = None
    resolution_m: float | None = None
    footprint: tuple[float, float, float, float] | None = None
    class_codes: list[int] = field(default_factory=list)
    # code -> name / colour where the source published them (colour table / GEE
    # asset properties); missing where it did not.
    class_names: dict[int, str] = field(default_factory=dict)
    class_colors: dict[int, str] = field(default_factory=dict)


#: Reading a whole band at full resolution is fine for a single tile (HRLC-scale,
#: a few billion pixels at most) but a mosaicked VRT over a whole continent can be
#: two orders of magnitude larger and OOMs a full read. Above this many pixels,
#: scan a decimated (overview-resolution) copy instead -- code *detection* only
#: needs to see every value that exists somewhere, not read every pixel, and a
#: missed rare code still surfaces later as "undeclared" by the reconciliation
#: check rather than silently disappearing.
_FULL_READ_PIXEL_LIMIT = 50_000_000


def detect_local_raster(path: str | Path, band: int = 1) -> Detected:
    """Auto-detect CRS, footprint, resolution, codes, and any embedded colour
    table / category names from a local GeoTIFF."""
    import numpy as np
    import rasterio
    from pyproj import Transformer

    det = Detected()
    with rasterio.open(path) as ds:
        det.crs = str(ds.crs) if ds.crs is not None else None
        # Resolution in metres if the raster is projected in metres; otherwise the
        # source doesn't cleanly give metres, so leave it for a human.
        try:
            det.resolution_m = float(abs(ds.transform.a))
        except Exception:  # pragma: no cover - defensive
            det.resolution_m = None
        # Footprint bounds reprojected to EPSG:4326.
        try:
            transformer = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
            left, bottom, right, top = ds.bounds
            xs, ys = transformer.transform([left, right], [bottom, top])
            det.footprint = (min(xs), min(ys), max(xs), max(ys))
        except Exception:  # pragma: no cover - defensive
            det.footprint = None
        # Class codes actually present. Full read for a normal-sized source;
        # decimated for a continent-scale mosaic (see _FULL_READ_PIXEL_LIMIT).
        total_pixels = ds.width * ds.height
        if total_pixels > _FULL_READ_PIXEL_LIMIT:
            decimation = (total_pixels / _FULL_READ_PIXEL_LIMIT) ** 0.5
            out_shape = (1, max(1, int(ds.height / decimation)), max(1, int(ds.width / decimation)))
            data = ds.read(band, out_shape=out_shape, resampling=rasterio.enums.Resampling.nearest)
        else:
            data = ds.read(band)
        vals = np.unique(data)
        if ds.nodata is not None:
            vals = vals[vals != ds.nodata]
        det.class_codes = sorted(int(v) for v in vals)
        # Embedded colour table -> per-class colours, if the GeoTIFF carries one.
        try:
            cmap = ds.colormap(band)  # {value: (r,g,b,a)}
            for code in det.class_codes:
                if code in cmap:
                    r, g, b = cmap[code][:3]
                    det.class_colors[code] = f"#{r:02x}{g:02x}{b:02x}"
        except (ValueError, KeyError):
            pass  # no colour table
        # Category names, if present (rasterio exposes band descriptions/tags).
        try:
            cats = ds.tags(band).get("CATEGORY_NAMES")
            if cats:
                pass  # unusual; left for the human unless clearly aligned
        except Exception:  # pragma: no cover - defensive
            pass
    return det


def detect_gee_asset(spec: ProductSpec, sample_aoi=None) -> Detected:
    """Auto-detect resolution, codes, and any published names/colours for a GEE
    label asset.

    Names and palette often live in the asset's band properties (e.g.
    ``<band>_class_values``, ``<band>_class_names``, ``<band>_class_palette``);
    those are pulled where present. Footprint is left ``None`` (GEE label maps in
    the MVP are global). ``sample_aoi`` bounds the code scan.
    """
    import ee

    from harmonizer.registry.adapters import _gee
    from harmonizer.sampling import _label_image_for, present_classes

    _gee.ensure_initialized()
    det = Detected()
    image = _label_image_for(spec.id)
    band = str(spec.band)

    if sample_aoi is not None:
        region = ee.Geometry.Rectangle(list(sample_aoi))
        det.class_codes = present_classes(image.clip(region), region)

    # Published class names / palette from the source band's properties.
    try:
        src = ee.ImageCollection(spec.access.asset_id).first()
        props = src.toDictionary().getInfo() or {}
        values = props.get(f"{band}_class_values")
        names = props.get(f"{band}_class_names")
        palette = props.get(f"{band}_class_palette")
        if values and names:
            for code, name in zip(values, names):
                det.class_names[int(code)] = str(name)
        if values and palette:
            for code, col in zip(values, palette):
                c = str(col)
                det.class_colors[int(code)] = c if c.startswith("#") else f"#{c}"
        if values and not det.class_codes:
            det.class_codes = sorted(int(v) for v in values)
    except Exception:  # pragma: no cover - network/asset shape varies
        pass
    return det


def build_registration_yaml(
    product_id: str,
    *,
    detected: Detected,
    role: str = "reference",
    kind: str = "label",
    access: dict | None = None,
) -> str:
    """Render a pre-filled registry YAML skeleton from what was auto-detected.

    Auto-detected fields are populated; anything the source did not provide is
    written as a ``TODO`` placeholder for a human to complete or confirm. Names and
    colours are filled per class where the source published them and left as
    ``TODO`` per class where it did not.
    """
    lines: list[str] = []
    a = access or {"method": "TODO", "asset_id": _TODO}

    def q(v):
        return _TODO if v is None else v

    lines.append(f"id: {product_id}")
    lines.append(f"display_name: {_TODO}")
    lines.append(f"provider: {_TODO}")
    lines.append(f"role: {role}")
    lines.append(f"kind: {kind}")
    lines.append("access:")
    for k, v in a.items():
        lines.append(f"  {k}: {v}")
    lines.append(f"band: {_TODO}")
    lines.append(f"resolution_m: {q(detected.resolution_m)}")
    lines.append(f"available_years: {_TODO}")
    lines.append(f"crs: {q(detected.crs)}")
    if detected.footprint is None:
        lines.append("footprint: null")
    else:
        lines.append(f"footprint: {list(detected.footprint)}")
    lines.append(f"licence: {_TODO}")
    lines.append(f"citation: {_TODO}")
    lines.append("legend:")
    for code in detected.class_codes:
        # Quote the values: a name may contain a colon and a colour begins with
        # '#', both of which would otherwise be mis-parsed in a YAML flow map.
        name = (
            _TODO if code not in detected.class_names
            else '"' + detected.class_names[code].replace('"', "'") + '"'
        )
        color = (
            _TODO if code not in detected.class_colors
            else '"' + detected.class_colors[code] + '"'
        )
        lines.append(f"  - {{code: {code}, name: {name}, color: {color}}}")
    if not detected.class_codes:
        lines.append(f"  []  # {_TODO}: no class codes detected")
    return "\n".join(lines) + "\n"


def _resolve_local_path(spec: ProductSpec) -> Path:
    """The GeoTIFF path for a local-raster product.

    Every local-raster product declares its own concrete file via ``access.path``
    (docs/PIPELINE.md, section 2.5) -- there is no shared directory to search,
    since two local products could otherwise collide on which file "the" local
    raster means.
    """
    raw = spec.access.path
    p = Path(raw) if raw else None
    if p is None or not p.is_file():
        raise FileNotFoundError(
            f"{spec.id!r}: access.path does not point at an existing file: {raw!r}"
        )
    return p
