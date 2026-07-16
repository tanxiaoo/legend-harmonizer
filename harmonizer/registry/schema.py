"""Product registry schema and YAML loader (docs/PIPELINE.md, section 2.5).

The product registry is the **single source of truth** for map metadata and
legends. It is one YAML file per map under ``harmonizer/registry/products/``; this
module parses those files into typed objects that the rest of the codebase reads
from -- the adapters, overlap logic, per-class toggles, matrix labels, and CSV
export all pull asset ids, footprints, resolutions, and legends (class codes,
names, colours) from here rather than from hardcoded values in ``config.py``,
``affinity.py``, ``tiles.py``, or the frontend.

Two layers per file:
  * map-level fields (id, display name, provider, access, band, resolution, years,
    CRS, footprint, licence/citation), and
  * a legend list, one entry per class (code, name, colour, description, optional
    shared-scheme mapping).

See docs/PIPELINE.md, section 2.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Directory holding one YAML file per map.
PRODUCTS_DIR = Path(__file__).resolve().parent / "products"

# A footprint box is (min_lon, min_lat, max_lon, max_lat) in EPSG:4326, or None
# for a global map.
Footprint = tuple[float, float, float, float] | None


# --------------------------------------------------------------------------- #
# Typed views over a registry file
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LegendClass:
    """One class in a map's legend.

    ``code`` is the raw integer class value in the raster / label band. ``name``
    and ``color`` (a ``#RRGGBB`` hex string) are the map's published legend.
    ``description`` and ``shared_scheme`` are optional.
    """

    code: int
    name: str
    color: str
    description: str | None = None
    shared_scheme: str | None = None


@dataclass(frozen=True)
class Access:
    """How a map is read: GEE asset or a local raster path."""

    method: str  # "gee" | "local_raster"
    asset_id: str | None = None      # GEE asset id (method == "gee")
    path: str | None = None          # local raster path (method == "local_raster")
    composite: str | None = None     # optional compositing note (e.g. "annual_modal")


@dataclass(frozen=True)
class ProductSpec:
    """A parsed registry file: the single source of truth for one map.

    Map-level metadata plus the legend list. Everything downstream reads map facts
    and legend (names, colours) from an instance of this rather than from
    hardcoded constants.
    """

    id: str
    display_name: str
    provider: str
    role: str           # "reference" | "compare" | "embedding"
    kind: str           # "label" | "embedding"
    access: Access
    band: str | int | None
    resolution_m: float | None
    available_years: tuple[int, ...]
    crs: str | None
    footprint: Footprint
    licence: str | None
    citation: str | None
    legend: tuple[LegendClass, ...]
    embedding_dims: int | None = None
    #: Absolute path to the YAML file this spec was loaded from.
    source_path: Path | None = field(default=None, repr=False)

    # -- convenience accessors (used by adapters / tiles / affinity) --------- #

    @property
    def legend_by_code(self) -> dict[int, LegendClass]:
        return {c.code: c for c in self.legend}

    @property
    def class_codes(self) -> list[int]:
        return [c.code for c in self.legend]

    def class_name(self, code: int) -> str:
        """Readable name for a class code, or the code itself if not in the legend."""
        entry = self.legend_by_code.get(int(code))
        return entry.name if entry is not None else str(code)

    def class_color(self, code: int) -> str | None:
        """``#RRGGBB`` hex for a class code, or None if not in the legend."""
        entry = self.legend_by_code.get(int(code))
        return entry.color if entry is not None else None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _norm_footprint(raw) -> Footprint:
    if raw is None:
        return None
    box = tuple(float(x) for x in raw)
    if len(box) != 4:
        raise ValueError(f"footprint must be 4 numbers or null, got: {raw!r}")
    return box  # type: ignore[return-value]


def _norm_color(raw: str | None) -> str:
    """Normalise a colour to a leading-``#`` lowercase hex string."""
    if raw is None:
        raise ValueError("legend class is missing a colour")
    s = str(raw).strip()
    if not s.startswith("#"):
        s = "#" + s
    return s.lower()


def _parse_legend(raw_legend) -> tuple[LegendClass, ...]:
    if not raw_legend:
        return ()
    out: list[LegendClass] = []
    for entry in raw_legend:
        out.append(
            LegendClass(
                code=int(entry["code"]),
                name=str(entry["name"]),
                color=_norm_color(entry.get("color")),
                description=entry.get("description"),
                shared_scheme=entry.get("shared_scheme"),
            )
        )
    return tuple(out)


def parse_product(doc: dict, source_path: Path | None = None) -> ProductSpec:
    """Build a :class:`ProductSpec` from a parsed YAML document."""

    access_raw = doc.get("access") or {}
    access = Access(
        method=str(access_raw.get("method", "")),
        asset_id=access_raw.get("asset_id"),
        path=access_raw.get("path"),
        composite=access_raw.get("composite"),
    )
    years = tuple(int(y) for y in (doc.get("available_years") or ()))
    return ProductSpec(
        id=str(doc["id"]),
        display_name=str(doc.get("display_name", doc["id"])),
        provider=str(doc.get("provider", "")),
        role=str(doc["role"]),
        kind=str(doc["kind"]),
        access=access,
        band=doc.get("band"),
        resolution_m=(
            float(doc["resolution_m"]) if doc.get("resolution_m") is not None else None
        ),
        available_years=years,
        crs=doc.get("crs"),
        footprint=_norm_footprint(doc.get("footprint")),
        licence=doc.get("licence"),
        citation=doc.get("citation"),
        legend=_parse_legend(doc.get("legend")),
        embedding_dims=(
            int(doc["embedding_dims"]) if doc.get("embedding_dims") is not None else None
        ),
        source_path=source_path,
    )


def load_product_file(path: Path) -> ProductSpec:
    """Load and parse a single product YAML file."""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        raise ValueError(f"registry file is not a mapping: {path}")
    return parse_product(doc, source_path=path)


def load_all_products(products_dir: Path | None = None) -> dict[str, ProductSpec]:
    """Load every ``*.yaml`` file in the products directory, keyed by id.

    This is how the registry is populated: the files on disk *are* the registry.
    """
    directory = products_dir or PRODUCTS_DIR
    specs: dict[str, ProductSpec] = {}
    for path in sorted(directory.glob("*.yaml")):
        spec = load_product_file(path)
        if spec.id in specs:
            raise ValueError(
                f"duplicate product id {spec.id!r} in {path} and "
                f"{specs[spec.id].source_path}"
            )
        specs[spec.id] = spec
    return specs
