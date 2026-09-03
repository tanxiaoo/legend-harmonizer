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


# --------------------------------------------------------------------------- #
# Semantic attributes (Stage 8a)
# --------------------------------------------------------------------------- #

# Allowed values per categorical attribute. ``any`` (or an omitted key) means
# *unspecified*, which scores 1.0 against anything -- the legend simply does not
# constrain that attribute. Anything else raises at load time, so a typo in a
# YAML encoding is caught when the registry loads rather than silently becoming
# an unmatchable value.
SEMANTIC_ENUMS: dict[str, frozenset[str]] = {
    "surface": frozenset({"vegetated", "built", "bare", "water", "snow"}),
    "cultivation": frozenset({"natural", "cultivated"}),
    "life_form": frozenset({"tree", "shrub", "herbaceous", "lichen_moss"}),
    "leaf_type": frozenset({"broadleaf", "needleleaf"}),
    "phenology": frozenset({"evergreen", "deciduous"}),
}

#: Interval-valued attributes, as ``[low, high]`` with ``null`` allowed on either
#: bound for open-ended. Their natural maxima live in ``config.py``.
SEMANTIC_INTERVALS: tuple[str, ...] = ("cover", "height", "flooding")

#: An interval bound pair; ``None`` on a side means open-ended there.
Interval = tuple[float | None, float | None]


@dataclass(frozen=True)
class SemanticAlternative:
    """One OR-branch of a class's LCCS attribute encoding.

    A legend class that means "either A or B" (WorldCover Cropland is dry *or*
    aquatic; HRLC 90 is tree *or* shrub) carries one alternative per branch. A
    single-meaning class carries exactly one.

    Categorical attributes are ``None`` when unspecified (written ``any`` or
    omitted in the YAML); interval attributes are ``None`` when unspecified and
    otherwise a ``(low, high)`` pair whose bounds may individually be ``None``
    for open-ended. Unspecified always scores 1.0 -- it is "no constraint", not
    "no match".
    """

    surface: str | None = None
    cultivation: str | None = None
    life_form: str | None = None
    leaf_type: str | None = None
    phenology: str | None = None
    cover: Interval | None = None
    height: Interval | None = None
    flooding: Interval | None = None


@dataclass(frozen=True)
class ClassSemantics:
    """A legend class's LCCS attribute encoding: one or more alternatives."""

    alternatives: tuple[SemanticAlternative, ...]


def _norm_enum(attr: str, raw, where: str) -> str | None:
    """Validate a categorical attribute value; ``any``/missing -> None."""
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value == "any":
        return None
    allowed = SEMANTIC_ENUMS[attr]
    if value not in allowed:
        raise ValueError(
            f"{where}: unknown {attr} value {raw!r}; "
            f"expected one of {sorted(allowed)} or 'any'"
        )
    return value


def _norm_interval(attr: str, raw, where: str) -> Interval | None:
    """Validate an interval attribute; missing -> None (unspecified)."""
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(
            f"{where}: {attr} must be a [low, high] pair (null allowed on a "
            f"bound), got {raw!r}"
        )
    low, high = (None if b is None else float(b) for b in raw)
    if low is not None and high is not None and low > high:
        raise ValueError(f"{where}: {attr} interval is inverted: {raw!r}")
    return (low, high)


def _parse_semantics(raw, where: str) -> ClassSemantics | None:
    """Parse a class's optional ``semantics`` block (docs/PIPELINE.md, 8a.1)."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: semantics must be a mapping, got {type(raw).__name__}")
    raw_alts = raw.get("alternatives")
    if not raw_alts:
        raise ValueError(f"{where}: semantics needs a non-empty 'alternatives' list")

    alts: list[SemanticAlternative] = []
    for n, entry in enumerate(raw_alts):
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: alternative {n} is not a mapping")
        unknown = set(entry) - set(SEMANTIC_ENUMS) - set(SEMANTIC_INTERVALS)
        if unknown:
            raise ValueError(
                f"{where}: alternative {n} has unknown attribute(s) {sorted(unknown)}"
            )
        loc = f"{where} alternative {n}"
        alts.append(
            SemanticAlternative(
                **{a: _norm_enum(a, entry.get(a), loc) for a in SEMANTIC_ENUMS},
                **{a: _norm_interval(a, entry.get(a), loc) for a in SEMANTIC_INTERVALS},
            )
        )
    return ClassSemantics(alternatives=tuple(alts))


@dataclass(frozen=True)
class LegendClass:
    """One class in a map's legend.

    ``code`` is the raw integer class value in the raster / label band. ``name``
    and ``color`` (a ``#RRGGBB`` hex string) are the map's published legend.
    ``description`` and ``shared_scheme`` are optional.

    ``observed`` records whether the class was actually **found in this
    dataset's pixels** when it was indexed (DESIGN.md 4.3). A class declared by
    the published legend but absent from the data is legitimate -- a regional
    subset simply does not contain every class of a global legend -- so it is
    kept in the legend and marked instead of dropped, which lets the UI grey it
    out and say why. ``None`` means "not determined", which is the honest answer
    for a GEE product or a hand-written entry that predates the check; only
    ``False`` asserts absence.
    """

    code: int
    name: str
    color: str
    description: str | None = None
    shared_scheme: str | None = None
    observed: bool | None = None
    #: Optional LCCS attribute encoding driving the Stage 8 semantic prior. None
    #: where the legend has not been encoded; such a product gets a uniform prior.
    semantics: ClassSemantics | None = None


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

    @property
    def has_semantics(self) -> bool:
        """True when **every** legend class carries an attribute encoding.

        The Stage 8 prior is only meaningful if the whole legend is encoded: a
        partially encoded product would silently mix real inclusion scores with
        uniform 1.0 rows, which reads as "everything matches this class". So a
        product is either fully encoded or treated as unencoded.
        """
        return bool(self.legend) and all(c.semantics is not None for c in self.legend)

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


def _parse_legend(raw_legend, product_id: str = "?") -> tuple[LegendClass, ...]:
    if not raw_legend:
        return ()
    out: list[LegendClass] = []
    for entry in raw_legend:
        code = int(entry["code"])
        out.append(
            LegendClass(
                code=code,
                name=str(entry["name"]),
                color=_norm_color(entry.get("color")),
                description=entry.get("description"),
                shared_scheme=entry.get("shared_scheme"),
                # Absent key -> None ("not determined"), never False: only an
                # explicit `observed: false` from the indexer asserts that the
                # class is missing from this dataset's pixels.
                observed=entry.get("observed"),
                semantics=_parse_semantics(
                    entry.get("semantics"), f"{product_id} class {code}"
                ),
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
        legend=_parse_legend(doc.get("legend"), str(doc.get("id", "?"))),
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
