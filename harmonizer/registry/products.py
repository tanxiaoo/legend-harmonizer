"""Product registry and adapter interface (Stage 1).

Defines the uniform way downstream code obtains, for any sample coordinate, the
class label from each map and the 64-dim AlphaEarth embedding vector. Concrete
adapters live in ``harmonizer.registry.adapters``.

The contract is deliberately narrow: an adapter takes a list of coordinates in
EPSG:4326 (the ``target_crs`` from section 2) and returns one result per input
coordinate, in the same order. Downstream stages read data *only* through this
layer, so the sampling order that avoids moving rasters (read HRLC locally first,
then send only coordinates to GEE) is expressed as a choice of which adapters a
caller invokes -- not baked into any single adapter.

See docs/PIPELINE.md, Stage 1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from harmonizer.registry.schema import ProductSpec, load_all_products

# A coordinate is a (longitude, latitude) pair in EPSG:4326 (the target CRS).
Coord = tuple[float, float]


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LabelResult:
    """The class label a label-producing adapter returns for one coordinate.

    ``label`` is the raw class code read from the map (an integer for both HRLC
    and Dynamic World). ``masked`` is True when the point has no usable label --
    a fill/no-data pixel, or a coordinate outside the map footprint -- in which
    case ``label`` is None and the point should be dropped downstream.
    """

    label: int | None
    masked: bool

    @classmethod
    def no_data(cls) -> "LabelResult":
        return cls(label=None, masked=True)

    @classmethod
    def of(cls, label: int) -> "LabelResult":
        return cls(label=int(label), masked=False)


@dataclass(frozen=True)
class EmbeddingResult:
    """The embedding vector an embedding adapter returns for one coordinate.

    ``vector`` holds the 64-dim AlphaEarth vector as a float array. ``masked`` is
    True when the embedding is no-data at that point, in which case ``vector`` is
    None and the point should be dropped downstream.
    """

    vector: np.ndarray | None
    masked: bool

    @classmethod
    def no_data(cls) -> "EmbeddingResult":
        return cls(vector=None, masked=True)

    @classmethod
    def of(cls, vector: Sequence[float]) -> "EmbeddingResult":
        return cls(vector=np.asarray(vector, dtype=float), masked=False)


# --------------------------------------------------------------------------- #
# Adapter interfaces
# --------------------------------------------------------------------------- #


class LabelAdapter(ABC):
    """An adapter that returns a class label per coordinate (HRLC, Dynamic World)."""

    #: Human-readable name for logging and the registry.
    name: str = "label-adapter"

    @abstractmethod
    def sample_labels(self, coords: Sequence[Coord]) -> list[LabelResult]:
        """Return one :class:`LabelResult` per input coordinate, in order."""
        raise NotImplementedError


class EmbeddingAdapter(ABC):
    """An adapter that returns an embedding vector per coordinate (AlphaEarth)."""

    name: str = "embedding-adapter"

    @abstractmethod
    def sample_embeddings(self, coords: Sequence[Coord]) -> list[EmbeddingResult]:
        """Return one :class:`EmbeddingResult` per input coordinate, in order."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Product:
    """A registry entry describing one map or embedding source.

    A ``Product`` is a thin runtime view over a :class:`ProductSpec` -- the parsed
    YAML file that is the single source of truth for this map's metadata and legend
    (docs/PIPELINE.md, section 2.5). Map facts (``name``, ``kind``, ``role``,
    ``footprint``, legend) are read from ``spec`` rather than hardcoded here.
    ``adapter_factory`` builds the adapter on demand; it is a callable so nothing
    touches the filesystem or the network at import time.

    ``footprint`` mirrors the spec's declared extent as a
    (min_lon, min_lat, max_lon, max_lat) box in EPSG:4326, or ``None`` for a
    global product. Per section 2, this declared box is *advisory*: the
    authoritative sampling/overlap footprint is derived from the loaded source at
    run time (see ``harmonizer.overlap`` / Stage 5), and for a local raster it may
    differ from whatever tile is currently present.
    """

    spec: ProductSpec
    adapter_factory: object = field(repr=False)

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def name(self) -> str:
        return self.spec.display_name

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def role(self) -> str:
        return self.spec.role

    @property
    def footprint(self) -> tuple[float, float, float, float] | None:
        return self.spec.footprint


class Registry:
    """A tiny in-memory registry of products keyed by id.

    Products (and their :class:`ProductSpec` metadata) come from the YAML files in
    ``harmonizer/registry/products/`` -- the single source of truth. Adapters and
    downstream code look up a product's spec here to read asset ids, bands,
    footprints, and legends.
    """

    def __init__(self) -> None:
        self._products: dict[str, Product] = {}

    def register(self, product: Product) -> None:
        self._products[product.id] = product

    def get(self, product_id: str) -> Product:
        return self._products[product_id]

    def spec(self, product_id: str) -> ProductSpec:
        """The parsed registry file (single source of truth) for a product."""
        return self._products[product_id].spec

    def all(self) -> list[Product]:
        return list(self._products.values())

    def by_kind(self, kind: str) -> list[Product]:
        return [p for p in self._products.values() if p.kind == kind]


#: Maps a registry product id to the adapter factory that reads it. Adapters are
#: imported lazily inside the factories so importing this module never requires
#: rasterio or GEE to be available. The adapters themselves read their asset id /
#: band / resolution from the product's ProductSpec (single source of truth).
def _adapter_factory_for(product_id: str):
    def _worldcover():
        from harmonizer.registry.adapters.worldcover_gee import WorldCoverAdapter

        return WorldCoverAdapter()

    def _hrlc():
        from harmonizer.registry.adapters.hrlc_local import HRLCLocalAdapter

        return HRLCLocalAdapter()

    def _dynamicworld():
        from harmonizer.registry.adapters.dynamicworld_gee import DynamicWorldAdapter

        return DynamicWorldAdapter()

    def _alphaearth():
        from harmonizer.registry.adapters.alphaearth_gee import AlphaEarthAdapter

        return AlphaEarthAdapter()

    factories = {
        "worldcover": _worldcover,
        "hrlc": _hrlc,
        "dynamicworld": _dynamicworld,
        "alphaearth": _alphaearth,
    }
    return factories.get(product_id)


def default_registry() -> Registry:
    """Build the registry by loading every product YAML file (section 2.5).

    The files in ``harmonizer/registry/products/`` are the single source of truth:
    each becomes a :class:`Product` whose metadata and legend come from the parsed
    :class:`ProductSpec`, paired with the adapter factory that knows how to read
    it. For the current test swap WorldCover is the reference; Dynamic World the
    compare; AlphaEarth the embedding; HRLC remains registered as the eventual real
    reference. A YAML file without a known adapter is registered with no factory so
    its metadata/legend are still available (e.g. for reconciliation).
    """

    reg = Registry()
    for product_id, spec in load_all_products().items():
        reg.register(
            Product(spec=spec, adapter_factory=_adapter_factory_for(product_id))
        )
    return reg
