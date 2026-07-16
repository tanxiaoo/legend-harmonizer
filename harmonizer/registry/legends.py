"""Registry-backed legend lookups (single source of truth).

Small helpers that resolve a product's legend -- class names, colours, ordered
entries -- from the per-map YAML files (docs/PIPELINE.md, section 2.5). Modules
that used to hardcode class names (``affinity.CLASS_NAMES``) or palettes
(``tiles._PALETTES``) read from here instead, so a legend edit lives in one place.

The parsed registry is cached at module load; call :func:`reload` after editing a
YAML file in a long-running process.
"""

from __future__ import annotations

from functools import lru_cache

from harmonizer.registry.schema import LegendClass, ProductSpec, load_all_products


@lru_cache(maxsize=1)
def _specs() -> dict[str, ProductSpec]:
    return load_all_products()


def reload() -> None:
    """Drop the cached registry so the next lookup re-reads the YAML files."""
    _specs.cache_clear()


def spec(product_id: str) -> ProductSpec | None:
    return _specs().get(product_id)


def class_name(product_id: str, class_value: int) -> str:
    """Human-readable label for a class value, or the value itself if unknown."""
    s = _specs().get(product_id)
    if s is None:
        return str(class_value)
    return s.class_name(int(class_value))


def class_color(product_id: str, class_value: int) -> str | None:
    """``#RRGGBB`` hex for a class value, or None if the product/class is unknown."""
    s = _specs().get(product_id)
    if s is None:
        return None
    return s.class_color(int(class_value))


def legend_classes(product_id: str) -> list[LegendClass]:
    """The product's legend entries in declared order, or [] if unknown."""
    s = _specs().get(product_id)
    return list(s.legend) if s is not None else []
