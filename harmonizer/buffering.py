"""Class-mask buffering (Stage 2).

Erodes each class mask inward by the configured pixels to remove boundary and
mixed pixels, leaving homogeneous cores, and confirms a single-class NxN
neighbourhood at each candidate point. Provides the relaxed (1-pixel) buffer used
when a class falls below the point floor.

All operations here are server-side Earth Engine image operations on an annual
label image built by Stage 2. Nothing but coordinates ever leaves GEE; this
module returns ``ee`` masks, not rasters.

The erosion is a morphological erosion of the binary class mask: a pixel survives
only if every pixel within the erosion radius shares the class. The homogeneous
NxN check is the same idea at a fixed 3x3 window, expressed independently so it
can be required on top of (or instead of) a larger erosion. Both are implemented
with ``reduceNeighborhood`` over a square kernel; requiring the neighbourhood
*minimum* of the binary mask to stay 1 is exactly "all neighbours share the
class".

See docs/PIPELINE.md, Stage 2.
"""

from __future__ import annotations

from harmonizer.config import CONFIG


def class_mask(label_image, class_value: int):
    """Binary mask (1 where ``label_image`` equals ``class_value``, else 0)."""
    return label_image.eq(int(class_value))


def _min_over_square(mask, radius_px: int):
    """Neighbourhood minimum of ``mask`` over a square of the given pixel radius.

    A square kernel of radius ``r`` spans ``2r+1`` pixels per side. The minimum
    over that window is 1 only where every pixel in the window is 1 -- i.e. the
    binary erosion of the mask by ``r`` pixels.
    """
    import ee

    kernel = ee.Kernel.square(radius=int(radius_px), units="pixels")
    # unmask(0) so pixels off the mask edge count as "not this class" rather than
    # dropping out and letting a boundary pixel survive erosion.
    return mask.unmask(0).reduceNeighborhood(
        reducer=ee.Reducer.min(), kernel=kernel
    )


def eroded_mask(label_image, class_value: int, erode_pixels: int | None = None):
    """Erode the class mask inward by ``erode_pixels`` (default from config).

    Returns a binary mask whose set pixels are homogeneous cores of the class:
    every pixel within ``erode_pixels`` of them is the same class. Passing
    ``erode_pixels`` explicitly is how the relaxed (1-pixel) buffer is applied to
    a starved class.
    """
    if erode_pixels is None:
        erode_pixels = CONFIG.buffering.erode_pixels
    mask = class_mask(label_image, class_value)
    if erode_pixels <= 0:
        return mask
    return _min_over_square(mask, erode_pixels)


def homogeneous_mask(label_image, class_value: int, window: int | None = None):
    """Mask of points whose NxN neighbourhood is entirely one class.

    ``window`` is the neighbourhood side (default the configured 3), so the
    radius is ``(window - 1) // 2``. The result is 1 only where the whole window
    shares ``class_value``. This is the "confirm a homogeneous 3x3 neighbourhood"
    constraint from section 2.
    """
    if window is None:
        window = CONFIG.buffering.homogeneous_window
    radius = (int(window) - 1) // 2
    mask = class_mask(label_image, class_value)
    if radius <= 0:
        return mask
    return _min_over_square(mask, radius)


def sampling_mask(label_image, class_value: int, erode_pixels: int | None = None):
    """Combined mask a class is sampled from: eroded core AND homogeneous window.

    Erodes by ``erode_pixels`` (default config) and additionally requires the
    configured homogeneous NxN neighbourhood, so candidate points sit in class
    interiors with a pure local window. Returns a binary mask self-masked to its
    set pixels, ready to hand to stratified sampling.
    """
    eroded = eroded_mask(label_image, class_value, erode_pixels=erode_pixels)
    homogeneous = homogeneous_mask(label_image, class_value)
    combined = eroded.min(homogeneous).rename("mask")
    # selfMask so sampleRegions only draws from set pixels.
    return combined.selfMask()
