"""
Module containing various stand-alone utility functions
"""

from typing import Optional

import galsim
import numpy as np


def expand_bbox(
    bbox: galsim.Bounds,
    mosaic_bbox: galsim.Bounds,
    center: Optional[galsim.PositionD] = None,
    min_size: int = 32,
) -> galsim.Bounds:
    """
    Expand the bounding box to be a power of 2.
    """
    if center is None:
        center: galsim.PositionD = bbox.center

    if not bbox.includes(center):
        raise RuntimeError("%s is not inside in %s" % (center, bbox))
    if mosaic_bbox is not None:
        if not mosaic_bbox.includes(center):
            raise RuntimeError("%s is not inside in %s" % (center, mosaic_bbox))

    # left_pad, right_pad, top_pad, bottom_pad = (min_size - 1) // 2
    ix = int(np.round(center.x))
    iy = int(np.round(center.y))

    pad = min_size // 2
    # expanded_bbox = galsim.BoundsI(ix - pad, ix + pad - 1, iy - pad, iy + pad - 1)

    # Initialize the expanded_bbox as a 4x4 box around the center.
    expanded_bbox = galsim.BoundsI(
        ix - (center.x < ix), ix + (center.x >= ix), iy - (center.y < iy), iy + (center.y >= iy)
    )

    # Grow the bounding box until it includes the entire footprint.
    while not expanded_bbox.includes(bbox):
        expanded_bbox = expanded_bbox.expand(2)

    # Grow the bounding box further if it is smaller than 32x32 pixels
    # and does not fall outside of the mosaic.
    while expanded_bbox.area() <= min_size**2 and mosaic_bbox.includes(expanded_bbox):
        expanded_bbox = expanded_bbox.expand(2)

    # If the while loop exited because the expanded_bbox grew outside of the
    # mosaic, then move the bbox.
    if not mosaic_bbox.includes(expanded_bbox):
        common_bbox = expanded_bbox & mosaic_bbox
        center_shift = common_bbox.center - expanded_bbox.center
        expanded_bbox = expanded_bbox.shift(2 * center_shift)

    # Check if this falls outside the full mosaic image

    # while (expanded_bbox.xmin > bbox.xmin) or (expanded_bbox.xmax < bbox.xmax):
    #     expanded_bbox = expanded_bbox.withBorder(dx=pad, dy=0)
    # while (expanded_bbox.ymin > bbox.ymin) or (expanded_bbox.ymax < bbox.ymax):
    #     expanded_bbox = expanded_bbox.withBorder(dx=0, dy=pad)

    return expanded_bbox
