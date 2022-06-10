"""
Module containing various stand-alone utility functions
"""

from typing import Optional

import galsim
import numpy as np


def expand_bbox(
    bbox: galsim.Bounds, center: Optional[galsim.PositionD] = None, min_size: int = 32
) -> galsim.Bounds:
    """
    Expand the bounding box to be a power of 2.
    """
    if center is None:
        center = bbox.center

    # left_pad, right_pad, top_pad, bottom_pad = (min_size - 1) // 2
    ix = int(np.round(center.x))
    iy = int(np.round(center.y))

    pad = min_size // 2
    expanded_bbox = galsim.BoundsI(ix - pad, ix + pad - 1, iy - pad, iy + pad - 1)
    while (expanded_bbox.xmin > bbox.xmin) or (expanded_bbox.xmax < bbox.xmax):
        expanded_bbox = expanded_bbox.withBorder(dx=pad, dy=0)
    while (expanded_bbox.ymin > bbox.ymin) or (expanded_bbox.ymax < bbox.ymax):
        expanded_bbox = expanded_bbox.withBorder(dx=0, dy=pad)

    return expanded_bbox
