from typing import Tuple

import galsim
import numpy as np
import pytest

from nirwl_metacal.utils import expand_bbox


@pytest.mark.parametrize(
    "corners",
    [(4, 6, 2, 6), (7, 9, 1, 5), (1, 34, 1, 34), (2, 36, 2, 28), (1, 6, 10, 40)],
)
def test_expand_bbox(corners: Tuple[int]):
    bbox = galsim.BoundsD(*corners)
    mosaic_bbox = galsim.BoundsI(-100, 100, -100, 100)
    expanded_bbox = expand_bbox(bbox, mosaic_bbox)
    # Check that the expanded bbox completely includes the bbox
    assert expanded_bbox.includes(bbox)
    # Check that the min_size is obeyed.
    assert expanded_bbox.area() >= 1024
    # Check that the dimensions are powers of 2
    nx = np.log2(expanded_bbox.xmax - expanded_bbox.xmin + 1)
    assert nx == int(nx)
    ny = np.log2(expanded_bbox.ymax - expanded_bbox.ymin + 1)
    assert ny == int(ny)
    # Check that the dimensions are just bigger than bbox, if bbox itself has
    # dimensions larger than min_size.
    if (bbox.xmax - bbox.xmin + 1) > 32:
        assert expanded_bbox.xmax - expanded_bbox.xmin + 1 == 2 ** np.ceil(np.log2(bbox.xmax - bbox.xmin + 1))
    if (bbox.ymax - bbox.ymin + 1) > 32:
        assert expanded_bbox.ymax - expanded_bbox.ymin + 1 == 2 ** np.ceil(np.log2(bbox.ymax - bbox.ymin + 1))


@pytest.mark.parametrize(
    "center",
    [(15.8, 15.2), (16.2, 16.7)],
)
def test_expand_bbox_with_center(center: Tuple[int]):
    center = galsim.PositionD(center)
    mosaic_bbox = galsim.BoundsI(-100, 100, -100, 100)
    bbox = galsim.BoundsI(xmin=1, xmax=31, ymin=1, ymax=31)
    expanded_bbox = expand_bbox(bbox, mosaic_bbox, center=center)
    # Check that the expanded bbox completely includes the bbox.
    assert expanded_bbox.includes(bbox)
    # Check that the min_size is obeyed.
    assert expanded_bbox.area() >= 1024
    # Check that the dimensions are powers of 2
    nx = np.log2(expanded_bbox.xmax - expanded_bbox.xmin + 1)
    assert nx == int(nx)
    ny = np.log2(expanded_bbox.ymax - expanded_bbox.ymin + 1)
    assert ny == int(ny)


@pytest.mark.parametrize(
    "bbox",
    [
        galsim.BoundsI(xmin=47, xmax=54, ymin=32, ymax=39),  # well inside the mosaic box
        galsim.BoundsI(xmin=32, xmax=48, ymin=2, ymax=8),  # close to the bottom edge
        galsim.BoundsI(xmin=44, xmax=58, ymin=90, ymax=98),  # close to the top edge
        galsim.BoundsI(xmin=5, xmax=21, ymin=30, ymax=64),  # close to the left edge
        galsim.BoundsI(xmin=2, xmax=10, ymin=91, ymax=98),  # close to a corners
    ],
)
def test_expand_bbox_corner_cases(bbox: galsim.BoundsI):
    mosaic_bbox = galsim.BoundsI(0, 100, 0, 100)
    expanded_bbox = expand_bbox(bbox, mosaic_bbox)
    # Check that the bbox  <- expanded bbox <- mosaic_bbox.
    assert expanded_bbox.includes(bbox)
    assert mosaic_bbox.includes(expanded_bbox)
