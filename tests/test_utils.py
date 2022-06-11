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
    expanded_bbox = expand_bbox(bbox)
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
    [(15.8, 15.2), (16.2, 16.7), (55.1, 58.9)],
)
def test_expand_bbox_with_center(center: Tuple[int]):
    center = galsim.PositionD(center)
    bbox = galsim.BoundsI(xmin=1, xmax=31, ymin=1, ymax=31)
    expanded_bbox = expand_bbox(bbox, center=center)
    # Check that the expanded bbox completely includes the bbox.
    assert expanded_bbox.includes(bbox)
    # Check that the min_size is obeyed.
    assert expanded_bbox.area() >= 1024
    # Check that the dimensions are powers of 2
    nx = np.log2(expanded_bbox.xmax - expanded_bbox.xmin + 1)
    assert nx == int(nx)
    ny = np.log2(expanded_bbox.ymax - expanded_bbox.ymin + 1)
    assert ny == int(ny)
