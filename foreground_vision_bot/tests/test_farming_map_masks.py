from __future__ import annotations

# pyright: reportImplicitRelativeImport=false
import numpy as np
import pytest
from farming.map_masks import dilate_mask, inflate_map_masks


def test_map_masks_keep_wall_and_teleport_buffers_distinct() -> None:
    traversable = np.ones((7, 7), dtype=np.bool_)
    traversable[3, 1] = False
    forbidden = np.zeros_like(traversable)
    forbidden[3, 5] = True

    masks = inflate_map_masks(
        traversable,
        forbidden,
        obstacle_radius_cells=1,
        teleport_radius_cells=2,
    )

    assert masks.obstacle_buffer[3, 2]
    assert not masks.teleport_buffer[3, 2]
    assert masks.teleport_buffer[3, 3]
    assert not masks.safe_traversable[3, 3]
    assert not masks.safe_traversable[0, 0]


def test_dilation_rejects_invalid_shapes_and_radii() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        dilate_mask(np.zeros(3, dtype=np.bool_), 1)
    with pytest.raises(ValueError, match="boolean"):
        dilate_mask(np.zeros((2, 2), dtype=np.bool_), True)
    with pytest.raises(ValueError, match="negative"):
        dilate_mask(np.zeros((2, 2), dtype=np.bool_), -1)
