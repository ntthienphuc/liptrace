from __future__ import annotations

import torch

from datamodule.transforms import DeterministicSpatialCrop


def test_locked_spatial_views_have_exact_offsets():
    image = torch.arange(96 * 96).reshape(1, 1, 96, 96)
    center = DeterministicSpatialCrop("center")(image)
    left = DeterministicSpatialCrop("left")(image)
    right = DeterministicSpatialCrop("right")(image)
    assert center.shape == left.shape == right.shape == (1, 1, 88, 88)
    assert center[0, 0, 0, 0] == image[0, 0, 4, 4]
    assert left[0, 0, 0, 0] == image[0, 0, 4, 2]
    assert right[0, 0, 0, 0] == image[0, 0, 4, 6]
