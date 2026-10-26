"""Spatial-only 3D pooling expressed as independent 2D frame pooling."""
import torch
from torch import nn


class SpatialPool(nn.Module):
    def forward(self, x):
        if x.ndim != 5:
            raise ValueError('Expected B C T H W pooling input')
        b,c,t,h,w = x.shape
        flat = x.permute(0,2,1,3,4).reshape(b*t,c,h,w)
        pooled = torch.nn.functional.max_pool2d(flat,3,2,1)
        return pooled.reshape(b,t,c,*pooled.shape[-2:]).permute(0,2,1,3,4)


def replace_spatial_pool(frontend):
    original = frontend.frontend3D[3]
    if not isinstance(original, nn.MaxPool3d):
        raise ValueError('Unexpected pretrained pooling module')
    if (original.kernel_size != (1,3,3) or original.stride != (1,2,2)
            or original.padding != (0,1,1) or original.dilation != 1
            or original.ceil_mode or original.return_indices):
        raise ValueError('Only the verified spatial pooling geometry is supported')
    frontend.frontend3D[3] = SpatialPool()
