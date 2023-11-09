from torch import nn
import torch
from einops.layers.torch import Rearrange


class ReshapeDecoder(nn.Module):
    def __init__(
            self,
            dims_in_stages: list[int],
            out_dim: int
    ):
        super().__init__()

        self.to_fused = nn.ModuleList([nn.Sequential(
            Rearrange("b c patch_num_in_h patch_num_in_w -> b patch_num_in_h patch_num_in_w c"),
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim * 4 ** i),
            # same order with patch embedding
            Rearrange(
                pattern="b patch_num_in_h patch_num_in_w (out_dim patch_size_in_h patch_size_in_w) -> "
                "b out_dim (patch_num_in_h patch_size_in_h) (patch_num_in_w patch_size_in_w) ",
                patch_size_in_h=2 ** i, patch_size_in_w=2 ** i)
        ) for i, dim in enumerate(dims_in_stages)])

        self.to_segmentation = nn.Conv2d(len(dims_in_stages) * out_dim, out_dim, kernel_size=1)

    def forward(self, x):
        fused = [to_fused(output) for output, to_fused in zip(x, self.to_fused)]
        fused = torch.cat(fused, dim=1)
        return self.to_segmentation(fused)
