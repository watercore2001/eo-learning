from torch import nn
import torch


class SegformerDecoder(nn.Module):
    def __init__(
            self,
            dims_in_stages: list[int],
            out_dim: int
    ):
        super().__init__()

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, out_dim, kernel_size=1),
            nn.Upsample(scale_factor=2 ** i, mode="bilinear")
        ) for i, dim in enumerate(dims_in_stages)])

        self.to_segmentation = nn.Conv2d(len(dims_in_stages) * out_dim, out_dim, kernel_size=1)

    def forward(self, x):
        fused = [to_fused(output) for output, to_fused in zip(x, self.to_fused)]
        fused = torch.cat(fused, dim=1)
        return self.to_segmentation(fused)
