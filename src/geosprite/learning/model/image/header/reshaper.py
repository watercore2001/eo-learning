from torch import nn
from einops.layers.torch import Rearrange
from .base_header import BaseHeader


class ReshapeHeader(BaseHeader):
    """
    for TSViT input, the first dimension of x is consist of batch size and num classes
    """

    def __init__(self, num_classes: int, embedding_dim: int, scale_factor: int):
        super().__init__(num_classes)
        # the first dimension is consist of batch size and num classes
        self.header = nn.Sequential(
            Rearrange("(b n) c patch_num_in_h patch_num_in_w -> b n patch_num_in_h patch_num_in_w c",
                      n=num_classes, c=embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, scale_factor ** 2),
            # same order with patch embedding
            Rearrange(
                "b n patch_num_in_h patch_num_in_w (patch_size_in_h patch_size_in_w) -> "
                "b n (patch_num_in_h patch_size_in_h) (patch_num_in_w patch_size_in_w)",
                patch_size_in_h=scale_factor, patch_size_in_w=scale_factor)
        )

    def forward(self, x):
        """
        Args:
            x: [Tensor(b n) c new_h new_w),]

        Returns:
            x: b n h w
        """
        return self.header(x)
