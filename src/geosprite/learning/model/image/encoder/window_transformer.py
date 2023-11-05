from torch import nn
import torch
from geosprite.learning.model.time_series.encoder import Transformer, TransformerArgs
from einops import rearrange
from einops.layers.torch import Rearrange

__all__ = ["WindowTransformer"]


class WindowTransformer(nn.Module):
    """ local transformer to simulate the model in TSViT paper
    the window size here is the image size in TSViT paper
    input: b c h w
    output: b c h w
    """

    def __init__(self, window_size: int, use_absolute_position_embedding: bool, transformer_args: TransformerArgs):
        super().__init__()
        self.window_size = window_size
        self.use_absolute_position_embedding = use_absolute_position_embedding
        self.window_partition = Rearrange(
            "b c (new_h window_size_in_h) (new_w window_size_in_w) -> (b new_h new_w) (window_size_in_h window_size_in_w) c",
            window_size_in_h=window_size, window_size_in_w=window_size)
        # use absolute positional embedding
        if use_absolute_position_embedding:
            self.space_position_embedding = nn.Parameter(
                torch.randn(window_size ** 2, transformer_args.embedding_dim))

        self.space_transformer = Transformer(transformer_args)

    def forward(self, x):
        """
        Args:
            x: b c h w
        Returns:
            [Tensor(b c h w),]
        """
        b, c, h, w = x.size()
        new_h = h // self.window_size
        new_w = w // self.window_size

        x = self.window_partition(x)
        if self.use_absolute_position_embedding:
            x += self.space_position_embedding

        x = self.space_transformer(x)
        x = rearrange(x,
                      "(b new_h new_w) (window_size_in_h window_size_in_w) c -> b c (new_h window_size_in_h) (new_w window_size_in_w)",
                      new_h=new_h, new_w=new_w, window_size_in_h=self.window_size, window_size_in_w=self.window_size)
        return [x]
