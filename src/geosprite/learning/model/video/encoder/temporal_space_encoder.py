from einops import repeat, rearrange
from einops.layers.torch import Rearrange
import torch
from torch import nn

__all__ = ["TemporalSpaceEncoder"]


class TemporalSpaceEncoder(nn.Module):
    """
    Temporal-Spatial Encoder, complete the following task:
    1. split dates from input x, and embedding dates as positional embedding
    2. reshape x and input x into temporal encoder
    3. reshape x and input x into space encoder
    4. reshape x ready for space decoder
    """

    def __init__(self, image_channels: int, patch_size: int, embedding_dim: int, num_classes: int,
                 position_encoder: nn.Module, temporal_encoder: nn.Module, space_encoder: nn.Module):
        super().__init__()

        self.patch_size = patch_size
        self.num_classes = num_classes

        # use linear layer to embedding channels
        patch_dim = image_channels * patch_size ** 2
        self.patch_embedding = nn.Sequential(
            Rearrange('b t c (patch_num_in_h patch_size_in_h) (patch_num_in_w patch_size_in_w) -> '
                      '(b patch_num_in_h patch_num_in_w) t (c patch_size_in_h patch_size_in_w)',
                      patch_size_in_h=patch_size, patch_size_in_w=patch_size),
            nn.Linear(patch_dim, embedding_dim))

        self.position_encoder = position_encoder

        self.temporal_class_token = nn.Parameter(torch.randn(num_classes, embedding_dim))

        # temporal encoder
        self.temporal_encoder = temporal_encoder
        # space encoder
        self.space_encoder = space_encoder

    def forward(self, x):
        """
        Args:
            x: consist of x (b, t, c, h, w) and dates (b, t)
                b: batch size
                l: sequence length
                c: dimension of input token
                days_of_year: days of years, 0 means the first day

        Returns:
            (b, n, new_c, new_h, new_w)
            n: number of classification classes
        """
        x, days_of_year = x
        b, t, c, h, w = x.size()

        # (b patch_num_in_h patch_num_in_w) t embedding_dim
        x = self.patch_embedding(x)
        patch_num_in_h = h // self.patch_size
        patch_num_in_w = w // self.patch_size

        pos_embedding = self.position_encoder(days_of_year)

        pos_embedding = repeat(pos_embedding, "b t d -> (b patch_num_in_h patch_num_in_w) t d",
                               patch_num_in_h=patch_num_in_h, patch_num_in_w=patch_num_in_w)

        x += pos_embedding

        temporal_class_tokens = repeat(self.temporal_class_token, 'n c -> new_b n c',
                                       new_b=b * patch_num_in_h * patch_num_in_w)
        x = torch.cat((temporal_class_tokens, x), dim=1)

        # (b patch_num_in_h patch_num_in_w) n+t embedding_dim
        x = self.temporal_encoder(x)
        x = x[:, :self.num_classes, :]

        x = rearrange(x, "(b patch_num_in_h patch_num_in_w) n c -> (b n) c patch_num_in_h patch_num_in_w",
                      patch_num_in_h=patch_num_in_h, patch_num_in_w=patch_num_in_w)

        x = self.space_encoder(x)
        return x
