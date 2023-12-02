import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import deepspeed
from torch.nn import functional
import dataclasses
from geosprite.learning.model.time_series.encoder import PostNorm, CosineAttention, MLP, TransformerArgs
from .util import add_tensor, window_partition, window_reverse, WindowShifter


__all__ = ["BaseSwinTransformerV2", "SwinTransformerV2Stages", "SwinTransformerV2StagesArgs"]


class SwinTransformerV2Block(nn.Module):
    """ Swin Transformer V2 Block: MSA + FeedForward
    one block is consist of:
        1. possible pad x to multipy of window
        2. possible window-shift
        3. window partition: prepare data for self-attention
        4. possible relative position embedding
        5. one self-attention layer with post norm and residual
        6. one feedforward layer with post norm and residual
        7. window reverse: prepare data for window-shift
        8. possible window-shift reverse
        9. possible remove padding part of x
    """

    def __init__(self, window_size: int, pretrained_window_size: int,
                 use_relative_position_embedding: bool, shift_size: int, embedding_dim: int,
                 heads: int, head_dim: int, mlp_ratio: int, dropout: float):
        """
        Args:
            window_size: window size
            pretrained_window_size:
            use_relative_position_embedding: bool
            shift_size: shift size for SW-MSA
            embedding_dim: number of input channels
            heads: number of attention heads
            head_dim: dimension of each head
            mlp_ratio: ratio of mlp hidden dim to embedding dim
            dropout: dropout rate
        """
        super().__init__()
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.shift_size = shift_size
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.use_relative_position_embedding = use_relative_position_embedding

        # assert head_dim == 32, f"head_dim in each stage should be 32 in my design, there is {head_dim}"
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0 -> window_size"

        if use_relative_position_embedding:
            self.relative_position_bias_mlp = nn.Sequential(nn.Linear(in_features=2, out_features=512, bias=True),
                                                            nn.ReLU(inplace=True),
                                                            nn.Linear(in_features=512, out_features=heads, bias=False))
            self.generate_relative_coords_table()
            self.generate_relative_position_index()

        self.cyclic_shifter = WindowShifter(window_size=window_size, shift_size=shift_size) if shift_size > 0 else None

        # use layer normalization after attention and feedforward
        self.attn = PostNorm(embedding_dim,
                             CosineAttention(embedding_dim=embedding_dim, heads=heads,
                                             head_dim=head_dim, dropout=dropout))
        self.mlp = PostNorm(embedding_dim, MLP(embedding_dim=embedding_dim,
                                               hidden_dim_ratio=mlp_ratio,
                                               dropout=dropout))

    def forward(self, x):
        """
        Args:
            x: b c h w
        return:
            b c h w
        """
        b, c, h, w = x.size()

        window_num_in_h = math.ceil(h / self.window_size)
        window_num_in_w = math.ceil(w / self.window_size)

        # 1.pad feature maps to multiples of window size
        pad_b = h - window_num_in_h * self.window_size
        pad_r = w - window_num_in_w * self.window_size

        # pad last two dimension
        x = functional.pad(x, pad=(0, pad_b, 0, pad_r))

        # 2.possible window-shift
        # important: cyclic shift window
        if self.cyclic_shifter is not None:
            # shift_mask is not parameter
            x, shift_mask = self.cyclic_shifter.shift(x)
            shift_mask = repeat(shift_mask,
                                pattern="window_num_per_image mm1 mm2 -> (b window_num_per_image) heads mm1 mm2",
                                b=b, heads=self.heads)
        else:
            shift_mask = None

        # 3.window partition
        # x and shift_mask are both generated from window_partition,
        # so they have same relationship between view and storage
        x = window_partition(x, window_size=self.window_size)

        # 4.relative position embedding
        if self.use_relative_position_embedding:
            # (2m-1)(2m-1) heads
            relative_position_bias_table = self.relative_position_bias_mlp(self.relative_coords_table)

            relative_position_index = rearrange(self.relative_position_index, pattern="mm1 mm2 -> (mm1 mm2)")

            # mm1 mm2 heads
            relative_position_embedding = relative_position_bias_table[relative_position_index]
            relative_position_embedding = repeat(relative_position_embedding,
                                                 pattern="(mm1 mm2) heads -> new_b heads mm1 mm2",
                                                 new_b=b * window_num_in_h * window_num_in_w,
                                                 mm1=self.window_size ** 2,
                                                 mm2=self.window_size ** 2,
                                                 heads=self.heads)
            relative_position_embedding = 16 * torch.sigmoid(relative_position_embedding)
        else:
            relative_position_embedding = None

        # 5.MSA with residual
        attn_mask = add_tensor(relative_position_embedding, shift_mask)
        x = self.attn(x, attn_mask=attn_mask) + x

        # 6.feed forward with residual
        x = self.mlp(x) + x

        # 7.reverse window
        x = window_reverse(x, window_num_in_h=window_num_in_h, window_num_in_w=window_num_in_w,
                           window_size=self.window_size)

        # 8.reverse cyclic shift
        if self.shift_size > 0:
            x = self.cyclic_shifter.reverse(x)

        # 9.remove padding part of x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :h, :w]

        return x

    def generate_relative_coords_table(self):
        coords_h = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        coords_w = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)

        relative_coords_table = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        relative_coords_table = rearrange(relative_coords_table, "c m1 m2 -> (m1 m2) c")

        # normalize to -8, 8
        relative_coords_table[:, 0] /= (self.pretrained_window_size - 1)
        relative_coords_table[:, 1] /= (self.pretrained_window_size - 1)
        relative_coords_table *= 8

        relative_coords_table = (torch.sign(relative_coords_table) *
                                 torch.log2(torch.abs(relative_coords_table) + 1.0) / np.log2(8))

        self.register_buffer("relative_coords_table", relative_coords_table)

    def generate_relative_position_index(self):
        """ Same as Swin transformer
        """
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        # important: the first dimension represent the point coordinate
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # (2, m, m)
        coords_flatten = rearrange(coords, "c m1 m2 -> c (m1 m2)")
        # important: use broadcasting to calculate distance of each pair of points
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, m*m, m*m)
        relative_coords = rearrange(relative_coords, "c mm1 mm2 -> mm1 mm2 c")
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        # important: use one number to index two relative axis
        # plus 2M-1, because relative_coords extent is 0 -> 2m - 2
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        # relative_position_index: (m*m, m*m)
        # element value extent: 0 -> (2m-2)*(2m-1) + (2m-2) = (2m-1)(2m-1) - 1
        relative_position_index = relative_coords.sum(-1)
        # relative position index should not be considered parameter but should store
        # the different between buffer and attribute: buffer will be saved into state_dict but attribute not
        self.register_buffer("relative_position_index", relative_position_index)


class SwinTransformerV2Stage(nn.Module):
    """ one swin transformer v2 stage
    one stage is consist of:
        1. possible patch merging block for down sample
        2. several Swin Transformer Block
    """

    def __init__(self,
                 use_patch_merging: bool,
                 window_size: int,
                 pretrain_window_size: int,
                 use_relative_position_embedding: bool,
                 transformer_args: TransformerArgs,
                 ):
        """
        Args:
            use_patch_merging: weather use patch merging
            window_size: int
            use_relative_position_embedding: bool
            transformer_args: combining several swin transformer blocks is like one Transformer model
        """
        super().__init__()

        self.patch_merging = nn.Sequential(
            # the order of (new_h p1) is really important!!!
            Rearrange("b c (new_h p1) (new_w p2) -> b new_h new_w (p1 p2 c)", p1=2, p2=2),
            nn.Linear(2 * transformer_args.embedding_dim, transformer_args.embedding_dim),
            Rearrange("b new_h new_w new_c -> b new_c new_h new_w")) if use_patch_merging else None

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerV2Block(window_size=window_size,
                                   pretrained_window_size=pretrain_window_size,
                                   shift_size=0 if (i % 2 == 0) else window_size // 2,
                                   use_relative_position_embedding=use_relative_position_embedding,
                                   embedding_dim=transformer_args.embedding_dim,
                                   heads=transformer_args.heads,
                                   head_dim=transformer_args.embedding_dim // transformer_args.heads,
                                   mlp_ratio=transformer_args.mlp_ratio,
                                   dropout=transformer_args.dropout
                                   ) for i in range(transformer_args.depth)])

        # in paper: Transformers without Tears: Improving the Normalization of Self-Attention
        # usr normalization at the end of Transformer
        self.norm = nn.LayerNorm(transformer_args.embedding_dim)

    def forward(self, x):
        """
        Args:
            x: b c h w
        Returns:
            (b 2c h/2 w/2) if patch merging, else (b c h w)
        """
        x = self.patch_merging(x) if self.patch_merging else x

        # calculate attention mask for SW-MSA
        for block in self.blocks:
            x = deepspeed.checkpointing.checkpoint(block, x)
            # x = block(x)

        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x


class BaseSwinTransformerV2(nn.Module):
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_embedding_table", "absolute_position_embedding",
                "logit_scale", "relative_position_bias_mlp"}


@dataclasses.dataclass
class SwinTransformerV2StagesArgs:
    """
        use_absolute_position_embedding: bool
        use_relative_position_embedding: bool
        embedding_dim: Number of linear projection output channels
        depth_in_stages: Depths of each Swin Transformer stage.
        heads_in_stages: Number of attention head of each stage.
        out_indices: Output from which stages
        window_size: Window size.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        dropout: Dropout rate.
    """
    use_absolute_position_embedding: bool
    use_relative_position_embedding: bool
    window_size: int
    pretrain_window_size: int
    embedding_dim: int
    depth_in_stages: list[int]
    heads_in_stages: list[int]
    out_indices: list[int]
    mlp_ratio: int
    dropout: float



class SwinTransformerV2Stages(BaseSwinTransformerV2):
    """ several swin transformer stages: Swin Transformer backbone without patch embedding
    Equivalent to combine several Transformer model
    """

    def __init__(self, stages_args: SwinTransformerV2StagesArgs):

        super().__init__()
        self.num_stages = len(stages_args.depth_in_stages)
        self.out_indices = stages_args.out_indices
        self.window_size = stages_args.window_size
        self.use_absolute_position_embedding = stages_args.use_absolute_position_embedding

        # build layers
        embedding_dim_in_stages = [stages_args.embedding_dim * 2 ** i for i in range(self.num_stages)]

        # important: must initiate position embedding in this class
        # or each self-attention layer will have independent parameter for their own position embedding

        # In paper AN IMAGE IS WORTH 16X16 WORDS: only use absolute position embedding after patch embedding
        if stages_args.use_absolute_position_embedding:
            self.absolute_position_embedding = nn.Parameter(torch.randn(stages_args.window_size ** 2,
                                                                        stages_args.embedding_dim))

        self.stages = nn.ModuleList([SwinTransformerV2Stage(
            window_size=stages_args.window_size,
            pretrain_window_size=stages_args.pretrain_window_size,
            use_patch_merging=(i > 0),  # no patch merging in the first stage
            use_relative_position_embedding=stages_args.use_relative_position_embedding,
            transformer_args=TransformerArgs(
                depth=stages_args.depth_in_stages[i],
                embedding_dim=embedding_dim_in_stages[i],
                heads=stages_args.heads_in_stages[i],
                head_dim=embedding_dim_in_stages[i] // stages_args.heads_in_stages[i],
                mlp_ratio=stages_args.mlp_ratio,
                dropout=stages_args.dropout,
            )) for i in range(self.num_stages)])

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.size()
        window_num_in_h = math.ceil(h / self.window_size)
        window_num_in_w = math.ceil(w / self.window_size)

        if self.use_absolute_position_embedding:
            # important: this operation is a little difficult to understand, it can be divided to two steps:
            # 1. reshape the sequence embedding into the shape of a window: (mm, c) -> (m, m, c)
            # 2. copy the window to cover the entire input x: (m, m, c) -> (b, c, n*m, n*m)
            # understanding the order in (n*m) is really important: combining n pieces of m
            absolute_position_embedding = repeat(self.absolute_position_embedding,
                                                 pattern="(m1 m2) c -> c (window_num_in_h m1) (window_num_in_w m2)",
                                                 c=c,
                                                 window_num_in_h=window_num_in_h,
                                                 window_num_in_w=window_num_in_w,
                                                 m1=self.window_size, m2=self.window_size)

            x += absolute_position_embedding[None, :, :h, :w]

        outs = []
        for i in range(self.num_stages):
            stage = self.stages[i]
            x = stage(x)

            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)


class SwinTransformerV2(BaseSwinTransformerV2):
    """ Swin Transformer backbone
    A PyTorch implement of : "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
    """

    def __init__(self,
                 image_channels: int,
                 patch_size: int,
                 stages_args: SwinTransformerV2StagesArgs):
        """
        Args:
            image_channels: Number of input image channels
            patch_size: Patch size
            stages_args: args for swin transformer stages
        """
        super().__init__()

        # split image into non-overlapping patches
        self.patch_embed = nn.Sequential(
            Rearrange(pattern="b c (p1 new_h) (p2 new_w) -> b new_h new_w (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.Linear(patch_size ** 2 * image_channels, stages_args.embedding_dim),
            Rearrange("b new_h new_w new_c -> b new_c new_h new_w"))

        self.stages = SwinTransformerV2Stages(stages_args)

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)

        outs = self.stages(x)

        return outs


