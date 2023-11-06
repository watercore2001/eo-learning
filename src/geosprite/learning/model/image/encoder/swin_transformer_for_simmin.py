import torch

from .swin_transformer import SwinTransformerStages, SwinTransformerStagesArgs
from torch import nn
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

__all__ = ["SwinTransformerForSimMIMDebug", "SwinTransformerForSimMIMBase"]


class SwinTransformerForSimMIM(nn.Module):
    def __init__(self,
                 image_channels: int,
                 patch_size: int,
                 stage_args: SwinTransformerStagesArgs):
        super().__init__()
        self.image_channels = image_channels
        self.patch_size = patch_size
        self.stage_args = stage_args

        self.patch_embedding_layers = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=self.stage_args.embedding_dim,
                      kernel_size=self.patch_size,
                      stride=self.patch_size)
            for _ in range(self.image_channels)
        ])

        # initial mask tokens for each channel
        self.mask_tokens = nn.Parameter(torch.zeros(self.image_channels, self.stage_args.embedding_dim))
        trunc_normal_(self.mask_tokens, mean=0., std=.02)

        self.fusion_channels = nn.Linear(in_features=self.image_channels*self.stage_args.embedding_dim,
                                         out_features=self.stage_args.embedding_dim)
        self.swin_transformer = SwinTransformerStages(stage_args)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"mask_tokens"} | self.swin_transformer.no_weight_decay_keywords()

    def forward(self, batch: dict):
        x, mask = batch["x"], batch["mask"]

        b, c, h, w = x.shape

        # 1. patch embedding for each channel
        x = torch.split(x, split_size_or_sections=1, dim=1)
        x = [embedding_layer(channel_x) for embedding_layer, channel_x in zip(self.patch_embedding_layers, x)]
        new_h, new_w = h // self.patch_size, w // self.patch_size
        x = rearrange(x, pattern="c b d new_h new_w -> b c (new_h new_w) d")

        # 2. add mask
        mask = rearrange(mask, pattern="b c new_h new_w -> b c (new_h new_w) 1")
        mask_tokens = repeat(self.mask_tokens, pattern="c d -> b c (new_h new_w) d", b=b, new_h=new_h, new_w=new_w)

        x = x * (1-mask) + mask_tokens * mask

        # 3. fusion all channels
        x = rearrange(x, pattern="b c l d -> b l (c d)")
        x = self.fusion_channels(x)
        x = rearrange(x, pattern="b (new_h new_w) d -> b d new_h new_w", new_h=new_h, new_w=new_w)

        # 4. swin transformer
        x = self.swin_transformer(x)

        return x


class SwinTransformerForSimMIMDebug(SwinTransformerForSimMIM):
    def __init__(self, image_channels: int):
        patch_size = 16
        stage_args = SwinTransformerStagesArgs(
            use_absolute_position_embedding=False,
            use_relative_position_embedding=True,
            window_size=4,
            embedding_dim=32,
            depth_in_stages=[2, 2],
            heads_in_stages=[4, 8],
            out_indices=[0, 1],
            mlp_ratio=1,
            dropout=0)
        super().__init__(image_channels=image_channels, patch_size=patch_size, stage_args=stage_args)


class SwinTransformerForSimMIMBase(SwinTransformerForSimMIM):
    def __init__(self, image_channels: int):
        patch_size = 4
        stages_args = SwinTransformerStagesArgs(
            use_absolute_position_embedding=False,
            use_relative_position_embedding=True,
            window_size=8,
            embedding_dim=128,
            depth_in_stages=[2, 2, 18, 2],
            heads_in_stages=[4, 8, 16, 32],
            out_indices=[0, 1, 2, 3],
            mlp_ratio=4,
            dropout=0)
        super().__init__(image_channels=image_channels, patch_size=patch_size, stage_args=stages_args)





