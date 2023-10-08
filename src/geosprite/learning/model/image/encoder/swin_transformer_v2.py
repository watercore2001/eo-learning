import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import functional

from .swin_transformer import MLP


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    """

    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0., proj_drop: float = 0., pretrained_window_size: int = 0):
        """
        Args:
            dim: Number of input channels
            window_size: The height and width of the window
            num_heads: Number of attention heads
            qkv_bias: If True, add a learnable bias to query, key, value
            attn_drop: Dropout ratio of attention weight
            proj_drop: Dropout ratio of output
            pretrained_window_size: The height and width of the window in pretraining
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(torch.ones((num_heads, 1, 1)).mul(10)), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.relative_position_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                                   nn.ReLU(inplace=True),
                                                   nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2M-1, 2*M-1, 2
        if pretrained_window_size > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) \
                                * torch.log2(torch.abs(relative_coords_table).add(1.0)) \
                                / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, M, M
        coords_flatten = torch.flatten(coords, 1)  # 2, M*M
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, M*M, M*M
        # transpose the cords dimension to the last dimension for calculate
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # M*M, M*M, 2
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # M*M, M*M
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: input features with shape of (window_num * batch size, n=M*M, c)
            mask: (0/-inf) mask with shape of (window_num, n, n) or None
        """
        window_num_in_batches, n, c = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = functional.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(window_num_in_batches, n, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torch script happy (cannot use tensor as tuple)

        # cosine attention
        attn = (functional.normalize(q, dim=-1) @ functional.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.relative_position_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # M*M, M*M, heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # heads, M*M, M*M
        relative_position_bias = torch.sigmoid(relative_position_bias).mul(16)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            window_num = mask.shape[0]
            attn = attn.view(window_num_in_batches // window_num, window_num, self.num_heads, n, n) \
                   + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(window_num_in_batches, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, n: int):
        """ calculate flops for one window
        Args:
            n: tokens in one window
        """
        flops = 0
        # qkv = self.qkv(x)
        flops += n * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * n * (self.dim // self.num_heads) * n
        #  x = (attn @ v)
        flops += self.num_heads * n * n * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += n * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block."""

    def __init__(self, dim: int, input_resolution: tuple[int, int], num_heads: int, window_size: int = 7,
                 shift_size: int = 0,
                 mlp_ratio: float = 4., qkv_bias: bool = True, proj_dropout: float = 0., attn_dropout: float = 0.,
                 drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm, pretrained_window_size: int = 0):
        """

        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            num_heads: Number of attention heads.
            window_size: Window size
            shift_size: Shift size for SW-MSA.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_dropout: Dropout rate
            attn_dropout: Attention dropout rate
            drop_path: Stochastic depth rate
            act_layer: Activation layer
            norm_layer: Normalization layer.
            pretrained_window_size: Window size in pretraining.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_dropout, proj_drop=proj_dropout,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(input_dim=dim, hidden_dim_ratio=mlp_ratio, act_layer=act_layer,
                       dropout=proj_dropout)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            h, w = self.input_resolution
            img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        h, w = self.input_resolution
        b, n, c = x.shape
        assert n == h * w, "input feature has wrong size"

        shortcut = x
        x = x.view(b, h, w, c)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nW*b, window_size*window_size, c

        # w-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*b, window_size*window_size, c

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(b, h * w, c)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        h, w = self.input_resolution
        # norm1
        flops += self.dim * h * w
        # w-MSA/SW-MSA
        window_num = h * w / self.window_size / self.window_size
        flops += window_num * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * h * w
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, n, c = x.shape
        assert n == h * w, "input feature has wrong size"
        assert h % 2 == 0 and w % 2 == 0, f"x size ({h}*{w}) are not even."

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        h, w = self.input_resolution
        flops = (h // 2) * (w // 2) * 4 * self.dim * 2 * self.dim
        flops += h * w * self.dim // 2
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage """

    def __init__(self, dim: int, input_resolution: tuple[int, int], depth: int, num_heads: int, window_size: int,
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop: float = 0., attn_drop: float = 0.,
                 drop_path: list[float] | float = 0., norm_layer: nn.Module = nn.LayerNorm,
                 down_sample: nn.Module = None,
                 use_checkpoint: bool = False, pretrained_window_size: int = 0):
        """
        Args:
            dim: Number of input channels
            input_resolution: Input resolution.
            depth: Number of blocks.
            num_heads: Number of attention heads.
            window_size: Local window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate
            norm_layer: Normalization layer.
            down_sample: Down sample layer at the end of the layer.
            use_checkpoint: Whether to use checkpointing to save memory.
            pretrained_window_size: Local window size in pretraining
        """

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 proj_dropout=drop, attn_dropout=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        if down_sample is not None:
            self.down_sample = down_sample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.down_sample = None

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (b, h*w, c)
        """
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.down_sample is not None:
            x_down = self.down_sample(x)
            return x, *self.input_resolution, x_down
        else:
            return x, *self.input_resolution, x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.down_sample is not None:
            flops += self.down_sample.flops()
        return flops

    def init_res_post_norm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """

    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96,
                 norm_layer: nn.Module = None):
        """

        Args:
            img_size: Image size.
            patch_size: Patch token size.
            in_chans: Number of input image channels.
            embed_dim: Number of linear projection output channels.
            norm_layer: Normalization layer.
        """
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.patch_num = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        b, c, h, w = x.shape
        # look at relaxing size constraints
        assert h == self.img_size[0] and w == self.img_size[1], \
            f"Input image size ({h}*{w}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # b patch_num c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        patch_num_in_height, patch_num_in_width = self.patches_resolution
        flops = patch_num_in_height * patch_num_in_width * self.embed_dim * self.in_chans * (
                self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += patch_num_in_height * patch_num_in_width * self.embed_dim
        return flops


class SwinTransformerV2(nn.Module):
    """ Swin Transformer v2 backbone
        A PyTorch implement of : "Swin Transformer V2: Scaling Up Capacity and Resolution"
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3,
                 embed_dim: int = 96, depths: tuple[int] = (2, 2, 6, 2), num_heads: tuple[int] = (3, 6, 12, 24),
                 window_size: int = 7, mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path_rate: float = 0.1,
                 norm_layer: nn.Module = nn.LayerNorm, ape: bool = False, patch_norm: bool = True,
                 use_checkpoint: bool = False, pretrained_window_sizes: tuple[int] = (0, 0, 0, 0),
                 out_indices: tuple[int] = (0, 1, 2, 3), ):
        """

        Args:
            img_size: Input image size
            patch_size: Patch size.
            in_chans: Number of classes for classification head.
            embed_dim: Patch embedding dimension.
            depths: Depth of each Swin Transformer layer.
            num_heads: Number of attention heads in different layers.
            window_size: Window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            drop_rate: Dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            norm_layer: Normalization layer.
            ape: If True, add absolute position embedding to the patch embedding.
            patch_norm: If True, add normalization after patch embedding.
            use_checkpoint: whether to use checkpointing to save memory.
            pretrained_window_sizes: Pretrained window sizes of each layer.
            out_indices: Output from which stages.
        """
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.out_indices = out_indices

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.patch_num
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               down_sample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)
        for layer in self.layers:
            layer.init_res_post_norm()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        outs = []
        for i, layer in enumerate(self.layers):
            x_out, h_out, w_out, x = layer(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, h_out, w_out, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return tuple(outs)

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
