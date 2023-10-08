import torch
from torch import nn
from einops import rearrange, einsum
import dataclasses
from torch.nn import functional


class PreNorm(nn.Module):
    """ pre-norm residual unit is more stable than post-norm residual unit in deep transformer
    References:
        paper: Learning Deep Transformer Models for Machine Translation
        paper: On layer normalization in the transformer architecture
    """

    def __init__(self, embedding_dim: int, layer: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.layer = layer

    def forward(self, x: torch.Tensor, **kwargs):
        return self.layer(self.norm(x), **kwargs)


class Attention(nn.Module):
    """ Self-Attention Layer in Attention is all you need
    """

    def __init__(self, embedding_dim, heads, head_dim, dropout):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.to_qkv = nn.Linear(embedding_dim, inner_dim * 3, bias=False)
        self.output = nn.Linear(inner_dim, embedding_dim)
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        """
        Args:
            x: (b, l, c)
            attn_mask: (b, h, l, l)

        Returns:
            (b, l, c)
        """
        b, l, _ = x.size()
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b l (n d) -> b n l d', n=self.heads), qkv)
        if x.device.type == "cuda":
            # memory efficient attention is only available for cuda
            # https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html
            x = functional.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=self.dropout_rate)
        else:
            attn = einsum(q, k, 'b n l1 d, b n l2 d -> b n l1 l2').mul(self.scale)
            attn += attn_mask if attn_mask is not None else 0
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            x = einsum(attn, v, 'b n l1 l2, b n l2 d -> b n l1 d')
        x = rearrange(x, 'b n l d -> b l (n d)')
        x = self.output(x)
        return x


class MLP(nn.Module):

    def __init__(self, embedding_dim: int, hidden_dim_ratio: int,
                 dropout: float, act_layer: nn.Module = nn.GELU):
        super().__init__()
        hidden_dim = embedding_dim * hidden_dim_ratio
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                 act_layer(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, embedding_dim))

    def forward(self, x):
        return self.mlp(x)


@dataclasses.dataclass
class TransformerArgs:
    """
    Args:
        depth: number of transformer layer
        embedding_dim: dimension of input token
        heads: head num
        head_dim: dimension of qkv in each head
        mlp_ratio: hidden dim ratio of embedding dim in mlp layer
        dropout: dropout ratio
    """
    depth: int
    embedding_dim: int
    heads: int
    head_dim: int
    mlp_ratio: int
    dropout: float


class Transformer(nn.Module):
    def __init__(self, transformer_args: TransformerArgs):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(transformer_args.depth):
            self.layers.append(nn.ModuleList([
                PreNorm(transformer_args.embedding_dim,
                        Attention(transformer_args.embedding_dim, heads=transformer_args.heads,
                                  head_dim=transformer_args.head_dim, dropout=transformer_args.dropout)),
                PreNorm(transformer_args.embedding_dim,
                        MLP(transformer_args.embedding_dim,
                            hidden_dim_ratio=transformer_args.mlp_ratio,
                            dropout=transformer_args.dropout))
            ]))
        # in paper: Transformers without Tears: Improving the Normalization of Self-Attention
        # In pre-norm residual unit, must append an additional normalization after both encoder and decoder
        # so their outputs are appropriately scaled.
        self.norm = nn.LayerNorm(transformer_args.embedding_dim)

    def forward(self, x):
        for attn, mlp in self.layers:
            x = attn(x) + x
            x = mlp(x) + x
        # layer normalization before return
        return self.norm(x)
