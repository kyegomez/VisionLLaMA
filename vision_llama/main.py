import torch
from torch import nn, Tensor
from zeta.nn import (
    SwiGLUStacked,
    MultiQueryAttention,
    img_to_text,
    
)
from local_attention import LocalAttention
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from math import sqrt, pi


# Pair
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class AS2DRoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        channels: int,
    ):
        super().__init__()
        self.dim = dim
        self.channels = channels

    def forward(self, x: Tensor) -> Tensor:
        pass


class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq=10):
        super().__init__()
        self.dim = dim
        scales = torch.linspace(1.0, max_freq / 2, self.dim // 4)
        self.register_buffer("scales", scales)

    def forward(self, x):
        device, dtype, n = x.device, x.dtype, int(sqrt(x.shape[-2]))

        seq = torch.linspace(-1.0, 1.0, steps=n, device=device)
        seq = seq.unsqueeze(-1)

        scales = self.scales[
            (*((None,) * (len(seq.shape) - 1)), Ellipsis)
        ]
        scales = scales.to(x)

        seq = seq * scales * pi

        x_sinu = repeat(seq, "i d -> i j d", j=n)
        y_sinu = repeat(seq, "j d -> i j d", i=n)

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim=-1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim=-1)

        sin, cos = map(
            lambda t: rearrange(t, "i j d -> (i j) d"), (sin, cos)
        )
        sin, cos = map(
            lambda t: repeat(t, "n d -> () n (d j)", j=2), (sin, cos)
        )
        return sin, cos


class VisionLlamaBlock(nn.Module):
    """
    VisionLlamaBlock is a module that represents a block in the VisionLLaMA model.

    Args:
        dim (int): The input dimension.
        depth (int): The depth of the block.
        channels (int): The number of channels in the input.
        heads (int): The number of attention heads.
        dim_head (int, optional): The dimension of each attention head. Defaults to 64.
        mlp_mult (int, optional): The multiplier for the hidden dimension in the MLP. Defaults to 4.
        dropout (float, optional): The dropout rate. Defaults to 0.

    Attributes:
        dim (int): The input dimension.
        depth (int): The depth of the block.
        channels (int): The number of channels in the input.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        mlp_mult (int): The multiplier for the hidden dimension in the MLP.
        dropout (float): The dropout rate.
        norm (nn.LayerNorm): The layer normalization module.
        act (SwiGLU): The SwiGLU activation module.
        attn (MultiQueryAttention): The multi-query attention module.

    """

    def __init__(
        self,
        dim: int,
        depth: int,
        channels: int = 3,
        heads: int = 12,
        dim_head: int = 64,
        mlp_mult: int = 4,
        dropout: float = 0.0,
        image_size: int = 224,
        patch_size: int = 16,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.channels = channels
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_mult = mlp_mult
        self.dropout = dropout
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        patch_dim = channels * patch_height * patch_width

        # Norm
        self.norm = nn.LayerNorm(dim)

        # Swiglu
        self.act = SwiGLUStacked(
            dim, dim * mlp_mult, dropout=dropout, *args, **kwargs
        )

        # Attention
        self.attn = MultiQueryAttention(
            dim,
            heads,
            *args,
            **kwargs,
            # qk_ln=True,
        )

        # AxialRotaryEmbedding
        self.axial_rotary = AxialRotaryEmbedding(dim)
        
        # To patch embeddings
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1 = patch_height,
                p2 = patch_width
            ),
            nn.Linear(patch_dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        
        # Patch Embedding
        x = self.to_patch_embedding(x)
        print(x.shape)
        
        # Reshape to text
        # x = img_to_text(x, self.channels, self.dim)
        print(x.shape)
        
        skip_1 = x
        print(x.shape)

        # Norm
        x = self.norm(x)
        print(x.shape)

        # as2d rope
        x, _ = self.axial_rotary(x)
        print(x.shape)

        # Attn
        x, _, _ = self.attn(x)
        x = x + skip_1
        print(x.shape)

        # norm
        x = self.norm(x)

        skip_2 = x

        # SWIGLU
        x = self.act(x)
        print(x.shape)

        return x + skip_2


class VisionLlamaPyramidBlock(nn.Module):
    """
    VisionLlamaPyramidBlock is a module that represents a single block in the Vision Llama Pyramid network.

    Args:
        dim (int): The input dimension.
        depth (int): The depth of the block.
        channels (int): The number of channels in the input image.
        heads (int): The number of attention heads.
        dim_head (int, optional): The dimension of each attention head. Defaults to 64.
        mlp_mult (int, optional): The multiplier for the MLP dimension. Defaults to 4.
        dropout (float, optional): The dropout rate. Defaults to 0.0.
        local_window_size (int, optional): The window size for local attention. Defaults to 512.

    Attributes:
        dim (int): The input dimension.
        depth (int): The depth of the block.
        channels (int): The number of channels in the input image.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        mlp_mult (int): The multiplier for the MLP dimension.
        dropout (float): The dropout rate.
        norm (nn.LayerNorm): The layer normalization module.
        act (SwiGLUStacked): The SwiGLU activation module.
        attn (MultiQueryAttention): The multi-query attention module.
        local_attn (LocalAttention): The local attention module.

    """

    def __init__(
        self,
        dim: int,
        depth: int,
        channels: int,
        heads: int,
        dim_head: int = 64,
        mlp_mult: int = 4,
        dropout: float = 0.0,
        local_window_size: int = 512,
        image_size: int = 224,
        patch_size: int = 16,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.channels = channels
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_mult = mlp_mult
        self.dropout = dropout
        
        # Image height
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        patch_dim = channels * patch_height * patch_width
        

        # Layernorm
        self.norm = nn.LayerNorm(dim)

        # Swiglu
        self.act = SwiGLUStacked(
            dim, dim * mlp_mult, dropout=dropout, *args, **kwargs
        )

        # Attention
        self.attn = MultiQueryAttention(
            dim,
            heads,
            *args,
            **kwargs,
            # qk_ln=True,
        )
        
        # AxialRotaryEmbedding
        self.rotary_embed = AxialRotaryEmbedding(dim)

        # Local Attention
        self.local_attn = LocalAttention(
            window_size=local_window_size,
            causal=True,
            dim=dim,
            autopad=True,
            shared_qk=True,
            use_rotary_pos_emb=True,
            # use_xpos=True,
        )
        
        # To patch embeddings
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1 = patch_height,
                p2 = patch_width
            ),
            nn.Linear(patch_dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        
        # Patch Embedding
        x = self.to_patch_embedding(x)

        # Convert image to text
        # x = img_to_text(x, self.channels, self.dim)

        # Skip connection
        skip_1 = x

        # Norm
        normed = self.norm(x)

        # as2drope
        normed, _ = self.rotary_embed(normed)

        # Local Attention with skip connect
        x = self.local_attn(normed, normed, normed) + skip_1

        # 2nd skip connection
        skip_2 = x

        # Norm2
        x = self.norm(x)

        # SWIGLU
        x = self.act(x) + skip_2

        # Now 2nd phase with norm
        x = self.norm(x)

        # as2drope
        x, _ = self.rotary_embed(x)

        # Global Attention
        x, _, _ = self.attn(x) 
        x = x + skip_1

        # residual connection
        skip_3 = x
        
        # Norm
        x = self.norm(x)

        # SWIGLU
        x = self.act(x) + skip_3

        # Skip connection

        return x
