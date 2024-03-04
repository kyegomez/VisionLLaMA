from torch import nn, Tensor
from zeta.nn import (
    SwiGLUStacked,
    MultiQueryAttention,
    img_to_text,
    LocalAttention,
)


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
        channels: int,
        heads: int,
        dim_head: int = 64,
        mlp_mult: int = 4,
        dropout: float = 0.0,
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

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape

        x = img_to_text(x, self.channels, self.dim)
        skip_1 = x
        print(x.shape)

        # Norm
        x = self.norm(x)

        # as2d rope

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

        # Local Attention
        self.local_attn = LocalAttention(
            window_size=local_window_size,
            causal=True,
            dim=dim,
            autopad=True,
            shared_qk=True,
            use_rotary_pos_emb=True,
            use_xpos=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape

        # Convert image to text
        x = img_to_text(x, self.channels, self.dim)

        # Skip connection
        skip_1 = x

        # Norm
        normed = self.norm(x)

        # as2drope
        # pass

        # Local Attention with skip connect
        x = self.local_attn(normed) + skip_1

        # 2nd skip connection
        skip_2 = x

        # Norm2
        x = self.norm(x)

        # SWIGLU
        x = self.act(x) + skip_2

        # Now 2nd phase with norm
        x = self.norm(x)

        # as2drope

        # Global Attention

        # residual connection

        # Norm

        # SWIGLU

        # Skip connection

        # return
