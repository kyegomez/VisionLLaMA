import math
import torch
from torch import nn, Tensor
from zeta.nn import (
    SwiGLUStacked,
    MultiQueryAttention,
)
from local_attention import LocalAttention
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from math import sqrt, pi


# Pair
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class AS2DRoPE:
    def __init__(self, dim, anchor_resolution: int = 56):
        """
        Initialize the AS2DRoPE class for 3D input (BATCH, SEQLENGTH, Dimension).

        Args:
        dim (int): The dimensionality of the model.
        anchor_resolution (int): The anchor resolution B used during training.
        """
        self.dim = dim
        self.anchor_resolution = anchor_resolution
        self.theta = self._get_theta(dim)

    def _get_theta(self, dim):
        """
        Calculate the theta values for the sin and cos functions based on the model dimensionality.

        Args:
        dim (int): The dimensionality of the model.

        Returns:
        torch.Tensor: The theta values.
        """
        theta = torch.tensor(
            [10000 ** (-2 * (i // 2) / dim) for i in range(dim)]
        )
        return theta

    def _get_positional_matrix(self, i, j, H):
        """
        Generate the positional matrix for a specific position and input resolution.

        Args:
        i (int): The row position.
        j (int): The column position.
        H (int): The current resolution H of the input image.

        Returns:
        torch.Tensor: The positional matrix R_i,j.
        """
        position_i = (i * self.anchor_resolution / H).float()
        position_j = (j * self.anchor_resolution / H).float()

        theta_i = self.theta * position_i
        theta_j = self.theta * position_j

        cos_i = torch.cos(theta_i)
        sin_i = torch.sin(theta_i)

        cos_j = torch.cos(theta_j)
        sin_j = torch.sin(theta_j)

        R_i = torch.eye(self.dim)
        R_j = torch.eye(self.dim)

        R_i[0::2, 0::2] = cos_i
        R_i[1::2, 1::2] = cos_i
        R_i[0::2, 1::2] = -sin_i
        R_i[1::2, 0::2] = sin_i

        R_j[2::4, 2::4] = cos_j
        R_j[3::4, 3::4] = cos_j
        R_j[2::4, 3::4] = -sin_j
        R_j[3::4, 2::4] = sin_j

        R = (
            R_i * R_j
        )  # Element-wise multiplication for combining the matrices
        return R

    def forward(self, x: Tensor, H: int):
        """
        Apply the AS2DRoPE encoding to the input tensor.

        Args:
        x (torch.Tensor): The input tensor with shape [batch_size, seq_length, dim].
        H (int): The side length of the square resolution of the input image.

        Returns:
        torch.Tensor: The positionally encoded tensor.
        """
        B, L, _ = x.shape
        H_sqrt = int(math.sqrt(L))  # Assuming L is a perfect square
        encoded_x = torch.zeros_like(x)

        # Convert linear sequence positions back to 2D grid positions
        for pos in range(L):
            i = pos // H_sqrt
            j = pos % H_sqrt
            R_ij = self._get_positional_matrix(
                torch.tensor(i), torch.tensor(j), torch.tensor(H)
            )
            encoded_x[:, pos, :] = torch.matmul(x[:, pos, :], R_ij)

        return encoded_x


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
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape

        # Patch Embedding
        x = self.to_patch_embedding(x)

        # Reshape to text
        # x = img_to_text(x, self.channels, self.dim)

        skip_1 = x

        # Norm
        x = self.norm(x)

        # as2d rope
        x, _ = self.axial_rotary(x)

        # Attn
        x, _, _ = self.attn(x)
        x = x + skip_1

        # norm
        x = self.norm(x)

        skip_2 = x

        # SWIGLU
        x = self.act(x)

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
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        # As2drope
        self.as2drope = AS2DRoPE(dim)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape

        # Patch Embedding
        x = self.to_patch_embedding(x)

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
