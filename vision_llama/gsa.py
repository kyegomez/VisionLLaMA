import torch
from torch import nn, Tensor
from einops import rearrange
from einops.layers.torch import Rearrange

class GSA(nn.Module):
    """
    Graph Self-Attention module.

    Args:
        dim (int): The input dimension.
        heads (int, optional): The number of attention heads. Defaults to 8.
        sub_sample_ratio (float, optional): The subsampling ratio for keys and values. Defaults to 0.5.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        sub_sample_ratio: int = 4,
    ):
        super().__init__()
        self.heads = heads
        self.sub_sample_ratio = sub_sample_ratio
        self.scale = (dim // heads) ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
        # Subsampling layer for keys and values
        self.subsample = Rearrange(
            "b n (h d) -> b h n d",
            h=heads
        )
    
    def forward(self, x: Tensor):
        """
        Forward pass of the GSA module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying self-attention.
        """
        b, n, _, h = *x.shape, self.heads
        
        
        # Generate queries, keys, and values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        # Apply subsampling on keys and values
        k, v = map(lambda t: self.subsample(t), (k, v))
        
        # Perform sub sampling by taking every sub sample ratio
        k = k[:, :, ::self.sub_sample_ratio, :]
        v = v[:, :, ::self.sub_sample_ratio, :]
        
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        
        # Cal attn
        attn = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        
        # Recombine heads
        out = rearrange(out, "b h n d -> b n (h d)")
        
        return self.to_out(out)
    
    
