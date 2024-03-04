import torch 
from torch import nn, Tensor, einsum
from zeta.nn import SwiGLUStacked, MultiQueryAttention, img_to_text

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
        dropout: float = 0.,
        *args,
        **kwargs
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
        self.act = SwiGLUStacked(dim, dim * mlp_mult, dropout=dropout, *args, **kwargs)
        
        # Attention
        self.attn = MultiQueryAttention(
            dim,
            heads,
            *args,
            **kwargs
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
    
    
x = torch.randn(1, 3, 224, 224)
model = VisionLlamaBlock(768, 12, 3, 12)
print(model(x).shape)

print(model(x))