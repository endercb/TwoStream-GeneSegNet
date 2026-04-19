import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelWiseCrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, reduction=16):
        super().__init__()
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.reduction = reduction
        
        # MLP to compute attention weights from concatenated global features
        hidden_dim = max(dim_q // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(dim_q + dim_kv, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim_q),
            nn.Sigmoid()
        )
        
    def forward(self, x_q, x_kv):
        """
        x_q: [B, C1, H, W]   (Stream 1 features)
        x_kv: [B, C2, H, W]  (Stream 2 features)
        Returns modulated x_q with same shape
        """
        B, C1, H, W = x_q.shape
        # Global average pooling to get channel descriptors
        q_global = F.adaptive_avg_pool2d(x_q, 1).squeeze(-1).squeeze(-1)   # [B, C1]
        k_global = F.adaptive_avg_pool2d(x_kv, 1).squeeze(-1).squeeze(-1)  # [B, C2]
        
        # Concatenate and compute attention weights per channel of x_q
        concat = torch.cat([q_global, k_global], dim=-1)   # [B, C1 + C2]
        weights = self.mlp(concat)  # [B, C1]
        
        # Modulate x_q
        x_q_modulated = x_q * weights.unsqueeze(-1).unsqueeze(-1)
        return x_q_modulated

class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim_q // num_heads) ** -0.5

        self.to_q  = nn.Conv2d(dim_q,  dim_q,  1)
        self.to_k  = nn.Conv2d(dim_kv, dim_q,  1)   # KV → Q uzayına projeksiyon
        self.to_v  = nn.Conv2d(dim_kv, dim_q,  1)
        self.out   = nn.Conv2d(dim_q,  dim_q,  1)
        self.norm  = nn.LayerNorm(dim_q)

    def forward(self, x_q, x_kv):
        """
        x_q  : [B, C1, H, W]  ← Bu stream sorguyu yollar
        x_kv : [B, C2, H, W]  ← Diğer stream anahtar/değer üretir
        """
        B, C, H, W = x_q.shape
        Q = self.to_q(x_q)   # [B, C1, H, W]
        K = self.to_k(x_kv)  # [B, C1, H, W]
        V = self.to_v(x_kv)  # [B, C1, H, W]

        # Spatial attention (flatten spatial dims)
        Q = Q.flatten(2)     # [B, C1, H*W]
        K = K.flatten(2)
        V = V.flatten(2)

        attn = torch.softmax(Q.transpose(1,2) @ K * self.scale, dim=-1)  # [B, H*W, H*W]
        out  = (attn @ V.transpose(1,2)).transpose(1,2)                   # [B, C1, H*W]
        out  = out.reshape(B, C, H, W)
        out  = self.out(out) + x_q   # Residual bağlantı
        return out