import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet_torch
from cross_attention import ChannelWiseCrossAttention

class TwoStreamDownsample(nn.Module):
    """
    Two-stream encoder with cross-attention at specified layers.
    Each stream has its own downsample path.
    Cross-attention applied at certain depths (e.g., Down2, Down3).
    """
    def __init__(self, nbase1, nbase2, sz, residual_on=True, cross_attn_layers=(2,3)):
        super().__init__()
        self.nbase1 = nbase1  # channel sizes for stream 1 (DAPI)
        self.nbase2 = nbase2  # channel sizes for stream 2 (HeatMap_all)
        self.sz = sz
        self.residual_on = residual_on
        self.cross_attn_layers = cross_attn_layers  # indices where cross-attention applied (0-indexed)
        
        # Build downsampling blocks for each stream
        self.down1 = nn.Sequential()
        self.down2 = nn.Sequential()
        self.maxpool = nn.MaxPool2d(2, 2)
        
        # Stream 1 encoder blocks
        for n in range(len(nbase1)-1):
            if residual_on:
                block = resnet_torch.resdown(nbase1[n], nbase1[n+1], sz)
            else:
                block = resnet_torch.convdown(nbase1[n], nbase1[n+1], sz)
            self.down1.add_module('down1_%d'%n, block)
        
        # Stream 2 encoder blocks
        for n in range(len(nbase2)-1):
            if residual_on:
                block = resnet_torch.resdown(nbase2[n], nbase2[n+1], sz)
            else:
                block = resnet_torch.convdown(nbase2[n], nbase2[n+1], sz)
            self.down2.add_module('down2_%d'%n, block)
        
        # Cross-attention modules at specified layers
        self.cross_attns = nn.ModuleDict()
        for layer_idx in cross_attn_layers:
            if layer_idx < len(nbase1)-1 and layer_idx < len(nbase2)-1:
                dim_q = nbase1[layer_idx+1]  # output channels of that layer for stream1
                dim_kv = nbase2[layer_idx+1] # output channels for stream2
                # Use channel-wise cross-attention for memory efficiency
                # Unidirectional: stream2 (RNA) modulates stream1 (DAPI)
                self.cross_attns[f'layer{layer_idx}_2to1'] = ChannelWiseCrossAttention(dim_kv, dim_q)
    
    def forward(self, x1, x2):
        """
        x1: [B, C1_in, H, W] stream1 input (DAPI)
        x2: [B, C2_in, H, W] stream2 input (HeatMap_all)
        Returns:
            x1_out: list of feature maps per layer for stream1
            x2_out: list of feature maps per layer for stream2
        """
        x1_out = []
        x2_out = []
        
        # Initial features before any downsampling
        y1 = x1
        y2 = x2
        
        for n in range(len(self.down1)):
            # Downsample if not first layer
            if n > 0:
                y1 = self.maxpool(y1)
                y2 = self.maxpool(y2)
            
            # Process through each stream's block
            y1 = self.down1[n](y1)
            y2 = self.down2[n](y2)
            
            # Apply cross-attention at this layer if specified
            if n in self.cross_attn_layers:
                # Unidirectional cross-attention: stream2 (RNA) modulates stream1 (DAPI)
                y1_mod = self.cross_attns[f'layer{n}_2to1'](y1, y2)  # stream2 modulates stream1
                # stream2 (RNA) remains unchanged
                y1 = y1_mod
            
            x1_out.append(y1)
            x2_out.append(y2)
        
        return x1_out, x2_out

class TwoStreamCPnet(nn.Module):
    def __init__(self, nbase1, nbase2, nout, sz,
                 residual_on=True, style_on=True, concatenation=False,
                 mkldnn=False, diam_mean=30., cross_attn_layers=(2,3)):
        super(TwoStreamCPnet, self).__init__()
        self.nbase1 = nbase1  # stream1 channels e.g., [1, 32, 64, 128, 256]
        self.nbase2 = nbase2  # stream2 channels e.g., [N, 32, 64, 128, 256]
        self.nout = nout
        self.sz = sz
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation
        self.mkldnn = mkldnn if mkldnn is not None else False
        self.cross_attn_layers = cross_attn_layers
        
        # Input stems to project to first channel size
        self.stem1 = nn.Conv2d(nbase1[0], nbase1[1], 3, padding=1)
        self.stem2 = nn.Conv2d(nbase2[0], nbase2[1], 3, padding=1)
        
        # Convert cross_attn_layers to stem‑after indices (since we use stem‑after channel list)
        # Original cross_attn_layers refer to down‑block indices in the full list (including stem).
        # Down1 is index 1, Down2 index 2, Down3 index 3.
        # We subtract 1 to get indices in the stem‑after list.
        cross_attn_layers_stem = tuple(l-1 for l in cross_attn_layers)
        # Two-stream encoder: stem‑after channels plus repeated last channel for extra down block
        down_channels1 = nbase1[1:] + [nbase1[-1]]
        down_channels2 = nbase2[1:] + [nbase2[-1]]
        self.downsample = TwoStreamDownsample(down_channels1, down_channels2, sz, residual_on, cross_attn_layers_stem)
        
        # Fusion after bottleneck: concatenate both stream's last features
        fused_channels = nbase1[-1] + nbase2[-1]
        # Reduce fused channels to original bottleneck size (nbase1[-1]) for decoder compatibility
        self.bottleneck_conv = nn.Conv2d(fused_channels, nbase1[-1], 1)
        # Decoder channel sizes: skip stem_out, use down1, down2, down3, bottleneck, plus extra bottleneck for up[3]
        # down_channels1 = [32,64,128,256,256]
        # nbaseup = [64,128,256,256,256] (5 elements) for 4 up blocks
        nbaseup = down_channels1[1:] + [down_channels1[-1]]  # [64,128,256,256,256]
        # Shared decoder (upsample)
        self.upsample = resnet_torch.upsample(nbaseup, sz, residual_on=residual_on, concatenation=concatenation)
        self.make_style = resnet_torch.make_style()
        self.output = resnet_torch.batchconv(nbaseup[0], nout, 1)
        
        self.diam_mean = nn.Parameter(data=torch.ones(1) * diam_mean, requires_grad=False)
        self.diam_labels = nn.Parameter(data=torch.ones(1) * diam_mean, requires_grad=False)
        self.style_on = style_on
        
    def forward(self, data, data2=None):
        """
        If data2 is provided:
            data: stream1 input (DAPI) [B, 1, H, W]
            data2: stream2 input (HeatMap_all) [B, N, H, W]
        Else:
            data: concatenated input [B, 1+N, H, W] where first channel is DAPI,
                  remaining N channels are HeatMap_all.
        """
        if data2 is None:
            # Split concatenated tensor
            n_dapi = self.nbase1[0]
            n_heat = self.nbase2[0]
            # data shape [B, 1+N, H, W]
            data1 = data[:, :n_dapi, :, :]
            data2 = data[:, n_dapi:, :, :]
            # Ensure channel counts match
            if data2.shape[1] != n_heat:
                raise ValueError(f"Expected {n_heat} heatmap channels, got {data2.shape[1]}")
        else:
            data1 = data
        
        if self.mkldnn:
            data1 = data1.to_mkldnn()
            data2 = data2.to_mkldnn()
        
        # Project input channels
        x1 = self.stem1(data1)
        x2 = self.stem2(data2)
        
        # Two-stream encoder
        x1_features, x2_features = self.downsample(x1, x2)
        
        # Take the last feature maps (bottleneck)
        x1_last = x1_features[-1]
        x2_last = x2_features[-1]
        
        # Fusion: concatenate
        fused = torch.cat([x1_last, x2_last], dim=1)  # [B, C1+C2, H/8, W/8]
        # Reduce to original bottleneck size
        bottleneck = self.bottleneck_conv(fused)  # [B, nbase1[-1], H/8, W/8]
        
        # Style vector from bottleneck (same as original CPnet)
        if self.mkldnn:
            style = self.make_style(bottleneck.to_dense())
        else:
            style = self.make_style(bottleneck)
        style0 = style
        if not self.style_on:
            style = style * 0
        
        # Prepare list of skip connections for decoder (down1, down2, down3, bottleneck)
        # down1 64, down2 128, down3 256, bottleneck 256
        # skip down4 (the last stream1 feature) because it's replaced by bottleneck
        xd = x1_features[:-1] + [bottleneck]  # 4 elements matching up[0]..up[3] skip connections

        
        # Upsample
        T0 = self.upsample(style, xd, self.mkldnn)
        T0 = self.output(T0)
        if self.mkldnn:
            T0 = T0.to_dense()
        
        return T0, style0
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
    
    def load_model(self, filename, cpu=False):
        if not cpu:
            state_dict = torch.load(filename)
        else:
            self.__init__(self.nbase1, self.nbase2, self.nout, self.sz,
                          self.residual_on, self.style_on, self.concatenation,
                          self.mkldnn, self.diam_mean, self.cross_attn_layers)
            state_dict = torch.load(filename, map_location=torch.device('cpu'))
        self.load_state_dict(dict([(name, param) for name, param in state_dict.items()]), strict=False)