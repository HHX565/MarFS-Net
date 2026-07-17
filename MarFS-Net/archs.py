#our model
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from kan import KANLinear


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        hidden = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, hidden, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = out + residual
        out = self.relu(out)
        return out


class PSA(nn.Module):
    def __init__(self, channel=512, reduction=4):
        super(PSA, self).__init__()
        self.channel = channel
        self.ch_conv_v = nn.Conv2d(channel, channel // 2, kernel_size=1, bias=False)
        self.ch_conv_q = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.ch_softmax = nn.Softmax(dim=1)
        self.ch_conv_z = nn.Conv2d(channel // 2, channel, kernel_size=1, bias=False)
        self.ch_ln = nn.LayerNorm([channel, 1, 1])
        self.ch_sigmoid = nn.Sigmoid()
        self.sp_conv_v = nn.Conv2d(channel, channel // 2, kernel_size=1, bias=False)
        self.sp_conv_q = nn.Conv2d(channel, channel // 2, kernel_size=1, bias=False)
        self.sp_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sp_softmax = nn.Softmax(dim=-1)
        self.sp_conv_z = nn.Conv2d(channel // 2, channel, kernel_size=1, bias=False)
        self.sp_ln = nn.LayerNorm(channel)
        self.sp_sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        ch_v = self.ch_conv_v(x).reshape(b, c // 2, -1)
        ch_q = self.ch_conv_q(x).reshape(b, 1, -1)
        ch_q = self.ch_softmax(ch_q)
        ch_context = torch.matmul(ch_v, ch_q.permute(0, 2, 1)).reshape(b, c // 2, 1, 1)
        ch_z = self.ch_conv_z(ch_context)
        ch_z = self.ch_ln(ch_z)
        ch_w = self.ch_sigmoid(ch_z)
        ch_out = x * ch_w

        sp_v = self.sp_conv_v(x)
        sp_q = self.sp_avg_pool(self.sp_conv_q(x))
        sp_q = self.sp_softmax(sp_q.view(b, c // 2, -1)).view(b, c // 2, 1, 1)
        sp_context = torch.mul(sp_v, sp_q)
        sp_z = self.sp_conv_z(sp_context)
        sp_z = sp_z.permute(0, 2, 3, 1)
        sp_z = self.sp_ln(sp_z)
        sp_z = sp_z.permute(0, 3, 1, 2)
        sp_w = self.sp_sigmoid(sp_z)
        return ch_out * sp_w


class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        kernels = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('kernels', kernels)
        self.requires_grad = False

    def forward(self, x):
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, 1, 0, 1), mode='reflect')
        weights = self.kernels.repeat(C, 1, 1, 1)
        out = F.conv2d(x, weights, stride=2, groups=C)
        out = out.reshape(B, C, 4, out.shape[2], out.shape[3])
        return out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]


class IDWT(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        kernels = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('kernels', kernels)
        self.requires_grad = False

    def forward(self, ll, lh, hl, hh):
        B, C, H_half, W_half = ll.shape
        x = torch.stack([ll, lh, hl, hh], dim=2).reshape(B, C * 4, H_half, W_half)
        weights = self.kernels.repeat(C, 1, 1, 1)
        return F.conv_transpose2d(x, weights, stride=2, groups=C)


class W_PSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwt = DWT()
        self.idwt = IDWT()
        self.high_freq_process = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            PSA(channel=dim),
            nn.Conv2d(dim, dim * 3, 1, bias=False)
        )
        self.low_freq_process = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )
        self.fusion = nn.Conv2d(dim, dim, 1, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        shortcut = x
        ll, lh, hl, hh = self.dwt(x)
        ll_feat = self.low_freq_process(ll)
        high_cat = torch.cat([lh, hl, hh], dim=1)
        high_refined = self.high_freq_process(high_cat)
        lh_new, hl_new, hh_new = torch.chunk(high_refined, 3, dim=1)
        out = self.idwt(ll_feat, lh_new, hl_new, hh_new)
        if out.shape != shortcut.shape:
            out = F.interpolate(out, size=shortcut.shape[2:], mode='bilinear', align_corners=False)
        out = self.fusion(out)
        out = self.act(out)
        return shortcut + out


def plain_mamba_scan(x):
    B, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1)
    x_scan = x.clone()
    x_scan[:, 1::2, :, :] = torch.flip(x[:, 1::2, :, :], dims=[2])
    return x_scan.reshape(B, H * W, C)


def plain_mamba_merge(x_seq, H, W):
    B, L, C = x_seq.shape
    assert L == H * W
    x = x_seq.view(B, H, W, C)
    x_rec = x.clone()
    x_rec[:, 1::2, :, :] = torch.flip(x[:, 1::2, :, :], dims=[2])
    return x_rec.permute(0, 3, 1, 2)


class PSA_Channel_KAN(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.ch_conv_v = nn.Conv2d(channel, channel // 2, kernel_size=1, bias=False)
        self.ch_conv_q = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.ch_softmax = nn.Softmax(dim=1)
        self.ch_conv_z = nn.Conv2d(channel // 2, channel, kernel_size=1, bias=False)
        self.ch_ln = nn.LayerNorm([channel, 1, 1])
        self.kan_refine = KANLinear(channel, channel)
        self.ch_sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        ch_v = self.ch_conv_v(x).reshape(b, c // 2, -1)
        ch_q = self.ch_conv_q(x).reshape(b, 1, -1)
        ch_q = self.ch_softmax(ch_q)
        ch_context = torch.matmul(ch_v, ch_q.permute(0, 2, 1)).reshape(b, c // 2, 1, 1)
        ch_z = self.ch_conv_z(ch_context)
        ch_z = self.ch_ln(ch_z)
        ch_z_flat = ch_z.flatten(1)
        ch_w = self.kan_refine(ch_z_flat)
        ch_w = self.ch_sigmoid(ch_w).view(b, c, 1, 1)
        return x * ch_w


class PSA_Spatial_KAN(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.sp_conv_v = nn.Conv2d(channel, channel // 2, kernel_size=1, bias=False)
        self.sp_conv_q = nn.Conv2d(channel, channel // 2, kernel_size=1, bias=False)
        self.sp_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sp_softmax = nn.Softmax(dim=-1)
        self.sp_conv_z = nn.Conv2d(channel // 2, channel, kernel_size=1, bias=False)
        self.sp_ln = nn.LayerNorm(channel)
        self.sp_sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        sp_v = self.sp_conv_v(x)
        sp_q = self.sp_avg_pool(self.sp_conv_q(x))
        sp_q = self.sp_softmax(sp_q.view(b, c // 2, -1)).view(b, c // 2, 1, 1)
        sp_context = torch.mul(sp_v, sp_q)
        sp_z = self.sp_conv_z(sp_context)
        sp_z = sp_z.permute(0, 2, 3, 1)
        sp_z = self.sp_ln(sp_z)
        sp_z = sp_z.permute(0, 3, 1, 2)
        sp_w = self.sp_sigmoid(sp_z)
        return x * sp_w


class Harmonic_Taylor_PSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.grad_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.lap_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self._init_operators()
        self.coeff_kan = KANLinear(dim, 2 * dim)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.psa = PSA_Channel_KAN(dim)
        self.fusion = nn.Conv2d(dim, dim, 1)

    def _init_operators(self):
        lap_k = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.lap_conv.weight.data = lap_k.repeat(self.dim, 1, 1, 1)
        grad_k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.grad_conv.weight.data = grad_k.repeat(self.dim, 1, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        feat_grad = self.grad_conv(x)
        feat_lap = self.lap_conv(x)
        stats = self.global_pool(torch.abs(feat_lap)).flatten(1)
        coeffs = self.coeff_kan(stats)
        alpha, beta = torch.split(coeffs, C, dim=1)
        alpha = self.sigmoid(alpha).view(B, C, 1, 1)
        beta = self.tanh(beta).view(B, C, 1, 1)
        x_harmonic = x + alpha * feat_grad - 0.5 * (1 + beta) * feat_lap
        feat_attn = self.psa(x_harmonic)
        return self.fusion(feat_attn)


class Entropy_Regularized_Mamba(nn.Module):
    def __init__(self, dim, target_size=(16, 16)):
        super().__init__()
        self.target_size = target_size
        self.pool = nn.AdaptiveAvgPool2d(target_size)
        self.std_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.ReLU()
        )
        self.kan_fwd = KANLinear(dim, dim)
        self.kan_bwd = KANLinear(dim, dim)
        self.ln_seq = nn.LayerNorm(dim)
        self.delta_kan = KANLinear(dim, dim)
        self.psa = PSA_Spatial_KAN(dim)
        self.out_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_small = self.pool(x)
        h, w = self.target_size
        local_mean = F.avg_pool2d(x_small, 3, stride=1, padding=1)
        local_sq_mean = F.avg_pool2d(x_small ** 2, 3, stride=1, padding=1)
        local_std = torch.sqrt(F.relu(local_sq_mean - local_mean ** 2) + 1e-6)
        entropy_seq = plain_mamba_scan(local_std).reshape(-1, C)
        x_seq = plain_mamba_scan(x_small).reshape(-1, C)
        delta_mod = torch.sigmoid(self.delta_kan(entropy_seq))
        x_regulated = x_seq * delta_mod
        out_fwd = self.kan_fwd(x_regulated).view(B, -1, C)
        x_seq_bwd = torch.flip(x_regulated.view(B, h * w, C), dims=[1]).reshape(-1, C)
        out_bwd = self.kan_bwd(x_seq_bwd).view(B, -1, C)
        out_bwd = torch.flip(out_bwd, dims=[1])
        out_seq = self.ln_seq(out_fwd + out_bwd)
        feat_small = plain_mamba_merge(out_seq, h, w)
        feat_global = F.interpolate(feat_small, size=(H, W), mode='bilinear', align_corners=True)
        return self.psa(self.out_proj(feat_global))


class Covariance_Manifold_Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj_l = nn.Conv2d(dim, dim // 4, 1)
        self.proj_g = nn.Conv2d(dim, dim // 4, 1)
        self.mi_kan = KANLinear(dim // 4, 2 * dim)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feat_l, feat_g):
        B, C, H, W = feat_l.shape
        f_l = self.proj_l(feat_l)
        f_g = self.proj_g(feat_g)
        covariance_map = f_l * f_g
        manifold_desc = self.pool(covariance_map).flatten(1)
        weights = self.mi_kan(manifold_desc)
        w_l, w_g = torch.split(weights, C, dim=1)
        w_l = self.sigmoid(w_l).view(B, C, 1, 1)
        w_g = self.sigmoid(w_g).view(B, C, 1, 1)
        return feat_l * w_l + feat_g * w_g


class SP_SKMod_HFMD(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.branch_local = Harmonic_Taylor_PSA(dim)
        self.branch_global = Entropy_Regularized_Mamba(dim)
        self.fusion = Covariance_Manifold_Fusion(dim)
        self.proj_final = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        feat_l = self.branch_local(x_norm)
        feat_g = self.branch_global(x_norm)
        out = self.fusion(feat_l, feat_g)
        return self.proj_final(out) + residual


class MSHNetCore(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, block=ResNet):
        super().__init__()
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)
        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block)
        self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block, param_blocks[0])
        self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block, param_blocks[1])
        self.encoder_3 = self._make_layer(param_channels[2], param_channels[3], block, param_blocks[2])
        self.middle_layer = self._make_layer(param_channels[3], param_channels[4], block, param_blocks[3])

        self.wpsa_e3 = W_PSA(param_channels[3])
        self.hfmd_m = SP_SKMod_HFMD(param_channels[4])

        self.decoder_3 = self._make_layer(param_channels[3] + param_channels[4], param_channels[3], block, param_blocks[2])
        self.decoder_2 = self._make_layer(param_channels[2] + param_channels[3], param_channels[2], block, param_blocks[1])
        self.decoder_1 = self._make_layer(param_channels[1] + param_channels[2], param_channels[1], block, param_blocks[0])
        self.decoder_0 = self._make_layer(param_channels[0] + param_channels[1], param_channels[0], block)

        self.output_0 = nn.Conv2d(param_channels[0], num_classes, 3, 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], num_classes, 3, 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], num_classes, 3, 1, 1)
        self.output_3 = nn.Conv2d(param_channels[3], num_classes, 3, 1, 1)
        self.final = nn.Conv2d(num_classes * 4, num_classes, 3, 1, 1)

    def _make_layer(self, in_channels: int, out_channels: int, block, block_num: int = 1) -> nn.Sequential:
        layers: List[nn.Module] = [block(in_channels, out_channels)]
        for _ in range(block_num - 1):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward_features(self, x: Tensor):
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))
        x_e3 = self.wpsa_e3(x_e3)
        x_m = self.middle_layer(self.pool(x_e3))
        x_m = self.hfmd_m(x_m)

        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], dim=1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], dim=1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], dim=1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], dim=1))
        return x_d0, x_d1, x_d2, x_d3

    def forward_original_style(self, x: Tensor, warm_flag: bool):
        x_d0, x_d1, x_d2, x_d3 = self.forward_features(x)
        mask0 = self.output_0(x_d0)
        mask1 = self.output_1(x_d1)
        mask2 = self.output_2(x_d2)
        mask3 = self.output_3(x_d3)
        fused = self.final(torch.cat([mask0, self.up(mask1), self.up_4(mask2), self.up_8(mask3)], dim=1))
        if warm_flag:
            return [mask0, mask1, mask2, mask3], fused
        return [], fused

    def forward(self, x: Tensor, deep_supervision: bool = False):
        x_d0, x_d1, x_d2, x_d3 = self.forward_features(x)
        mask0 = self.output_0(x_d0)
        mask1 = self.output_1(x_d1)
        mask2 = self.output_2(x_d2)
        mask3 = self.output_3(x_d3)
        fused = self.final(torch.cat([mask0, self.up(mask1), self.up_4(mask2), self.up_8(mask3)], dim=1))
        if not deep_supervision:
            return fused
        return [fused, mask0, mask1, mask2, mask3]


class MSHNet_Official(MSHNetCore):
    def __init__(
        self,
        num_classes: int = 1,
        input_channels: int = 3,
        deep_supervision: bool = False,
        embed_dims: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(input_channels=input_channels, num_classes=num_classes)
        self.deep_supervision = deep_supervision
        self.embed_dims = embed_dims

    def forward(self, x: Tensor):
        return super().forward(x, deep_supervision=self.deep_supervision)


class MSHNet(MSHNet_Official):
    pass


__all__ = [
    'ChannelAttention',
    'SpatialAttention',
    'ResNet',
    'PSA',
    'DWT',
    'IDWT',
    'W_PSA',
    'PSA_Channel_KAN',
    'PSA_Spatial_KAN',
    'Harmonic_Taylor_PSA',
    'Entropy_Regularized_Mamba',
    'Covariance_Manifold_Fusion',
    'SP_SKMod_HFMD',
    'MSHNetCore',
    'MSHNet_Official',
    'MSHNet',
]

