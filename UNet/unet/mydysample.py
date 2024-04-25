import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class Dy_UpSample(nn.Module):
    def __init__(self, c_in, style='lp', ratio=2, groups=4, dyScope=True):
        super(Dy_UpSample, self).__init__()
        self.ratio =ratio
        self.style = style
        self.groups = groups
        self.dySample = dyScope
        assert style in ['lp', 'pl']
        assert c_in % groups == 0

        # upsampling 是分为 linear+pixel-shuffle 和 pixel-shuffle+linear
        # downsampling 分为 linear+pixel-unshuffle 和 pixel-unshuffle
        if style == 'lp':
            c_out = int(2 * groups * ratio**2)
        else:
            assert c_in >= groups * ratio**2
            c_out = 2 * groups
            c_in = int(c_in // ratio**2)
        
        
        if dyScope:
            self.scope = nn.Conv2d(c_in, c_out, kernel_size=1, groups=groups)
            constant_init(self.scope, val=0.)

        self.offset = nn.Conv2d(c_in, c_out, kernel_size=1)
        # self.offset = nn.Conv2d(c_in, c_out, kernel_size=1, groups=groups)
        normal_init(self.offset, std=0.001)


    def Sample(self, x, offset):
        _, _, h, w = offset.size()
        x = einops.rearrange(x, 'b (c grp) h w -> (b grp) c h w', grp=self.groups)
        offset = einops.rearrange(offset, 'b (grp two) h w -> (b grp) h w two',
                                  two=2, grp=self.groups)
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 1, 1, 2)
        
        # offset = torch.zeros_like(offset)

        h = torch.linspace(0.5, h - 0.5, h)
        w = torch.linspace(0.5, w - 0.5, w)
        pos = torch.stack(torch.meshgrid(w, h, indexing='xy')).to(x.device)
        pos = einops.rearrange(pos, 'two h w -> 1 h w two')
        pos = 2 * (pos + offset) / normalizer - 1


        out = F.grid_sample(x, pos, align_corners=False, mode='bilinear', padding_mode="border")
        out = einops.rearrange(out, '(b grp) c h w -> b (c grp) h w', grp=self.groups)
        return out

    def forward_lp(self, x):
        offset = self.offset(x)
        if self.dySample:
            offset = F.sigmoid(self.scope(x)) * 0.5 * offset
        else:
            offset = 0.25 * offset
        offset = F.pixel_shuffle(offset, upscale_factor=self.ratio)
        return self.Sample(x, offset)
    
    def forward_pl(self, x):
        y = F.pixel_shuffle(x, upscale_factor=self.ratio)
        offset = self.offset(y)
        if self.dySample:
            offset = F.sigmoid(self.scope(y)) * 0.5 * offset
        else:
            offset = 0.25 * offset
        return self.Sample(x, offset)

    def forward(self, x):
        if self.style == 'lp':
            return self.forward_lp(x)
        return self.forward_pl(x)

class Dy_DownSample(nn.Module):
    def __init__(self, c_in, style='lp', ratio=2, groups=4, dyScope=True):
        super(Dy_DownSample, self).__init__()
        self.ratio =ratio
        self.style = style
        self.groups = groups
        self.dySample = dyScope
        assert style in ['lp', 'pl']
        assert c_in % groups == 0

        # upsampling 是分为 linear+pixel-shuffle 和 pixel-shuffle+linear
        # downsampling 分为 linear+pixel-unshuffle 和 pixel-unshuffle+linear
        # if ratio > 1:
        if style == 'lp':
            assert 2 * groups % ratio**2 == 0
            c_out = 2 * int(groups / ratio**2)
        else:
            # assert c_in >= groups / ratio**2
            c_out = 2 * groups
            c_in = c_in * ratio**2


        if dyScope:
            self.scope = nn.Conv2d(c_in, c_out, kernel_size=1)
            constant_init(self.scope, val=0.)

        self.offset = nn.Conv2d(c_in, c_out, kernel_size=1)

        normal_init(self.offset, std=0.001)


    def Sample(self, x, offset):
        _, _, h, w = offset.size()
        x = einops.rearrange(x, 'b (c grp) h w -> (b grp) c h w', grp=self.groups)
        offset = einops.rearrange(offset, 'b (grp two) h w -> (b grp) h w two',
                                  two=2, grp=self.groups)
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 1, 1, 2)


        h = torch.linspace(0.5, h - 0.5, h)
        w = torch.linspace(0.5, w - 0.5, w)
        pos = torch.stack(torch.meshgrid(w, h, indexing='xy')).to(x.device)
        pos = einops.rearrange(pos, 'two h w -> 1 h w two')
        pos = 2 * (pos + offset) / normalizer - 1


        out = F.grid_sample(x, pos, align_corners=False, mode='bilinear', padding_mode="border")
        out = einops.rearrange(out, '(b grp) c h w -> b (c grp) h w', grp=self.groups)
        return out

    def forward_lp(self, x):
        offset = self.offset(x)
        if self.dySample:
            offset = F.sigmoid(self.scope(x)) * 0.5 * offset
        else:
            offset = 0.25 * offset
        offset = F.pixel_unshuffle(offset, downscale_factor=self.ratio)
        return self.Sample(x, offset)
    
    def forward_pl(self, x):
        y = F.pixel_unshuffle(x, downscale_factor=self.ratio)
        offset = self.offset(y)
        if self.dySample:
            offset = F.sigmoid(self.scope(y)) * 0.5 * offset
        else:
            offset = 0.25 * offset
        return self.Sample(x, offset)

    def forward(self, x):
        _, _, h, w = x.size()
        padh = self.ratio - h % self.ratio
        padw = self.ratio - w % self.ratio
        x = F.pad(x, (padw//2, (padw+1)//2, padh//2, (padh+1)//2), mode='replicate')
        if self.style == 'lp':
            return self.forward_lp(x)
        return self.forward_pl(x)


if __name__ == '__main__':
    x = torch.randn(size=(2, 16, 4, 7))
    dy_samp = Dy_DownSample(16, style='pl', ratio=2)
    x = dy_samp(x)
    print(x.size())

# if __name__ == '__main__':
#     x = torch.tensor([[[[0., 0.1],
#                        [0.2, 0.3]],
#                        [[0., 0.1],
#                        [0.2, 0.3]],
#                        [[0., 0.1],
#                        [0.2, 0.3]],
#                        [[0., 0.1],
#                        [0.2, 0.3]]]])
#     print(x.size())
#     dys = DySample(4)
#     x1 = dys(x)
#     x2 = F.interpolate(x, scale_factor=2, mode='bilinear')
#     print(x1)
#     print(x2)
#     print(torch.allclose(x1, x2))
