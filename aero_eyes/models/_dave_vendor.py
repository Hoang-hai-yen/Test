"""Copied (not cloned) from DAVE -- A Detect-and-Verify Paradigm for
Low-Shot Counting (Pelhan, Lukezic, Zavrtanik, Kristan; CVPR 2024;
arXiv:2404.16622), https://github.com/jerpelhan/DAVE, commit 8aeeb9c.
MIT License, Copyright (c) 2024 Jer Pelhan -- see DAVE/LICENSE in this
project's paper-reference checkout for the full license text (reproduced
below per its own terms, which require keeping the notice with any copy).

Only the 2 classes aero_eyes.models.dave_verification.DaveVerificationExtractor
actually needs (the ResNet50+SWaV backbone and the verify-stage feature
projection) are copied here, VERBATIM (no behavior changes) from
DAVE/models/backbone.py::Backbone and DAVE/models/feat_comparison.py::
Feature_Transform -- everything else in the original DAVE repo (training
scripts, FSC147 data loaders, the detection head, eval tooling) is
irrelevant to this project's own use (it only ever reuses DAVE's verify
stage) and is deliberately NOT vendored, so this project does not need the
full DAVE checkout (or any git submodule for it) at runtime -- only this
file, plus a manually-downloaded verification.pth (see DaveVerificationConfig,
aero_eyes/config.py).

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.ops.misc import FrozenBatchNorm2d


class Backbone(nn.Module):
    """Verbatim copy of DAVE/models/backbone.py::Backbone."""

    def __init__(
        self,
        name: str,
        pretrained: bool,
        dilation: bool,
        reduction: int,
        swav: bool,
        requires_grad: bool
    ):

        super(Backbone, self).__init__()

        resnet = getattr(models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained, norm_layer=FrozenBatchNorm2d
        )

        self.backbone = resnet
        self.reduction = reduction

        if name == 'resnet50' and swav:
            checkpoint = torch.hub.load_state_dict_from_url(
                'https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar',
                map_location="cpu"
            )
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            self.backbone.load_state_dict(state_dict, strict=False)

        # concatenation of layers 2, 3 and 4
        self.num_channels = 896 if name in ['resnet18', 'resnet34'] else 3584

        for n, param in self.backbone.named_parameters():
            if 'layer2' not in n and 'layer3' not in n and 'layer4' not in n:
                param.requires_grad_(False)
            else:
                param.requires_grad_(requires_grad)

    def forward(self, x):
        size = x.size(-2) // self.reduction, x.size(-1) // self.reduction
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = layer1 = self.backbone.layer1(x)
        x = layer2 = self.backbone.layer2(x)
        x = layer3 = self.backbone.layer3(x)
        x = layer4 = self.backbone.layer4(x)

        x = torch.cat([
            F.interpolate(f, size=size, mode='bilinear', align_corners=True)
            for f in [layer2, layer3, layer4]
        ], dim=1)

        return x


class ConvBlock1(torch.nn.Module):
    """Verbatim copy of DAVE/models/feat_comparison.py::ConvBlock1."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation='relu'):
        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, 256, kernel_size=1)

    def forward(self, input):
        x = self.conv1(input)
        if self.activation == 'relu':
            return self.conv2(self.relu(self.bn(x)))
        else:
            return x


class Feature_Transform(nn.Module):
    """Verbatim copy of DAVE/models/feat_comparison.py::Feature_Transform."""

    def __init__(self):
        self.in_channels = 3584
        super(Feature_Transform, self).__init__()
        self.conv_block1 = ConvBlock1(self.in_channels, self.in_channels // 8, kernel_size=1)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.flat(x)
        return x
