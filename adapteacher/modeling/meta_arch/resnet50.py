from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.modeling.backbone import (
    Backbone,
    BACKBONE_REGISTRY
)
from detectron2.layers.blocks import FrozenBatchNorm2d
from detectron2.layers import ShapeSpec

from abc import ABCMeta, abstractmethod
import torch.nn as nn

from detectron2.layers import ShapeSpec

__all__ = ["Backbone"]

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, norm_type='FronzenBN'):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn1 = FrozenBatchNorm2d(planes) # nn.BatchNorm2d(planes)
        elif norm_type == 'SyncBN':
            self.bn1 = nn.SyncBatchNorm(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn2 = FrozenBatchNorm2d(planes) # nn.BatchNorm2d(planes)
        elif norm_type == 'SyncBN':
            self.bn2 = nn.SyncBatchNorm(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn3 = FrozenBatchNorm2d(planes * self.expansion) # nn.BatchNorm2d(planes * self.expansion)
        elif norm_type == 'SyncBN':
            self.bn3 = nn.SyncBatchNorm(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            if norm_type == 'FronzenBN':
                this_norm = FrozenBatchNorm2d(planes * self.expansion) #("1", nn.BatchNorm2d(planes * self.expansion))
            elif norm_type == 'SyncBN':
                this_norm = nn.SyncBatchNorm(planes * self.expansion)
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", this_norm), #("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(Backbone):
    """
    Extended from CLIP implementation. It contains following changes:
    1. change all nn.BatchNorm2d() to FrozenBatchNorm2d(), due to small batch size of detection training
    2. add self._out_feature_strides according to standard ResNet
    2. modify forward() to be compatible with Detectron2
    3. add freeze() and output_shape() to be compatible with Detectron2
    4. add build_clip_resnet_backbone() to build this ModifiedResNet

    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, 
        out_features=None, freeze_at=0, depth=None, pool_vec=True, create_att_pool=False, norm_type='FronzenBN'):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.norm_type = norm_type

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn1 = FrozenBatchNorm2d(width // 2)  # nn.BatchNorm2d(width // 2)
        elif norm_type == 'SyncBN':
            self.bn1 = nn.SyncBatchNorm(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn2 = FrozenBatchNorm2d(width // 2)  # nn.BatchNorm2d(width // 2)
        elif norm_type == 'SyncBN':
            self.bn2 = nn.SyncBatchNorm(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn3 = FrozenBatchNorm2d(width) # nn.BatchNorm2d(width)
        elif norm_type == 'SyncBN':
            self.bn3 = nn.SyncBatchNorm(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        if 'res5' in out_features:  # FPN
            self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        else:  # C4, layer4 created here won't be used in backbone, but used in roi_head
            self.layer4 = self._make_layer(width * 8, layers[3], stride=2) # None
        
        self.pool_vec = pool_vec
        if self.pool_vec or create_att_pool:  # pool a vector representation for an image
            embed_dim = width * 32  # the ResNet feature dimension
            self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

        self._out_features = out_features if out_features else []
        if depth in [50,101]: # resnet50 or resnet 101
            # FPN: ["res2", "res3", "res4", "res5"]; C4: ["res4"]
            self._out_feature_channels = {'stem': 64, 'res2': 256, 'res3': 512, 'res4': 1024, 'res5': 2048} if 'res5' in self._out_features \
                else {'stem': 64, 'res2': 256, 'res3': 512, 'res4': 1024}
            self._out_feature_strides = {'stem': 4, 'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32} if 'res5' in self._out_features \
                else  {'stem': 4, 'res2': 4, 'res3': 8, 'res4': 16}  # anti-aliasing strided conv???        
        elif depth in [200]: # resnet50x4
            # FPN: ["res2", "res3", "res4", "res5"]; C4: ["res4"]
            self._out_feature_channels = {'stem': 80, 'res2': 320, 'res3': 640, 'res4': 1280, 'res5': 2560} if 'res5' in self._out_features \
                else {'stem': 80, 'res2': 320, 'res3': 640, 'res4': 1280}
            self._out_feature_strides = {'stem': 4, 'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32} if 'res5' in self._out_features \
                else  {'stem': 4, 'res2': 4, 'res3': 8, 'res4': 16}  # anti-aliasing strided conv???        
        self.freeze(freeze_at)


    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride, norm_type=self.norm_type)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, norm_type=self.norm_type))

        return nn.Sequential(*layers)

    def forward(self, x, depth):
        # DEPTH
        # d = depth[:, 0:1, :, :].float()
        # d = (d - d.amin((2,3), True)) / (d.amax((2,3), True) - d.amin((2,3), True) + 1e-8)
        # x = x * (1 + d)

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x
        
        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = x.type(self.conv1.weight.dtype) # det2 resnet50: [3, 800, 1216]; CLIP resnet50: [3, 224, 224]
        x = stem(x) # det2 resnet50: [64, 200, 304]; CLIP resnet50: [64, 56, 56]
        if "stem" in self._out_features:
            outputs["stem"] = x
        x = self.layer1(x) # det2 resnet50: [256, 200, 304]; CLIP resnet50: [256, 56, 56]
        outputs['res2'] = x if "res2" in self._out_features else None
        x = self.layer2(x) # det2 resnet50: [512, 100, 152]; CLIP resnet50: [512, 28, 28]
        outputs['res3'] = x if "res3" in self._out_features else None
        x = self.layer3(x) # det2 resnet50: [1024, 50, 76]; CLIP resnet50: [1024, 14, 14]
        outputs['res4'] = x if "res4" in self._out_features else None
        x = self.layer4(x)  if "res5" in self._out_features else x # det2 resnet50: [2048, 25, 38]; CLIP resnet50: [2048, 7, 7]
        outputs['res5'] = x if "res5" in self._out_features else None
        return outputs

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ResNet itself
        """
        def cnnblockbase_freeze(nn_module):
            """
            Make this block not trainable.
            This method sets all parameters to `requires_grad=False`,
            and convert all BatchNorm layers to FrozenBatchNorm

            Returns:
                the block itself
            """
            for p in nn_module.parameters():
                p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(nn_module)
        
        if freeze_at >= 1: # stem
            cnnblockbase_freeze(self.conv1)
            cnnblockbase_freeze(self.bn1)
            cnnblockbase_freeze(self.conv2)
            cnnblockbase_freeze(self.bn2)
            cnnblockbase_freeze(self.conv3)
            cnnblockbase_freeze(self.bn3)
        # each stage is a torch.nn.modules.container.Sequential
        for idx, stage in enumerate([self.layer1, self.layer2, self.layer3, self.layer4], start=2): 
            if freeze_at >= idx:
                for block in stage.children():  # each block is a Bottleneck
                    cnnblockbase_freeze(block)  
        return self

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

@BACKBONE_REGISTRY.register()
def build_clip_resnet_backbone(cfg, input_shape):
    """
    Create a CLIP-version ResNet instance from config.

    Returns:
        ModifiedResNet: a :class:`ModifiedResNet` instance.
    """
    depth = 50
    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [4, 6, 10, 6], # flag for ResNet50x4
    }[depth]
    vision_layers = num_blocks_per_stage
    vision_width = {
        50: 64,
        101: 64,
        200: 80, # flag for ResNet50x4
    }[depth]  # cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    
    # default configs of CLIP ModifiedResNet, but not used if only building ModifiedResNet as backbone
    embed_dim = {
        50: 1024,
        101: 512,
        200: 640, # flag for ResNet50x4
    }[depth] 
    vision_heads = vision_width * 32 // 64
    image_resolution = {
        50: 224,
        101: 224,
        200: 288, # flag for ResNet50x4
    }[depth] 

    # if combine {ModifiedResNet of CLIP, C4, text emb as classifier}, then has to use att_pool to match dimension
    create_att_pool = False

    return ModifiedResNet(layers=vision_layers, 
                output_dim=embed_dim,
                out_features=["res2", "res3", "res4", "res5"],
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                depth=depth,
                pool_vec=False,
                create_att_pool=create_att_pool,
                )

# resnet = build_clip_resnet_backbone(50)