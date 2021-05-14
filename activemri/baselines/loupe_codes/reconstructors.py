"""
Portion of this code is from fastmri(https://github.com/facebookresearch/fastMRI) 

Copyright (c) Facebook, Inc. and its affiliates.

Licensed under the MIT License.
"""

import torch
from torch import nn
from torch.nn import functional as F
from activemri.baselines.loupe_codes.transforms import *
from activemri.baselines.loupe_codes.layers import *

class LOUPEUNet(nn.Module):
    """
        PyTorch implementation of a U-Net model.
        This is based on:
            Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
            for biomedical image segmentation. In International Conference on Medical image
            computing and computer-assisted intervention, pages 234–241. Springer, 2015.   

        The model takes a real or complex value image and use a UNet to denoise the image. 
        A residual connection is applied to stablize training.       
    """
    def __init__(self,
                 in_chans,
                 out_chans,
                 chans,
                 num_pool_layers,
                 drop_prob,
#                  mask_length,
                 bi_dir=False,
                 old_recon=False,
                 with_uncertainty=False):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()
        self.old_recon = old_recon 
        if old_recon:
            assert 0 
            in_chans = in_chans+1 # add mask dim and old reconstruction dim  

        self.with_uncertainty = with_uncertainty

        if with_uncertainty:
            out_chans = out_chans+1

        self.in_chans = in_chans 
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

        nn.init.normal_(self.conv2[-1].weight, mean=0, std=0.001) 
        self.conv2[-1].bias.data.fill_(0)

    def forward(self, input, old_recon=None, eps=1e-8):
        # input: NCHW 
        # output: NCHW 

        if self.old_recon:
            assert 0 
            output = torch.cat([input, old_recon], dim=1)
        else:
            output = input 
        
        stack = []
        # print(input.shape, mask.shape)

        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            downsample_layer = stack.pop()
            layer_size = (downsample_layer.shape[-2], downsample_layer.shape[-1])
            output = F.interpolate(output, size=layer_size, mode='bilinear', align_corners=False)
            output = torch.cat([output, downsample_layer], dim=1)
            output = layer(output)

        out_conv2 = self.conv2(output)

        img_residual = out_conv2[:, :1]

        if self.with_uncertainty:
            map = out_conv2[:, 1:]
        else:
            map = torch.zeros_like(out_conv2)
    
        if self.old_recon:
            return img_residual + old_recon
        else:
            return img_residual + torch.norm(input, dim=1, keepdim=True)