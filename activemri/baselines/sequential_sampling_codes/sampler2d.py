from activemri.baselines.sequential_sampling_codes.conv_sampler import BasicBlock
import torch
from torch import nn
from torch.nn import functional as F
from activemri.baselines.loupe_codes import transforms
from activemri.baselines.loupe_codes.reconstructors import ConvBlock

class Sampler2D(nn.Module):
    def __init__(self, num_blocks=[1, 1, 1], fixed_input=False):
        super(Sampler2D, self).__init__()
        
        in_chans = 5
        out_chans = 1
        chans = 64
        num_pool_layers = 4
        drop_prob = 0

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

        self.fixed_input = fixed_input
        if fixed_input:
            print('generate random input tensor')
            fixed_input_tensor = torch.randn(size=[1, 5, 128, 128])
            self.fixed_input_tensor = nn.Parameter(fixed_input_tensor, requires_grad=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, mask):
        """
            This function takes the observed kspace data and sampling trajectories as input
            and output next sampling probability mask. We additionally mask out all
            previous sampled locations.
            input: NHWC
            mask:  N1HW
        """
        # NHWC -> NCHW
        x = x.permute(0, -1, 1, 2)

        # concatenate with previous sampling mask
        x = torch.cat([x, mask], dim=1)

        if self.fixed_input:
            x = self.fixed_input_tensor

        output = x 
        
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

        out = self.conv2(output)

        out = F.softplus(out)

        out = out / torch.max(out.reshape(out.shape[0], -1), dim=1)[0].reshape(-1, 1, 1, 1)

        # Don't need to select an existed measurement
        new_mask = out * (1-mask)

        return new_mask
