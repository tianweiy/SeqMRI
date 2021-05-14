import torch
from torch import nn
from torch.nn import functional as F
from activemri.baselines.loupe_codes import transforms

# ResNet code is modified from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
# Licensed under MIT License

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, fixed_input=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        # down sample layers

        self.conv1 = nn.Conv2d(5, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        # up sample layers

        self.up_convs = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, stride=1)
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
            x = self.fixed_input_tensor.repeat(x.shape[0], 1, 1, 1).to(x.device) 

        # down convs
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.up_convs(out)
        out = self.conv_last(out)

        out = torch.relu(out) / torch.max(out.reshape(out.shape[0], -1), dim=1)[0].reshape(-1, 1, 1, 1)

        # Don't need to select an existed measurement
        new_mask = out * (1-mask)

        return new_mask

def ConvSamplerSmall(fixed_input=False):
    return ResNet(BasicBlock, [1, 1, 1], fixed_input=fixed_input)



class LineConstrainedSampler(nn.Module):
    """
        A line constrained convolutional sampler which uses the same architecture
        as the evaluator in https://arxiv.org/pdf/1902.03051.pdf
        In this module, we first create a pseudo image containing spectral maps
        corresponding to every kspace line. We then pass this pseudo image into
        a classification network to predict the probability of selecting
        each kspace line. Refer to the above paper for details
    """

    def __init__(self, shape=[32]):
        super().__init__()
        self.mask_conv = nn.Linear(shape[0], 6)
        self.middle_convs = nn.Sequential(
            nn.Conv2d(6+shape[0], 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(1024),
            nn.LeakyReLU(0.2),
        )
        self.conv_last = nn.Linear(1024, shape[0])

        # 1xWxHxWxC
        # spectral maps of each kspace column
        self.masks = torch.zeros(1, shape[1], shape[0], shape[1], 1, requires_grad=False).cuda()
        for i in range(shape[1]):
            self.masks[:, i, :, i] = 1

        print("Finish Mask Initialization")

    def _to_spectral(self, kspace):
        """
        Args:
            kspace (torch.Tensor): Already sampled measurements shape NHWC

        Returns:
            spectral_map (torch.Tensor): NWHW
        """
        spectral_map = transforms.complex_abs(transforms.fftshift(transforms.ifft2(kspace.unsqueeze(1) * self.masks)))

        return spectral_map

    def forward(self, kspace, mask):
        """
        Args:
            kspace (torch.Tensor): Already sampled measurements shape NHWC
            mask (torch.Tensor): previous sampling trajectories shape N1HW

        Returns:
            new_mask (torch.Tensor): probability map of next iteration's sampling locations
            shape N1HW <- broadcasted from N11W
        """
        N, C, H, W = mask.shape
        spectral_map = self._to_spectral(kspace)
        # mask = mask[:, 0, 0]
        mask = (mask.sum(dim=-2) == H).float().squeeze()

        mask_embedding = self.mask_conv(mask).reshape(N, 6, 1, 1).repeat(1, 1, H, W)

        spectral_map = torch.cat([spectral_map, mask_embedding], dim=1)

        out = self.middle_convs(spectral_map)

        # global average pooling
        out = out.mean([-2, -1])
        out = self.conv_last(out)

        out = torch.sigmoid(out)

        # Don't need to select an existed measurement
        new_mask = out * (1-mask)

        # don't use repeat here. Pytorch can automatically broadcast.
        return new_mask.reshape(N, 1, 1, W)

class KspaceLineConstrainedSampler(nn.Module):
    """
        A line constrained convolutional sampler which uses the same architecture
        as the evaluator in https://arxiv.org/pdf/1902.03051.pdf
        In this module, we first create a pseudo image containing spectral maps
        corresponding to every kspace line. We then pass this pseudo image into
        a classification network to predict the probability of selecting
        each kspace line. Refer to the above paper for details
    """

    def __init__(self, in_chans, out_chans, clamp=100, with_uncertainty=False, fixed_input=False):
        super().__init__()
        """self.middle_convs = nn.Sequential(
            nn.Conv2d(1+2, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
        )"""

        if with_uncertainty:
            self.flatten_size = int((in_chans) **2 * 6)
        else:
            self.flatten_size = int((in_chans) **2 * 5)

        self.conv_last = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_chans)
        )

        self.clamp = clamp
        self.with_uncertainty = with_uncertainty
        self.fixed_input = fixed_input 

        if fixed_input:
            print("Generate random input tensor")
            fixed_input_tensor = torch.randn(size=[1, 5, in_chans, in_chans])
            self.fixed_input_tensor = nn.Parameter(fixed_input_tensor, requires_grad=False)

        print("Finish Mask Initialization")

    def forward(self, kspace, mask, uncertainty_map=None):
        """
        Args:
            kspace (torch.Tensor): Already sampled measurements shape NHWC
            mask (torch.Tensor): previous sampling trajectories shape N1HW

        Returns:
            new_mask (torch.Tensor): probability map of next iteration's sampling locations
            shape N1HW <- broadcasted from N11W
        """
        N, C, H, W = mask.shape

        if self.with_uncertainty:
            assert 0 
            feat_map = torch.cat([kspace.permute(0, 3, 1, 2), uncertainty_map], dim=1)
        else:
            feat_map = torch.cat([kspace.permute(0, 3, 1, 2), mask], dim=1)

        if self.fixed_input:
            feat_map = self.fixed_input_tensor.repeat(N, 1, 1, 1).to(kspace.device) 

        out = feat_map.flatten(start_dim=1)

        out = self.conv_last(out)

        out = F.softplus(out)

        # out = out / torch.clamp(out / torch.clamp(torch.max(out.reshape(out.shape[0], -1), dim=1)[0].reshape(-1, 1), min=1e-10), max=1-1e-10, min=1e-10)
        out = out / torch.max(out.reshape(out.shape[0], -1), dim=1)[0].reshape(-1, 1)

        # Don't need to select an existed measurement
        if out.shape[-1] == mask.shape[-1]:
            vertical_mask = (mask.sum(dim=-2) == H).float().reshape(N, -1)
            new_mask = out * (1-vertical_mask)
        else:
            # verify horizontal / vertical separately
            vertical_mask = (mask.sum(dim=-2) == H).float().reshape(N, -1)
            horizontal_mask = (mask.transpose(-2, -1).sum(dim=-2)==W).float().reshape(N, -1)

            length = out.shape[1] // 2
            new_mask = torch.zeros_like(out)
            new_mask[:, :length] = out[:, :length] * (1-vertical_mask)
            new_mask[:, length:] = out[:, length:] * (1-horizontal_mask)

        # don't use repeat here. Pytorch can automatically broadcast.
        return new_mask.reshape(N, 1, 1, -1)


class BiLineConstrainedSampler(nn.Module):
    """
        A line constrained convolutional sampler which uses the same architecture
        as the evaluator in https://arxiv.org/pdf/1902.03051.pdf
        In this module, we first create a pseudo image containing spectral maps
        corresponding to every kspace line. We then pass this pseudo image into
        a classification network to predict the probability of selecting
        each kspace line. Refer to the above paper for details
    """

    def __init__(self, in_chans, clamp, with_uncertainty=False, fixed_input=False):
        super().__init__()

        self.sampler = KspaceLineConstrainedSampler(in_chans=in_chans, out_chans=in_chans*2, clamp=clamp,
            with_uncertainty=with_uncertainty, fixed_input=fixed_input)

    def forward(self, kspace, mask, uncertainty_map=None):
        """
        Args:
            kspace (torch.Tensor): Already sampled measurements shape NHWC
            mask (torch.Tensor): previous sampling trajectories shape N1HW

        Returns:
            new_mask (torch.Tensor): probability map of next iteration's sampling locations
            shape N1HW <- broadcasted from N11W
        """
        return self.sampler.forward(kspace, mask, uncertainty_map)
