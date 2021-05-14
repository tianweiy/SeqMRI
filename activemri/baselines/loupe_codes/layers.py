"""
Portion of this code is from fastmri(https://github.com/facebookresearch/fastMRI)

Copyright (c) Facebook, Inc. and its affiliates.

Licensed under the MIT License.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from activemri.baselines.loupe_codes import transforms

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """
    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super(ConvBlock, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class ProbMask(nn.Module):
    """
    A learnable probablistic mask with the same shape as the kspace measurement.
    This learned mask samples measurements in the whole kspace
    """
    def __init__(self, shape=[320, 320], slope=5, preselect=False, preselect_num=0):
        """
            shape ([int. int]): Shape of the reconstructed image
            slope (float): Slope for the Loupe probability mask. Larger slopes make the mask converge faster to
                           deterministic state.
        """
        super(ProbMask, self).__init__()

        self.slope = slope
        self.preselect = preselect 
        self.preselect_num_one_side = preselect_num // 2 

        init_tensor = self._slope_random_uniform(shape)
        self.mask = nn.Parameter(init_tensor)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape NHWC

        Returns:
            (torch.Tensor): Output tensor of shape NHWC
        """
        logits = self.mask.view(1, input.shape[1], input.shape[2], 1)
        return torch.sigmoid(self.slope * logits)

    def _slope_random_uniform(self, shape, eps=1e-2):
        """
            uniform random sampling mask
        """
        temp = torch.zeros(shape).uniform_(eps, 1-eps)

        # logit with slope factor
        logits = -torch.log(1./temp-1.) / self.slope

        logits = logits.reshape(1, shape[0], shape[1], 1) 

        logits[:, :self.preselect_num_one_side, :self.preselect_num_one_side] = -1e2
        logits[:, :self.preselect_num_one_side, -self.preselect_num_one_side:] = -1e2 
        logits[:, -self.preselect_num_one_side:, :self.preselect_num_one_side] = -1e2 
        logits[:, -self.preselect_num_one_side:, -self.preselect_num_one_side:] = -1e2  

        return logits 

class HalfProbMask(nn.Module):
    """
    A learnable probablistic mask with the same shape as half of the kspace measurement (to force the model
    to not take conjugate symmetry points)
    """
    def __init__(self, shape=[320, 320], slope=5):
        """
            shape ([int. int]): Shape of the reconstructed image
            slope (float): Slope for the Loupe probability mask. Larger slopes make the mask converge faster to
                           deterministic state.
        """
        super(HalfProbMask, self).__init__()

        self.slope = slope
        init_tensor = self._slope_random_uniform(shape)
        self.mask = nn.Parameter(init_tensor)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape NHWC

        Returns:
            (torch.Tensor): Output tensor of shape NHWC
        """
        mask =  torch.sigmoid(self.slope * self.mask).to(input.device).view(1, input.shape[1], input.shape[2]//2, 1) # only half of the kspace

        zero_mask = torch.zeros((1, input.shape[1], input.shape[2], 1))
        zero_mask[:, :, :input.shape[2]//2] = mask

        return zero_mask.to(input.device)

    def _slope_random_uniform(self, shape, eps=1e-2):
        """
            uniform random sampling mask with the shape as half of the kspace measurement
        """
        temp = torch.zeros([shape[0], shape[1]//2]).uniform_(eps, 1-eps)

        # logit with slope factor
        return -torch.log(1./temp-1.) / self.slope

class LineConstrainedProbMask(nn.Module):
    """
    A learnable probablistic mask with the same shape as the kspace measurement.
    The mask is constrinaed to include whole kspace lines in the readout direction
    """
    def __init__(self, shape=[32], slope=5, preselect=False, preselect_num=2):
        super(LineConstrainedProbMask, self).__init__()

        if preselect:
            length = shape[0] - preselect_num 
        else:
            length = shape[0]

        self.preselect_num = preselect_num 
        self.preselect = preselect 
        self.slope = slope
        init_tensor = self._slope_random_uniform(length)
        self.mask = nn.Parameter(init_tensor)

    def forward(self, input, eps=1e-10):
        """
        Args:
            input (torch.Tensor): Input tensor of shape NHWC

        Returns:
            (torch.Tensor): Output tensor of shape NHWC
        """
        logits = self.mask
        mask = torch.sigmoid(self.slope * logits).view(1, 1, self.mask.shape[0], 1) 

        if self.preselect:
            if self.preselect_num % 2 ==0:
                zeros = torch.zeros(1, 1, self.preselect_num // 2, 1).to(input.device) 
                mask = torch.cat([zeros, mask, zeros], dim=2)
            else:
                raise NotImplementedError()

        return mask 

    def _slope_random_uniform(self, shape, eps=1e-2):
        """
            uniform random sampling mask with the same shape as the kspace measurement
        """
        temp = torch.zeros(shape).uniform_(eps, 1-eps)

        # logit with slope factor
        return -torch.log(1./temp-1.) / self.slope

class BiLineConstrainedProbMask(LineConstrainedProbMask):
    def __init__(self, shape, slope, preselect, preselect_num):
        super().__init__(shape=shape, slope=slope, preselect=preselect, preselect_num=preselect_num)

    def forward(self, input, eps=1e-10):
        """
        Args:
            input (torch.Tensor): Input tensor of shape NHWC

        Returns:
            (torch.Tensor): Output tensor of shape NHWC
        """
        logits = self.mask
        mask = torch.sigmoid(self.slope * logits).view(1, 1, self.mask.shape[0], 1) 

        if self.preselect:
            zeros = torch.zeros(1, 1, self.preselect_num, 1).to(input.device) 
            mask = torch.cat([zeros, mask], dim=2)
        return mask 

class HalfLineConstrainedProbMask(nn.Module):
    """
    A learnable probablistic mask with the same shape as half of the kspace measurement (to force the model
    to not take conjugate symmetry points).
    The mask is constrained to include whole kspace lines in the readout direction
    """
    def __init__(self, shape=[32], slope=5):
        """
            shape ([int. int]): Shape of the reconstructed image
            slope (float): Slope for the Loupe probability mask. Larger slopes make the mask converge faster to
                           deterministic state.
        """
        super(HalfLineConstrainedProbMask, self).__init__()

        self.slope = slope
        init_tensor = self._slope_random_uniform(shape[0])
        self.mask = nn.Parameter(init_tensor)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape NHWC

        Returns:
            (torch.Tensor): Output tensor of shape NHWC
        """
        mask =  torch.sigmoid(self.slope * self.mask).to(input.device).view(1, 1, input.shape[2]//2, 1) # only half of the kspace

        zero_mask = torch.zeros((1, 1, input.shape[2], 1))
        zero_mask[:, :, :input.shape[2]//2] = mask

        return zero_mask.to(input.device)

    def _slope_random_uniform(self, shape, eps=1e-2):
        """
            uniform random sampling mask with the shape as half of the kspace measurement
        """
        temp = torch.zeros(shape//2).uniform_(eps, 1-eps)

        # logit with slope factor
        return -torch.log(1./temp-1.) / self.slope


def RescaleProbMap(batch_x, sparsity):
    """
        Rescale Probability Map
        given a prob map x, rescales it so that it obtains the desired sparsity

        if mean(x) > sparsity, then rescaling is easy: x' = x * sparsity / mean(x)
        if mean(x) < sparsity, one can basically do the same thing by rescaling
                                (1-x) appropriately, then taking 1 minus the result.
    """
    batch_size = len(batch_x)
    ret = []
    for i in range(batch_size):
        x = batch_x[i:i+1]
        xbar = torch.mean(x)
        r = sparsity / (xbar)
        beta = (1-sparsity) / (1-xbar)

        # compute adjucement
        le = torch.le(r, 1).float()
        ret.append(le * x * r + (1-le) * (1 - (1 - x) * beta))

    return torch.cat(ret, dim=0)


class ThresholdRandomMaskSigmoidV1(Function):
    def __init__(self):
        """
            Straight through estimator.
            The forward step stochastically binarizes the probability mask.
            The backward step estimate the non differentiable > operator using sigmoid with large slope (10).
        """
        super(ThresholdRandomMaskSigmoidV1, self).__init__()

    @staticmethod
    def forward(ctx, input):
        batch_size = len(input)
        probs = [] 
        results = [] 

        for i in range(batch_size):
            x = input[i:i+1]

            count = 0 
            while True:
                prob = x.new(x.size()).uniform_()
                result = (x > prob).float()

                if torch.isclose(torch.mean(result), torch.mean(x), atol=1e-3):
                    break

                count += 1 

                if count > 1000:
                    print(torch.mean(prob), torch.mean(result), torch.mean(x))
                    assert 0 

            probs.append(prob)
            results.append(result)

        results = torch.cat(results, dim=0)
        probs = torch.cat(probs, dim=0)
        ctx.save_for_backward(input, probs)

        return results  

    @staticmethod
    def backward(ctx, grad_output):
        slope = 10
        input, prob = ctx.saved_tensors

        # derivative of sigmoid function
        current_grad = slope * torch.exp(-slope * (input - prob)) / torch.pow((torch.exp(-slope*(input-prob))+1), 2)

        return current_grad * grad_output

def MaximumBinarize(input):
    batch_size = len(input)
    results = [] 

    for i in range(batch_size):
        x = input[i:i+1]
        num = torch.sum(x).round().int()

        indices = torch.topk(x.reshape(-1), k=num)[1]

        mask = torch.zeros_like(x).reshape(-1)
        
        mask[indices] = 1

        mask = mask.reshape(*x.shape) 

        results.append(mask)

    results = torch.cat(results, dim=0)

    return results  
