from enum import auto
import torch
from torch import nn
from torch.nn import functional as F
from activemri.baselines.loupe_codes.layers import *

from activemri.baselines.loupe_codes import transforms

from torch.autograd import Function
import matplotlib.pyplot as plt
import numpy as np
import sigpy 
import sigpy.mri 

class LOUPESampler(nn.Module):
    """
        LOUPE Sampler
    """
    def __init__(self, shape=[320, 320], slope=5, sparsity=0.25, line_constrained=False, 
        conjugate_mask=False, preselect=False, preselect_num=2, random=False, poisson=False,
        spectrum=False, equispaced=False):
        """
            shape ([int. int]): Shape of the reconstructed image
            slope (float): Slope for the Loupe probability mask. Larger slopes make the mask converge faster to
                           deterministic state.
            sparsity (float): Predefined sparsity of the learned probability mask. 1 / acceleration_ratio
            line_constrained (bool): Sample kspace measurements column by column
            conjugate_mask (bool): For real image, the corresponding kspace measurements have conjugate symmetry property
                (point reflection). Therefore, the information in the left half of the kspace image is the same as the
                other half. To take advantage of this, we can force the model to only sample right half of the kspace
                (when conjugate_mask is set to True)
            preselect: preselect center regions  
        """
        super().__init__()

        assert conjugate_mask is False

        # probability mask
        if line_constrained:
            self.gen_mask = LineConstrainedProbMask(shape, slope, preselect=preselect, preselect_num=preselect_num)
        else:
            self.gen_mask = ProbMask(shape, slope, preselect_num=preselect_num)

        self.rescale = RescaleProbMap
        self.binarize = ThresholdRandomMaskSigmoidV1.apply # FIXME

        self.preselect =preselect 
        self.preselect_num_one_side = preselect_num // 2
        self.shape = shape
        self.line_constrained = line_constrained
        self.random_baseline = random 
        self.poisson_baseline = poisson 
        self.spectrum_baseline = spectrum 
        self.equispaced_baseline = equispaced

        if self.poisson_baseline:
            self.acc = 1 / (sparsity + (self.preselect_num_one_side*2)**2 / (128*128))
            print("generate variable density mask with acceleration {}".format(self.acc))

        if self.spectrum_baseline:
            acc = 1 / (sparsity + (self.preselect_num_one_side*2)**2 / (128*128))
            print("generate spectrum mask with acceleration {}".format(acc))
            mask = torch.load('resources/spectrum_{}x_128.pt'.format(int(acc)))
            mask = mask.reshape(1, 128, 128, 1).float()
            self.spectrum_mask = nn.Parameter(mask, requires_grad=False)

        if self.equispaced_baseline:
            acc = 1 / (sparsity + (self.preselect_num_one_side)*2 / (128))

            assert self.line_constrained
            assert acc == 4 
            mask = torch.load('resources/equispaced_4x_128.pt').reshape(1, 128, 128, 1).float()
            self.equispaced_mask = nn.Parameter(mask, requires_grad=False)

    def _gen_poisson_mask(self):
        mask = sigpy.mri.poisson((128, 128), self.acc, dtype='int32', crop_corner=False) 
        mask = torch.tensor(mask).reshape(1, 128, 128, 1).float()
        return mask 

    def _mask_neg_entropy(self, mask, eps=1e-10):
        # negative of pixel wise entropy
        entropy = mask * torch.log(mask+eps) + (1-mask) * torch.log(1-mask+eps)
        return entropy

    def forward(self, kspace, sparsity):
        # kspace: NHWC
        # sparsity (float)
        prob_mask = self.gen_mask(kspace)

        if self.random_baseline:
            prob_mask = torch.ones_like(prob_mask) / 4 

            if not self.line_constrained:
                prob_mask[:, :self.preselect_num_one_side, :self.preselect_num_one_side] = 0
                prob_mask[:, :self.preselect_num_one_side, -self.preselect_num_one_side:] = 0 
                prob_mask[:, -self.preselect_num_one_side:, :self.preselect_num_one_side] = 0 
                prob_mask[:, -self.preselect_num_one_side:, -self.preselect_num_one_side:] = 0
            else:
                prob_mask[..., :self.preselect_num_one_side, :] = 0
                prob_mask[..., -self.preselect_num_one_side:, :] = 0 


        if not self.preselect:
            rescaled_mask = self.rescale(prob_mask, sparsity)
            binarized_mask = self.binarize(rescaled_mask)
        else:
            rescaled_mask = self.rescale(prob_mask, sparsity)     
            if self.training:
                binarized_mask = self.binarize(rescaled_mask)
            else:
                binarized_mask = self.binarize(rescaled_mask)

            if not self.line_constrained:
                binarized_mask[:, :self.preselect_num_one_side, :self.preselect_num_one_side] = 1
                binarized_mask[:, :self.preselect_num_one_side, -self.preselect_num_one_side:] = 1 
                binarized_mask[:, -self.preselect_num_one_side:, :self.preselect_num_one_side] = 1 
                binarized_mask[:, -self.preselect_num_one_side:, -self.preselect_num_one_side:] = 1 
            else:
                binarized_mask[..., :self.preselect_num_one_side, :] = 1
                binarized_mask[..., -self.preselect_num_one_side:, :] = 1 

        neg_entropy = self._mask_neg_entropy(rescaled_mask)

        if self.poisson_baseline:
            assert not self.line_constrained
            binarized_mask = transforms.fftshift(self._gen_poisson_mask(), dim=(1,2)).to(kspace.device)

        if self.spectrum_baseline:
            assert not self.line_constrained
            binarized_mask = self.spectrum_mask # DC are in the corners 

        if self.equispaced_baseline:
            binarized_mask = transforms.fftshift(self.equispaced_mask, dim=(1, 2))

        masked_kspace = binarized_mask * kspace

        data_to_vis_sampler = {'prob_mask': transforms.fftshift(prob_mask[0,:,:,0],dim=(0,1)).cpu().detach().numpy(), 
                               'rescaled_mask': transforms.fftshift(rescaled_mask[0,:,:,0],dim=(0,1)).cpu().detach().numpy(), 
                               'binarized_mask': transforms.fftshift(binarized_mask[0,:,:,0],dim=(0,1)).cpu().detach().numpy()}

        return masked_kspace, binarized_mask, neg_entropy, data_to_vis_sampler

class BiLOUPESampler(nn.Module):
    """
        LOUPE Sampler
    """
    def __init__(self, shape=[320, 320], slope=5, sparsity=0.25, line_constrained=False, 
        conjugate_mask=False, preselect=False, preselect_num=2):
        """
            shape ([int. int]): Shape of the reconstructed image
            slope (float): Slope for the Loupe probability mask. Larger slopes make the mask converge faster to
                           deterministic state.
            sparsity (float): Predefined sparsity of the learned probability mask. 1 / acceleration_ratio
            line_constrained (bool): Sample kspace measurements column by column
            conjugate_mask (bool): For real image, the corresponding kspace measurements have conjugate symmetry property
                (point reflection). Therefore, the information in the left half of the kspace image is the same as the
                other half. To take advantage of this, we can force the model to only sample right half of the kspace
                (when conjugate_mask is set to True)
            preselect: preselect center regions  
        """
        super().__init__()

        assert conjugate_mask is False
        assert line_constrained

        # probability mask
        if line_constrained:
            self.gen_mask = BiLineConstrainedProbMask([shape[0]*2], slope, preselect=preselect, preselect_num=preselect_num)
        else:
            assert 0 
            self.gen_mask = ProbMask(shape, slope)

        self.rescale = RescaleProbMap
        self.binarize = ThresholdRandomMaskSigmoidV1.apply # FIXME

        self.preselect =preselect 
        self.preselect_num = preselect_num 
        self.shape = shape
        
    def _mask_neg_entropy(self, mask, eps=1e-10):
        # negative of pixel wise entropy
        entropy = mask * torch.log(mask+eps) + (1-mask) * torch.log(1-mask+eps)
        return entropy

    def forward(self, kspace, sparsity):
        # kspace: NHWC
        # sparsity (float)
        prob_mask = self.gen_mask(kspace)

        batch_size = kspace.shape[0]

        if not self.preselect:
            assert 0 
        else:
            rescaled_mask = self.rescale(prob_mask, sparsity/2)
            if self.training:
                binarized_mask = self.binarize(rescaled_mask)
            else:
                binarized_mask = self.binarize(rescaled_mask)

            # always preselect vertical lines 
            binarized_vertical_mask, binarized_horizontal_mask = torch.chunk(binarized_mask, dim=2, chunks=2)

            binarized_horizontal_mask = binarized_horizontal_mask.transpose(1, 2)

            binarized_mask = torch.clamp(binarized_vertical_mask + binarized_horizontal_mask, max=1, min=0)

            binarized_mask[..., :self.preselect_num, :] = 1  

        masked_kspace = binarized_mask * kspace
        neg_entropy = self._mask_neg_entropy(rescaled_mask)

        # for visualization purpose 
        vertical_mask, horizontal_mask = torch.chunk(prob_mask.reshape(1, -1), dim=-1, chunks=2)
        prob_mask =vertical_mask.reshape(1, 1, 1, -1)+horizontal_mask.reshape(1, 1, -1, 1)

        rescaled_vertical_mask, rescaled_horizontal_mask = torch.chunk(rescaled_mask.reshape(1, -1), dim=-1, chunks=2)
        rescaled_mask = rescaled_vertical_mask.reshape(1, 1, 1, -1)+rescaled_horizontal_mask.reshape(1, 1, -1, 1)
    
        
        data_to_vis_sampler = {'prob_mask': transforms.fftshift(prob_mask[0, 0],dim=(0,1)).cpu().detach().numpy(), 
                               'rescaled_mask': transforms.fftshift(rescaled_mask[0, 0],dim=(0,1)).cpu().detach().numpy(), 
                               'binarized_mask': transforms.fftshift(binarized_mask[0,:,:,0],dim=(0,1)).cpu().detach().numpy()}

        return masked_kspace, binarized_mask, neg_entropy, data_to_vis_sampler
