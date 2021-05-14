import logging
import argparse
import time
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from tensorboardX import SummaryWriter

import activemri.envs.loupe_envs as loupe_envs
from activemri.baselines.loupe_codes.samplers import *
from activemri.baselines.loupe_codes.reconstructors import *
from activemri.baselines.loupe_codes.layers import *
from activemri.baselines.loupe_codes.transforms import *
from ..envs.util import compute_ssim_torch, compute_psnr_torch
import os 
from torch.autograd import Function
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Optional, Tuple


class LOUPE(nn.Module):
    """
        Reimplementation of Loupe (https://arxiv.org/abs/1907.11374) sampling-reconstruction framework
        with straight through estimator (https://arxiv.org/abs/1308.3432).

        The model gets two components: A learned probability mask (Sampler) and a UNet reconstructor.
        
        Args:
            in_chans (int): Number of channels in the input to the reconstructor (2 for complex image, 1 for real image).
            out_chans (int): Number of channels in the output to the reconstructor (default is 1 for real image).
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
            shape ([int. int]): Shape of the reconstructed image
            slope (float): Slope for the Loupe probability mask. Larger slopes make the mask converge faster to
                           deterministic state.
            sparsity (float): Predefined sparsity of the learned probability mask. 1 / acceleration_ratio
            line_constrained (bool): Sample kspace measurements column by column
            conjugate_mask (bool): For real image, the corresponding kspace measurements have conjugate symmetry property
                (point reflection). Therefore, the information in the left half of the kspace image is the same as the
                other half. To take advantage of this, we can force the model to only sample right half of the kspace
                (when conjugate_mask is set to True)
            preselect (bool): preselect DC components
            bi_dir (bool): sample from both vertical and horizontal lines 
    """
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 64,
        num_pool_layers: int = 4,
        drop_prob: float = 0,
        shape: List[int] = [320, 320],
        slope: float = 5,
        sparsity: float = 0.25,
        line_constrained: bool = False,
        conjugate_mask: bool = False,
        preselect: bool = False,
        preselect_num: int = 2,
        bi_dir: bool = False,
        random: bool = False,
        poisson: bool = False,
        spectrum: bool = False,
        equispaced: bool = False  
    ):
        super().__init__()
        assert conjugate_mask is False 

        self.preselect =preselect 

        if not line_constrained:
            sparsity = (sparsity - preselect_num**2 / (shape[0]*shape[1])) if preselect else sparsity
        else:
            sparsity = (sparsity - preselect_num / shape[1]) if preselect else sparsity

        # for backward compatability
        self.samplers = nn.ModuleList()

        if bi_dir:
            assert 0 
            self.samplers.append(BiLOUPESampler(shape, slope, sparsity, line_constrained, conjugate_mask, preselect, preselect_num))
        else:
            self.samplers.append(LOUPESampler(shape, slope, sparsity, line_constrained, conjugate_mask, preselect, preselect_num, 
                random=random, poisson=poisson, spectrum=spectrum, equispaced=equispaced))
        
        self.reconstructor = LOUPEUNet(in_chans, out_chans, chans, num_pool_layers, drop_prob)
        
        self.sparsity = sparsity
        self.conjugate_mask = conjugate_mask
        self.data_for_vis = {}

        if in_chans == 1:
            assert self.conjugate_mask, "Reconstructor (denoiser) only take the real part of the ifft output"

    def forward(self, target, kspace, seed=None):
        """
        Args:
            kspace (torch.Tensor): Input tensor of shape NHWC (kspace data)
        Returns:
            (torch.Tensor): Output tensor of shape NCHW (reconstructed image )
        """

        # choose kspace sampling location
        # masked_kspace: NHWC
        masked_kspace, mask, neg_entropy, data_to_vis_sampler = self.samplers[0](kspace, self.sparsity)
        
        self.data_for_vis.update(data_to_vis_sampler)

        # Inverse Fourier Transform to get zero filled solution
        # NHWC to NCHW
        zero_filled_recon = transforms.fftshift(transforms.ifft2(masked_kspace),dim=(1,2)).permute(0, -1, 1, 2)
        
        if self.conjugate_mask:
            # only the real part of the ifft output is the same as the original image
            # when you only sample in one half of the kspace
            recon = self.reconstructor(zero_filled_recon[:,0:1,:,:])
        else:
            recon = self.reconstructor(zero_filled_recon, 0)
        
        self.data_for_vis.update({'input': target[0,0,:,:].cpu().detach().numpy(),
         'kspace': transforms.complex_abs(transforms.fftshift(kspace[0,:,:,:],dim=(0,1))).cpu().detach().numpy(),
         'masked_kspace': transforms.complex_abs(transforms.fftshift(masked_kspace[0,:,:,:],dim=(0,1))).cpu().detach().numpy(),
         'zero_filled_recon': zero_filled_recon[0,0,:,:].cpu().detach().numpy(),
         'recon': recon[0,0,:,:].cpu().detach().numpy()})
        
        pred_dict = {'output': recon.norm(dim=1, keepdim=True), 'energy': neg_entropy, 'mask': mask}

        return pred_dict

    def loss(self, pred_dict, target_dict, meta, loss_type):
        """
        Args:
            pred_dict:
                output: reconstructed image from downsampled kspace measurement
                energy: negative entropy of the probability mask
                mask: the binazried sampling mask (used for visualization)

            target_dict:
                target: original fully sampled image

            meta:
                recon_weight: weight of reconstruction loss
                entropy_weight: weight of the entropy loss (to encourage exploration)
        """
        target = target_dict['target']
        pred = pred_dict['output']
        energy = pred_dict['energy']
        
        if loss_type == 'l1':
            reconstruction_loss = F.l1_loss(pred, target, size_average=True) 
        elif loss_type == 'ssim':
            reconstruction_loss = -torch.mean(compute_ssim_torch(pred, target))
        elif loss_type == 'psnr':
            reconstruction_loss = - torch.mean(compute_psnr_torch(pred, target))
        else:
            raise NotImplementedError

        entropy_loss = torch.mean(energy)

        loss = entropy_loss * meta['entropy_weight'] + reconstruction_loss * meta['recon_weight']

        log_dict = {'Total Loss': loss.item(), 'Entropy': entropy_loss.item(), 'Reconstruction': reconstruction_loss.item()}

        return loss, log_dict

    def show_mask(self):
        """
            Return:
                (np.ndarray) the learned undersampling mask shape HW
        """
        H, W = self.samplers[0].shape
        pseudo_image = torch.zeros(1, H, W, 1)

        return self.samplers[0].mask(pseudo_image, self.sparsity)

    def visualize_and_save(self, options, epoch, data_for_vis_name):
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=[18, 6])
        
        cmap = "viridis" if options.resolution[1] == 32 else "gray"
        
        sp1 = ax[0, 0].imshow(self.data_for_vis['input'], cmap=cmap)
        sp2 = ax[0, 1].imshow(np.log(self.data_for_vis['kspace']))
        sp3 = ax[0, 2].imshow(np.log(self.data_for_vis['masked_kspace']))
        sp4 = ax[0, 3].imshow(self.data_for_vis['zero_filled_recon'], cmap=cmap)
        sp5 = ax[0, 4].imshow(self.data_for_vis['recon'], cmap=cmap)
        sp6 = ax[1, 0].imshow(self.data_for_vis['prob_mask'], aspect='auto')
        sp7 = ax[1, 1].imshow(self.data_for_vis['rescaled_mask'], aspect='auto')
        sp8 = ax[1, 2].imshow(self.data_for_vis['binarized_mask'], aspect='auto')
        ax[0, 0].title.set_text('Input image')
        ax[0, 1].title.set_text('Log k-space of the input')
        ax[0, 2].title.set_text('Undersampled log k-space')
        ax[0, 3].title.set_text('Zero-filled reconstruction')
        ax[0, 4].title.set_text('Reconstruction')
        ax[1, 0].title.set_text('Probabilistic mask')
        ax[1, 1].title.set_text('Rescaled mask')
        ax[1, 2].title.set_text('Binary mask')
        fig.colorbar(sp1, ax=ax[0, 0])
        fig.colorbar(sp2, ax=ax[0, 1])
        fig.colorbar(sp3, ax=ax[0, 2])
        fig.colorbar(sp4, ax=ax[0, 3])
        fig.colorbar(sp5, ax=ax[0, 4])
        fig.colorbar(sp6, ax=ax[1, 0])
        fig.colorbar(sp7, ax=ax[1, 1])
        fig.colorbar(sp8, ax=ax[1, 2])
        ax[1, 3].axis('off')
        ax[1, 4].axis('off')
        plt.suptitle('Epoch = [{}/{}]'.format(1 + epoch, options.num_epochs), fontsize=20)
        
        if not os.path.isdir(options.visualization_dir):
            os.mkdir(options.visualization_dir)

        plt.savefig(str(options.visualization_dir)+'/'+data_for_vis_name+'.png')
        plt.close()
