from activemri.baselines.sequential_sampling_codes.sampler2d import Sampler2D
from activemri.baselines.sequential_sampling_codes.conv_sampler import BiLineConstrainedSampler, KspaceLineConstrainedSampler
import torch
from typing import Any
from torch import nn
import numpy as np
from torch.nn import functional as F
from activemri.baselines.sequential_sampling_codes import ConvSamplerSmall, LineConstrainedSampler
from activemri.baselines.loupe_codes.reconstructors import LOUPEUNet
import matplotlib.pyplot as plt
import fastmri
import activemri.baselines.loupe_codes.transforms as transforms
from ..envs.util import compute_gaussian_nll_loss, compute_ssim_torch, compute_psnr_torch
from activemri.baselines.loupe_codes.layers import RescaleProbMap, ThresholdRandomMaskSigmoidV1, MaximumBinarize
import os 

class Sampler(nn.Module):
    """
        Convolutional Sampler.
        It contains three components:
            mask_net: A convolutional mask generator
            rescale_mask: Rescale the generated mask to specific sparsity
            binarizer: Sampling from the bernouli distribution
    """

    def __init__(self, shape=[320, 320], line_constrained=False, bi_direction=False, binary_sampler=False,
        with_uncertainty=False, clamp=100, detach_kspace=False, fixed_input=False):
        """
            shape ([int. int]): Shape of the reconstructed image
            line_constrained (bool): Sample kspace measurements column by column
        """
        super().__init__()
        self.binary_sampler = binary_sampler

        self.detach_kspace = detach_kspace

        # probability mask
        if line_constrained:
            if bi_direction:
                if self.binary_sampler:
                    assert 0
                else:
                    self.mask_net = BiLineConstrainedSampler(shape[0], clamp=clamp, with_uncertainty=with_uncertainty, fixed_input=fixed_input)
            else:
                self.mask_net = KspaceLineConstrainedSampler(shape[0], shape[0], with_uncertainty=with_uncertainty, fixed_input=fixed_input)
        else:
            self.mask_net = Sampler2D(fixed_input=fixed_input)

        self.bidirection = bi_direction
        self.rescale = RescaleProbMap
        self.binarize = ThresholdRandomMaskSigmoidV1.apply
        self.shape = shape
        self.line_constrained = line_constrained
        self.with_uncertainty = with_uncertainty

    def forward(self, full_kspace, observed_kspace, old_mask, budget, uncertainty_mask=None):
        """
        Args:
            full_kspace (torch.Tensor): The ground truth kspace to sample measurements
                                        shape NHWC
            observed_kspace (torch.Tensor): Already sampled measurements shape NHWC
            old_mask (torch.Tensor): previous sampling trajectories shape NCHW
            budget (int): number of sampling lines or pixels
        Returns:
            masked_kspace (torch.Tensor): Measured Kspace shape NHWC
            mask (torch.Tensor): Updated sampling mask (which contains all previous sampling locations) shape NCHW
        """
        if self.detach_kspace:
            observed_kspace = observed_kspace.detach()

        if self.with_uncertainty:
            temp = torch.cat([observed_kspace, full_kspace*old_mask.permute(0, 2, 3, 1)], dim=-1)
            mask = self.mask_net(temp, old_mask, uncertainty_mask)
        else:
            temp = torch.cat([observed_kspace, full_kspace*old_mask.permute(0, 2, 3, 1)], dim=-1)
            mask = self.mask_net(temp, old_mask)

            # mask = self.mask_net(full_kspace*old_mask.permute(0, 2, 3, 1), old_mask)
            # temp = torch.cat([observed_kspace, full_kspace*old_mask.permute(0, 2, 3, 1)], dim=-1)

        sparsity = budget / (self.shape[0] * self.shape[1]) if not self.line_constrained else (budget / self.shape[0])

        batch_size = mask.shape[0]

        if self.bidirection:
            rescaled_mask = self.rescale(mask, sparsity/2)
            if self.training:
                binary_mask = self.binarize(rescaled_mask)
            else:
                binary_mask = self.binarize(rescaled_mask)

            binarized_vertical_mask, binarized_horizontal_mask = torch.chunk(binary_mask, dim=-1, chunks=2)

            binarized_vertical_mask = binarized_vertical_mask.reshape(batch_size, 1, 1, -1)
            binarized_horizontal_mask = binarized_horizontal_mask.reshape(batch_size, 1, -1, 1)
            binary_mask = torch.clamp(binarized_vertical_mask + binarized_horizontal_mask, max=1, min=0)

            if self.line_constrained:
                # for visualization purpose
                vertical_mask, horizontal_mask = torch.chunk(mask, dim=-1, chunks=2)
                mask =vertical_mask.reshape(batch_size, 1, 1, -1)+horizontal_mask.reshape(batch_size, 1, -1, 1)

                rescaled_vertical_mask, rescaled_horizontal_mask = torch.chunk(rescaled_mask, dim=-1, chunks=2)
                rescaled_mask = rescaled_vertical_mask.reshape(batch_size, 1, 1, -1)+rescaled_horizontal_mask.reshape(batch_size, 1, -1, 1)
        else:
            rescaled_mask = self.rescale(mask, sparsity)
            if self.training:
                binary_mask = self.binarize(rescaled_mask)
            else:
                binary_mask = self.binarize(rescaled_mask)
                # binary_mask = MaximumBinarize(rescaled_mask)

        binary_mask = old_mask + binary_mask

        # double check to ensure the mask is within range
        binary_mask = torch.clamp(binary_mask, min=0, max=1)

        # NCHW -> NHWC
        binary_mask = binary_mask.permute(0, 2, 3, 1)
        masked_kspace = binary_mask * full_kspace

        data_to_vis_sampler = {
            'pred_kspace': transforms.complex_abs(transforms.fftshift(observed_kspace[0,:,:,:],dim=(0,1))).cpu().detach().numpy(),
            'prob_mask': transforms.fftshift(mask[0,0,:,:],dim=(0,1)).cpu().detach().numpy(),
            'rescaled_mask': transforms.fftshift(rescaled_mask[0,0,:,:],dim=(0,1)).cpu().detach().numpy(),
            'binarized_mask': transforms.fftshift(binary_mask[0,:,:,0],dim=(0,1)).cpu().detach().numpy()}


        return masked_kspace, binary_mask.permute(0, -1, 1, 2), data_to_vis_sampler # NCHW

class SequentialUnet(nn.Module):
    """
    PyTorch implementation of a Sequential Sampling model.
    We sequentially sample measurements from the kspace for image reconstruction.
    Later sampling locations are determined by previous observations using a neural
    network.
    By default, in the 2d case, we select the center 4x4(2x4) regions before any adaptive sampling
    in the 1d case, we select 1 or 2 center lines before any adaptive sampling
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, shape=[320, 320],
                 sparsity=0.25, line_constrained=False, num_step=4, preselect=True, preselect_num=2, conjugate_mask=False,
                 bi_direction=False, binary_sampler=False, clamp=100, old_recon=False, with_uncertainty=False,
                 detach_kspace=False, fixed_input=False, pretrained_recon=False):
        """
        Args:
            in_chans (int): Number of channels in the input to the reconstructor (2 for complex image, 1 for real image).
            out_chans (int): Number of channels in the output to the reconstructor (default is 1 for real image).
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
            shape ([int. int]): Shape of the reconstructed image
            sparsity (float): Predefined sparsity of the learned probability mask. 1 / acceleration_ratio
            line_constrained (bool): Sample kspace measurements column by column
            num_step (int): Number of sequential sampling steps
            conjugate_mask (bool): For real image, the corresponding kspace measurements have conjugate symmetry property
                (point reflection). Therefore, the information in the left half of the kspace image is the same as the
                other half. To take advantage of this, we can force the model to only sample right half of the kspace
                (when conjugate_mask is set to True)
        """
        super().__init__()

        assert conjugate_mask is False

        self.reconstructor = LOUPEUNet(in_chans, out_chans, chans, num_pool_layers, drop_prob,
                bi_dir=bi_direction, old_recon=old_recon, with_uncertainty=with_uncertainty)
        self.sampler = Sampler(shape, line_constrained, bi_direction=bi_direction, binary_sampler=binary_sampler,
            clamp=clamp, with_uncertainty=with_uncertainty, detach_kspace=detach_kspace, fixed_input=fixed_input)

        self.num_step = num_step
        self.shape = shape
        self.preselect = preselect

        if pretrained_recon:
            if sparsity == 0.25:
                path = 'reconstructors/4x_random_recon.pt'
            else:
                assert 0 

            print("Use Pretrained Reconstructor from {} and freeze Reconstructor Weights".format(path))
            self.reconstructor.load_state_dict(torch.load(path, map_location='cpu')['model'])
            self.reconstructor.requires_grad_(False)
            self.reconstructor.eval()

        # original sparsity - center region's area
        if not line_constrained:
            self.sparsity = sparsity - (preselect_num*preselect_num) / (shape[0]*shape[1]) if preselect else sparsity
        else:
            self.sparsity = (sparsity - preselect_num / shape[1]) if preselect else sparsity

        if line_constrained:
            self.budget = self.sparsity * shape[0] 
        else:
            self.budget = self.sparsity * shape[0] * shape[1]

        self.preselect_num_one_side = preselect_num // 2
        self.bidirection = bi_direction
        self.conjugate_mask = conjugate_mask
        self.line_constrained = line_constrained
        self.with_uncertainty = with_uncertainty
        self.data_for_vis = []

        assert old_recon is False 

        for i in range(self.num_step):
            self.data_for_vis.append(dict())

        print("Number of rnn iterations ", self.num_step)
        print("Sparsity ", sparsity)
        print("Reuse Reconstruction {}".format(old_recon))



    def step_forward(self, full_kspace, pred_kspace, old_mask, old_recon, target, step, uncertainty_map=None):
        """
        Args:
            full_kspace (torch.Tensor): The ground truth kspace to sample measurements
                                        shape NHWC
            pred_kspace (torch.Tensor): Kspace of the UNet output shape NHWC
            old_mask (torch.Tensor): previous sampling trajectories shape NCHW
            step: current sampling iteration index
            eps: epsilon value to avoid undefined gradients
        Returns:
            out (torch.Tensor): Reconstructed image shape NCHW
            mask (torch.Tensor): Updated sampling mask (which contains all previous sampling locations) shape NHWC
            before_unet_img (torch.Tensor): visualization usage shape NCHW
            before_unet_kspace (torch.Tensor) shape NCHW
        """
        # evenly divide sampling budget across different iterations
        budget = self.budget / self.num_step

        # to NHWC
        masked_kspace, mask, data_to_vis_sampler = self.sampler(full_kspace, pred_kspace, old_mask, budget, uncertainty_map)
        self.data_for_vis[step].update(data_to_vis_sampler)

        # Inverse Fourier Transform to get zero filled solution NHWC -> NCHW
        # zero_filled_recon = transforms.fftshift(transforms.ifft2(masked_kspace),dim=(1,2)).permute(0, -1, 1, 2)
        zero_filled_recon = transforms.fftshift(transforms.ifft2(masked_kspace),dim=(1,2)).permute(0, -1, 1, 2)
        zero_recon = torch.norm(zero_filled_recon, dim=1, keepdim=True)

        recon = self.reconstructor(zero_filled_recon)

        self.data_for_vis[step].update({'input': target[0,0,:,:].cpu().detach().numpy(),
         'kspace': transforms.complex_abs(transforms.fftshift(full_kspace[0,:,:,:],dim=(0,1))).cpu().detach().numpy(),
         'masked_kspace': transforms.complex_abs(transforms.fftshift(masked_kspace[0,:,:,:],dim=(0,1))).cpu().detach().numpy(),
         'zero_filled_recon': zero_recon[0,0,:,:].cpu().detach().numpy(),
         'recon': recon[0,0,:,:].cpu().detach().numpy(), 'uncertainty_map': uncertainty_map[0, 0].cpu().detach().numpy()})

        pred_dict = {'output': recon, 'mask': mask, 'zero_recon': zero_recon, 'uncertainty_map': uncertainty_map}

        return pred_dict

    def _init_mask(self, data):
        """
            Take the center 4*4 region (or 2*4 in the conjugate symmetry case)
            Up-Left, Up-right, Down-Left, Down-right + FFTShift to center
            data: NHWC (INPUT)
            a: N1HW (OUTPUT)
        """
        a = torch.zeros([data.shape[0], 1, self.shape[0], self.shape[1]]).to(data.device)

        if self.preselect:
            if not self.line_constrained:
                a[:, 0, :self.preselect_num_one_side, :self.preselect_num_one_side] = 1 
                a[:, 0, :self.preselect_num_one_side, -self.preselect_num_one_side:] = 1 
                a[:, 0, -self.preselect_num_one_side:, :self.preselect_num_one_side] = 1 
                a[:, 0, -self.preselect_num_one_side:, -self.preselect_num_one_side:] = 1 
            else:
                """
                    In the line constrained case, we pre-select the center lines.
                """
                """if self.bidirection:
                    # a[:, :, :, 5] = 1
                    a[:, :, :, :self.preselect_num_one_side] = 1
                    a[:, :, :self.preselect_num_one_side, :] = 1
                else:
                """
                a[:, :, :, :self.preselect_num_one_side] = 1
                a[:, :, :, -self.preselect_num_one_side:] = 1

        return a

    def _init_kspace(self, data, mask):
        """
            data: NHWC (input image in kspace)
            mask: NCHW
            kspace: NHW2

        """
        kspace =  data * mask.permute(0, 2, 3, 1)
        init_img = transforms.fftshift(transforms.ifft2(kspace),dim=(1,2)).permute(0, -1, 1, 2)
        recon = self.reconstructor(init_img, None, mask.detach())

        image = torch.cat([recon, torch.zeros_like(recon)], dim=1).permute(0, 2, 3, 1)
        image_for_kspace = fastmri.ifftshift(image, dim=(1, 2))
        pred_kspace = image_for_kspace.fft(2, normalized=False)

        return pred_kspace

    def forward(self, target, full_kspace, idx=None):
        """
        Args:
            input (torch.Tensor): Input tensor of shape NHWC
        Returns:
            (torch.Tensor): Output tensor of shape NCHW
        """
        old_mask = self._init_mask(full_kspace)
        pred_kspace = self._init_kspace(full_kspace, old_mask)

        old_recon = torch.zeros_like(target)
        uncertainty_map = torch.zeros_like(old_mask)

        reconstructions = []
        zero_filled = []
        uncertainty_maps = []

        for i in range(self.num_step):
            pred_dict = self.step_forward(full_kspace, pred_kspace, old_mask, old_recon, target, i, uncertainty_map)

            new_img, old_recon, old_mask, uncertainty_map = (pred_dict['output'], pred_dict['output'],
                pred_dict['mask'].detach(), pred_dict['uncertainty_map'])

            reconstructions.append(new_img)
            zero_filled.append(pred_dict['zero_recon'])
            uncertainty_maps.append(uncertainty_map)

            # transform back to kspace
            # NCHW -> NHWC
            image = torch.cat([new_img, torch.zeros_like(new_img)], dim=1).permute(0, 2, 3, 1)
            image_for_kspace = fastmri.ifftshift(image, dim=(1, 2))
            pred_kspace = image_for_kspace.fft(2, normalized=False)

        pred_dict = {'output': reconstructions, 'mask': old_mask, 'zero_filled_recon': zero_filled,
                    'uncertainty_maps': uncertainty_maps}

        return pred_dict

    def loss(self, pred_dict, target_dict, meta, loss_type):
        """
        Args:
            pred_dict:
                output: reconstructed image from downsampled kspace measurement NCHW
                energy: negative entropy of the probability mask
                mask: the binazried sampling mask (used for visualization)

            target_dict:
                target: original fully sampled image NCHW

            meta:
                recon_weight: weight of reconstruction loss
                entropy_weight: weight of the entropy loss (to encourage exploration)
        """
        target = target_dict['target']
        label = target_dict['label']
        pred = pred_dict['output'][-1]
        zero_filled = pred_dict['zero_filled_recon'][-1]
        gt_kspace = target_dict['kspace']

        nll_loss =0
        if self.with_uncertainty:
            for i in range(len(pred_dict['output'])):
                pred = pred_dict['output'][i]
                uncertainty_map = pred_dict['uncertainty_maps'][i]
                nll_loss += torch.mean(compute_gaussian_nll_loss(pred, target, uncertainty_map))

        if loss_type == 'l1':
            reconstruction_loss = F.l1_loss(pred, target, size_average=True)
            zero_loss = F.l1_loss(zero_filled, target, size_average=True)
        elif loss_type == 'ssim':
            reconstruction_loss = -torch.mean(compute_ssim_torch(pred, target))
            zero_loss = -torch.mean(compute_ssim_torch(zero_filled, target))
        elif loss_type == 'psnr':
            reconstruction_loss = - torch.mean(compute_psnr_torch(pred, target))
            zero_loss = -torch.mean(compute_psnr_torch(zero_filled, target))
        elif loss_type == 'xentropy':
            criterion = nn.CrossEntropyLoss()
            reconstruction_loss = criterion(pred, label)
            zero_loss = torch.from_numpy(np.array([0]))
        else:
            raise NotImplementedError

        # k-space loss 
        image = torch.cat([pred, torch.zeros_like(pred)], dim=1).permute(0, 2, 3, 1)
        image_for_kspace = fastmri.ifftshift(image, dim=(1, 2))
        pred_kspace = image_for_kspace.fft(2, normalized=False)

        pred_kspace = torch.norm(pred_kspace, dim=-1, keepdim=True)
        gt_kspace = torch.norm(gt_kspace, dim=-1, keepdim=True)
        pred_kspace = pred_kspace.permute(0, 3, 1, 2)
        gt_kspace = gt_kspace.permute(0, 3, 1, 2)

        kspace_loss = -torch.mean(compute_ssim_torch(pred_kspace, gt_kspace)) 

        loss = reconstruction_loss * meta['recon_weight'] +  kspace_loss * meta['kspace_weight'] # + 10*zero_loss

        log_dict = {'Total Loss': loss.item(), 'Zero Filled Loss': zero_loss.item(), 
            'K-space Loss': kspace_loss.item(), 'Recon loss': reconstruction_loss.item()}

        if self.with_uncertainty:
            loss += nll_loss * meta['uncertainty_weight']
            log_dict.update({'Uncertainty Loss': nll_loss.item()})

        return loss, log_dict

    def visualize_and_save(self, options, epoch, data_for_vis_name):
        for step in range(self.num_step):
            fig, ax = plt.subplots(nrows=2, ncols=5, figsize=[18, 6])

            cmap = "viridis" if options.resolution[1] == 32 else "gray"

            sp1 = ax[0, 0].imshow(self.data_for_vis[step]['input'], cmap=cmap)
            sp2 = ax[0, 1].imshow(np.log(self.data_for_vis[step]['kspace']))
            sp3 = ax[0, 2].imshow(np.log(self.data_for_vis[step]['masked_kspace']))
            sp4 = ax[0, 3].imshow(self.data_for_vis[step]['zero_filled_recon'], cmap=cmap)
            sp5 = ax[0, 4].imshow(self.data_for_vis[step]['recon'], cmap=cmap)
            sp6 = ax[1, 0].imshow(self.data_for_vis[step]['prob_mask'], aspect='auto')
            sp7 = ax[1, 1].imshow(self.data_for_vis[step]['rescaled_mask'], aspect='auto')
            sp8 = ax[1, 2].imshow(self.data_for_vis[step]['binarized_mask'], aspect='auto')
            sp9 = ax[1, 3].imshow(np.log(self.data_for_vis[step]['pred_kspace']), aspect='auto')
            sp10 = ax[1, 4].imshow(np.exp(self.data_for_vis[step]['uncertainty_map']), cmap=cmap)

            ax[0, 0].title.set_text('Input image')
            ax[0, 1].title.set_text('Log k-space of the input')
            ax[0, 2].title.set_text('Undersampled log k-space')
            ax[0, 4].title.set_text('Reconstruction')
            ax[1, 0].title.set_text('Probabilistic mask')
            ax[1, 1].title.set_text('Rescaled mask')
            ax[1, 2].title.set_text('Binary mask')
            ax[1, 3].title.set_text('Log k-space of Predicted Kspace')
            ax[1, 4].title.set_text('Uncertainty of the reconstruction')

            fig.colorbar(sp1, ax=ax[0, 0])
            fig.colorbar(sp2, ax=ax[0, 1])
            fig.colorbar(sp3, ax=ax[0, 2])
            fig.colorbar(sp4, ax=ax[0, 3])
            fig.colorbar(sp5, ax=ax[0, 4])
            fig.colorbar(sp6, ax=ax[1, 0])
            fig.colorbar(sp7, ax=ax[1, 1])
            fig.colorbar(sp8, ax=ax[1, 2])
            fig.colorbar(sp9, ax=ax[1, 3])
            fig.colorbar(sp10, ax=ax[1, 4])

            plt.suptitle('Epoch = [{}/{}] Step = [{}/{}]'.format(1 + epoch, options.num_epochs, step+1, options.num_step), fontsize=20)
            if not os.path.isdir(options.visualization_dir):
                os.mkdir(options.visualization_dir)

            plt.savefig(str(options.visualization_dir)+'/'+data_for_vis_name+'_step{}.png'.format(step))
            plt.close()


            path = str(options.visualization_dir)+'/'+data_for_vis_name+'_step{}.pkl'.format(step)

            import pickle 
            with open(path, 'wb') as f:
                pickle.dump(self.data_for_vis[step], f)