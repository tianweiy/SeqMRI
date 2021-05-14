from collections import defaultdict
import torch, argparse
from typing import List
import activemri.envs.envs as mri_envs
import numpy as np
import logging
import os
from torch.nn import functional as F
from tensorboardX import SummaryWriter
import time
from .loupe_codes.evaluate import Metrics, METRIC_FUNCS
from  .loupe_codes import transforms
import activemri.envs.loupe_envs as loupe_envs
import pickle
from .differential_sampler import SequentialUnet
from .loupe import LOUPE
import random
import shutil
import matplotlib.pyplot as plt
from ..envs.util import compute_gaussian_nll_loss, compute_ssim_torch, compute_psnr_torch

def build_optimizer(lr, weight_decay, model_parameters, type='adam'):
    if type == 'adam':
        optimizer = torch.optim.Adam(model_parameters, lr, weight_decay=weight_decay)
    elif type == 'sgd':
        optimizer = torch.optim.SGD(model_parameters, lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError()

    return optimizer

def build_lr_scheduler(options, optimizer):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, options.lr_step_size, options.lr_gamma)
    return scheduler

def build_model(options):
    if options.model == 'LOUPE':
        model = LOUPE(
            in_chans=options.input_chans,
            out_chans=options.output_chans,
            chans=options.num_chans,
            num_pool_layers=options.num_pools,
            drop_prob=options.drop_prob,
            sparsity=1.0/options.accelerations[0],
            shape=options.resolution,
            conjugate_mask=options.conjugate_mask,
            line_constrained=options.line_constrained,
            bi_dir=options.bi_dir,
            preselect_num=options.preselect_num,
            preselect=options.preselect,
            random=options.random if 'random' in options.__dict__ else False,
            poisson=options.poisson if 'poisson' in options.__dict__ else False,
            spectrum=options.spectrum if 'spectrum' in options.__dict__ else False,
            equispaced=options.equispaced if 'equispaced' in options.__dict__ else False,).to(options.device)
    elif options.model == 'SequentialSampling':
        model = SequentialUnet(
            in_chans=options.input_chans,
            out_chans=options.output_chans,
            chans=options.num_chans,
            num_pool_layers=options.num_pools,
            drop_prob=options.drop_prob,
            sparsity=1.0/options.accelerations[0],
            shape=options.resolution,
            conjugate_mask=options.conjugate_mask,
            num_step=options.num_step,
            preselect=options.preselect,
            bi_direction=options.bi_dir,
            preselect_num=options.preselect_num,
            binary_sampler=options.binary_sampler,
            clamp=options.clamp,
            line_constrained=options.line_constrained,
            old_recon=options.old_recon,
            with_uncertainty=options.uncertainty_loss,
            detach_kspace=options.detach_kspace,
            fixed_input=options.fixed_input,
            pretrained_recon=options.pretrained_recon if 'pretrained_recon' in options.__dict__ else False).to(options.device)
    else:
        raise NotImplementedError()

    return model

def get_mask_stats(masks):
    masks = np.array(masks)

    return np.mean(masks, axis=0), np.std(masks, axis=0)


class NonRLTrainer:
    """Differentiabl Sampler Trainer for active MRI acquisition.

    Configuration for the trainer is provided by argument ``options``. Must contain the
    following fields:

    Args:
        options(``argparse.Namespace``): Options for the trainer.
        env(``activemri.envs.ActiveMRIEnv``): Env for which the policy is trained.
        device(``torch.device``): Device to use.
    """
    def __init__(
        self,
        options: argparse.Namespace,
        env: mri_envs.ActiveMRIEnv,
        device: torch.device
    ):
        self.options = options
        self.env = env
        self.device = device
        self.model = build_model(self.options)
        if options.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        self.optimizer = build_optimizer(self.options.lr, self.options.weight_decay, self.model.parameters())

        self.train_loader, self.dev_loader, self.display_loader, _ = self.env._setup_data_handlers()

        self.scheduler = build_lr_scheduler(self.options, self.optimizer)
        self.best_dev_loss = np.inf
        self.epoch = 0
        self.start_epoch = 0
        self.end_epoch = options.num_epochs

        # setup saving, writer, and logging
        options.exp_dir.mkdir(parents=True, exist_ok=True)

        with open(os.path.join(str(options.exp_dir), 'args.pkl'), "wb") as f:
            pickle.dump(options.__dict__, f)

        options.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=options.exp_dir / 'summary')

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(self.options)
        self.logger.info(self.model)

    def load_checkpoint_if_needed(self):
        if self.options.resume:
            self.load()

    def evaluate(self):
        self.model.eval()
        losses = []
        sparsity = []
        targets, preds = [], []
        metrics = Metrics(METRIC_FUNCS)
        start = time.perf_counter()

        with torch.no_grad():
            for iter, data in enumerate(self.dev_loader):
                # input: [batch_size, num_channels, height, width] denoted as NCHW in other places
                # label: label of the current image (0~9 for mnist/fashion-mnist) default: -1
                # target: a copy of the input image for computing reconstruction loss in [NCHW]
                kspace, _, input, label, *ignored = data

                # adapt data to loupe
                target = input.clone().detach()
                target = transforms.complex_abs(target).unsqueeze(1)

                input = input.to(self.options.device)
                target = target.to(self.options.device)
                kspace = kspace.to(self.options.device)
                # label = label.to(self.options.device)

                pred_dict = self.model(target, kspace)

                if (self.epoch == 0 or (self.epoch+1) % 1 == 0) and iter == 0:
                    data_for_vis_name = 'eval_epoch=' + str(self.epoch+1)
                    self.model.visualize_and_save(self.options, self.epoch, data_for_vis_name)

                output = pred_dict['output']
                # only use the last reconstructed image to compute loss
                if isinstance(output, list):
                    output = output[-1]

                target_dict = {'target': target, 'label': label, 'kspace':kspace}
                meta = {'entropy_weight': self.options.entropy_weight, 'recon_weight': self.options.recon_weight,
                'uncertainty_weight': 0, 'kspace_weight': self.options.kspace_weight}

                loss, log_dict = self.model.loss(pred_dict, target_dict, meta, self.options.loss_type)

                mask = pred_dict['mask']
                sparsity.append(torch.mean(mask).item())
                losses.append(loss.item())

                # target: 16*1*32*32
                # output: 16*1*32*32
                target = target.cpu().numpy()
                pred = output.cpu().numpy()

                for t, p in zip(target, pred):
                    metrics.push(t, p)

            print(metrics)
            self.writer.add_scalar('Dev_MSE', metrics.means()['MSE'], self.epoch)
            self.writer.add_scalar('Dev_NMSE', metrics.means()['NMSE'], self.epoch)
            self.writer.add_scalar('Dev_PSNR', metrics.means()['PSNR'], self.epoch)
            self.writer.add_scalar('Dev_SSIM', metrics.means()['SSIM'], self.epoch)

            self.writer.add_scalar('Dev_Loss', np.mean(losses), self.epoch)

        return np.mean(losses), np.mean(sparsity), time.perf_counter() - start

    def train_epoch(self):
        self.model.train()
        losses = []
        targets, preds = [], []
        metrics = Metrics(METRIC_FUNCS)
        avg_loss = 0.
        start_epoch = start_iter = time.perf_counter()
        global_step = self.epoch * len(self.train_loader)

        for iter, data in enumerate(self.train_loader):
            # self.scheduler.step()
            # input: [batch_size, num_channels, height, width] denoted as NCHW in other places
            # label: label of the current image (0~9 for mnist/fashion-mnist) default: -1
            # target: a copy of the input image for computing reconstruction loss in [NCHW]
            kspace, _, input, label, *ignored= data

            # adapt data to loupe
            target = input.clone().detach()
            target = transforms.complex_abs(target).unsqueeze(1)

            input = input.to(self.options.device)
            target = target.to(self.options.device)
            kspace = kspace.to(self.options.device)
            # label = label.to(self.options.device)

            """if self.options.noise_type == 'gaussian':
                kspace = transforms.add_gaussian_noise(self.options, kspace, mean=0., std=self.options.noise_level)
            """

            pred_dict = self.model(target, kspace)

            if (self.epoch == 0 or (self.epoch+1) % 1 == 0) and iter ==  0:
                data_for_vis_name = 'train_epoch={}_iter={}'.format(str(self.epoch+1), str(iter+1))
                self.model.visualize_and_save(self.options, self.epoch, data_for_vis_name)

            output = pred_dict['output']
            target_dict = {'target': target, 'label': label, 'kspace': kspace}
            meta = {'entropy_weight': self.options.entropy_weight, 'recon_weight': self.options.recon_weight,
                'kspace_weight': self.options.kspace_weight,
                'uncertainty_weight': self.options.uncertainty_weight if 'uncertainty_weight' in self.options.__dict__ else 0}

            loss, log_dict = self.model.loss(pred_dict, target_dict, meta, self.options.loss_type)

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)

            self.optimizer.step()

            self.writer.add_scalar('Train_Loss', loss.item(), global_step + iter)

            losses.append(loss.item())

            # target: 16*1*32*32
            # output: 16*1*32*32

            if isinstance(output, list):
                output = output[-1]

            target = target.cpu().detach().numpy()
            pred = output.cpu().detach().numpy()

            if iter % self.options.report_interval == 0:
                self.logger.info(
                    f'Epoch = [{1 + self.epoch:3d}/{self.options.num_epochs:3d}] '
                    f'Iter = [{iter:4d}/{len(self.train_loader):4d}] '
                    f'Time = {time.perf_counter() - start_iter:.4f}s',
                )
                for key, val in log_dict.items():
                    print('{} = {}'.format(key, val))

            start_iter = time.perf_counter()

            for t, p in zip(target, pred):
                metrics.push(t, p)

        print(metrics)
        self.writer.add_scalar('Train_MSE', metrics.means()['MSE'], self.epoch)
        self.writer.add_scalar('Train_NMSE', metrics.means()['NMSE'], self.epoch)
        self.writer.add_scalar('Train_PSNR', metrics.means()['PSNR'], self.epoch)
        self.writer.add_scalar('Train_SSIM', metrics.means()['SSIM'], self.epoch)

        return np.mean(np.array(losses)), time.perf_counter() - start_epoch

    def _train_loupe(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            train_loss, train_time = self.train_epoch()
            self.scheduler.step(epoch)
            dev_loss, mean_sparsity, dev_time = self.evaluate()

            is_new_best = dev_loss < self.best_dev_loss
            self.best_dev_loss = min(self.best_dev_loss, dev_loss)
            if self.options.save_model:
                self.save_model(is_new_best)
            self.logger.info(
                f'Epoch = [{1 + self.epoch:4d}/{self.options.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
                f'DevLoss = {dev_loss:.4g} MeanSparsity = {mean_sparsity:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
            )
        self.writer.close()

    def __call__(self):
        self.load_checkpoint_if_needed()
        return self._train_loupe()

    def save_model(self,is_new_best):
        exp_dir = self.options.exp_dir
        torch.save(
            {
                'epoch': self.epoch,
                'options': self.options,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_dev_loss': self.best_dev_loss,
                'exp_dir': exp_dir
            },
            f = exp_dir / 'model.pt'
        )
        if is_new_best:
            shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


    def load(self):
        self.model = build_model(self.options)
        if self.options.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        checkpoint1 = torch.load(self.options.checkpoint1)
        print("Load checkpoint {} with loss {}".format(checkpoint1['epoch'], checkpoint1['best_dev_loss']))

        self.model.load_state_dict(checkpoint1['model'])

        self.optimizer = build_optimizer(self.options.lr, self.options.weight_decay, self.model.parameters())
        self.optimizer.load_state_dict(checkpoint1['optimizer'])

        self.best_dev_loss = checkpoint1['best_dev_loss']
        self.start_epoch = checkpoint1['epoch'] + 1
        del checkpoint1


class NonRLTester:
    def __init__(
        self,
        env: loupe_envs.LOUPEDataEnv,
        exp_dir: str,
        options: argparse.Namespace,
        label_range: List[int]
    ):
        self.env = env
        # load options and model
        self.load(os.path.join(exp_dir , 'best_model.pt'))
        self.options.label_range = label_range
        self.options.exp_dir = exp_dir
        self.options.test_visual_frequency = options.test_visual_frequency
        self.options.visualization_dir = options.visualization_dir
        self.options.batch_size = 1
        _, _, self.dev_loader, _  = self.env._setup_data_handlers()

        # setup saving and logging
        self.options.exp_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(self.options)
        self.logger.info(self.model)

    def evaluate(self):
        self.model.eval()
        losses = []
        sparsity = []
        targets, preds = [], []
        metrics = Metrics(METRIC_FUNCS)
        masks = defaultdict(list)
        sample_image = dict()

        with torch.no_grad():
            for iter, data in enumerate(self.dev_loader):
                # input: [batch_size, num_channels, height, width] denoted as NCHW in other places
                # label: label of the current image (0~9 for mnist/fashion-mnist) default: -1
                # target: a copy of the input image for computing reconstruction loss in [NCHW]
                kspace, _, input, label, *ignored = data

                # adapt data to loupe
                target = input.clone().detach()
                target = transforms.complex_abs(target).unsqueeze(1)

                input = input.to(self.options.device)
                target = target.to(self.options.device)
                kspace = kspace.to(self.options.device)

                """if self.options.noise_type == 'gaussian':
                    kspace = transforms.add_gaussian_noise(self.options, kspace, mean=0., std=self.options.noise_level)
                """

                pred_dict = self.model(target, kspace)

                if iter % self.options.test_visual_frequency == 0:
                    data_for_vis_name = 'test_iter=' + str(iter+1)
                    print("visualize {}".format(data_for_vis_name))
                    self.model.visualize_and_save(self.options, iter, data_for_vis_name)

                output = pred_dict['output']
                if isinstance(output, list):
                    output = output[-1]

                loss = compute_ssim_torch(output, target) #  F.l1_loss(output, target, size_average=True)

                mask = pred_dict['mask']
                # masks[label.item()].append(transforms.fftshift(mask[0, 0], dim=(0, 1)).cpu().detach().numpy())

                sparsity.append(torch.mean(mask).item())
                losses.append(loss.item())

                targets.extend(target.cpu().numpy())
                preds.extend(output.cpu().numpy())

                # sample_image[label.item()] = target[0, 0].cpu().detach().numpy()

        """for key, val in masks.items():
            # get average mask and standard deviation
            mask_mean, mask_std = get_mask_stats(val)
            mask_mean[16] = 0
            mask_mean[:, 16] = 0
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=[10, 3])
            sp1 = ax[0].imshow(sample_image[key], cmap='viridis')
            sp2 = ax[1].imshow(mask_mean)
            sp3 = ax[2].imshow(mask_std)
            fig.colorbar(sp1, ax=ax[0])
            fig.colorbar(sp2, ax=ax[1])
            fig.colorbar(sp3, ax=ax[2])

            plt.savefig(str(self.options.visualization_dir)+'/label_{}_mask_stats.png'.format(key))
            plt.close()
        """

        print("Done Prediction")

        for t, p in zip(targets, preds):
            metrics.push(t, p)

        return losses, np.mean(sparsity), metrics

    def __call__(self):
        dev_loss, sparsity, metrics = self.evaluate()

        print("L1 Loss {} STD {} Sparsity {}".format(np.mean(dev_loss), np.std(dev_loss), sparsity))

        with open(os.path.join(self.options.exp_dir, 'statistics.txt'), 'w') as f:
            f.write('L1 Loss {} +- {}\n'.format(np.mean(dev_loss), np.std(dev_loss)))
            f.write(str(metrics))

        print(metrics)

        with open(os.path.join(self.options.exp_dir, 'loss.pkl'), 'wb') as f:
            pickle.dump(dev_loss, f)

    def load(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        print("Load checkpoint {} with loss {}".format(checkpoint['epoch'], checkpoint['best_dev_loss']))
        self.options = checkpoint['options']
        random.seed(self.options.seed)
        np.random.seed(self.options.seed)
        torch.manual_seed(self.options.seed)


        if 'bi_dir' not in self.options.__dict__:
            self.options.bi_dir = False

        if 'clamp' not in self.options.__dict__:
            self.options.clamp = 100

        if 'fixed_input' not in self.options.__dict__:
            self.options.fixed_input = False

        print('clamp value {}'.format(self.options.clamp))

        self.model = build_model(self.options)
        self.model.eval()
        self.model.load_state_dict(checkpoint['model'])

        torch.save({'model':self.model.reconstructor.state_dict()}, os.path.join(*checkpoint_file.split('/')[:-1], 'best_recon.pt'))

        # checkpoint = torch.load('checkpoints/Reconstructors/dicom_ckpt/PG_DICOM_Knee_Reconstructor_UNet/best_model.pt', map_location='cpu')
        # self.model.reconstructor.load_state_dict(checkpoint['model'])
