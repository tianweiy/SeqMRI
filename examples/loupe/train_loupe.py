# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import random
import numpy as np
import torch
import uuid
import activemri.envs.loupe_envs as loupe_envs
from activemri.baselines.non_rl import NonRLTrainer, NonRLTester
import matplotlib
matplotlib.use('Agg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MRI Reconstruction Example')
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--num-step', type=int, default=2, help='Number of LSTM iterations')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=64, help='Number of U-Net channels')

    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--noise-type', type=str, default='none', help='Type of additive noise to measurements')
    parser.add_argument('--noise-level', type=float, default=0, help='Noise level')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, required=True,
                        help='Path where model and results should be saved')
    parser.add_argument('--checkpoint1', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--entropy_weight', type=float, default=0.0,
                        help='weight for the entropy/diversity loss')
    parser.add_argument('--recon_weight', type=float, default=1.0,
                        help='weight for the reconsturction loss')
    parser.add_argument('--sparsity_weight', type=float, default=0.0,
                        help='weight for the sparsity loss')
    parser.add_argument('--save-model', type=bool, default=False, help='save model every iteration or not')

    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--resolution', default=[128, 128], nargs='+', type=int, help='Resolution of images')

    parser.add_argument('--dataset-name', type=str, choices=['fashion-mnist', 'dicom-knee', 'real-knee'],
        required=True, help='name of the dataset')
    parser.add_argument('--sample-rate', type=float, default=1.,
                        help='Fraction of total volumes to include')

    # Mask parameters
    parser.add_argument('--accelerations', nargs='+', default=[4], type=float,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                            'provided, then one of those is chosen uniformly at random for '
                            'each volume.')
    parser.add_argument('--label_range', nargs='+', type=int, help='train using images of specific class')
    parser.add_argument('--model', type=str, help='name of the model to run', required=True)
    parser.add_argument('--input_chans', type=int, choices=[1, 2], required=True, help='number of input channels. One for real image, 2 for compelx image')
    parser.add_argument('--output_chans', type=int, default=1, help='number of output channels. One for real image')
    parser.add_argument('--line-constrained', type=int, default=0)
    parser.add_argument('--unet', action='store_true')
    parser.add_argument('--conjugate_mask', action='store_true', help='force loupe model to use conjugate symmetry.')
    parser.add_argument('--bi-dir', type=int, default=0)
    parser.add_argument('--loss_type', type=str, choices=['l1', 'ssim', 'psnr'], default='l1')
    parser.add_argument('--test_visual_frequency', type=int, default=1000)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--preselect', type=int, default=0)
    parser.add_argument('--preselect_num', type=int, default=2)
    parser.add_argument('--random_rotate', type=int, default=0)
    parser.add_argument('--random_baseline', type=int, default=0)
    parser.add_argument('--poisson', type=int, default=0)
    parser.add_argument('--spectrum', type=int, default=0)
    parser.add_argument("--equispaced", type=int, default=0)


    args = parser.parse_args()
    args.equispaced = args.equispaced > 0 
    args.spectrum = args.spectrum > 0 
    args.poisson = args.poisson > 0 
    args.random = args.random_baseline > 0 
    args.random_rotate = args.random_rotate > 0 
    args.kspace_weight = 0 
    args.line_constrained = args.line_constrained > 0

    if args.checkpoint1 is not None:
        args.resume = True
    else:
        args.resume = False

    noise_str = ''
    if args.noise_type is 'none':
        noise_str = '_no_noise_'
    else:
        noise_str = '_' + args.noise_type + str(args.noise_level) + '_'

    if args.preselect > 0:
        args.preselect = True
    else:
        args.preselect = False

    if args.bi_dir > 0 :
        args.bi_dir = True
    else:
        args.bi_dir = False

    if str(args.exp_dir) is 'auto':
        args.exp_dir =('checkpoints/'+args.dataset_name + '_' + str(float(args.accelerations[0])) +
             'x_' + args.model + '_bi_dir_{}'.format(args.bi_dir)+ '_preselect_{}'.format(args.preselect) +
             noise_str + 'lr=' + str(args.lr) + '_bs=' + str(args.batch_size) + '_loss_type='+args.loss_type +
              '_epochs=' + str(args.num_epochs))

        args.exp_dir = pathlib.Path(args.exp_dir+'_uuid_'+uuid.uuid4().hex.upper()[0:6])

    print('save logs to {}'.format(args.exp_dir))


    args.visualization_dir = args.exp_dir / 'visualizations'

    if args.test:
        args.batch_size = 1

    if args.dataset_name == 'real-knee':
        args.data_path = 'datasets/knee'
        # args.resolution = [128, 128]
        env = loupe_envs.LOUPERealKspaceEnv(args)
    else:
        raise NotImplementedError

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.test:
        policy = NonRLTester(env, args.exp_dir, args, None)
    else:
        policy = NonRLTrainer(args, env, torch.device(args.device))
    policy()
