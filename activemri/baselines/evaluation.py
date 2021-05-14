# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import argparse
import numpy.matlib
import activemri.baselines as baselines
import activemri.envs as envs

import matplotlib.pyplot as plt


def evaluate(
    args: argparse.Namespace,
    env: envs.envs.ActiveMRIEnv,
    policy: baselines.Policy,
    num_episodes: int,
    seed: int,
    split: str,
    verbose: Optional[bool] = False,
) -> Tuple[Dict[str, np.ndarray], List[Tuple[Any, Any]]]:
    env.seed(seed)
    if split == "test":
        env.set_test()
    elif split == "val":
        env.set_val()
    else:
        raise ValueError(f"Invalid evaluation split: {split}.")

    score_keys = env.score_keys()
    all_scores = dict(
        (k, np.zeros((num_episodes * env.num_parallel_episodes, env.budget + 1)))
        for k in score_keys
    )
    all_img_ids = []
    trajectories_written = 0
    for episode in range(num_episodes):
        print('Now running episode: '+str(episode))
        step = 0
        obs, meta = env.reset()
#         plt.imshow(obs["reconstruction"][0,:,:,0])
#         plt.savefig('random.png')
#         input("Press Enter to continue...")
        if not obs:
            break  # no more images
        # in case the last batch is smaller
        actual_batch_size = len(obs["reconstruction"])
        if verbose:
            msg = ", ".join(
                [
                    f"({meta['fname'][i]}, {meta['slice_id'][i]})"
                    for i in range(actual_batch_size)
                ]
            )
            print(f"Read images: {msg}")
        for i in range(actual_batch_size):
            all_img_ids.append((meta["fname"][i], meta["slice_id"][i]))
        batch_idx = slice(
            trajectories_written, trajectories_written + actual_batch_size
        )
        for k in score_keys:
            all_scores[k][batch_idx, step] = meta["current_score"][k]
        trajectories_written += actual_batch_size
        all_done = False
        while not all_done:
            step += 1
            action = policy.get_action(obs)
            obs, reward, done, meta, gt = env.step(action)

            for k in score_keys:
                all_scores[k][batch_idx, step] = meta["current_score"][k]
            all_done = all(done)

            # visualize the first image
            if episode == 0:
                print('Visualize')
                num_row, num_col = obs["reconstruction"][0,:,:,0].cpu().detach().numpy().shape

                kspace_t = obs["reconstruction"][0,:,:,:].fft(2,normalized=False)
                kspace_t = (kspace_t ** 2).sum(dim=-1).sqrt().cpu().detach().numpy()
                action_t = np.zeros((num_row, num_col))
                action_play = action[0] if type(action)==list else action
                action_t[:,action_play] = 1
                reward_play = reward[0]
                # gt["gt_kspace"] is a List
                gt_kspace_t = np.sqrt((gt["gt_kspace"][0] ** 2).sum(-1))

                eval_vis = {"phi_t": obs["reconstruction"][0,:,:,0].cpu().detach().numpy(),
                            "kspace_t": kspace_t,
                            "mask_t": np.matlib.repmat(obs["mask"][0:1,:].cpu().detach().numpy(),num_row,1),
                            "action_t": action_t,
                            "action_t_val": action_play,
                            "reward_t_val": reward_play,
                            "gt_t": gt["gt"][0,:,:,0].numpy(),
                            "gt_kspace_t": gt_kspace_t
                           }
                if not 'output_dir' in vars(args).keys():
                    args.output_dir = args.checkpoints_dir
                visualize_and_save(eval_vis, step, episode, args.budget, args.output_dir)

                if step == 1:
                    import os
                    os.makedirs(args.output_dir+'mask/', exist_ok=True)

                left = obs['mask'][0][:len(obs['mask'][0])//2].numpy()
                right = np.flip(obs['mask'][0][len(obs['mask'][0])//2:].numpy())
                visualize_symmetry(left, right, step, episode, env.budget, args.output_dir)

            elif all_done:
                left = obs['mask'][0][:len(obs['mask'][0])//2].numpy()
                right = np.flip(obs['mask'][0][len(obs['mask'][0])//2:].numpy())
                visualize_symmetry(left, right, step, episode, env.budget, args.output_dir)

#           print('-------')
#           print(step)
#           print(obs['mask'])
#           print(list(map(int,left+right)))
#           print(meta["current_score"][k])
#           input("Press Enter to continue...")
        if episode % 10 == 0:
            np.save(os.path.join(args.output_dir, "scores_0-{}.npy".format(episode)), all_scores)
    
    for k in score_keys:
        all_scores[k] = all_scores[k][: len(all_img_ids), :]
    return all_scores, all_img_ids


def visualize_and_save(eval_vis: Dict[str, Any], step: int, episode: int, num_train_steps: int, checkpoints_dir: str):
    _, num_col = eval_vis["phi_t"].shape

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=[20, 8])

    cmap = "viridis" if num_col == 32 else "gray"

    sp1 = ax[0, 0].imshow(eval_vis["phi_t"], cmap=cmap)
    sp2 = ax[0, 1].imshow(np.fft.fftshift(np.log(eval_vis["kspace_t"])))
    sp3 = ax[0, 2].imshow(eval_vis["gt_t"], cmap=cmap)
    sp4 = ax[0, 3].imshow(np.fft.fftshift(np.log(eval_vis["gt_kspace_t"])))
    sp5 = ax[1, 0].imshow(np.fft.fftshift(eval_vis["mask_t"]-eval_vis["action_t"]))
    sp6 = ax[1, 1].imshow(np.fft.fftshift(eval_vis["action_t"]))
    sp7 = ax[1, 2].imshow(np.fft.fftshift(eval_vis["mask_t"]))
    ax[0, 0].title.set_text('Recon. at time t (phi_t)')
    ax[0, 1].title.set_text('Log k-space of phi_t')
    ax[0, 2].title.set_text('Ground truth of phi_t')
    ax[0, 3].title.set_text('GT log k-space of phi_t')
    ax[1, 0].title.set_text('Mask at time t-1')
    ax[1, 1].title.set_text('Action at time t')
    ax[1, 2].title.set_text('Mask at time t')
    fig.colorbar(sp1, ax=ax[0, 0])
    fig.colorbar(sp2, ax=ax[0, 1])
    fig.colorbar(sp3, ax=ax[0, 2])
    fig.colorbar(sp4, ax=ax[0, 3])
    fig.colorbar(sp5, ax=ax[1, 0])
    fig.colorbar(sp6, ax=ax[1, 1])
    fig.colorbar(sp7, ax=ax[1, 2])
    ax[1, 3].text(0, 0.8, r'action_t: ', fontsize=15)
    ax[1, 3].text(0.1, 0.65, str(eval_vis["action_t_val"])+' in [1,'+str(num_col)+']', fontsize=15)
    ax[1, 3].text(0.1, 0.5, str((eval_vis["action_t_val"]+num_col/2)%num_col)+' after fftshift', fontsize=15)
    ax[1, 3].text(0, 0.3, r'reward_t: ', fontsize=15)
    ax[1, 3].text(0.1, 0.15, str(eval_vis["reward_t_val"])+' (SSIM)', fontsize=15)
    ax[1, 3].axis('off')
    plt.suptitle('Step = [{}/{}]'.format(step, num_train_steps), fontsize=20)

    plt.savefig(checkpoints_dir+'episode='+str(episode)+'_step='+str(step)+'.png')
    plt.close()

def visualize_symmetry(left, right, step, episode, budget, save_dir):
    plt.plot(list(map(int,left+right)),'.')
    plt.xlabel('Columns')
    plt.ylabel('2: Both, 1: One, 0: Neither')
    plt.title('Step = [{}/{}]\nEpisode = {}'.format(step, budget, 1+episode))
    plt.savefig(save_dir+'mask/episode='+str(1+episode)+'_step='+str(step)+'.png')
    plt.close()
