# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import json
import pathlib

from typing import Dict, Tuple

import numpy as np
import skimage.metrics
import torch


def get_user_dir() -> pathlib.Path:
    # return pathlib.Path.home() / ".activemri"
    return pathlib.Path.cwd() / ".activemri"


def maybe_create_datacache_dir() -> pathlib.Path:
    datacache_dir = get_user_dir() / "__datacache__"
    if not datacache_dir.is_dir():
        datacache_dir.mkdir()
    return datacache_dir


def get_defaults_json() -> Tuple[Dict[str, str], str]:
    defaults_path = get_user_dir() / "defaults.json"
    if not pathlib.Path.exists(defaults_path):
        parent = defaults_path.parents[0]
        parent.mkdir(exist_ok=True)
        content = {"data_location": "", "saved_models_dir": ""}
        with defaults_path.open("w", encoding="utf-8") as f:
            json.dump(content, f)
    else:
        with defaults_path.open("r", encoding="utf-8") as f:
            content = json.load(f)
    return content, str(defaults_path)


def import_object_from_str(classname: str):
    the_module, the_object = classname.rsplit(".", 1)
    the_object = classname.split(".")[-1]
    module = importlib.import_module(the_module)
    return getattr(module, the_object)


def compute_ssim(xs: torch.Tensor, ys: torch.Tensor) -> np.ndarray:
    ssims = []
    for i in range(xs.shape[0]):
        ssim = skimage.metrics.structural_similarity(
            xs[i].cpu().numpy(),
            ys[i].cpu().numpy(),
            data_range=ys[i].cpu().numpy().max(),
        )
        ssims.append(ssim)
    return np.array(ssims, dtype=np.float32)


def compute_psnr(xs: torch.Tensor, ys: torch.Tensor) -> np.ndarray:
    psnrs = []
    for i in range(xs.shape[0]):
        psnr = skimage.metrics.peak_signal_noise_ratio(
            xs[i].cpu().numpy(),
            ys[i].cpu().numpy(),
            data_range=ys[i].cpu().numpy().max(),
        )
        psnrs.append(psnr)
    return np.array(psnrs, dtype=np.float32)


def compute_mse(xs: torch.Tensor, ys: torch.Tensor) -> np.ndarray:
    dims = tuple(range(1, len(xs.shape)))
    return np.mean((ys.cpu().numpy() - xs.cpu().numpy()) ** 2, axis=dims)

def compute_nmse(xs: torch.Tensor, ys: torch.Tensor) -> np.ndarray:
    ys_numpy = ys.cpu().numpy()
    nmses = []
    for i in range(xs.shape[0]):
        x = xs[i].cpu().numpy()
        y = ys_numpy[i]
        nmse = np.linalg.norm(y - x) ** 2 / np.linalg.norm(y) ** 2
        nmses.append(nmse)
    return np.array(nmses, dtype=np.float32)

def compute_mse_torch(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, len(xs.shape)))
    return torch.mean((ys - xs) ** 2, dim=dims)

def compute_nmse_torch(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, len(xs.shape)))
    nmse = torch.linalg.norm(ys-xs, dim=dims)**2 / torch.linalg.norm(ys, dim=dims)**2

    return nmse 

from torch import nn 
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        dims = tuple(range(1, len(X.shape)))
        
        return S.mean(dim=dims)

SSIM = SSIMLoss()

def compute_ssim_torch(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    global SSIM
    SSIM = SSIM.to(xs.device)
    data_range = [y.max() for y in ys]
    data_range = torch.stack(data_range, dim=0)
    
    return SSIM(xs, ys, data_range=data_range.detach())

def compute_psnr_torch(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    mse = compute_mse_torch(xs, ys)
    data_range = [y.max() for y in ys]
    data_range = torch.stack(data_range, dim=0)
    
    return 10 * torch.log10((data_range ** 2) / mse)

def compute_gaussian_nll_loss(reconstruction, target, logvar):
    l2 = F.mse_loss(reconstruction, target, reduce=False)
    # Clip logvar to make variance in [0.0001, 0.1], for numerical stability

    logvar = logvar.clamp(min=-9.2, max=1.609)
    one_over_var = torch.exp(-logvar)


    assert len(l2) == len(logvar)
    return 0.5 * (one_over_var * l2 + logvar)
