#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run MNIST-style CT inverse problem with LPN proximal prior across multiple angle counts.

Key fixes:
- Do NOT define parse_args inside a for-loop. Instead, pass num_angles explicitly.
- Iterate over a user-provided list of angles and call main for each.
- All comments are in English, per user request.
"""

import argparse
import os
import sys
import json
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim

from scico import functional, linop, loss  # noqa: F401 (functional currently unused but kept)
from lpn.operators.tomo import get_tomo_A, get_operators
from lpn.utils import get_imgs
from inverse_mayoct_cs import (
    _save,
    load_model,
    lpn_denoise_patch,
    HelperFunctional,
    nonjit_scico,
    measure,
)
from lpn.utils.admm import ADMM


# ------------------------
# Config and utilities
# ------------------------
def parse_cli():
    """Parse CLI; accepts seed index and angles list."""
    p = argparse.ArgumentParser(description="LPN-based ADMM CT inversion")
    p.add_argument("var_seed", type=int, help="Index into predefined seeds list")
    p.add_argument(
        "--angles",
        type=str,
        default="11,22,33",
        help="Comma-separated list of angle counts, e.g., '11,22,33'",
    )
    p.add_argument(
        "--exp_data",
        type=str,
        default="mnist",
        choices=["mnist"],
        help="Dataset config switch (currently only 'mnist' used)",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="noAug_1",
        help="Base model name (will be suffixed with seed for MNIST)",
    )
    return p.parse_args()


def get_seed(var_seed_idx: int) -> int:
    """Pick a seed from a fixed list using the provided index."""
    seeds = [1234, 5678, 7890, 2024, 2025, 6092, 7220, 7478, 3508, 7879]
    if var_seed_idx < 0 or var_seed_idx >= len(seeds):
        raise IndexError(f"var_seed index {var_seed_idx} out of range [0, {len(seeds)-1}]")
    return seeds[var_seed_idx]


def build_model_name(base_name: str, exp_data: str, seed: int) -> str:
    """Compose model name (for MNIST we suffix with seed)."""
    if exp_data == "mnist":
        return f"{base_name}_{seed}"
    return base_name


def make_args_for_run(exp_data: str, img_size: int, num_angles: int, det_shape: int,
                      model_name: str, seed: int):
    """Create an OmegaConf args bundle for a specific num_angles run."""
    args = OmegaConf.create()
    args.operator_config = OmegaConf.create(
        {
            "operator": "tomo",
            "space_range": 7,
            "img_size": img_size,
            "num_angles": num_angles,
            "det_shape": det_shape,
        }
    )

    # Load model config JSON into OmegaConf
    args.model_config_path = 'exps/mnist/configs/mnist/model.json'
    with open(args.model_config_path, "r") as f:
        model_config = json.load(f)
    args.model_config = OmegaConf.create(model_config)

    # Prox configuration for LPN
    args.prox_config = OmegaConf.create(
        {
            "prox": "lpn",
            "model_path": f"exps/mayoct/models_mnist/lpn/s=0.1_{model_name}/model.pt",
            "patch_size": img_size,   # will be overridden for ct_for_mnist
            "stride_size": 64,
        }
    )

    # Dataset configuration
    args.dataset_config = OmegaConf.create(
        {
            "dataset": "ct_for_mnist",
            "root": "data/mayoct/",
            "split": "idood_valid",
            "squeeze": True,
            "start_idx": 0,
            "num_imgs": 100,
        }
    )

    # Adjust sizes for the ct_for_mnist dataset
    if args.dataset_config['dataset'] == 'ct_for_mnist':
        args.prox_config['patch_size'] = 28
        args.operator_config['img_size'] = 28

    # Output directory includes num_angles and seed; also fixes det_shape in path
    args.out_dir = (
        f"exps/mayoct/results/inverse/mayoct/tomo/"
        f"{args.dataset_config['split']}_num_angles={args.operator_config['num_angles']}"
        f"_det_shape={args.operator_config['det_shape']}_{seed}_noise=2.0"
    )
    return args


def get_A(operator_config):
    """Get forward operator."""
    if operator_config.operator == "tomo":
        return get_tomo_A(
            operator_config.space_range,
            operator_config.img_size,
            operator_config.num_angles,
            operator_config.det_shape,
        )
    raise NotImplementedError(f"Unknown operator {operator_config.operator}")


# ------------------------
# Main inverse routine
# ------------------------
def main_mayoct_tomo(args, seed: int):
    """Run ADMM with LPN proximal on a single (num_angles) configuration."""
    # Fetch dataset and operators
    A = get_A(args.operator_config)
    x_list = get_imgs(args.dataset_config)

    space_range = args.operator_config.space_range
    img_size = args.operator_config.img_size
    num_angles = args.operator_config.num_angles
    det_shape = args.operator_config.det_shape

    fwd_op_numpy, adjoint_op_numpy, fbp_op_numpy = get_operators(
        space_range, img_size, num_angles, det_shape
    )

    # Simulate measurements
    noise_std_dev = 2.0
    y_list = measure(x_list, A, noise_std_dev, seed=seed)
    fbp_list = [fbp_op_numpy(y) for y in y_list]

    # Build proximal operator via LPN denoiser
    model = load_model(args.model_config, args.prox_config.model_path)
    model = model.eval()
    prox = lambda x, lam=1.0, **kwargs: lpn_denoise_patch(
        np.asarray(x), model, args.prox_config.patch_size, args.prox_config.stride_size
    )

    # ADMM hyperparameters
    rho = 0.05                   # ADMM penalty parameter
    maxiter = 15                 # ADMM iterations
    subprob_tol = 1e-5           # Subproblem tolerance (CG)
    subprob_maxiter = 30         # Subproblem max iters (CG)
    scale_f = 8.0 / 200 / 400    # Data fidelity scaling
    zero_one = False             # Clip to [0, 1] at each iteration if True

    psnr_list, ssim_list = [], []

    for i, y in enumerate(tqdm(y_list, desc=f"angles={num_angles}")):
        # Initialize from FBP
        x0 = fbp_list[i]
        x_shape = x0.shape

        # Data fidelity term
        f = loss.SquaredL2Loss(y=y, A=A, scale=scale_f)

        # Proximal regularization term(s)
        g = HelperFunctional(prox)
        g_list = [g] if not zero_one else [g, nonjit_scico.ZeroOneIndicator()]

        # Identity linops and rho assignments for ADMM splitting
        C_list = [linop.Identity(x_shape) for _ in range(len(g_list))]
        rho_list = [rho] * len(g_list)

        solver = ADMM(
            f=f,
            g_list=g_list,
            C_list=C_list,
            rho_list=rho_list,
            x0=x0,
            maxiter=maxiter,
            subproblem_solver=nonjit_scico.LinearSubproblemSolver(
                cg_kwargs={"tol": subprob_tol, "maxiter": subprob_maxiter}
            ),
            itstat_options={"display": True},
            order="132",
        )
        xhat = solver.solve()
        xhat = np.clip(np.asarray(xhat), 0, 1)

        # Metrics
        psnr_val = skimage_psnr(x_list[i], xhat, data_range=1.0)
        ssim_val = skimage_ssim(x_list[i], xhat, data_range=1.0)
        print(f"[angles={num_angles}] idx={i:03d} | PSNR: {psnr_val:.3f} | SSIM: {ssim_val:.4f}")

        # Save outputs
        _save(xhat,      os.path.join(args.out_dir, args.prox_config.prox, "xhat"), i)
        _save(y,         os.path.join(args.out_dir, args.prox_config.prox, "y"), i)
        _save(x_list[i], os.path.join(args.out_dir, args.prox_config.prox, "x"), i)
        _save(fbp_list[i], os.path.join(args.out_dir, args.prox_config.prox, "fbp"), i)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    recon_log = (
        f"[angles={num_angles}] average performance: "
        "PSNR {:.3f} +/- {:.3f}, SSIM {:.4f} +/- {:.4f}".format(
            np.mean(psnr_list), np.std(psnr_list), np.mean(ssim_list), np.std(ssim_list)
        )
    )
    print(recon_log)
    os.makedirs(os.path.join(args.out_dir, args.prox_config.prox, "xhat"), exist_ok=True)
    with open(
        os.path.join(args.out_dir, args.prox_config.prox, "xhat", "recon_log.txt"), "w"
    ) as f:
        f.write(recon_log + "\n")


# ------------------------
# Entry point
# ------------------------
if __name__ == "__main__":
    cli = parse_cli()
    seed = get_seed(cli.var_seed)
    print(f"[Seed index={cli.var_seed}] -> seed={seed}")

    exp_data = cli.exp_data
    base_model_name = cli.model_name

    # Compose model name (MNIST appends seed)
    # model_name = build_model_name(base_model_name, exp_data, seed)
    model_name = build_model_name(base_model_name, exp_data, 7890)

    # Parse angles: allow comma-separated list
    angles_list = [11, 22, 33]
    print(angles_list)
    
    # Fixed dataset-dependent image size and detector shape
    # (kept consistent with your original code)
    base_img_size = 28 if exp_data == "mnist" else 512
    det_shape = 22

    # Run for each angles value
    for num_angles in angles_list:
        args = make_args_for_run(
            exp_data=exp_data,
            img_size=base_img_size,
            num_angles=num_angles,
            det_shape=det_shape,
            model_name=model_name,
            seed=seed,
        )
        main_mayoct_tomo(args, seed)
