import argparse
import os
import numpy as np
from scico import functional, linop, loss

# from scico.optimize.admm import ADMM
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim
from omegaconf import OmegaConf

from lpn.operators.tomo import get_tomo_A, get_operators
from lpn.utils import load_config, get_imgs
from inverse_mayoct_cs import (
    _save,
    load_model,
    lpn_denoise_patch,
    HelperFunctional,
    nonjit_scico,
    load_config,
    measure,
)
from lpn.utils.admm import ADMM
import json
import sys
var_seed = sys.argv[1]
print(var_seed)

# model_name = 'noAug_1'
model_name = 'mnist'
exp_data = 'mnist'
seeds = [2025, 0, 5678, 7890, 2024, 7618]
seeds = [1234, 5678, 7890, 2024, 2025, 6092, 7220, 7478, 3508, 7879]
seed = seeds[int(var_seed)]
if seed == 0:
    pass
else:
    if exp_data == 'mnist':
        model_name = model_name + f'_{seed}'


img_size = 28

if exp_data == 'mnist':
    
    def parse_args():
        args = OmegaConf.create()
        args.operator_config = OmegaConf.create(
            {
                "operator": "tomo",
                "space_range": 7,
                "img_size": img_size,
                "num_angles": 33,
                "det_shape": 22,
            }
        )
        args.model_config_path = 'exps/mnist/configs/mnist/model.json'
        with open(args.model_config_path, "r") as f:
            model_config = json.load(f)
        args.model_config = OmegaConf.create(model_config)
        args.prox_config = OmegaConf.create(
            {
                "prox": "lpn",
                "model_path": f"exps/mayoct/models_mnist/lpn/s=0.1_{model_name}/model.pt",
                "patch_size": img_size,
                "stride_size": 64,
            }
        )
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
        if args.dataset_config['dataset'] == 'ct_for_mnist':
            args.prox_config['patch_size'] = 28
            args.operator_config['img_size'] = 28
        args.out_dir = (
            f"exps/mayoct/results/inverse/mayoct/tomo/{args.dataset_config['split']}_num_angles={args.operator_config['num_angles']}_det_shape={args.operator_config['det_shape']}_{seed}_noise=2.0"
        )
        return args

def main_mayoct_tomo(args):
    idx_list = np.arange(128)
    A = get_A(args.operator_config)
    x_list = get_imgs(args.dataset_config)
    space_range = args.operator_config.space_range
    img_size = args.operator_config.img_size
    num_angles = args.operator_config.num_angles
    det_shape = args.operator_config.det_shape
    fwd_op_numpy, adjoint_op_numpy, fbp_op_numpy = get_operators(
        space_range, img_size, num_angles, det_shape
    )
    noise_std_dev = 2.0
    y_list = measure(x_list, A, noise_std_dev, seed=seed)
    fbp_list = [fbp_op_numpy(y) for y in y_list]

    # get LPN proximal step
    model = load_model(args.model_config, args.prox_config.model_path)
    model = model.eval()
    prox = lambda x, lam=1.0, **kwargs: lpn_denoise_patch(
        np.asarray(x), model, args.prox_config.patch_size, args.prox_config.stride_size
    )
    # print(prox.shape)
    img_size = 512
    if args.dataset_config['dataset'] == 'ct_for_mnist':
        img_size = 28
    elif args.dataset_config['dataset'] == 'cifar10':
        img_size = 32

    # parse ADMM arguments
    rho = 0.05  # ADMM penalty parameter
    maxiter = 15  # number of ADMM iterations
    subprob_tol = 1e-5  # tolerance for subproblem
    subprob_maxiter = 30  # maximum iteration for subproblem
    scale_f = 8.0 / 200 / 400  # scale for data fidelity term
    zero_one = False  # clip to [0, 1] after each iteration

    psnr_list, ssim_list = [], []
    for i, y in enumerate(tqdm(y_list)):
        x0 = fbp_list[i]  # initialize with FBP
        x_shape = x0.shape

        y_ = y
        A_ = A
        f = loss.SquaredL2Loss(y=y_, A=A_, scale=scale_f)

        g = HelperFunctional(prox)
        # print(g)
        if zero_one:
            g_list = [g, nonjit_scico.ZeroOneIndicator()]
        else:
            g_list = [g]
        # print('g list:',g_list)
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
        # print(x_list[i].shape, xhat.shape)
        psnr_val = skimage_psnr(x_list[i], xhat, data_range=1.0)
        ssim_val = skimage_ssim(x_list[i], xhat, data_range=1.0)
        print(
            f"PSNR: {psnr_val}",
            f"SSIM: {ssim_val}",
        )
        # save
        _save(xhat, os.path.join(args.out_dir, args.prox_config.prox, "xhat"), i)
        _save(y, os.path.join(args.out_dir, args.prox_config.prox, "y"), i)
        _save(x_list[i], os.path.join(args.out_dir, args.prox_config.prox, "x"), i)
        _save(fbp_list[i], os.path.join(args.out_dir, args.prox_config.prox, "fbp"), i)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    recon_log = (
        "average performance: "
        "PSNR {:.3f} +/- {:.3f}, SSIM {:.4f} +/- {:.4f}".format(
            np.mean(psnr_list),
            np.std(psnr_list),
            np.mean(ssim_list),
            np.std(ssim_list),
        )
    )
    print(recon_log)
    with open(
        os.path.join(args.out_dir, args.prox_config.prox, "xhat", "recon_log.txt"), "w"
    ) as f:
        f.write(recon_log)


def get_A(operator_config):
    """Get forward operator"""
    if operator_config.operator == "tomo":
        space_range = operator_config.space_range
        img_size = operator_config.img_size
        num_angles = operator_config.num_angles
        det_shape = operator_config.det_shape
        A = get_tomo_A(space_range, img_size, num_angles, det_shape)
    else:
        raise NotImplementedError
    return A


if __name__ == "__main__":
    args = parse_args()
    main_mayoct_tomo(args)
