# coding=utf-8
# Modifications Copyright 2021 The PlenOctree Authors.
# Original Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Utility functions."""
import collections
import os
from os import path
from absl import flags
import math

import torch
import torch.nn.functional as F

import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm
from octree.nerf import datasets

INTERNAL = False

Rays = collections.namedtuple("Rays", ("origins", "directions", "viewdirs"))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*map(fn, tup))


def define_flags():
    """Define flags for both training and evaluation modes."""
    flags.DEFINE_string("train_dir", None, "where to store ckpts and logs")
    flags.DEFINE_string("data_dir", None, "input data directory.")
    flags.DEFINE_string("config", None, "using config files to set hyperparameters.")

    # Dataset Flags
    # TODO(pratuls): rename to dataset_loader and consider cleaning up
    flags.DEFINE_enum(
        "dataset",
        "blender",
        list(k for k in datasets.dataset_dict.keys()),
        "The type of dataset feed to nerf.",
    )
    flags.DEFINE_bool(
        "image_batching", False, "sample rays in a batch from different images."
    )
    flags.DEFINE_bool(
        "white_bkgd",
        True,
        "using white color as default background." "(used in the blender dataset only)",
    )
    flags.DEFINE_integer(
        "batch_size", 1024, "the number of rays in a mini-batch (for training)."
    )
    flags.DEFINE_integer(
        "factor", 4, "the downsample factor of images, 0 for no downsample."
    )
    flags.DEFINE_bool("spherify", False, "set for spherical 360 scenes.")
    flags.DEFINE_bool(
        "render_path",
        False,
        "render generated path if set true." "(used in the llff dataset only)",
    )
    flags.DEFINE_integer(
        "llffhold",
        8,
        "will take every 1/N images as LLFF test set."
        "(used in the llff dataset only)",
    )

    # Model Flags
    flags.DEFINE_string("model", "nerf", "name of model to use.")
    flags.DEFINE_float("near", 2.0, "near clip of volumetric rendering.")
    flags.DEFINE_float("far", 6.0, "far clip of volumentric rendering.")
    flags.DEFINE_integer("net_depth", 8, "depth of the first part of MLP.")
    flags.DEFINE_integer("net_width", 256, "width of the first part of MLP.")
    flags.DEFINE_integer("net_depth_condition", 1, "depth of the second part of MLP.")
    flags.DEFINE_integer("net_width_condition", 128, "width of the second part of MLP.")
    flags.DEFINE_float("weight_decay_mult", 0, "The multiplier on weight decay")
    flags.DEFINE_integer(
        "skip_layer",
        4,
        "add a skip connection to the output vector of every" "skip_layer layers.",
    )
    flags.DEFINE_integer("num_rgb_channels", 3, "the number of RGB channels.")
    flags.DEFINE_integer("num_sigma_channels", 1, "the number of density channels.")
    flags.DEFINE_bool("randomized", True, "use randomized stratified sampling.")
    flags.DEFINE_integer(
        "min_deg_point", 0, "Minimum degree of positional encoding for points."
    )
    flags.DEFINE_integer(
        "max_deg_point", 10, "Maximum degree of positional encoding for points."
    )
    flags.DEFINE_integer("deg_view", 4, "Degree of positional encoding for viewdirs.")
    flags.DEFINE_integer(
        "num_coarse_samples",
        64,
        "the number of samples on each ray for the coarse model.",
    )
    flags.DEFINE_integer(
        "num_fine_samples", 128, "the number of samples on each ray for the fine model."
    )
    flags.DEFINE_bool("use_viewdirs", True, "use view directions as a condition.")
    flags.DEFINE_integer("sh_deg", -1, "set to use spherical harmonics output of given order.")
    flags.DEFINE_integer("sg_dim", -1, "set to use spherical gaussians (SG).")
    flags.DEFINE_float(
        "noise_std",
        None,
        "std dev of noise added to regularize sigma output."
        "(used in the llff dataset only)",
    )
    flags.DEFINE_bool(
        "lindisp", False, "sampling linearly in disparity rather than depth."
    )
    flags.DEFINE_string(
        "net_activation", "ReLU", "activation function used within the MLP."
    )
    flags.DEFINE_string(
        "rgb_activation", "Sigmoid", "activation function used to produce RGB."
    )
    flags.DEFINE_string(
        "sigma_activation", "ReLU", "activation function used to produce density."
    )
    flags.DEFINE_bool(
        "legacy_posenc_order",
        False,
        "If True, revert the positional encoding feature order to an older version of this codebase.",
    )

    # Train Flags
    flags.DEFINE_float("lr_init", 5e-4, "The initial learning rate.")
    flags.DEFINE_float("lr_final", 5e-6, "The final learning rate.")
    flags.DEFINE_integer(
        "lr_delay_steps",
        0,
        "The number of steps at the beginning of "
        "training to reduce the learning rate by lr_delay_mult",
    )
    flags.DEFINE_float(
        "lr_delay_mult",
        1.0,
        "A multiplier on the learning rate when the step " "is < lr_delay_steps",
    )
    flags.DEFINE_integer("max_steps", 1000000, "the number of optimization steps.")
    flags.DEFINE_integer(
        "save_every", 5000, "the number of steps to save a checkpoint."
    )
    flags.DEFINE_integer(
        "print_every", 500, "the number of steps between reports to tensorboard."
    )
    flags.DEFINE_integer(
        "render_every",
        10000,
        "the number of steps to render a test image,"
        "better to be x00 for accurate step time record.",
    )
    flags.DEFINE_integer(
        "gc_every", 10000, "the number of steps to run python garbage collection."
    )
    flags.DEFINE_float(
        "sparsity_weight",
        1e-3,
        "Sparsity loss weight",
    )
    flags.DEFINE_float(
        "sparsity_length",
        0.05,
        "Sparsity loss 'length' for alpha calculation",
    )
    flags.DEFINE_float(
        "sparsity_radius",
        1.5,
        "Sparsity loss point sampling box 1/2 side length",
    )
    flags.DEFINE_integer(
        "sparsity_npoints",
        10000,
        "Number of samples for sparsity loss",
    )

    # Eval Flags
    flags.DEFINE_bool(
        "eval_once",
        True,
        "evaluate the model only once if true, otherwise keeping evaluating new"
        "checkpoints if there's any.",
    )
    flags.DEFINE_bool("save_output", True, "save predicted images to disk if True.")
    flags.DEFINE_integer(
        "chunk",
        81920,
        "the size of chunks for evaluation inferences, set to the value that"
        "fits your GPU/TPU memory.",
    )

    # Octree flags
    flags.DEFINE_float(
        'renderer_step_size',
        1e-4,
        'step size epsilon in volume render.'
        '1e-3 = fast setting, 1e-4 = usual setting, 1e-5 = high setting, lower is better')
    flags.DEFINE_bool(
        'no_early_stop',
        False,
        'If set, does not use early stopping; slows down rendering slightly')


def update_flags(args):
    """Update the flags in `args` with the contents of the config YAML file."""
    if args.config is None:
        return
    pth = path.join(args.config + ".yaml")
    with open_file(pth, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    # Only allow args to be updated if they already exist.
    invalid_args = list(set(configs.keys()) - set(dir(args)))
    if invalid_args:
        raise ValueError(f"Invalid args {invalid_args} in {pth}.")
    args.__dict__.update(configs)


def check_flags(args, require_data=True, require_batch_size_div=False):
    if args.train_dir is None:
        raise ValueError("train_dir must be set. None set now.")
    if require_data and args.data_dir is None:
        raise ValueError("data_dir must be set. None set now.")
    if require_batch_size_div and args.batch_size % torch.cuda.device_count() != 0:
        raise ValueError("Batch size must be divisible by the number of devices.")


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def open_file(pth, mode="r"):
    if not INTERNAL:
        pth = path.expanduser(pth)
        return open(pth, mode=mode)


def file_exists(pth):
    if not INTERNAL:
        return path.exists(pth)


def listdir(pth):
    if not INTERNAL:
        return os.listdir(pth)


def isdir(pth):
    if not INTERNAL:
        return path.isdir(pth)


def makedirs(pth):
    if not INTERNAL:
        os.makedirs(pth, exist_ok=True)


@torch.no_grad()
def eval_points(fn, points, chunk=720720, to_cpu=True):
    """Evaluate at given points (in test mode).
    Currently not supporting viewdirs.

    Args:
      fn: function
      points: torch.tensor [..., 3]
      chunk: int, the size of chunks to render sequentially.

    Returns:
      rgb: torch.tensor or np.array.
      sigmas: torch.tensor or np.array.
    """
    num_points = points.shape[0]
    rgbs, sigmas = [], []

    for i in tqdm(range(0, num_points, chunk)):
        chunk_points = points[i : i + chunk]
        rgb, sigma = fn(chunk_points, None)
        if to_cpu:
            rgb = rgb.detach().cpu().numpy()
            sigma = sigma.detach().cpu().numpy()
        rgbs.append(rgb)
        sigmas.append(sigma)
    if to_cpu:
        rgbs = np.concatenate(rgbs, axis=0)
        sigmas = np.concatenate(sigmas, axis=0)
    else:
        rgbs = torch.cat(rgbs, dim=0)
        sigmas = torch.cat(sigmas, dim=0)
    return rgbs, sigmas


def compute_psnr(mse):
    """Compute psnr value given mse (we assume the maximum pixel value is 1).

    Args:
      mse: float, mean square error of pixels.

    Returns:
      psnr: float, the psnr value.
    """
    return -10.0 * torch.log(mse) / np.log(10.0)


def compute_ssim(
    img0,
    img1,
    max_val,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,
):
    """Computes SSIM from two images.

    This function was modeled after tf.image.ssim, and should produce comparable
    output.

    Args:
      img0: torch.tensor. An image of size [..., width, height, num_channels].
      img1: torch.tensor. An image of size [..., width, height, num_channels].
      max_val: float > 0. The maximum magnitude that `img0` or `img1` can have.
      filter_size: int >= 1. Window size.
      filter_sigma: float > 0. The bandwidth of the Gaussian used for filtering.
      k1: float > 0. One of the SSIM dampening parameters.
      k2: float > 0. One of the SSIM dampening parameters.
      return_map: Bool. If True, will cause the per-pixel SSIM "map" to returned

    Returns:
      Each image's mean SSIM, or a tensor of individual values if `return_map`.
    """
    device = img0.device
    ori_shape = img0.size()
    width, height, num_channels = ori_shape[-3:]
    img0 = img0.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
    img1 = img1.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
    batch_size = img0.shape[0]

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((torch.arange(filter_size, device=device) - hw + shift) / filter_sigma) ** 2
    filt = torch.exp(-0.5 * f_i)
    filt /= torch.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    # z is a tensor of size [B, H, W, C]
    filt_fn1 = lambda z: F.conv2d(
        z, filt.view(1, 1, -1, 1).repeat(num_channels, 1, 1, 1),
        padding=[hw, 0], groups=num_channels)
    filt_fn2 = lambda z: F.conv2d(
        z, filt.view(1, 1, 1, -1).repeat(num_channels, 1, 1, 1),
        padding=[0, hw], groups=num_channels)

    # Vmap the blurs to the tensor size, and then compose them.
    filt_fn = lambda z: filt_fn1(filt_fn2(z))
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0 ** 2) - mu00
    sigma11 = filt_fn(img1 ** 2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = torch.clamp(sigma00, min=0.0)
    sigma11 = torch.clamp(sigma11, min=0.0)
    sigma01 = torch.sign(sigma01) * torch.min(
        torch.sqrt(sigma00 * sigma11), torch.abs(sigma01)
    )

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = torch.mean(ssim_map.reshape([-1, num_channels*width*height]), dim=-1)
    return ssim_map if return_map else ssim


def generate_rays(w, h, focal, camtoworlds, equirect=False):
    """
    Generate perspective camera rays. Principal point is at center.
    Args:
        w: int image width
        h: int image heigth
        focal: float real focal length
        camtoworlds: jnp.ndarray [B, 4, 4] c2w homogeneous poses
        equirect: if true, generates spherical rays instead of pinhole
    Returns:
        rays: Rays a namedtuple(origins [B, 3], directions [B, 3], viewdirs [B, 3])
    """
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(w, dtype=np.float32),  # X-Axis (columns)
        np.arange(h, dtype=np.float32),  # Y-Axis (rows)
        indexing="xy",
    )

    if equirect:
        uv = np.stack([x * (2.0 / w) - 1.0, y * (2.0 / h) - 1.0], axis=-1)
        camera_dirs = equirect2xyz(uv)
    else:
        camera_dirs = np.stack(
            [
                (x - w * 0.5) / focal,
                -(y - h * 0.5) / focal,
                -np.ones_like(x),
            ],
            axis=-1,
        )

    #  camera_dirs = camera_dirs / np.linalg.norm(camera_dirs, axis=-1, keepdims=True)

    c2w = camtoworlds[:, None, None, :3, :3]
    camera_dirs = camera_dirs[None, Ellipsis, None]
    directions = np.matmul(c2w, camera_dirs)[Ellipsis, 0]
    origins = np.broadcast_to(
        camtoworlds[:, None, None, :3, -1], directions.shape
    )
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    viewdirs = directions / norms
    rays = Rays(
        origins=origins, directions=directions, viewdirs=viewdirs
    )
    return rays


def eval_octree(t, dataset, args, want_lpips=True, want_frames=False):
    import svox
    w, h, focal = dataset.w, dataset.h, dataset.focal
    if 'llff' in args.config and (not args.spherify):
        ndc_config = svox.NDCConfig(width=w, height=h, focal=focal)
    else:
        ndc_config = None

    r = svox.VolumeRenderer(
        t, step_size=args.renderer_step_size, ndc=ndc_config)

    print('Evaluating octree')
    device = t.data.device
    if want_lpips:
        import lpips
        lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0
    out_frames = []
    for idx in tqdm(range(dataset.size)):
        c2w = torch.from_numpy(dataset.camtoworlds[idx]).float().to(device)
        im_gt_ten = torch.from_numpy(dataset.images[idx]).float().to(device)

        im = r.render_persp(
            c2w, width=w, height=h, fx=focal, fast=not args.no_early_stop)
        im.clamp_(0.0, 1.0)

        mse = ((im - im_gt_ten) ** 2).mean()
        psnr = compute_psnr(mse).mean()
        ssim = compute_ssim(im, im_gt_ten, max_val=1.0).mean()

        avg_psnr += psnr.item()
        avg_ssim += ssim.item()
        if want_lpips:
            lpips_i = lpips_vgg(im_gt_ten.permute([2, 0, 1]).contiguous(),
                    im.permute([2, 0, 1]).contiguous(), normalize=True)
            avg_lpips += lpips_i.item()

        if want_frames:
            im = im.cpu()
            # vis = np.hstack((im_gt_ten.cpu().numpy(), im.cpu().numpy()))
            vis = im.cpu().numpy()  # for lpips calculation
            vis = (vis * 255).astype(np.uint8)
            out_frames.append(vis)

    avg_psnr /= dataset.size
    avg_ssim /= dataset.size
    avg_lpips /= dataset.size
    return avg_psnr, avg_ssim, avg_lpips, out_frames


def memlog(device='cuda'):
    # Memory debugging
    print(torch.cuda.memory_summary(device))
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                    hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if str(obj.device) != 'cpu':
                    print(obj.device, '{: 10}'.format(obj.numel()),
                            obj.dtype,
                            obj.size(), type(obj))
        except:
            pass
