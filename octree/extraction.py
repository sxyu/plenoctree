#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
"""Extract a plenoctree from a trained NeRF-SH model.

Usage:

export DATA_ROOT=./data/NeRF/nerf_synthetic/
export CKPT_ROOT=./data/PlenOctree/checkpoints/syn_sh16
export SCENE=chair
export CONFIG_FILE=nerf_sh/config/blender

python -m octree.extraction \
    --train_dir $CKPT_ROOT/$SCENE/ --is_jaxnerf_ckpt \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/octrees/tree.npz
"""
import os
# Get rid of ugly TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import torch.nn.functional as F
import numpy as np
import os.path as osp

from absl import app
from absl import flags

from octree.nerf import models
from octree.nerf import utils
from octree.nerf import datasets
from octree.nerf import sh_proj

from svox import N3Tree
from svox import NDCConfig, VolumeRenderer
from svox.helpers import _get_c_extension
from tqdm import tqdm

_C = _get_c_extension()

FLAGS = flags.FLAGS

utils.define_flags()

flags.DEFINE_string(
    "output",
    "./tree.npz",
    "Output file",
)
flags.DEFINE_string(
    "center",
    "0 0 0",
    "Center of volume in x y z OR single number",
)
flags.DEFINE_string(
    "radius",
    "1.5",
    "1/2 side length of volume",
)
flags.DEFINE_float(
    "alpha_thresh",
    0.01,
    "Alpha threshold to keep a voxel in initial sigma thresholding",
)
flags.DEFINE_float(
    "max_refine_prop",
    0.5,
    "Max proportion of cells to refine",
)
flags.DEFINE_float(
    "z_min",
    None,
    "Discard z axis points below this value, for NDC use",
)
flags.DEFINE_float(
    "z_max",
    None,
    "Discard z axis points above this value, for NDC use",
)
flags.DEFINE_integer(
    "tree_branch_n",
    2,
    "Tree branch factor (2=octree)",
)
flags.DEFINE_integer(
    "init_grid_depth",
    8,
    "Initial evaluation grid (2^{x+1} voxel grid)",
)
flags.DEFINE_integer(
    "samples_per_cell",
    8,
    "Samples per cell in step 2 (3D antialiasing)",
    short_name='S',
)
flags.DEFINE_bool(
    "is_jaxnerf_ckpt",
    False,
    "Whether the ckpt is from jaxnerf or not.",
)
flags.DEFINE_enum(
    "masking_mode",
    "weight",
    ["sigma", "weight"],
    "How to calculate mask when building the octree",
)
flags.DEFINE_float(
    "weight_thresh",
    0.001,
    "Weight threshold to keep a voxel",
)
flags.DEFINE_integer(
    "projection_samples",
    10000,
    "Number of rays to sample for SH projection.",
)

# Load bbox from dataset
flags.DEFINE_bool(
    "bbox_from_data",
    False,
    "Use bounding box from dataset if possible",
)
flags.DEFINE_float(
    "data_bbox_scale",
    1.0,
    "Scaling factor to apply to the bounding box from dataset (before autoscale), " +
    "if bbox_from_data is used",
)
flags.DEFINE_bool(
    "autoscale",
    False,
    "Automatic scaling, after bbox_from_data",
)
flags.DEFINE_bool(
    "bbox_cube",
    False,
    "Force bbox to be a cube",
)
flags.DEFINE_float(
    "bbox_scale",
    1.0,
    "Scaling factor to apply to the bounding box at the end (after load, autoscale)",
)
flags.DEFINE_float(
    "scale_alpha_thresh",
    0.01,
    "Alpha threshold to keep a voxel in initial sigma thresholding for autoscale",
)
# For integrated eval (to avoid slow load)
flags.DEFINE_bool(
    "eval",
    True,
    "Evaluate after building the octree",
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_grid_weights(dataset, sigmas, reso, invradius, offset):
    w, h, focal = dataset.w, dataset.h, dataset.focal

    opts = _C.RenderOptions()
    opts.step_size = FLAGS.renderer_step_size
    opts.sigma_thresh = 0.0
    if 'llff' in FLAGS.config and (not FLAGS.spherify):
        ndc_config = NDCConfig(width=w, height=h, focal=focal)
        opts.ndc_width = ndc_config.width
        opts.ndc_height = ndc_config.height
        opts.ndc_focal = ndc_config.focal
    else:
        opts.ndc_width = -1

    cam = _C.CameraSpec()
    cam.fx = focal
    cam.fy = focal
    cam.width = w
    cam.height = h

    grid_data = sigmas.reshape((reso, reso, reso))
    maximum_weight = torch.zeros_like(grid_data)
    for idx in tqdm(range(dataset.size)):
        cam.c2w = torch.from_numpy(dataset.camtoworlds[idx]).float().to(sigmas.device)
        grid_weight, grid_hit = _C.grid_weight_render(
            grid_data,
            cam,
            opts,
            offset,
            invradius,
        )
        maximum_weight = torch.max(maximum_weight, grid_weight)

    return maximum_weight


def project_nerf_to_sh(nerf, sh_deg, points):
    """
    Args:
        points: [N, 3]
    Returns:
        coeffs for rgb. [N, C * (sh_deg + 1)**2]
    """
    nerf.use_viewdirs = True

    def _sperical_func(viewdirs):
        # points: [num_points, 3]
        # viewdirs: [num_rays, 3]
        # raw_rgb: [num_points, num_rays, 3]
        # sigma: [num_points]
        raw_rgb, sigma = nerf.eval_points_raw(points, viewdirs, cross_broadcast=True)
        return raw_rgb, sigma

    coeffs, sigma = sh_proj.ProjectFunctionNeRF(
        order=sh_deg,
        sperical_func=_sperical_func,
        batch_size=points.shape[0],
        sample_count=FLAGS.projection_samples,
        device=points.device)

    return coeffs.reshape([points.shape[0], -1]), sigma


def auto_scale(args, center, radius, nerf):
    print('* Step 0: Auto scale')
    reso = 2 ** args.init_grid_depth

    radius = torch.tensor(radius, dtype=torch.float32)
    center = torch.tensor(center, dtype=torch.float32)
    scale = 0.5 / radius
    offset = 0.5 * (1.0 - center / radius)

    arr = (torch.arange(0, reso, dtype=torch.float32) + 0.5) / reso
    xx = (arr - offset[0]) / scale[0]
    yy = (arr - offset[1]) / scale[1]
    zz = (arr - offset[2]) / scale[2]
    if args.z_min is not None:
        zz = zz[zz >= args.z_min]
    if args.z_max is not None:
        zz = zz[zz <= args.z_max]

    grid = torch.stack(torch.meshgrid(xx, yy, zz)).reshape(3, -1).T

    out_chunks = []
    for i in tqdm(range(0, grid.shape[0], args.chunk)):
        grid_chunk = grid[i:i+args.chunk].cuda()
        if nerf.use_viewdirs:
            fake_viewdirs = torch.zeros([grid_chunk.shape[0], 3], device=grid_chunk.device)
        else:
            fake_viewdirs = None
        rgb, sigma = nerf.eval_points_raw(grid_chunk, fake_viewdirs)
        del grid_chunk
        out_chunks.append(sigma.squeeze(-1))
    sigmas = torch.cat(out_chunks, 0)
    del out_chunks

    approx_delta = 2.0 / reso
    sigma_thresh = -np.log(1.0 - args.scale_alpha_thresh) / approx_delta
    mask = sigmas >= sigma_thresh

    grid = grid[mask]
    del mask

    lc = grid.min(dim=0)[0] - 0.5 / reso
    uc = grid.max(dim=0)[0] + 0.5 / reso
    return ((lc + uc) * 0.5).tolist(), ((uc - lc) * 0.5).tolist()

def step1(args, tree, nerf, dataset):
    print('* Step 1: Grid eval')
    reso = 2 ** (args.init_grid_depth + 1)
    offset = tree.offset.cpu()
    scale = tree.invradius.cpu()

    arr = (torch.arange(0, reso, dtype=torch.float32) + 0.5) / reso
    xx = (arr - offset[0]) / scale[0]
    yy = (arr - offset[1]) / scale[1]
    zz = (arr - offset[2]) / scale[2]
    if args.z_min is not None:
        zz = zz[zz >= args.z_min]
    if args.z_max is not None:
        zz = zz[zz <= args.z_max]

    grid = torch.stack(torch.meshgrid(xx, yy, zz)).reshape(3, -1).T
    print('init grid', grid.shape)

    approx_delta = 2.0 / reso
    sigma_thresh = -np.log(1.0 - args.alpha_thresh) / approx_delta

    out_chunks = []
    for i in tqdm(range(0, grid.shape[0], args.chunk)):
        grid_chunk = grid[i:i+args.chunk].cuda()
        if nerf.use_viewdirs:
            fake_viewdirs = torch.zeros([grid_chunk.shape[0], 3], device=grid_chunk.device)
        else:
            fake_viewdirs = None
        rgb, sigma = nerf.eval_points_raw(grid_chunk, fake_viewdirs)
        del grid_chunk
        out_chunks.append(sigma.squeeze(-1))
    sigmas = torch.cat(out_chunks, 0)
    del out_chunks

    if FLAGS.masking_mode == "sigma":
        mask = sigmas >= sigma_thresh
    elif FLAGS.masking_mode == "weight":
        print ("* Calculating grid weights")
        grid_weights = calculate_grid_weights(dataset,
            sigmas, reso, tree.invradius, tree.offset)
        mask = grid_weights.reshape(-1) >= FLAGS.weight_thresh
        del grid_weights
    else:
        raise ValueError
    del sigmas

    grid = grid[mask]
    del mask
    print(grid.shape, grid.min(), grid.max())
    grid = grid.cuda()

    torch.cuda.empty_cache()
    print(' Building octree')
    for i in range(args.init_grid_depth - 1):
        tree[grid].refine()
    refine_chunk = 2000000
    if grid.shape[0] <= refine_chunk:
        tree[grid].refine()
    else:
        # Do last layer separately
        grid = grid.cpu()
        for j in tqdm(range(0, grid.shape[0], refine_chunk)):
            tree[grid[j:j+refine_chunk].cuda()].refine()
    print(tree)

    assert tree.max_depth == args.init_grid_depth

def step2(args, tree, nerf):
    print('* Step 2: AA', args.samples_per_cell)

    leaf_mask = tree.depths.cpu() == tree.max_depth
    leaf_ind = torch.where(leaf_mask)[0]
    del leaf_mask

    if args.use_viewdirs:
        chunk_size = args.chunk // (args.samples_per_cell * args.projection_samples // 10)
    else:
        chunk_size = args.chunk // (args.samples_per_cell)

    for i in tqdm(range(0, leaf_ind.size(0), chunk_size)):
        chunk_inds = leaf_ind[i:i+chunk_size]
        points = tree[chunk_inds].sample(args.samples_per_cell)  # (n_cells, n_samples, 3)
        points = points.view(-1, 3)

        if not args.use_viewdirs:  # trained NeRF-SH/SG model returns rgb as coeffs
            rgb, sigma = nerf.eval_points_raw(points)
        else:  # vanilla NeRF model returns rgb, so we project them into coeffs (only SH supported)
            rgb, sigma = project_nerf_to_sh(nerf, args.sh_deg, points)

        if tree.data_format.format == tree.data_format.RGBA:
            rgb = rgb.reshape(-1, args.samples_per_cell, tree.data_dim - 1);
            sigma = sigma.reshape(-1, args.samples_per_cell, 1);
            sigma_avg = sigma.mean(dim=1)

            reso = 2 ** (args.init_grid_depth + 1)
            approx_delta = 2.0 / reso
            alpha = 1.0 - torch.exp(-approx_delta * sigma)
            msum = alpha.sum(dim=1)
            rgb_avg = (rgb * alpha).sum(dim=1) / msum
            rgb_avg[msum[..., 0] < 1e-3] = 0
            rgba = torch.cat([rgb_avg, sigma_avg], dim=-1)
            del rgb, sigma
        else:
            rgba = torch.cat([rgb, sigma], dim=-1)
            del rgb, sigma
            rgba = rgba.reshape(-1, args.samples_per_cell, tree.data_dim).mean(dim=1)
        tree[chunk_inds] = rgba

def euler2mat(angle):
    """Convert euler angles to rotation matrix.

    Args:
        angle: rotation angle along 3 axis (in radians). [..., 3]
    Returns:
        Rotation matrix corresponding to the euler angles. [..., 3, 3]
    """
    x, y, z = angle[..., 0], angle[..., 1], angle[..., 2]
    cosz = torch.cos(z)
    sinz = torch.sin(z)
    cosy = torch.cos(y)
    siny = torch.sin(y)
    cosx = torch.cos(x)
    sinx = torch.sin(x)
    zeros = torch.zeros_like(z)
    ones = torch.ones_like(z)
    zmat = torch.stack([torch.stack([cosz,  -sinz, zeros], dim=-1),
                        torch.stack([sinz,   cosz, zeros], dim=-1),
                        torch.stack([zeros, zeros,  ones], dim=-1)], dim=-1)
    ymat = torch.stack([torch.stack([ cosy, zeros,  siny], dim=-1),
                        torch.stack([zeros,  ones, zeros], dim=-1),
                        torch.stack([-siny, zeros,  cosy], dim=-1)], dim=-1)
    xmat = torch.stack([torch.stack([ ones, zeros, zeros], dim=-1),
                        torch.stack([zeros,  cosx, -sinx], dim=-1),
                        torch.stack([zeros,  sinx,  cosx], dim=-1)], dim=-1)
    rotMat = torch.einsum("...ij,...jk,...kq->...iq", xmat, ymat, zmat)
    return rotMat

@torch.no_grad()
def main(unused_argv):
    utils.set_random_seed(20200823)
    utils.update_flags(FLAGS)

    print('* Loading NeRF')
    nerf = models.get_model_state(FLAGS, device=device, restore=True)
    nerf.eval()

    data_format = None
    extra_data = None
    if FLAGS.sg_dim > 0:
        data_format = f'SG{FLAGS.sg_dim}'
        assert FLAGS.sg_global
        extra_data = torch.cat((
                            F.softplus(nerf.sg_lambda[:, None]),
                            sh_proj.spher2cart(nerf.sg_mu_spher[:, 0], nerf.sg_mu_spher[:, 1])
                         ), dim=-1)
    elif FLAGS.sh_deg > 0:
        data_format = f'SH{(FLAGS.sh_deg + 1) ** 2}'
    if data_format is not None:
        print('Detected format:', data_format)

    base_dir = osp.dirname(FLAGS.output)
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)

    assert FLAGS.data_dir  # Dataset is required now
    dataset = datasets.get_dataset("train", FLAGS)

    if FLAGS.bbox_from_data:
        assert dataset.bbox is not None  # Dataset must be NSVF
        center = (dataset.bbox[:3] + dataset.bbox[3:6]) * 0.5
        radius = (dataset.bbox[3:6] - dataset.bbox[:3]) * 0.5 * FLAGS.data_bbox_scale
        print('Bounding box from data: c', center, 'r', radius)
    else:
        center = list(map(float, FLAGS.center.split()))
        if len(center) == 1:
            center *= 3
        radius = list(map(float, FLAGS.radius.split()))
        if len(radius) == 1:
            radius *= 3

    if FLAGS.autoscale:
        center, radius = auto_scale(FLAGS, center, radius, nerf)
        print('Autoscale result center', center, 'radius', radius)

    radius = [r * FLAGS.bbox_scale for r in radius]
    if FLAGS.bbox_cube:
        radius = [max(radius)] * 3

    num_rgb_channels = FLAGS.num_rgb_channels
    if FLAGS.sh_deg >= 0:
        assert FLAGS.sg_dim == -1, (
            "You can only use up to one of: SH or SG")
        num_rgb_channels *= (FLAGS.sh_deg + 1) ** 2
    elif FLAGS.sg_dim > 0:
        assert FLAGS.sh_deg == -1, (
            "You can only use up to one of: SH or SG")
        num_rgb_channels *= FLAGS.sg_dim
    data_dim =  1 + num_rgb_channels  # alpha + rgb
    print('data dim is', data_dim)

    print('* Creating model')
    tree = N3Tree(N=FLAGS.tree_branch_n,
                  data_dim=data_dim,
                  init_refine=0,
                  init_reserve=500000,
                  geom_resize_fact=1.0,
                  depth_limit=FLAGS.init_grid_depth,
                  radius=radius,
                  center=center,
                  data_format=data_format,
                  extra_data=extra_data,
                  map_location=device)

    step1(FLAGS, tree, nerf, dataset)
    step2(FLAGS, tree, nerf)
    tree[:, -1:].relu_()
    tree.shrink_to_fit()
    print(tree)

    del dataset.images
    print('* Saving', FLAGS.output)
    tree.save(FLAGS.output, compress=False)  # Faster saving

    if FLAGS.eval:
        dataset = datasets.get_dataset("test", FLAGS)
        print('* Evaluation (before fine tune)')
        avg_psnr, avg_ssim, avg_lpips, out_frames = utils.eval_octree(tree,
                dataset, FLAGS, want_lpips=True)
        print('Average PSNR', avg_psnr, 'SSIM', avg_ssim, 'LPIPS', avg_lpips)


if __name__ == "__main__":
    app.run(main)
