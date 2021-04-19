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
"""Different model implementation plus a general port for all the models."""
import os, glob
import inspect
from typing import Any, Callable
import math

import torch
import torch.nn as nn

from octree.nerf import model_utils


def get_model(args):
    """A helper function that wraps around a 'model zoo'."""
    model_dict = {
        "nerf": construct_nerf,
    }
    return model_dict[args.model](args)


def get_model_state(args, device="cpu", restore=True):
    """
    Helper for loading model with get_model & creating optimizer &
    optionally restoring checkpoint to reduce boilerplate
    """
    model = get_model(args).to(device)
    if restore:
        if args.is_jaxnerf_ckpt:
            model = restore_model_state_from_jaxnerf(args, model)
        else:
            model = restore_model_state(args, model)
    return model


def restore_model_state(args, model):
    """
    Helper for restoring checkpoint.
    """
    ckpt_paths = sorted(
        glob.glob(os.path.join(args.train_dir, "*.ckpt")))
    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt["model"])
        print (f"* restore ckpt from {ckpt_path}.")
    return model


def restore_model_state_from_jaxnerf(args, model):
    """
    Helper for restoring checkpoint for jaxnerf.
    """
    from flax.training import checkpoints
    ckpt_paths = sorted(
        glob.glob(os.path.join(args.train_dir, "checkpoint_*")))

    if len(ckpt_paths) > 0:
        ckpt_dict = checkpoints.restore_checkpoint(
            args.train_dir, target=None)["optimizer"]["target"]["params"]
        state_dict = {}

        def _init_layer(from_name, to_name):
            state_dict[f"MLP_0.{to_name}.weight"] = \
                ckpt_dict["MLP_0"][f"{from_name}"]["kernel"].T
            state_dict[f"MLP_0.{to_name}.bias"] = \
                ckpt_dict["MLP_0"][f"{from_name}"]["bias"]
            state_dict[f"MLP_1.{to_name}.weight"] = \
                ckpt_dict["MLP_1"][f"{from_name}"]["kernel"].T
            state_dict[f"MLP_1.{to_name}.bias"] = \
                ckpt_dict["MLP_1"][f"{from_name}"]["bias"]
            pass

        # init all layers
        for i in range(model.net_depth):
            _init_layer(f"Dense_{i}", f"input_layers.{i}")
        i += 1
        _init_layer(f"Dense_{i}", f"sigma_layer")
        if model.use_viewdirs:
            i += 1
            _init_layer(f"Dense_{i}", f"bottleneck_layer")
            for j in range(model.net_depth_condition):
                i += 1
                _init_layer(f"Dense_{i}", f"condition_layers.{j}")
        i += 1
        _init_layer(f"Dense_{i}", f"rgb_layer")

        # support SG
        if model.sg_dim > 0:
            state_dict["sg_lambda"] = ckpt_dict["sg_lambda"]
            state_dict["sg_mu_spher"] = ckpt_dict["sg_mu_spher"]

        for key in state_dict.keys():
            state_dict[key] = torch.from_numpy(state_dict[key].copy())
        model.load_state_dict(state_dict)
        print (f"* restore ckpt from {args.train_dir}")
    return model


class NerfModel(nn.Module):
    """Nerf NN Model with both coarse and fine MLPs."""

    def __init__(
        self,
        num_coarse_samples: int = 64,  # The number of samples for the coarse nerf.
        num_fine_samples: int = 128,  # The number of samples for the fine nerf.
        use_viewdirs: bool = True,  # If True, use viewdirs as an input.
        sh_deg: int = -1,  # If != -1, use spherical harmonics output of given order
        sg_dim: int = -1, # If != -1, use spherical gaussians output of given dimension
        near: float = 2.0,  # The distance to the near plane
        far: float = 6.0,  # The distance to the far plane
        noise_std: float = 0.0,  # The std dev of noise added to raw sigma.
        net_depth: int = 8,  # The depth of the first part of MLP.
        net_width: int = 256,  # The width of the first part of MLP.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
        net_activation: Callable[Ellipsis, Any] = nn.ReLU(),  # MLP activation
        skip_layer: int = 4,  # How often to add skip connections.
        num_rgb_channels: int = 3,  # The number of RGB channels.
        num_sigma_channels: int = 1,  # The number of density channels.
        white_bkgd: bool = True,  # If True, use a white background.
        min_deg_point: int = 0,  # The minimum degree of positional encoding for positions.
        max_deg_point: int = 10,  # The maximum degree of positional encoding for positions.
        deg_view: int = 4,  # The degree of positional encoding for viewdirs.
        lindisp: bool = False,  # If True, sample linearly in disparity rather than in depth.
        rgb_activation: Callable[Ellipsis, Any] = nn.Sigmoid(),  # Output RGB activation.
        sigma_activation: Callable[Ellipsis, Any] = nn.ReLU(),  # Output sigma activation.
        legacy_posenc_order: bool = False,  # Keep the same ordering as the original tf code.
    ):
        super(NerfModel, self).__init__()
        self.num_coarse_samples = num_coarse_samples
        self.num_fine_samples = num_fine_samples
        self.use_viewdirs = use_viewdirs
        self.sh_deg = sh_deg
        self.sg_dim = sg_dim
        self.near = near
        self.far = far
        self.noise_std = noise_std
        self.net_depth = net_depth
        self.net_width = net_width
        self.net_depth_condition = net_depth_condition
        self.net_width_condition = net_width_condition
        self.net_activation = net_activation
        self.skip_layer = skip_layer
        self.num_rgb_channels = num_rgb_channels
        self.num_sigma_channels = num_sigma_channels
        self.white_bkgd = white_bkgd
        self.min_deg_point = min_deg_point
        self.max_deg_point = max_deg_point
        self.deg_view = deg_view
        self.lindisp = lindisp
        self.rgb_activation = rgb_activation
        self.sigma_activation = sigma_activation
        self.legacy_posenc_order = legacy_posenc_order
        # Construct the "coarse" MLP. Weird name is for
        # compatibility with 'compact' version
        self.MLP_0 = model_utils.MLP(
            net_depth = self.net_depth,
            net_width = self.net_width,
            net_depth_condition = self.net_depth_condition,
            net_width_condition = self.net_width_condition,
            net_activation = self.net_activation,
            skip_layer = self.skip_layer,
            num_rgb_channels = self.num_rgb_channels,
            num_sigma_channels = self.num_sigma_channels,
            input_dim=3 * (1 + 2 * (self.max_deg_point - self.min_deg_point)),
            condition_dim=3 * (1 + 2 * self.deg_view) if self.use_viewdirs else 0)
        # Construct the "fine" MLP.
        self.MLP_1 = model_utils.MLP(
            net_depth = self.net_depth,
            net_width = self.net_width,
            net_depth_condition = self.net_depth_condition,
            net_width_condition = self.net_width_condition,
            net_activation = self.net_activation,
            skip_layer = self.skip_layer,
            num_rgb_channels = self.num_rgb_channels,
            num_sigma_channels = self.num_sigma_channels,
            input_dim=3 * (1 + 2 * (self.max_deg_point - self.min_deg_point)),
            condition_dim=3 * (1 + 2 * self.deg_view) if self.use_viewdirs else 0)

        # Construct learnable variables for spherical gaussians.
        if self.sg_dim > 0:
            self.register_parameter(
                "sg_lambda",
                nn.Parameter(torch.ones([self.sg_dim]))
            )
            self.register_parameter(
                "sg_mu_spher",
                nn.Parameter(torch.stack([
                    torch.rand([self.sg_dim]) * math.pi,  # theta
                    torch.rand([self.sg_dim]) * math.pi * 2  # phi
                ], dim=-1))
            )

    def eval_points_raw(self, points, viewdirs=None, coarse=False, cross_broadcast=False):
        """
        Evaluate at points, returing rgb and sigma.
        If sh_deg >= 0 then this will return spherical harmonic
        coeffs for RGB. Please see eval_points for alternate
        version which always returns RGB.

        Args:
          points: torch.tensor [B, 3]
          viewdirs: torch.tensor [B, 3]. if cross_broadcast = True, it can be [M, 3].
          coarse: if true, uses coarse MLP.
          cross_broadcast: if true, cross broadcast between points and viewdirs.

        Returns:
          raw_rgb: torch.tensor [B, 3 * (sh_deg + 1)**2 or 3]. if cross_broadcast = True, it
            returns [B, M, 3 * (sh_deg + 1)**2 or 3]
          raw_sigma: torch.tensor [B, 1]
        """
        points = points[None]
        points_enc = model_utils.posenc(
            points,
            self.min_deg_point,
            self.max_deg_point,
            self.legacy_posenc_order,
        )
        if self.num_fine_samples > 0 and not coarse:
            mlp = self.MLP_1
        else:
            mlp = self.MLP_0
        if self.use_viewdirs:
            assert viewdirs is not None
            viewdirs = viewdirs[None]
            viewdirs_enc = model_utils.posenc(
                viewdirs,
                0,
                self.deg_view,
                self.legacy_posenc_order,
            )
            raw_rgb, raw_sigma = mlp(points_enc, viewdirs_enc, cross_broadcast=cross_broadcast)
        else:
            raw_rgb, raw_sigma = mlp(points_enc)
        return raw_rgb[0], raw_sigma[0]


def construct_nerf(args):
    """Construct a Neural Radiance Field.

    Args:
      args: FLAGS class. Hyperparameters of nerf.

    Returns:
      model: nn.Model. Nerf model with parameters.
      state: flax.Module.state. Nerf model state for stateful parameters.
    """
    net_activation = getattr(nn, str(args.net_activation))
    if inspect.isclass(net_activation):
        net_activation = net_activation()
    rgb_activation = getattr(nn, str(args.rgb_activation))
    if inspect.isclass(rgb_activation):
        rgb_activation = rgb_activation()
    sigma_activation = getattr(nn, str(args.sigma_activation))
    if inspect.isclass(sigma_activation):
        sigma_activation = sigma_activation()

    # Assert that rgb_activation always produces outputs in [0, 1], and
    # sigma_activation always produce non-negative outputs.
    x = torch.exp(torch.linspace(-90, 90, 1024))
    x = torch.cat([-x, x], dim=0)

    rgb = rgb_activation(x)
    if torch.any(rgb < 0) or torch.any(rgb > 1):
        raise NotImplementedError(
            "Choice of rgb_activation `{}` produces colors outside of [0, 1]".format(
                args.rgb_activation
            )
        )

    sigma = sigma_activation(x)
    if torch.any(sigma < 0):
        raise NotImplementedError(
            "Choice of sigma_activation `{}` produces negative densities".format(
                args.sigma_activation
            )
        )

    num_rgb_channels = args.num_rgb_channels
    if not args.use_viewdirs:
        if args.sh_deg >= 0:
            assert args.sg_dim == -1, (
                "You can only use up to one of: SH or SG.")
            num_rgb_channels *= (args.sh_deg + 1) ** 2
        elif args.sg_dim > 0:
            assert args.sh_deg == -1, (
                "You can only use up to one of: SH or SG.")
            num_rgb_channels *= args.sg_dim

    model = NerfModel(
        min_deg_point=args.min_deg_point,
        max_deg_point=args.max_deg_point,
        deg_view=args.deg_view,
        num_coarse_samples=args.num_coarse_samples,
        num_fine_samples=args.num_fine_samples,
        use_viewdirs=args.use_viewdirs,
        sh_deg=args.sh_deg,
        sg_dim=args.sg_dim,
        near=args.near,
        far=args.far,
        noise_std=args.noise_std,
        white_bkgd=args.white_bkgd,
        net_depth=args.net_depth,
        net_width=args.net_width,
        net_depth_condition=args.net_depth_condition,
        net_width_condition=args.net_width_condition,
        skip_layer=args.skip_layer,
        num_rgb_channels=num_rgb_channels,
        num_sigma_channels=args.num_sigma_channels,
        lindisp=args.lindisp,
        net_activation=net_activation,
        rgb_activation=rgb_activation,
        sigma_activation=sigma_activation,
        legacy_posenc_order=args.legacy_posenc_order,
    )
    return model
