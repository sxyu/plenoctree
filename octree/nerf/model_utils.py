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
"""Helper functions/classes for model definition."""

import functools
from typing import Any, Callable
import math

import torch
import torch.nn as nn


def dense_layer(in_features, out_features):
    layer = nn.Linear(in_features, out_features)
    # The initialization matters!
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class MLP(nn.Module):
    """A simple MLP."""

    def __init__(
        self,
        net_depth: int = 8,  # The depth of the first part of MLP.
        net_width: int = 256,  # The width of the first part of MLP.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
        net_activation: Callable[Ellipsis, Any] = nn.ReLU(),  # The activation function.
        skip_layer: int = 4,  # The layer to add skip layers to.
        num_rgb_channels: int = 3,  # The number of RGB channels.
        num_sigma_channels: int = 1,  # The number of sigma channels.
        input_dim: int = 63,  # The number of input tensor channels.
        condition_dim: int = 27,  # The number of conditional tensor channels.
    ):
        super(MLP, self).__init__()
        self.net_depth = net_depth
        self.net_width = net_width
        self.net_depth_condition = net_depth_condition
        self.net_width_condition = net_width_condition
        self.net_activation = net_activation
        self.skip_layer = skip_layer
        self.num_rgb_channels = num_rgb_channels
        self.num_sigma_channels = num_sigma_channels
        self.input_dim = input_dim
        self.condition_dim = condition_dim

        self.input_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.input_layers.append(
                dense_layer(in_features, self.net_width)
            )
            if i % self.skip_layer == 0 and i > 0:
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        self.sigma_layer = dense_layer(in_features, self.num_sigma_channels)

        if self.condition_dim > 0:
            self.bottleneck_layer = dense_layer(in_features, self.net_width)
            self.condition_layers = nn.ModuleList()
            in_features = self.net_width + self.condition_dim
            for i in range(self.net_depth_condition):
                self.condition_layers.append(
                    dense_layer(in_features, self.net_width_condition)
                )
                in_features = self.net_width_condition
        self.rgb_layer = dense_layer(in_features, self.num_rgb_channels)

    def forward(self, x, condition=None, cross_broadcast=False):
        """Evaluate the MLP.

        Args:
          x: torch.tensor(float32), [batch, num_samples, feature], points.
          condition: torch.tensor(float32),
            [batch, feature] or [batch, num_samples, feature] or [batch, num_rays, feature],
            if not None, this variable will be part of the input to the second part of the MLP
            concatenated with the output vector of the first part of the MLP. If
            None, only the first part of the MLP will be used with input x. In the
            original paper, this variable is the view direction. Note when the shape of this
            tensor is [batch, num_rays, feature], where `num_rays` != `num_samples`, this
            function will cross broadcast all rays with all samples. And the `cross_broadcast`
            option must be set to `True`.
          cross_broadcast: if true, cross broadcast the x tensor and the condition
            tensor.

        Returns:
          raw_rgb: torch.tensor(float32), with a shape of
            [batch, num_samples, num_rgb_channels]. If `cross_broadcast` is true, the return
            shape would be [batch, num_samples, num_rays, num_rgb_channels].
          raw_sigma: torch.tensor(float32), with a shape of
            [batch, num_samples, num_sigma_channels]. If `cross_broadcast` is true, the return
            shape woudl be [batch, num_samples, num_rays, num_sigma_channels].
        """
        batch_size = x.shape[0]
        feature_dim = x.shape[-1]
        num_samples = x.shape[1]
        x = x.view([-1, feature_dim])
        inputs = x
        for i in range(self.net_depth):
            x = self.input_layers[i](x)
            x = self.net_activation(x)
            if i % self.skip_layer == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
        raw_sigma = self.sigma_layer(x).view(
            [-1, num_samples, self.num_sigma_channels]
        )

        if condition is not None:
            # Output of the first part of MLP.
            bottleneck = self.bottleneck_layer(x)
            # Broadcast condition from [batch, feature] to
            # [batch, num_samples, feature] since all the samples along the same ray
            # have the same viewdir.
            if len(condition.shape) == 2 and (not cross_broadcast):
                condition = condition[:, None, :].repeat(1, num_samples, 1)
            # Broadcast samples from [batch, num_samples, feature]
            # and condition from [batch, num_rays, feature] to
            # [batch, num_samples, num_rays, feature] since in this case each point
            # is passed by all the rays. This option is used for projecting an
            # trained vanilla NeRF to NeRF-SH.
            if cross_broadcast:
                condition = condition.view([batch_size, -1, condition.shape[-1]])
                num_rays = condition.shape[1]
                condition = condition[:, None, :, :].repeat(1, num_samples, 1, 1)
                bottleneck = bottleneck.view([batch_size, -1, bottleneck.shape[-1]])
                bottleneck = bottleneck[:, :, None, :].repeat(1, 1, num_rays, 1)
            # Collapse the [batch, num_samples, (num_rays,) feature] tensor to
            # [batch * num_samples (* num_rays), feature] so that it can be fed into nn.Dense.
            x = torch.cat([
                bottleneck.view([-1, bottleneck.shape[-1]]),
                condition.view([-1, condition.shape[-1]])], dim=-1)
            # Here use 1 extra layer to align with the original nerf model.
            for i in range(self.net_depth_condition):
                x = self.condition_layers[i](x)
                x = self.net_activation(x)
        raw_rgb = self.rgb_layer(x).view(
            [batch_size, num_samples, self.num_rgb_channels] if not cross_broadcast else \
            [batch_size, num_samples, num_rays, self.num_rgb_channels]
        )
        return raw_rgb, raw_sigma


def posenc(x, min_deg, max_deg, legacy_posenc_order=False):
    """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].

    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    Args:
      x: torch.tensor, variables to be encoded. Note that x should be in [-pi, pi].
      min_deg: int, the minimum (inclusive) degree of the encoding.
      max_deg: int, the maximum (exclusive) degree of the encoding.
      legacy_posenc_order: bool, keep the same ordering as the original tf code.

    Returns:
      encoded: torch.tensor, encoded variables.
    """
    if min_deg == max_deg:
        return x
    scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)],
                          dtype=x.dtype, device=x.device)
    if legacy_posenc_order:
        xb = x[Ellipsis, None, :] * scales[:, None]
        four_feat = torch.reshape(
            torch.sin(torch.stack([xb, xb + 0.5 * math.pi], -2)), list(x.shape[:-1]) + [-1]
        )
    else:
        xb = torch.reshape(
            (x[Ellipsis, None, :] * scales[:, None]), list(x.shape[:-1]) + [-1]
        )
        four_feat = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)

