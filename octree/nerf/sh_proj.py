# Modifications Copyright 2021 The PlenOctree Authors.
# Original Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sperical harmonics projection related functions

Some codes are borrowed from:
https://github.com/google/spherical-harmonics/blob/master/sh/spherical_harmonics.cc
"""
from typing import Callable
import math
import torch

kHardCodedOrderLimit = 4


def spher2cart(theta, phi):
  """Convert spherical coordinates into Cartesian coordinates (radius 1)."""
  r = torch.sin(theta)
  x = r * torch.cos(phi)
  y = r * torch.sin(phi)
  z = torch.cos(theta)
  return torch.stack([x, y, z], dim=-1)


# Get the total number of coefficients for a function represented by
# all spherical harmonic basis of degree <= @order (it is a point of
# confusion that the order of an SH refers to its degree and not the order).
def GetCoefficientCount(order: int):
  return (order + 1) ** 2


# Get the one dimensional index associated with a particular degree @l
# and order @m. This is the index that can be used to access the Coeffs
# returned by SHSolver.
def GetIndex(l: int, m: int):
  return l * (l + 1) + m


# Hardcoded spherical harmonic functions for low orders (l is first number
# and m is second number (sign encoded as preceeding 'p' or 'n')).
#
# As polynomials they are evaluated more efficiently in cartesian coordinates,
# assuming that @{dx, dy, dz} is unit. This is not verified for efficiency.

def HardcodedSH00(dx, dy, dz):
  # 0.5 * sqrt(1/pi)
  return 0.28209479177387814 + (dx * 0)  # keep the shape

def HardcodedSH1n1(dx, dy, dz):
  # -sqrt(3/(4pi)) * y
  return -0.4886025119029199 * dy

def HardcodedSH10(dx, dy, dz):
  # sqrt(3/(4pi)) * z
  return 0.4886025119029199 * dz

def HardcodedSH1p1(dx, dy, dz):
  # -sqrt(3/(4pi)) * x
  return -0.4886025119029199 * dx

def HardcodedSH2n2(dx, dy, dz):
  # 0.5 * sqrt(15/pi) * x * y
  return 1.0925484305920792 * dx * dy

def HardcodedSH2n1(dx, dy, dz):
  # -0.5 * sqrt(15/pi) * y * z
  return -1.0925484305920792 * dy * dz

def HardcodedSH20(dx, dy, dz):
  # 0.25 * sqrt(5/pi) * (-x^2-y^2+2z^2)
  return 0.31539156525252005 * (-dx * dx - dy * dy + 2.0 * dz * dz)

def HardcodedSH2p1(dx, dy, dz):
  # -0.5 * sqrt(15/pi) * x * z
  return -1.0925484305920792 * dx * dz

def HardcodedSH2p2(dx, dy, dz):
  # 0.25 * sqrt(15/pi) * (x^2 - y^2)
  return 0.5462742152960396 * (dx * dx - dy * dy)

def HardcodedSH3n3(dx, dy, dz):
  # -0.25 * sqrt(35/(2pi)) * y * (3x^2 - y^2)
  return -0.5900435899266435 * dy * (3.0 * dx * dx - dy * dy)

def HardcodedSH3n2(dx, dy, dz):
  # 0.5 * sqrt(105/pi) * x * y * z
  return 2.890611442640554 * dx * dy * dz

def HardcodedSH3n1(dx, dy, dz):
  # -0.25 * sqrt(21/(2pi)) * y * (4z^2-x^2-y^2)
  return -0.4570457994644658 * dy * (4.0 * dz * dz - dx * dx - dy * dy)

def HardcodedSH30(dx, dy, dz):
  # 0.25 * sqrt(7/pi) * z * (2z^2 - 3x^2 - 3y^2)
  return 0.3731763325901154 * dz * (2.0 * dz * dz - 3.0 * dx * dx - 3.0 * dy * dy)

def HardcodedSH3p1(dx, dy, dz):
  # -0.25 * sqrt(21/(2pi)) * x * (4z^2-x^2-y^2)
  return -0.4570457994644658 * dx * (4.0 * dz * dz - dx * dx - dy * dy)

def HardcodedSH3p2(dx, dy, dz):
  # 0.25 * sqrt(105/pi) * z * (x^2 - y^2)
  return 1.445305721320277 * dz * (dx * dx - dy * dy)

def HardcodedSH3p3(dx, dy, dz):
  # -0.25 * sqrt(35/(2pi)) * x * (x^2-3y^2)
  return -0.5900435899266435 * dx * (dx * dx - 3.0 * dy * dy)

def HardcodedSH4n4(dx, dy, dz):
  # 0.75 * sqrt(35/pi) * x * y * (x^2-y^2)
  return 2.5033429417967046 * dx * dy * (dx * dx - dy * dy)

def HardcodedSH4n3(dx, dy, dz):
  # -0.75 * sqrt(35/(2pi)) * y * z * (3x^2-y^2)
  return -1.7701307697799304 * dy * dz * (3.0 * dx * dx - dy * dy)

def HardcodedSH4n2(dx, dy, dz):
  # 0.75 * sqrt(5/pi) * x * y * (7z^2-1)
  return 0.9461746957575601 * dx * dy * (7.0 * dz * dz - 1.0)

def HardcodedSH4n1(dx, dy, dz):
  # -0.75 * sqrt(5/(2pi)) * y * z * (7z^2-3)
  return -0.6690465435572892 * dy * dz * (7.0 * dz * dz - 3.0)

def HardcodedSH40(dx, dy, dz):
  # 3/16 * sqrt(1/pi) * (35z^4-30z^2+3)
  z2 = dz * dz
  return 0.10578554691520431 * (35.0 * z2 * z2 - 30.0 * z2 + 3.0)

def HardcodedSH4p1(dx, dy, dz):
  # -0.75 * sqrt(5/(2pi)) * x * z * (7z^2-3)
  return -0.6690465435572892 * dx * dz * (7.0 * dz * dz - 3.0)

def HardcodedSH4p2(dx, dy, dz):
  # 3/8 * sqrt(5/pi) * (x^2 - y^2) * (7z^2 - 1)
  return 0.47308734787878004 * (dx * dx - dy * dy) * (7.0 * dz * dz - 1.0)

def HardcodedSH4p3(dx, dy, dz):
  # -0.75 * sqrt(35/(2pi)) * x * z * (x^2 - 3y^2)
  return -1.7701307697799304 * dx * dz * (dx * dx - 3.0 * dy * dy)

def HardcodedSH4p4(dx, dy, dz):
  # 3/16*sqrt(35/pi) * (x^2 * (x^2 - 3y^2) - y^2 * (3x^2 - y^2))
  x2 = dx * dx
  y2 = dy * dy
  return 0.6258357354491761 * (x2 * (x2 - 3.0 * y2) - y2 * (3.0 * x2 - y2))


def EvalSH(l: int, m: int, dirs):
  """
  Args:
    dirs: array [..., 3]. works with torch/jnp/np
  Return:
    array [...]
  """
  if l <= kHardCodedOrderLimit:
    # Validate l and m here (don't do it generally since EvalSHSlow also
    # checks it if we delegate to that function).
    assert l >= 0, "l must be at least 0."
    assert -l <= m and m <= l, "m must be between -l and l."
    dx = dirs[..., 0]
    dy = dirs[..., 1]
    dz = dirs[..., 2]

    if l == 0:
      return HardcodedSH00(dx, dy, dz)
    elif l == 1:
      if m == -1:
        return HardcodedSH1n1(dx, dy, dz)
      elif m == 0:
        return HardcodedSH10(dx, dy, dz)
      elif m == 1:
        return HardcodedSH1p1(dx, dy, dz)
    elif l == 2:
      if m == -2:
        return HardcodedSH2n2(dx, dy, dz)
      elif m == -1:
        return HardcodedSH2n1(dx, dy, dz)
      elif m == 0:
        return HardcodedSH20(dx, dy, dz)
      elif m == 1:
        return HardcodedSH2p1(dx, dy, dz)
      elif m == 2:
        return HardcodedSH2p2(dx, dy, dz)
    elif l == 3:
      if m == -3:
        return HardcodedSH3n3(dx, dy, dz)
      elif m == -2:
            return HardcodedSH3n2(dx, dy, dz)
      elif m == -1:
            return HardcodedSH3n1(dx, dy, dz)
      elif m == 0:
            return HardcodedSH30(dx, dy, dz)
      elif m == 1:
            return HardcodedSH3p1(dx, dy, dz)
      elif m == 2:
            return HardcodedSH3p2(dx, dy, dz)
      elif m == 3:
            return HardcodedSH3p3(dx, dy, dz)
    elif l == 4:
      if m == -4:
        return HardcodedSH4n4(dx, dy, dz)
      elif m == -3:
        return HardcodedSH4n3(dx, dy, dz)
      elif m == -2:
        return HardcodedSH4n2(dx, dy, dz)
      elif m == -1:
        return HardcodedSH4n1(dx, dy, dz)
      elif m == 0:
        return HardcodedSH40(dx, dy, dz)
      elif m == 1:
        return HardcodedSH4p1(dx, dy, dz)
      elif m == 2:
        return HardcodedSH4p2(dx, dy, dz)
      elif m == 3:
        return HardcodedSH4p3(dx, dy, dz)
      elif m == 4:
        return HardcodedSH4p4(dx, dy, dz)

    # This is unreachable given the CHECK's above but the compiler can't tell.
    return None

  else:
    # Not hard-coded so use the recurrence relation (which will convert this
    # to spherical coordinates).
    # return EvalSHSlow(l, m, dx, dy, dz)
    raise NotImplementedError


def spherical_uniform_sampling(sample_count, device="cpu"):
  # See: https://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
  theta = torch.acos(2.0 * torch.rand([sample_count]) - 1.0)
  phi = 2.0 * math.pi * torch.rand([sample_count])
  return theta.to(device), phi.to(device)


def ProjectFunction(order: int, sperical_func: Callable, sample_count: int, device="cpu"):
  assert order >= 0, "Order must be at least zero."
  assert sample_count > 0, "Sample count must be at least one."

  # This is the approach demonstrated in [1] and is useful for arbitrary
  # functions on the sphere that are represented analytically.
  coeffs = torch.zeros([GetCoefficientCount(order)], dtype=torch.float32).to(device)

  # generate sample_count uniformly and stratified samples over the sphere
  # See http://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
  theta, phi = spherical_uniform_sampling(sample_count, device=device)
  dirs = spher2cart(theta, phi)

  # evaluate the analytic function for the current spherical coords
  func_value = sperical_func(dirs)

  # evaluate the SH basis functions up to band O, scale them by the
  # function's value and accumulate them over all generated samples
  for l in range(order + 1):  # end inclusive
    for m in range(-l, l + 1):  # end inclusive
      coeffs[GetIndex(l, m)] = sum(func_value * EvalSH(l, m, dirs))

  # scale by the probability of a particular sample, which is
  # 4pi/sample_count. 4pi for the surface area of a unit sphere, and
  # 1/sample_count for the number of samples drawn uniformly.
  weight = 4.0 * math.pi / sample_count
  coeffs *= weight
  return coeffs


def ProjectFunctionNeRF(order: int, sperical_func: Callable, batch_size: int, sample_count: int, device="cpu"):
  assert order >= 0, "Order must be at least zero."
  assert sample_count > 0, "Sample count must be at least one."
  C = 3  # rgb channels

  # This is the approach demonstrated in [1] and is useful for arbitrary
  # functions on the sphere that are represented analytically.
  coeffs = torch.zeros([batch_size, C, GetCoefficientCount(order)], dtype=torch.float32).to(device)

  # generate sample_count uniformly and stratified samples over the sphere
  # See http://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
  theta, phi = spherical_uniform_sampling(sample_count, device=device)
  dirs = spher2cart(theta, phi)

  # evaluate the analytic function for the current spherical coords
  func_value, others = sperical_func(dirs)  # [batch_size, sample_count, C]

  # evaluate the SH basis functions up to band O, scale them by the
  # function's value and accumulate them over all generated samples
  for l in range(order + 1):  # end inclusive
    for m in range(-l, l + 1):  # end inclusive
      coeffs[:, :, GetIndex(l, m)] = torch.einsum("bsc,s->bc", func_value, EvalSH(l, m, dirs))

  # scale by the probability of a particular sample, which is
  # 4pi/sample_count. 4pi for the surface area of a unit sphere, and
  # 1/sample_count for the number of samples drawn uniformly.
  weight = 4.0 * math.pi / sample_count
  coeffs *= weight
  return coeffs, others

def ProjectFunctionNeRFSparse(
    order: int,
    spherical_func: Callable,
    sample_count: int,
    device="cpu",
):
    assert order >= 0, "Order must be at least zero."
    assert sample_count > 0, "Sample count must be at least one."
    C = 3  # rgb channels

    # generate sample_count uniformly and stratified samples over the sphere
    # See http://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
    theta, phi = spherical_uniform_sampling(sample_count, device=device)
    dirs = spher2cart(theta, phi)  # [sample_count, 3]

    # evaluate the analytic function for the current spherical coords
    func_value, others = spherical_func(dirs)  # func_value [batch_size, sample_count, C]

    batch_size = func_value.shape[0]

    coeff_count = GetCoefficientCount(order)
    basis_vals = torch.empty(
        [sample_count, coeff_count], dtype=torch.float32
    ).to(device)

    # evaluate the SH basis functions up to band O, scale them by the
    # function's value and accumulate them over all generated samples
    for l in range(order + 1):  # end inclusive
        for m in range(-l, l + 1):  # end inclusive
            basis_vals[:, GetIndex(l, m)] = EvalSH(l, m, dirs)

    basis_vals = basis_vals.view(
           sample_count, coeff_count) # [sample_count, coeff_count]
    func_value = func_value.transpose(0, 1).reshape(
           sample_count, batch_size * C) # [sample_count, batch_size * C]
    soln = torch.lstsq(func_value, basis_vals).solution[:basis_vals.size(1)]
    soln = soln.T.reshape(batch_size, C, -1)
    return soln, others
 
