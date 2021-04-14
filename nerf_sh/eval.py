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
"""Evaluation script for Nerf."""

import os
# Get rid of ugly TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import functools
from os import path

from absl import app
from absl import flags
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import random
import numpy as np

from nerf_sh.nerf import datasets
from nerf_sh.nerf import models
from nerf_sh.nerf import utils

FLAGS = flags.FLAGS

utils.define_flags()


def main(unused_argv):
    rng = random.PRNGKey(20200823)
    rng, key = random.split(rng)

    utils.update_flags(FLAGS)
    utils.check_flags(FLAGS)

    dataset = datasets.get_dataset("test", FLAGS)
    model, state = models.get_model_state(key, FLAGS, restore=False)

    # Rendering is forced to be deterministic even if training was randomized, as
    # this eliminates "speckle" artifacts.
    render_pfn = utils.get_render_pfn(model, randomized=False)

    # Compiling to the CPU because it's faster and more accurate.
    ssim_fn = jax.jit(functools.partial(utils.compute_ssim, max_val=1.0), backend="cpu")

    last_step = 0
    out_dir = path.join(
        FLAGS.train_dir, "path_renders" if FLAGS.render_path else "test_preds"
    )
    if not FLAGS.eval_once:
        summary_writer = tensorboard.SummaryWriter(path.join(FLAGS.train_dir, "eval"))
    while True:
        print('Loading model')
        state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)
        step = int(state.optimizer.state.step)
        if step <= last_step:
            continue
        if FLAGS.save_output and (not utils.isdir(out_dir)):
            utils.makedirs(out_dir)
        psnrs = []
        ssims = []
        if not FLAGS.eval_once:
            showcase_index = np.random.randint(0, dataset.size)
        for idx in range(dataset.size):
            print(f"Evaluating {idx+1}/{dataset.size}")
            batch = next(dataset)
            if idx % FLAGS.approx_eval_skip != 0:
                continue
            pred_color, pred_disp, pred_acc = utils.render_image(
                functools.partial(render_pfn, state.optimizer.target),
                batch["rays"],
                rng,
                FLAGS.dataset == "llff",
                chunk=FLAGS.chunk,
            )
            if jax.host_id() != 0:  # Only record via host 0.
                continue
            if not FLAGS.eval_once and idx == showcase_index:
                showcase_color = pred_color
                showcase_disp = pred_disp
                showcase_acc = pred_acc
                if not FLAGS.render_path:
                    showcase_gt = batch["pixels"]
            #  if not FLAGS.render_path:
            #      psnr = utils.compute_psnr(((pred_color - batch["pixels"]) ** 2).mean())
            #      ssim = ssim_fn(pred_color, batch["pixels"])
            #      print(f"PSNR = {psnr:.4f}, SSIM = {ssim:.4f}")
            #      psnrs.append(float(psnr))
            #      ssims.append(float(ssim))
            if FLAGS.save_output:
                utils.save_img(pred_color, path.join(out_dir, "{:03d}.png".format(idx)))
                utils.save_img(
                    pred_disp[Ellipsis, 0],
                    path.join(out_dir, "disp_{:03d}.png".format(idx)),
                )
        if (not FLAGS.eval_once) and (jax.host_id() == 0):
            summary_writer.image("pred_color", showcase_color, step)
            summary_writer.image("pred_disp", showcase_disp, step)
            summary_writer.image("pred_acc", showcase_acc, step)
            #  if not FLAGS.render_path:
            #      summary_writer.scalar("psnr", np.mean(np.array(psnrs)), step)
            #      summary_writer.scalar("ssim", np.mean(np.array(ssims)), step)
            #      summary_writer.image("target", showcase_gt, step)
        #  if FLAGS.save_output and (not FLAGS.render_path) and (jax.host_id() == 0):
        #      with utils.open_file(path.join(out_dir, "psnr.txt"), "w") as f:
        #          f.write("{}".format(np.mean(np.array(psnrs))))
        #      with utils.open_file(path.join(out_dir, "ssim.txt"), "w") as f:
        #          f.write("{}".format(np.mean(np.array(ssims))))
        #      with utils.open_file(path.join(out_dir, f"psnrs_{step}.txt"), "w") as f:
        #          f.write(" ".join([str(v) for v in psnrs]))
        #      with utils.open_file(path.join(out_dir, f"ssims_{step}.txt"), "w") as f:
        #          f.write(" ".join([str(v) for v in ssims]))
        if FLAGS.eval_once:
            break
        if int(step) >= FLAGS.max_steps:
            break
        last_step = step


if __name__ == "__main__":
    app.run(main)
