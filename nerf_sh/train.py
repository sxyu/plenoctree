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
"""Training script for Nerf."""

import os
# Get rid of ugly TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import functools
import gc
import time
from absl import app
from absl import flags
import flax
import flax.linen as nn
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import config
from jax import random
import jax.numpy as jnp
import numpy as np


from nerf_sh.nerf import datasets
from nerf_sh.nerf import models
from nerf_sh.nerf import utils
from nerf_sh.nerf.utils import host0_print as h0print

FLAGS = flags.FLAGS

utils.define_flags()
config.parse_flags_with_absl()


def train_step(model, rng, state, batch, lr):
    """One optimization step.

    Args:
      model: The linen model.
      rng: jnp.ndarray, random number generator.
      state: utils.TrainState, state of the model/optimizer.
      batch: dict, a mini-batch of data for training.
      lr: float, real-time learning rate.

    Returns:
      new_state: utils.TrainState, new training state.
      stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
      rng: jnp.ndarray, updated random number generator.
    """
    rng, key_0, key_1, key_2 = random.split(rng, 4)

    def loss_fn(variables):
        rays = batch["rays"]
        ret = model.apply(variables, key_0, key_1, rays, FLAGS.randomized)
        if len(ret) not in (1, 2):
            raise ValueError(
                "ret should contain either 1 set of output (coarse only), or 2 sets"
                "of output (coarse as ret[0] and fine as ret[1])."
            )

        if FLAGS.sparsity_weight > 0.0:
            rng, key = random.split(key_2)
            sp_points = random.uniform(key, (FLAGS.sparsity_npoints, 3), minval=-FLAGS.sparsity_radius, maxval=FLAGS.sparsity_radius)
            sp_rgb, sp_sigma = model.apply(variables, sp_points, method=model.eval_points_raw)
            del sp_rgb
            sp_sigma = nn.relu(sp_sigma)
            loss_sp = FLAGS.sparsity_weight * (1.0 - jnp.exp(- FLAGS.sparsity_length * sp_sigma).mean())
        else:
            loss_sp = 0.0

        # The main prediction is always at the end of the ret list.
        rgb, unused_disp, unused_acc = ret[-1]
        loss = ((rgb - batch["pixels"][Ellipsis, :3]) ** 2).mean()
        psnr = utils.compute_psnr(loss)
        if len(ret) > 1:
            # If there are both coarse and fine predictions, we compute the loss for
            # the coarse prediction (ret[0]) as well.
            rgb_c, unused_disp_c, unused_acc_c = ret[0]
            loss_c = ((rgb_c - batch["pixels"][Ellipsis, :3]) ** 2).mean()
            psnr_c = utils.compute_psnr(loss_c)
        else:
            loss_c = 0.0
            psnr_c = 0.0

        def tree_sum_fn(fn):
            return jax.tree_util.tree_reduce(
                lambda x, y: x + fn(y), variables, initializer=0
            )

        weight_l2 = tree_sum_fn(lambda z: jnp.sum(z ** 2)) / tree_sum_fn(
            lambda z: jnp.prod(jnp.array(z.shape))
        )

        stats = utils.Stats(
            loss=loss, psnr=psnr, loss_c=loss_c, loss_sp=loss_sp,
            psnr_c=psnr_c, weight_l2=weight_l2
        )
        return loss + loss_c + loss_sp + FLAGS.weight_decay_mult * weight_l2, stats

    (_, stats), grad = jax.value_and_grad(loss_fn, has_aux=True)(state.optimizer.target)
    grad = jax.lax.pmean(grad, axis_name="batch")
    stats = jax.lax.pmean(stats, axis_name="batch")
    new_optimizer = state.optimizer.apply_gradient(grad, learning_rate=lr)
    new_state = state.replace(optimizer=new_optimizer)
    return new_state, stats, rng


def main(unused_argv):
    rng = random.PRNGKey(20200823)
    # Shift the numpy random seed by host_id() to shuffle data loaded by different
    # hosts.
    np.random.seed(20201473 + jax.host_id())
    rng, key = random.split(rng)

    utils.update_flags(FLAGS)
    utils.check_flags(FLAGS, require_batch_size_div=True)

    utils.makedirs(FLAGS.train_dir)
    render_dir = os.path.join(FLAGS.train_dir, 'render')
    utils.makedirs(render_dir)

    # TEMP
    timings_file = open(os.path.join(FLAGS.train_dir, 'timings.txt'), 'a')
    from datetime import datetime
    def write_ts_now(step):
        timings_file.write(f"{step} {datetime.now().isoformat()}\n")
        timings_file.flush()
    write_ts_now(0)

    h0print('* Load train data')
    dataset = datasets.get_dataset("train", FLAGS)
    h0print('* Load test data')
    test_dataset = datasets.get_dataset("test", FLAGS)

    h0print('* Load model')
    model, state = models.get_model_state(key, FLAGS)

    learning_rate_fn = functools.partial(
        utils.learning_rate_decay,
        lr_init=FLAGS.lr_init,
        lr_final=FLAGS.lr_final,
        max_steps=FLAGS.max_steps,
        lr_delay_steps=FLAGS.lr_delay_steps,
        lr_delay_mult=FLAGS.lr_delay_mult,
    )

    train_pstep = jax.pmap(
        functools.partial(train_step, model),
        axis_name="batch",
        in_axes=(0, 0, 0, None),
        donate_argnums=(2,),
    )

    render_pfn = utils.get_render_pfn(model, randomized=FLAGS.randomized)

    # Compiling to the CPU because it's faster and more accurate.
    ssim_fn = jax.jit(functools.partial(utils.compute_ssim, max_val=1.0), backend="cpu")

    # Resume training a the step of the last checkpoint.
    init_step = state.optimizer.state.step + 1
    state = flax.jax_utils.replicate(state)

    if jax.host_id() == 0:
        summary_writer = tensorboard.SummaryWriter(FLAGS.train_dir)

    h0print('* Prefetch')
    # Prefetch_buffer_size = 3 x batch_size
    pdataset = flax.jax_utils.prefetch_to_device(dataset, 3)
    n_local_deices = jax.local_device_count()
    rng = rng + jax.host_id()  # Make random seed separate across hosts.
    keys = random.split(rng, n_local_deices)  # For pmapping RNG keys.
    gc.disable()  # Disable automatic garbage collection for efficiency.
    stats_trace = []


    reset_timer = True
    for step, batch in zip(range(init_step, FLAGS.max_steps + 1), pdataset):
        if reset_timer:
            t_loop_start = time.time()
            reset_timer = False
        lr = learning_rate_fn(step)
        state, stats, keys = train_pstep(keys, state, batch, lr)
        if jax.host_id() == 0:
            stats_trace.append(stats)
        if step % FLAGS.gc_every == 0:
            gc.collect()

        # Log training summaries. This is put behind a host_id check because in
        # multi-host evaluation, all hosts need to run inference even though we
        # only use host 0 to record results.
        if jax.host_id() == 0:
            if step % FLAGS.print_every == 0:
                summary_writer.scalar("train_loss", stats.loss[0], step)
                summary_writer.scalar("train_psnr", stats.psnr[0], step)
                summary_writer.scalar("train_loss_coarse", stats.loss_c[0], step)
                summary_writer.scalar("train_psnr_coarse", stats.psnr_c[0], step)
                if FLAGS.sparsity_weight > 0.0:
                    summary_writer.scalar("train_sparse_loss", stats.loss_sp[0], step)
                summary_writer.scalar("weight_l2", stats.weight_l2[0], step)
                avg_loss = np.mean(np.concatenate([s.loss for s in stats_trace]))
                avg_psnr = np.mean(np.concatenate([s.psnr for s in stats_trace]))
                stats_trace = []
                summary_writer.scalar("train_avg_loss", avg_loss, step)
                summary_writer.scalar("train_avg_psnr", avg_psnr, step)
                summary_writer.scalar("learning_rate", lr, step)
                steps_per_sec = FLAGS.print_every / (time.time() - t_loop_start)
                reset_timer = True
                rays_per_sec = FLAGS.batch_size * steps_per_sec
                summary_writer.scalar("train_steps_per_sec", steps_per_sec, step)
                summary_writer.scalar("train_rays_per_sec", rays_per_sec, step)
                precision = int(np.ceil(np.log10(FLAGS.max_steps))) + 1
                print(
                    ("{:" + "{:d}".format(precision) + "d}").format(step)
                    + f"/{FLAGS.max_steps:d}: "
                    + f"i_loss={stats.loss[0]:0.4f}, "
                    + f"avg_loss={avg_loss:0.4f}, "
                    + f"weight_l2={stats.weight_l2[0]:0.2e}, "
                    + f"lr={lr:0.2e}, "
                    + f"{rays_per_sec:0.0f} rays/sec"
                )
            if step % FLAGS.save_every == 0:
                print('* Saving')
                state_to_save = jax.device_get(jax.tree_map(lambda x: x[0], state))
                checkpoints.save_checkpoint(
                    FLAGS.train_dir, state_to_save, int(step), keep=200
                )

        # Test-set evaluation.
        if FLAGS.render_every > 0 and step % FLAGS.render_every == 0:
            # We reuse the same random number generator from the optimization step
            # here on purpose so that the visualization matches what happened in
            # training.
            h0print('\n* Rendering')
            t_eval_start = time.time()
            eval_variables = jax.device_get(
                jax.tree_map(lambda x: x[0], state)
            ).optimizer.target
            test_case = next(test_dataset)
            pred_color, pred_disp, pred_acc = utils.render_image(
                functools.partial(render_pfn, eval_variables),
                test_case["rays"],
                keys[0],
                FLAGS.dataset == "llff",
                chunk=FLAGS.chunk,
            )

            # Log eval summaries on host 0.
            if jax.host_id() == 0:
                write_ts_now(step)
                psnr = utils.compute_psnr(
                    ((pred_color - test_case["pixels"]) ** 2).mean()
                )
                ssim = ssim_fn(pred_color, test_case["pixels"])
                eval_time = time.time() - t_eval_start
                num_rays = jnp.prod(jnp.array(test_case["rays"].directions.shape[:-1]))
                rays_per_sec = num_rays / eval_time
                summary_writer.scalar("test_rays_per_sec", rays_per_sec, step)
                print(f"Eval {step}: {eval_time:0.3f}s., {rays_per_sec:0.0f} rays/sec")
                summary_writer.scalar("test_psnr", psnr, step)
                summary_writer.scalar("test_ssim", ssim, step)
                #  print(pred_color.shape, pred_disp.shape, pred_acc.shape,
                #          test_case["pixels"].shape)
                #  print(pred_color.dtype, pred_disp.dtype, pred_acc.dtype,
                #          test_case["pixels"].dtype)
                #  print(pred_color.min(), pred_color.max(),
                #        pred_disp.min(), pred_disp.max(),
                #        pred_acc.min(), pred_acc.max(),
                #        test_case['pixels'].min(), test_case['pixels'].max())
                #  0 1.  0.0 1.0  0.90906805 1.0000007  0.0 1.0

                #  (800, 800, 3) (800, 800, 1) (800, 800, 1) (800, 800, 3)
                #  float32 float32 float32 float32

                vis_list= [test_case["pixels"],
                           pred_color,
                           np.repeat(pred_disp, 3, axis=-1),
                           np.repeat(pred_acc, 3, axis=-1)]
                out_path = os.path.join(render_dir, '{:010}.png'.format(step))
                utils.save_img(np.hstack(vis_list), out_path)
                print(' Rendering saved to ', out_path)

                # I am saving rendering to disk instead of Tensorboard
                # Since Tensorboard begins to load very slowly when it has many images

                #  summary_writer.image("test_pred_color", pred_color, step)
                #  summary_writer.image("test_pred_disp", pred_disp, step)
                #  summary_writer.image("test_pred_acc", pred_acc, step)
                #  summary_writer.image("test_target", test_case["pixels"], step)

    if FLAGS.max_steps % FLAGS.save_every != 0:
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        checkpoints.save_checkpoint(
            FLAGS.train_dir, state, int(FLAGS.max_steps), keep=200
        )


if __name__ == "__main__":
    app.run(main)
