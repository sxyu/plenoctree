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
"""
Simple video generation script (for 360 Blender scene only)
"""
import os
# Get rid of ugly TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from absl import app
from absl import flags

import jax
from jax import config
from jax import random
import jax.numpy as jnp
import numpy as np
import flax
from flax.training import checkpoints

import functools

from nerf_sh.nerf import models
from nerf_sh.nerf import utils
from nerf_sh.nerf.utils import host0_print as h0print

import imageio

FLAGS = flags.FLAGS

utils.define_flags()

flags.DEFINE_float(
    "elevation",
    -30.0,
    "Elevation angle (negative is above)",
)
flags.DEFINE_integer(
    "num_views",
    40,
    "The number of views to generate.",
)
flags.DEFINE_integer(
    "height",
    800,
    "The size of images to generate.",
)
flags.DEFINE_integer(
    "width",
    800,
    "The size of images to generate.",
)
flags.DEFINE_float(
    "camera_angle_x",
    0.7,
    "The camera angle in rad in x direction (used to get focal length).",
    short_name='A',
)
flags.DEFINE_string(
    "intrin",
    None,
    "Intrinsics file. If set, overrides camera_angle_x",
)
flags.DEFINE_float(
    "radius",
    4.0,
    "Radius to origin of camera path.",
)
flags.DEFINE_integer(
    "fps",
    20,
    "FPS of generated video",
)
flags.DEFINE_integer(
    "up_axis",
    1,
    "up axis for camera views; 1-6: Z up/Z down/Y up/Y down/X up/X down; " +
    "same effect as pressing number keys in volrend",
)
flags.DEFINE_string(
    "write_poses",
    None,
    "Specify to write poses to given file (4N x 4), does not write poses else",
)

config.parse_flags_with_absl()

def main(unused_argv):
    rng = random.PRNGKey(20200823)

    utils.update_flags(FLAGS)
    utils.check_flags(FLAGS, require_data=False)

    rng, key = random.split(rng)

    h0print('* Generating poses')
    render_poses = np.stack(
        [
            utils.pose_spherical(angle, FLAGS.elevation, FLAGS.radius, FLAGS.up_axis - 1)
            for angle in np.linspace(-180, 180, FLAGS.num_views + 1)[:-1]
        ],
        0,
    )  # (NV, 4, 4)

    if FLAGS.write_poses:
        np.savetxt(FLAGS.write_poses, render_poses.reshape(-1, 4))
        print('Saved poses to', FLAGS.write_poses)

    h0print('* Generating rays')
    focal = 0.5 * FLAGS.width / np.tan(0.5 * FLAGS.camera_angle_x)

    if FLAGS.intrin is not None:
        print('Load focal length from intrin file')
        K : np.ndarray = np.loadtxt(FLAGS.intrin)
        focal = (K[0, 0] + K[1, 1]) * 0.5

    rays = utils.generate_rays(FLAGS.width, FLAGS.height, focal, render_poses)

    h0print('* Creating model')
    model, state = models.get_model_state(key, FLAGS)
    render_pfn = utils.get_render_pfn(model, randomized=False)

    h0print('* Rendering')

    vid_name = "e{:03}".format(int(-FLAGS.elevation * 10))
    video_dir = os.path.join(FLAGS.train_dir, 'video', vid_name)
    frames_dir = os.path.join(video_dir, 'frames')
    h0print(' Saving to', video_dir)
    utils.makedirs(frames_dir)

    frames = []
    for i in range(FLAGS.num_views):
        h0print(f'** View {i+1}/{FLAGS.num_views} = {i / FLAGS.num_views * 100}%')
        pred_color, pred_disp, pred_acc = utils.render_image(
            functools.partial(render_pfn, state.optimizer.target),
            utils.to_device(utils.namedtuple_map(lambda x: x[i], rays)),
            rng,
            FLAGS.dataset == "llff",
            chunk=FLAGS.chunk,
        )
        if jax.host_id() == 0:
            utils.save_img(pred_color, os.path.join(frames_dir, f'{i:04}.png'))
            frames.append(np.array(pred_color))

    if jax.host_id() == 0:
        frames = np.stack(frames)
        vid_path = os.path.join(video_dir, "video.mp4")
        print('* Writing video', vid_path)
        imageio.mimwrite(
            vid_path, (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8),
            fps=FLAGS.fps, quality=8
        )
        print('* Done')

if __name__ == "__main__":
    app.run(main)
