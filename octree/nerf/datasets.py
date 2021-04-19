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
"""Different datasets implementation plus a general port for all the datasets."""
INTERNAL = False  # pylint: disable=g-statement-before-imports
import json
import os
from os import path

if not INTERNAL:
    import cv2  # pylint: disable=g-import-not-at-top
import numpy as np
from PIL import Image
from tqdm import tqdm

from octree.nerf import utils


def get_dataset(split, args):
    return dataset_dict[args.dataset](split, args)


def convert_to_ndc(origins, directions, focal, w, h, near=1.0):
    """Convert a set of rays to NDC coordinates."""
    # Shift ray origins to near plane
    t = -(near + origins[Ellipsis, 2]) / directions[Ellipsis, 2]
    origins = origins + t[Ellipsis, None] * directions

    dx, dy, dz = tuple(np.moveaxis(directions, -1, 0))
    ox, oy, oz = tuple(np.moveaxis(origins, -1, 0))

    # Projection
    o0 = -((2 * focal) / w) * (ox / oz)
    o1 = -((2 * focal) / h) * (oy / oz)
    o2 = 1 + 2 * near / oz

    d0 = -((2 * focal) / w) * (dx / dz - ox / oz)
    d1 = -((2 * focal) / h) * (dy / dz - oy / oz)
    d2 = -2 * near / oz

    origins = np.stack([o0, o1, o2], -1)
    directions = np.stack([d0, d1, d2], -1)
    return origins, directions


class Dataset():
    """Dataset Base Class."""

    def __init__(self, split, args, prefetch=True):
        super(Dataset, self).__init__()
        self.split = split
        self._general_init(args)

    @property
    def size(self):
        return self.n_examples

    def _general_init(self, args):
        bbox_path = path.join(args.data_dir, 'bbox.txt')
        if os.path.isfile(bbox_path):
            self.bbox = np.loadtxt(bbox_path)[:-1]
        else:
            self.bbox = None
        self._load_renderings(args)


class Blender(Dataset):
    """Blender Dataset."""

    def _load_renderings(self, args):
        """Load images from disk."""
        if args.render_path:
            raise ValueError("render_path cannot be used for the blender dataset.")
        with utils.open_file(
            path.join(args.data_dir, "transforms_{}.json".format(self.split)), "r"
        ) as fp:
            meta = json.load(fp)
        images = []
        cams = []
        print(' Load Blender', args.data_dir, 'split', self.split)
        for i in tqdm(range(len(meta["frames"]))):
            frame = meta["frames"][i]
            fname = os.path.join(args.data_dir, frame["file_path"] + ".png")
            with utils.open_file(fname, "rb") as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.0
                if args.factor == 2:
                    [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
                    image = cv2.resize(
                        image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA
                    )
                elif args.factor > 0:
                    raise ValueError(
                        "Blender dataset only supports factor=0 or 2, {} "
                        "set.".format(args.factor)
                    )
            cams.append(frame["transform_matrix"])
            if args.white_bkgd:
                mask = image[..., -1:]
                image = image[..., :3] * mask + (1.0 - mask)
            else:
                image = image[..., :3]
            images.append(image)
        self.images = np.stack(images, axis=0)
        self.h, self.w = self.images.shape[1:3]
        self.resolution = self.h * self.w
        self.camtoworlds = np.stack(cams, axis=0).astype(np.float32)
        camera_angle_x = float(meta["camera_angle_x"])
        self.focal = 0.5 * self.w / np.tan(0.5 * camera_angle_x)
        self.n_examples = self.images.shape[0]


class LLFF(Dataset):
    """LLFF Dataset."""

    def _load_renderings(self, args):
        """Load images from disk."""
        args.data_dir = path.expanduser(args.data_dir)
        print(' Load LLFF', args.data_dir, 'split', self.split)
        # Load images.
        imgdir_suffix = ""
        if args.factor > 0:
            imgdir_suffix = "_{}".format(args.factor)
            factor = args.factor
        else:
            factor = 1
        imgdir = path.join(args.data_dir, "images" + imgdir_suffix)
        if not utils.file_exists(imgdir):
            raise ValueError("Image folder {} doesn't exist.".format(imgdir))
        imgfiles = [
            path.join(imgdir, f)
            for f in sorted(utils.listdir(imgdir))
            if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
        ]
        images = []
        for imgfile in imgfiles:
            with utils.open_file(imgfile, "rb") as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.0
                images.append(image)
        images = np.stack(images, axis=-1)

        # Load poses and bds.
        with utils.open_file(path.join(args.data_dir, "poses_bounds.npy"), "rb") as fp:
            poses_arr = np.load(fp)
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = poses_arr[:, -2:].transpose([1, 0])
        if poses.shape[-1] != images.shape[-1]:
            raise RuntimeError(
                "Mismatch between imgs {} and poses {}".format(
                    images.shape[-1], poses.shape[-1]
                )
            )

        # Update poses according to downsampling.
        poses[:2, 4, :] = np.array(images.shape[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1.0 / factor

        # Correct rotation matrix ordering and move variable dim to axis 0.
        poses = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1
        )
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        images = np.moveaxis(images, -1, 0)
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)

        # Rescale according to a default bd factor.
        scale = 1.0 / (bds.min() * 0.75)
        poses[:, :3, 3] *= scale
        bds *= scale

        # Recenter poses.
        poses = self._recenter_poses(poses)

        # Generate a spiral/spherical ray path for rendering videos.
        if args.spherify:
            poses = self._generate_spherical_poses(poses, bds)
            self.spherify = True
        else:
            self.spherify = False
        if not args.spherify and self.split == "test":
            self._generate_spiral_poses(poses, bds)

        # Select the split.
        i_test = np.arange(images.shape[0])[:: args.llffhold]
        i_train = np.array(
            [i for i in np.arange(int(images.shape[0])) if i not in i_test]
        )
        if self.split == "train":
            indices = i_train
        else:
            indices = i_test
        images = images[indices]
        poses = poses[indices]

        self.images = images
        self.camtoworlds = poses[:, :3, :4]
        self.focal = poses[0, -1, -1]
        self.h, self.w = images.shape[1:3]
        self.resolution = self.h * self.w
        if args.render_path:
            self.n_examples = self.render_poses.shape[0]
        else:
            self.n_examples = images.shape[0]

    def _generate_rays(self):
        """Generate normalized device coordinate rays for llff."""
        if self.split == "test":
            n_render_poses = self.render_poses.shape[0]
            self.camtoworlds = np.concatenate(
                [self.render_poses, self.camtoworlds], axis=0
            )

        super()._generate_rays()

        if not self.spherify:
            ndc_origins, ndc_directions = convert_to_ndc(
                self.rays.origins, self.rays.directions, self.focal, self.w, self.h
            )
            self.rays = utils.Rays(
                origins=ndc_origins,
                directions=ndc_directions,
                viewdirs=self.rays.viewdirs,
            )

        # Split poses from the dataset and generated poses
        if self.split == "test":
            self.camtoworlds = self.camtoworlds[n_render_poses:]
            split = [np.split(r, [n_render_poses], 0) for r in self.rays]
            split0, split1 = zip(*split)
            self.render_rays = utils.Rays(*split0)
            self.rays = utils.Rays(*split1)

    def _recenter_poses(self, poses):
        """Recenter poses according to the original NeRF code."""
        poses_ = poses.copy()
        bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
        c2w = self._poses_avg(poses)
        c2w = np.concatenate([c2w[:3, :4], bottom], -2)
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)
        poses = np.linalg.inv(c2w) @ poses
        poses_[:, :3, :4] = poses[:, :3, :4]
        poses = poses_
        return poses

    def _poses_avg(self, poses):
        """Average poses according to the original NeRF code."""
        hwf = poses[0, :3, -1:]
        center = poses[:, :3, 3].mean(0)
        vec2 = self._normalize(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate([self._viewmatrix(vec2, up, center), hwf], 1)
        return c2w

    def _viewmatrix(self, z, up, pos):
        """Construct lookat view matrix."""
        vec2 = self._normalize(z)
        vec1_avg = up
        vec0 = self._normalize(np.cross(vec1_avg, vec2))
        vec1 = self._normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m

    def _normalize(self, x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)

    def _generate_spiral_poses(self, poses, bds):
        """Generate a spiral path for rendering."""
        c2w = self._poses_avg(poses)
        # Get average pose.
        up = self._normalize(poses[:, :3, 1].sum(0))
        # Find a reasonable "focus depth" for this dataset.
        close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
        dt = 0.75
        mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
        focal = mean_dz
        # Get radii for spiral path.
        tt = poses[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        n_views = 120
        n_rots = 2
        # Generate poses for spiral path.
        render_poses = []
        rads = np.array(list(rads) + [1.0])
        hwf = c2w_path[:, 4:5]
        zrate = 0.5
        for theta in np.linspace(0.0, 2.0 * np.pi * n_rots, n_views + 1)[:-1]:
            c = np.dot(
                c2w[:3, :4],
                (
                    np.array(
                        [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]
                    )
                    * rads
                ),
            )
            z = self._normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
            render_poses.append(np.concatenate([self._viewmatrix(z, up, c), hwf], 1))
        self.render_poses = np.array(render_poses).astype(np.float32)[:, :3, :4]

    def _generate_spherical_poses(self, poses, bds):
        """Generate a 360 degree spherical path for rendering."""
        # pylint: disable=g-long-lambda
        p34_to_44 = lambda p: np.concatenate(
            [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
        )
        rays_d = poses[:, :3, 2:3]
        rays_o = poses[:, :3, 3:4]

        def min_line_dist(rays_o, rays_d):
            a_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
            b_i = -a_i @ rays_o
            pt_mindist = np.squeeze(
                -np.linalg.inv((np.transpose(a_i, [0, 2, 1]) @ a_i).mean(0))
                @ (b_i).mean(0)
            )
            return pt_mindist

        pt_mindist = min_line_dist(rays_o, rays_d)
        center = pt_mindist
        up = (poses[:, :3, 3] - center).mean(0)
        vec0 = self._normalize(up)
        vec1 = self._normalize(np.cross([0.1, 0.2, 0.3], vec0))
        vec2 = self._normalize(np.cross(vec0, vec1))
        pos = center
        c2w = np.stack([vec1, vec2, vec0, pos], 1)
        poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])
        rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
        sc = 1.0 / rad
        poses_reset[:, :3, 3] *= sc
        bds *= sc
        rad *= sc
        centroid = np.mean(poses_reset[:, :3, 3], 0)
        zh = centroid[2]
        radcircle = np.sqrt(rad ** 2 - zh ** 2)
        new_poses = []

        for th in np.linspace(0.0, 2.0 * np.pi, 120):
            camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
            up = np.array([0, 0, -1.0])
            vec2 = self._normalize(camorigin)
            vec0 = self._normalize(np.cross(vec2, up))
            vec1 = self._normalize(np.cross(vec2, vec0))
            pos = camorigin
            p = np.stack([vec0, vec1, vec2, pos], 1)
            new_poses.append(p)

        new_poses = np.stack(new_poses, 0)
        new_poses = np.concatenate(
            [
                new_poses,
                np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape),
            ],
            -1,
        )
        poses_reset = np.concatenate(
            [
                poses_reset[:, :3, :4],
                np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape),
            ],
            -1,
        )
        if self.split == "test":
            self.render_poses = new_poses[:, :3, :4]
        return poses_reset


class NSVF(Dataset):
    """NSVF Generic Dataset."""

    def _load_renderings(self, args):
        """Load images from disk."""
        if args.render_path:
            raise ValueError("render_path cannot be used for the NSVF dataset.")
        args.data_dir = path.expanduser(args.data_dir)
        K : np.ndarray = np.loadtxt(path.join(args.data_dir, "intrinsics.txt"))
        pose_files = sorted(os.listdir(path.join(args.data_dir, 'pose')))
        img_files = sorted(os.listdir(path.join(args.data_dir, 'rgb')))

        if self.split == 'train':
            pose_files = [x for x in pose_files if x.startswith('0_')]
            img_files = [x for x in img_files if x.startswith('0_')]
        elif self.split == 'val':
            pose_files = [x for x in pose_files if x.startswith('1_')]
            img_files = [x for x in img_files if x.startswith('1_')]
        elif self.split == 'test':
            test_pose_files = [x for x in pose_files if x.startswith('2_')]
            test_img_files = [x for x in img_files if x.startswith('2_')]
            if len(test_pose_files) == 0:
                test_pose_files = [x for x in pose_files if x.startswith('1_')]
                test_img_files = [x for x in img_files if x.startswith('1_')]
            pose_files = test_pose_files
            img_files = test_img_files

        images = []
        cams = []

        cam_trans = np.diag(np.array([1, -1, -1, 1], dtype=np.float32))

        assert len(img_files) == len(pose_files)
        print(' Load NSVF', args.data_dir, 'split', self.split, 'num_images', len(img_files))
        for img_fname, pose_fname in tqdm(zip(img_files, pose_files), total=len(img_files)):
            img_fname = path.join(args.data_dir, 'rgb', img_fname)
            with utils.open_file(img_fname, "rb") as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.0
            cam_mtx = np.loadtxt(path.join(args.data_dir, 'pose', pose_fname)) @ cam_trans
            cams.append(cam_mtx)  # C2W
            if image.shape[-1] == 4:
                # Alpha channel available
                if args.white_bkgd:
                    mask = image[..., -1:]
                    image = image[..., :3] * mask + (1.0 - mask)
                else:
                    image = image[..., :3]
            if args.factor > 1:
                [rsz_h, rsz_w] = [hw // args.factor for hw in image.shape[:2]]
                image = cv2.resize(
                    image, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA
                )

            images.append(image)
        self.images = np.stack(images, axis=0)
        self.n_examples, self.h, self.w = self.images.shape[:3]
        self.resolution = self.h * self.w
        self.camtoworlds = np.stack(cams, axis=0).astype(np.float32)
        # We assume fx and fy are same
        self.focal = (K[0, 0] + K[1, 1]) * 0.5
        if args.factor > 1:
            self.focal /= args.factor


dataset_dict = {
    "blender": Blender,
    "llff": LLFF,
    "nsvf": NSVF,
}
