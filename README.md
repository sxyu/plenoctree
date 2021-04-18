# PlenOctrees Official Repo: NeRF-SH training and conversion

This repository contains code to train NeRF-SH and
to extract the PlenOctree, constituting part of the code release for:

PlenOctrees for Real Time Rendering of Neural Radiance Fields<br>
Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, Angjoo Kanazawa

https://alexyu.net/plenoctrees

Please see the following repository for our C++ PlenOctrees volume renderer:
<https://github.com/sxyu/volrend>

## Setup

Please use conda for a replicable environment.
```
conda env create -f environment.yml
conda activate plenoctree
pip install --upgrade pip
```

Or you can install the dependencies manually by:
```
conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
conda install svox tqdm
pip install tensorboard imageio imageio-ffmpeg ipdb lpips jax jaxlib flax opencv-python Pillow pyyaml tensorflow pymcubes
```

[Optional] Install GPU and TPU support for Jax. Only useful for NeRF-SH training.
Remember to **change cuda110 to your CUDA version**, e.g. cuda102 for CUDA 10.2.
```
pip install --upgrade jax jaxlib==0.1.65+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

## NeRF-SH Training

## PlenOctrees Conversion

```
export DATA_ROOT=$HOME/data/NeRF/nerf_synthetic/
export CKPT_ROOT=$HOME/checkpoints/plenoctree/
export SCENE=chair 
export CONFIG_FILE=nerf_sh/config/blender.conf
```

### From NeRF-SH

An example command line to extract the plenoctree from a trained NeRF-SH model:
```
python octree/extract_octree.py \
    --train_dir $CKPT_ROOT/$SCENE/ --is_jaxnerf_ckpt \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/tree.npz \
```

An example command line to optimize (finetune) the extracted plenoctree:
```
python octree/optimize_octree.py \
    --input checkpoints/chair/octrees/tree.npz \
    --config configs/sh \
    --data_dir <DATA_ROOT>/nerf_synthetic/chair/ \
    --output checkpoints/chair/octrees/tree_opt.npz
```

An example command line to evaluate the plenoctree:
```
python octree/eval_octree.py \
    --input checkpoints/chair/octrees/tree_opt.npz \
    --config configs/sh \
    --data_dir <DATA_ROOT>/nerf_synthetic/chair/
```

### From NeRF (Projection)

An example command line to project a trained vanilla NeRF to SH and extract the plenoctree.
```
python octree/extract_octree.py \
    --train_dir checkpoints_nerf/chair/ --is_jaxnerf_ckpt \
    --config configs/sh \
    --data_dir <DATA_ROOT>/nerf_synthetic/chair/ \
    --output checkpoints/chair/octrees/tree.npz \
    --projection_samples 10000
```

### Optimization

### Evaluation
