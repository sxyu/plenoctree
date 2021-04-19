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

[Optional] Install GPU and TPU support for Jax. This is useful for NeRF-SH training only.
Remember to **change cuda110 to your CUDA version**, e.g. cuda102 for CUDA 10.2.
```
pip install --upgrade jax jaxlib==0.1.65+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

## NeRF-SH Training

We release our trained NeRF-SH models at [here](). You can also use the following commands to reproduce the results.

Training and evaluation on [NeRF-Synthetic dataset]():
```
export DATA_ROOT=./data/NeRF/nerf_synthetic/
export CKPT_ROOT=./data/Plenoctree/checkpoints/syn_sh16/
export SCENE=chair
export CONFIG_FILE=nerf_sh/config/blender

python -m nerf_sh.train \
    --train_dir $CKPT_ROOT/$SCENE/ \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/

python -m nerf_sh.eval \
    --chunk 4096 \
    --train_dir $CKPT_ROOT/$SCENE/ \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/
```
Note for `SCENE=mic`, we adopt a warmup learning rate schedule (`--lr_delay_steps 50000 --lr_delay_mult 0.01`) to avoid unstable initialization.


Training and evaluation on [TanksAndTemple dataset]():
```
export DATA_ROOT=./data/TanksAndTemple/
export CKPT_ROOT=./data/PlenOctree/checkpoints/tt_sh25/
export SCENE=Barn
export CONFIG_FILE=nerf_sh/config/tt

python -m nerf_sh.train \
    --train_dir $CKPT_ROOT/$SCENE/ \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/

python -m nerf_sh.eval \
    --chunk 4096 \
    --train_dir $CKPT_ROOT/$SCENE/ \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/
```

## PlenOctrees Conversion and Optimization

Before converting the NeRF-SH models into plenoctrees, you should already have the 
NeRF-SH models trained and placed at `./data/PlenOctree/checkpoints/{syn_sh16, tt_sh25}/`. 
Also make sure you have the training data placed at `./data/{NeRF/nerf_synthetic, TanksAndTemple}`.

To reproduce our results in the paper, you can simplly run:
```
# NeRF-Synthetic dataset
python -m octree.task_manager octree/config/syn_sh16.json --gpus=0,1,2,3

# TanksAndTemple dataset
python -m octree.task_manager octree/config/tt_sh25.json --gpus=0,1,2,3
```
The above command will parallel all scenes in the dataset across the gpus you set. The json files 
contain dedicated hyper-parameters towards better performance (PSNR, SSIM, LPIPS). So in this setting, a 32GB GPU is
needed for each scene and in averange the process takes about 15 minutes to finish. The converted plenoctree
will be saved to `./data/PlenOctree/checkpoints/{syn_sh16, tt_sh25}/$SCENE/octrees/`.


Below is a more straight-forward script for demonstration purpose:
```
export DATA_ROOT=./data/NeRF/nerf_synthetic/
export CKPT_ROOT=./data/PlenOctree/checkpoints/syn_sh16
export SCENE=chair
export CONFIG_FILE=nerf_sh/config/blender

python -m octree.extraction \
    --train_dir $CKPT_ROOT/$SCENE/ --is_jaxnerf_ckpt \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/octrees/tree.npz

python -m octree.optimization \
    --input $CKPT_ROOT/$SCENE/tree.npz \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/octrees/tree_opt.npz

python -m octree.evaluation \
    --input $CKPT_ROOT/$SCENE/octrees/tree_opt.npz \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/

# [Optional] Only used for in-browser viewing.
python -m octree.compression \
    $CKPT_ROOT/$SCENE/octrees/tree_opt.npz \
    --out_dir $CKPT_ROOT/$SCENE/ \
    --overwrite
```

## MISC

### Project Vanilla NeRF to PlenOctree

(TODO)
