# PlenOctrees Official Repo: NeRF-SH training and conversion

This repository contains code to train NeRF-SH and
to extract the PlenOctree, constituting part of the code release for:

PlenOctrees for Real Time Rendering of Neural Radiance Fields<br>
Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, Angjoo Kanazawa

https://alexyu.net/plenoctrees

```
@inproceedings{yu2021plenoctrees,
      title={{PlenOctrees} for Real-time Rendering of Neural Radiance Fields},
      author={Alex Yu and Ruilong Li and Matthew Tancik and Hao Li and Ren Ng and Angjoo Kanazawa},
      year={2021},
      booktitle={ICCV},
}
```

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
conda install tqdm
pip install -r requirements.txt
```

[Optional] Install GPU and TPU support for Jax. This is useful for NeRF-SH training.
Remember to **change cuda110 to your CUDA version**, e.g. cuda102 for CUDA 10.2.
```
pip install --upgrade jax jaxlib==0.1.65+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

## NeRF-SH Training

We release our trained NeRF-SH models as well as converted plenoctrees at 
[Google Drive](https://drive.google.com/drive/folders/1J0lRiDn_wOiLVpCraf6jM7vvCwDr9Dmx?usp=sharing). 
You can also use the following commands to reproduce the NeRF-SH models.

Training and evaluation on the **NeRF-Synthetic dataset** ([Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)):
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


Training and evaluation on **TanksAndTemple dataset** 
([Download Link](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip)) from the [NSVF](https://github.com/facebookresearch/NSVF) paper:
```
export DATA_ROOT=./data/TanksAndTemple/
export CKPT_ROOT=./data/Plenoctree/checkpoints/tt_sh25/
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
NeRF-SH models trained/downloaded and placed at `./data/Plenoctree/checkpoints/{syn_sh16, tt_sh25}/`. 
Also make sure you have the training data placed at 
`./data/NeRF/nerf_synthetic` and/or `./data/TanksAndTemple`.

To reproduce our results in the paper, you can simplly run:
```
# NeRF-Synthetic dataset
python -m octree.task_manager octree/config/syn_sh16.json --gpus="0 1 2 3"

# TanksAndTemple dataset
python -m octree.task_manager octree/config/tt_sh25.json --gpus="0 1 2 3"
```
The above command will parallel all scenes in the dataset across the gpus you set. The json files 
contain dedicated hyper-parameters towards better performance (PSNR, SSIM, LPIPS). So in this setting, a 24GB GPU is
needed for each scene and in averange the process takes about 15 minutes to finish. The converted plenoctree
will be saved to `./data/Plenoctree/checkpoints/{syn_sh16, tt_sh25}/$SCENE/octrees/`.


Below is a more straight-forward script for demonstration purpose:
```
export DATA_ROOT=./data/NeRF/nerf_synthetic/
export CKPT_ROOT=./data/Plenoctree/checkpoints/syn_sh16
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

A vanilla trained NeRF can also be converted to a plenoctree for fast inference. To mimic the 
view-independency propertity as in a NeRF-SH model, we project the vanilla NeRF model to SH basis functions
by sampling view directions for every points in the space. Though this makes converting vanilla NeRF to
a plenoctree possible, the projection process inevitability loses the quality of the model, even with a large amount 
of sampling view directions (which takes hours to finish). So we recommend to just directly train a NeRF-SH model end-to-end.

Below is a example of projecting a trained vanilla NeRF model from 
[JaxNeRF repo](https://github.com/google-research/google-research/tree/master/jaxnerf) 
([Download Link](http://storage.googleapis.com/gresearch/jaxnerf/jaxnerf_pretrained_models.zip)) to a plenoctree. 
After extraction, you can optimize & evaluate & compress the plenoctree just like usual:
```
export DATA_ROOT=./data/NeRF/nerf_synthetic/ 
export CKPT_ROOT=./data/JaxNeRF/jaxnerf_models/blender/ 
export SCENE=drums
export CONFIG_FILE=nerf_sh/config/misc/proj

python -m octree.extraction \
    --train_dir $CKPT_ROOT/$SCENE/ --is_jaxnerf_ckpt \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/octrees/tree.npz \
    --projection_samples 100 \
    --radius 1.3
```
Note `--projection_samples` controls how many sampling view directions are used. More sampling view directions give better
projection quality but takes longer time to finish. For example, for the `drums` scene 
in the NeRF-Synthetic dataset, `100 / 10000` sampling view directions takes about `2 mins / 2 hours` to finish the plenoctree extraction. 
It produce *raw* plenoctrees with `PSNR=22.49 / 23.84` (before optimization). Note that extraction from a NeRF-SH model produce 
a *raw* plenoctree with `PSNR=25.01`.

### List of possible improvements

In the interst reproducibility, the parameters used in the paper are also used here.
For future work we recommend trying the changes in mip-NeRF <https://jonbarron.info/mipnerf/> 
for improved stability and quality:

- Centered pixels (+ 0.5 on x, y) when generating rays
- Use shifted SoftPlus instead of ReLU for density (including for octree optimization)
- Pad the RGB sigmoid output (avoid low gradient region near 0/1 color)
- Multi-scale training from mip-NeRF


