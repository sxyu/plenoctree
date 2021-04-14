# PlenOctrees Official Repo: NeRF-SH training and conversion

This repository contains code to train NeRF-SH and
to extract the PlenOctree, constituting part of the code release for:

PlenOctrees for Real Time Rendering of Neural Radiance Fields<br>
Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, Angjoo Kanazawa

https://alexyu.net/plenoctrees

Please see the following repository for our C++ PlenOctrees volume renderer:
<https://github.com/sxyu/volrend>

## Setup

Please use conda.
```
conda env create -f environment.yml
conda activate plenoctree
pip install --upgrade pip
```

[Optional] Install GPU and TPU support for Jax.
Remember to **change cuda102 to your CUDA version**, e.g. cuda110 for CUDA 11.0.
```
pip install --upgrade jax jaxlib==0.1.59+cuda102 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

## NeRF-SH Training


## PlenOctrees Conversion

### From NeRF-SH

### From NeRF (Projection)

### Optimization

### Evaluation
