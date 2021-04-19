export DATA_ROOT=./data/NeRF/nerf_synthetic/
export CKPT_ROOT=./checkpoints/plenoctree/
export SCENE=chair
export CONFIG_FILE=nerf_sh/config/blender

python -m nerf_sh.train \
    --train_dir $CKPT_ROOT/$SCENE/ \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ 
