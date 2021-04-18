export DATA_ROOT=$HOME/data/NeRF/nerf_synthetic/
export CKPT_ROOT=$HOME/checkpoints/plenoctree/
export SCENE=chair
export CONFIG_FILE=config/blender

python -m nerf_sh.eval \
    --chunk 4096 \
    --train_dir $CKPT_ROOT/$SCENE/ \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ 
