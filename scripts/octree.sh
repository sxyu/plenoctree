export DATA_ROOT=$HOME/data/NeRF/nerf_synthetic/
export CKPT_ROOT=$HOME/checkpoints/plenoctree/
export SCENE=chair
export CONFIG_FILE=config/blender 

python -m octree.extraction \
    --train_dir $CKPT_ROOT/$SCENE/ --is_jaxnerf_ckpt \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/tree.npz
