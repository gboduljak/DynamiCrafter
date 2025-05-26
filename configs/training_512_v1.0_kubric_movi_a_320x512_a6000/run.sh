# args
name="training_512_v1.0_kubric_movi_a_320x512_a6000"
config_file=configs/${name}/config.yaml
# save root dir for logs, checkpoints, tensorboard record, etc.
save_root="/scratch/shared/beegfs/gabrijel/dynamicrafter"

mkdir -p $save_root/$name

## run
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch \
--nproc_per_node=2 \
--nnodes=1 \
--master_addr=127.0.0.1 \
--master_port=12352 \
--node_rank=0 \
./main/trainer.py \
--base $config_file \
--train \
--name $name \
--logdir $save_root \
--devices 2 \
lightning.trainer.num_nodes=1