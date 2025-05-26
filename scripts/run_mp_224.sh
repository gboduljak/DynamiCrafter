version=512
seed=42

ckpt=/scratch/shared/beegfs/gabrijel/hf/models--Doubiiu--DynamiCrafter_512/snapshots/9eb874455b7903c0244fe7bf53ca21cb979cd05a/model.ckpt 
config=configs/inference_512_v1.0.yaml

prompt_dir=prompts/kubric_224/
res_dir="results"
name=kubric_224_dynamicrafter_512_mp_seed${seed}


H=320
FS=12


## multi-cond CFG: the <unconditional_guidance_scale> is s_txt, <cfg_img> is s_img
#--multiple_cond_cfg --cfg_img 7.5
#--loop

## inference using single node with multi-GPUs:

CUDA_VISIBLE_DEVICES=2 python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=23456 --node_rank=0 \
scripts/evaluation/ddp_wrapper.py \
--module 'inference' \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height ${H} --width 512 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--video_length 24 \
--frame_stride ${FS} \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae