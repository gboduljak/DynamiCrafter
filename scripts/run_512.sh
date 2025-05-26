version=512
seed=42

ckpt=/scratch/shared/beegfs/gabrijel/hf/models--Doubiiu--DynamiCrafter_512/snapshots/9eb874455b7903c0244fe7bf53ca21cb979cd05a/model.ckpt 
config=configs/inference_512_v1.0.yaml

prompt_dir=prompts/kubric_512/
res_dir="results"

FS=12 ## This model adopts FPS=24, range recommended: 15-30 (smaller value -> larger motion)

CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/inference.py \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--text_input \
--video_length 24 \
--frame_stride ${FS} \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae