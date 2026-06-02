echo "=================================="
echo "Script: $0"
echo "Started at: $(TZ='Asia/Seoul' date)"
echo "=================================="
echo ""
echo "Script contents:"
cat "$0"
echo ""
echo "=================================="
echo "Execution begins:"
echo "=================================="
CUDA_VISIBLE_DEVICES=0 python ./train_diffusion.py \
--env ARCLE \
--solar_dir /home/jovyan/beomi/yunho/ldcq_arc_working/ARC_Single/segment/train.74dd1130-half.s10.H5.26.01.25 \
--data_dir /home/jovyan/beomi/yunho/ldcq_arc_working/LDCQ_for_SOLAR/data/gpu0_02.26 \
--checkpoint_dir /home/jovyan/beomi/yunho/ldcq_arc_working/LDCQ_for_SOLAR/checkpoints/gpu0_02.26 \
--skill_model_filename gpu0_skill_model_ARCLE_02.26_400_.pth \
--n_epoch 400 \
--save_cycle 10 \
--diffusion_steps 500 \
--gpu_name gpu0 \
--s_dim 512 \
--batch_size 64 \
--max_grid_size 10 \
--use_in_out 1 \
--use_enhanced_pair_encoding 0 \
--use_shared_grid_embedding 0 \
--disable_pair_encoding 0 \
--use_split_pair_trajectory_encoding 0 \
--normalize_latent 0 \
--use_concept_guidance 0 \
--use_discrete_concepts 0 \
--use_cfg_for_concept 0 \
--cfg_weight 0.0 \
--use_concept_in_encoder 0 \
--num_concepts 0 \
--concept_scale 0.0 \
--use_amp 1


:<<"OPTIONS"
explanation of arguments
-env: RL environment. If you change this, the data type and functions are all changed. 
-data_dir: diffusion model 학습하는 데 필요한 데이터를 저장하는 directory.
-checkpoint_dir: vae 모델 저장된 directory.
-skill_model_filename: vae 모델 파일 이름. pth 확장자로 된 것.
-diffusion_steps: diffusion model 학습 시 diffusion step. 앞의 이전단계들의 diffusion step과는 다른 것.
-s_dim: state embedding layer usaually same with h_dim.
-max_grid_size: maximum grid dim h
OPTIONS