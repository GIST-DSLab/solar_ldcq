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
CUDA_VISIBLE_DEVICES=0 python ./collect_offline_q_learning_dataset.py \
--env ARCLE \
--solar_dir /home/jovyan/beomi/yunho/ldcq_arc_working/ARC_Single/segment/train.74dd1130-half.s10.H5.26.01.25 \
--data_dir /home/jovyan/beomi/yunho/ldcq_arc_working/LDCQ_for_SOLAR/data/gpu0_02.26 \
--checkpoint_dir /home/jovyan/beomi/yunho/ldcq_arc_working/LDCQ_for_SOLAR/checkpoints/gpu0_02.26 \
--skill_model_filename gpu0_skill_model_ARCLE_02.26_400_.pth \
--diffusion_model_filename gpu0_skill_model_ARCLE_02.26_400__diffusion_prior_best.pt \
--num_diffusion_samples 100 \
--num_prior_samples 100 \
--diffusion_steps 500 \
--a_dim 36 \
--z_dim 256 \
--h_dim 512 \
--skill_model_diffusion_steps 100 \
--train_diffusion_prior 1 \
--conditional_prior 1 \
--normalize_latent 0 \
--batch_size 256 \
--max_grid_size 10  \
--gamma 0.7 \
--horizon 5 \
--use_in_out 1 \
--use_ddim 1 \
--ddim_steps 100 \
--ddim_eta 1.0 \
--ddim_discr uniform \
--noise_temperature 1.0 \
--use_enhanced_pair_encoding 0 \
--use_shared_grid_embedding 0 \
--use_split_pair_trajectory_encoding 0 \
--use_direct_output_predictor 0 \
--use_direct_output_for_diffusion 0 \
--encoder_type gru \
--disable_pair_encoding 0 \
--use_concept_guidance 0 \
--use_discrete_concepts 0 \
--use_cfg_for_concept 0 \
--cfg_weight 0.0 \
--use_concept_in_encoder 0 \
--num_concepts 0 \
--state_decoder_type mlp


:<<"OPTIONS"
explanation of arguments
-env: RL environment. If you change this, the data type and functions are all changed. 
-solar_dir: train dataset directory.
-data_dir: diffusion model 학습하는 데 필요한 데이터를 저장하는 directory.
-checkpoint_dir: vae 모델 저장된 directory.
-skill_model_filename: vae 모델 파일 이름. pth 확장자로 된 것.
-diffusion_model_filename: diffusion 모델 파일 이름. checkpoint에 같이 저장된 pt 확장자 파일 중 선택.
-num_diffusion_samples, num_prior_samples sample 몇개 뽑을 지인데 diffusion 썻냐 아니냐 차이에 따라 들어감. default는 500.
-diffusion_steps: 3번 diffusion model 학습에 사용한 diffusion model.
-horizon: step length of segment trace
-a_dim: operation 갯수 ARCLE 기준 0~34, 35는 None
-z_dim: size of latent
-h_dim: hidden layer. usually 2*z_dim
-skill_model_diffusion_steps 1~2번에서 사용한 skill model 학습시 사용한 diffusion step.
-train_diffusion_prior: vae 학습 시 diffusion prior도 같이 학습 시킬것인지. true로 하는 것이 vae 학습에 도움이 된다고 함.
-conditional_prior: vae에서 prior모듈을 별도로 학습시킬 것인지. default는 true.
-normalize_latent: latent normalize할 것인지. default는 true.
-gamma: discount factor.
-max_grid_size: maximum grid dim h
-use_ddim: DDIM 샘플링 사용 여부 (0=DDPM, 1=DDIM). 기본값: 0
-ddim_steps: DDIM 샘플링 스텝 수 (작을수록 빠름). 기본값: 50  
-ddim_eta: DDIM 확률성 조절 (0=deterministic, 1=DDPM-like). 기본값: 0.0
-ddim_discr: DDIM 타임스텝 분할 방식 (uniform/quad). 기본값: uniform
OPTIONS