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
CUDA_VISIBLE_DEVICES=0 python ./plan_skills_diffusion_ARCLE.py \
--env ARCLE/O2ARCv2Env-v0 \
--test_solar_dir /five_task/whole_2:1.01.25/test \
--checkpoint_dir /LDCQ_for_SOLAR/checkpoints/gpu0_02.04 \
--skill_model_filename gpu0_skill_model_ARCLE_02.04_400_.pth \
--diffusion_model_filename gpu0_skill_model_ARCLE_02.04_400__diffusion_prior_best.pt \
--q_checkpoint_dir /LDCQ_for_SOLAR/q_checkpoints/gpu0_02.04 \
--policy_decoder_type mlp \
--num_diffusion_samples 100 \
--q_checkpoint_steps 150 \
--diffusion_steps 500 \
--num_parallel_envs 1 \
--skill_model_diffusion_steps 100 \
--a_dim 36 \
--z_dim 256 \
--h_dim 512 \
--s_dim 512 \
--train_diffusion_prior 1 \
--conditional_prior 1 \
--normalize_latent 0 \
--exec_horizon 1 \
--horizon 5 \
--policy q \
--render None \
--beta 0.1 \
--max_grid_size 10 \
--use_in_out 1 \
--max_episode_steps 30 \
--use_ddim 1 \
--ddim_steps 100 \
--ddim_eta 0.0 \
--ddim_discr uniform \
--noise_temperature 1.0 \
--use_mlp_embed_q 1 \
--use_enhanced_pair_encoding 0 \
--use_shared_grid_embedding 0 \
--disable_pair_encoding_skill 0 \
--use_split_pair_trajectory_encoding 0 \
--use_direct_output_predictor 0 \
--use_direct_output_for_diffusion 0 \
--disable_pair_encoding_q 0 \
--use_split_pair_trajectory_encoding 0 \
--use_direct_output_predictor 0 \
--use_direct_output_for_diffusion 0 \
--num_evals 500 \
--update_in_grid_on_fail 1 \
--encoder_type gru \
--repetition_threshold 5 \
--use_vae_prior_for_latent 0
