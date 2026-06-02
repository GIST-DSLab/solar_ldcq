import os
import sys

curr_folder=os.path.abspath(__file__)
parent_folder=os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder) 

from argparse import ArgumentParser
import os
import pickle
# import gym
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.diffusion_models import (
    Model_mlp,
    Model_cnn_mlp,
    Model_Cond_Diffusion,
)
from models.skill_model import SkillModel
# from models.concept_encoder import ConceptEncoder  # imported conditionally below
# from models.discrete_concept_encoder import DiscreteConceptEncoder  # imported conditionally below
from utils.utils import get_dataset, ARC_Segment_Dataset
import json

def collect_data(args):
    # dataset_file = parent_folder+'/data/'+args.env+'.pkl'
    # with open(dataset_file, "rb") as f:
    #     dataset = pickle.load(f)

    state_dim = args.h_dim
    a_dim = args.a_dim

    skill_model_path = os.path.join(args.checkpoint_dir, args.skill_model_filename)

    checkpoint = torch.load(skill_model_path)

    skill_model = SkillModel(
        state_dim,
        a_dim,
        args.z_dim,
        args.h_dim,
        args.horizon,
        a_dist=args.a_dist,
        beta=args.beta,
        fixed_sig=None,
        encoder_type=args.encoder_type,
        state_decoder_type=args.state_decoder_type,
        policy_decoder_type=args.policy_decoder_type,
        per_element_sigma=args.per_element_sigma,
        conditional_prior=args.conditional_prior,
        train_diffusion_prior=args.train_diffusion_prior,
        diffusion_steps=args.skill_model_diffusion_steps,
        normalize_latent=args.normalize_latent,
        color_num=11,	# 0~9 색깔, 10은 배경
        action_num=args.a_dim,	# 36개의 action
        max_grid_size=args.max_grid_size,
        use_in_out=args.use_in_out,
        use_enhanced_pair_encoding=args.use_enhanced_pair_encoding,
        disable_pair_encoding=args.disable_pair_encoding,
        use_shared_grid_embedding=args.use_shared_grid_embedding,
        use_split_pair_trajectory_encoding=args.use_split_pair_trajectory_encoding,
        use_direct_output_predictor=bool(args.use_direct_output_predictor),
        use_direct_output_for_diffusion=bool(args.use_direct_output_for_diffusion),
        use_concept_guidance=bool(args.use_concept_guidance)
    ).to(args.device)
    skill_model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    skill_model.eval()

    # Apply torch.compile to skill_model if enabled
    use_compile = bool(args.use_compile)
    if use_compile:
        print(f"Compiling skill_model with mode='{args.compile_mode}'...")
        skill_model = torch.compile(skill_model, mode=args.compile_mode)
        print("Skill model compiled!")

    if args.do_diffusion:
        diffusion_nn_model = torch.load(os.path.join(args.checkpoint_dir, args.diffusion_model_filename),weights_only=False).to(args.device)
        if not hasattr(diffusion_nn_model, 'use_in_out'):
            diffusion_nn_model.use_in_out = args.use_in_out  # 또는 원하는 기본값
        if not hasattr(diffusion_nn_model, 'use_enhanced_pair_encoding'):
            diffusion_nn_model.use_enhanced_pair_encoding = args.use_enhanced_pair_encoding
        if hasattr(diffusion_nn_model, 'nn') and not hasattr(diffusion_nn_model.nn, 'use_in_out'):
            diffusion_nn_model.nn.use_in_out = args.use_in_out
        if hasattr(diffusion_nn_model, 'nn') and not hasattr(diffusion_nn_model.nn, 'use_enhanced_pair_encoding'):
            diffusion_nn_model.nn.use_enhanced_pair_encoding = args.use_enhanced_pair_encoding
        
        diffusion_model = Model_Cond_Diffusion(
            diffusion_nn_model,
            betas=(1e-4, 0.02),
            n_T=args.diffusion_steps,
            device=args.device,
            x_dim=state_dim,
            y_dim=args.z_dim,
            drop_prob=None,
            guide_w=args.cfg_weight,
            use_in_out=args.use_in_out,
            normalize_latent=args.normalize_latent
        )
        diffusion_model.eval()

        # Apply torch.compile to diffusion_model if enabled
        if use_compile:
            print(f"Compiling diffusion_model with mode='{args.compile_mode}'...")
            diffusion_model = torch.compile(diffusion_model, mode=args.compile_mode)
            print("Diffusion model compiled!")

    # Initialize concept encoder if using concept guidance
    use_concept_guidance = getattr(args, 'use_concept_guidance', False)
    use_discrete_concepts = getattr(args, 'use_discrete_concepts', False)  # Default to text-based (0)

    if use_concept_guidance and args.do_diffusion:
        from models.concept_encoder import ConceptEncoder
        from models.discrete_concept_encoder import DiscreteConceptEncoder
        print("Initializing concept encoder for concept guidance...")
        if use_discrete_concepts:
            # Load discrete concept mapping from checkpoint directory
            mapping_path = os.path.join(args.checkpoint_dir,
                                       args.skill_model_filename[:-4] + '_concept_mapping.json')
            if os.path.exists(mapping_path):
                print(f"Loading discrete concept mapping from {mapping_path}")
                with open(mapping_path, 'r') as f:
                    concept_to_id = json.load(f)
                concept_encoder = DiscreteConceptEncoder(
                    num_concepts=len(concept_to_id),
                    embedding_dim=args.z_dim,
                    concept_to_id=concept_to_id,
                    device=args.device
                )
                print(f"Loaded discrete concept encoder with {len(concept_to_id)} concepts")
            else:
                print(f"Warning: Discrete concept mapping not found at {mapping_path}")
                print("Falling back to text-based concept encoder")
                use_discrete_concepts = False
                concept_encoder = ConceptEncoder(
                    model_name='all-MiniLM-L6-v2',
                    projection_dim=args.z_dim,
                    device=args.device
                )
        else:
            # Use text-based concept encoder (default)
            concept_encoder = ConceptEncoder(
                model_name='all-MiniLM-L6-v2',
                projection_dim=args.z_dim,
                device=args.device
            )
        concept_encoder.eval()
    else:
        concept_encoder = None

    # Dataset returns concept only if using concept guidance
    dataset = ARC_Segment_Dataset(
		data_path=args.solar_dir,
		return_concept=use_concept_guidance
	)
    len_train_dataset = dataset.__len__()
    print("Length dataset: {0}".format(len_train_dataset))
    
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=8)

    states_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    clip_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    in_grid_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    
    pair_in_gt = np.zeros((len_train_dataset, 3, args.max_grid_size, args.max_grid_size))
    pair_out_gt = np.zeros((len_train_dataset, 3, args.max_grid_size, args.max_grid_size))
    
    latent_gt = np.zeros((len_train_dataset, args.z_dim))
    if args.save_z_dist:
        latent_std_gt = np.zeros((len_train_dataset, args.z_dim))
    sT_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    clip_T_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    rewards_gt = np.zeros((len_train_dataset, 1))

    if args.do_diffusion:
        diffusion_latents_gt = np.zeros((len_train_dataset, args.num_diffusion_samples, args.z_dim))
    else:
        prior_latents_gt = np.zeros((len_train_dataset, args.num_prior_samples, args.z_dim))

    if not 'maze' in args.env and not 'kitchen' in args.env:
        terminals_gt = np.zeros((len_train_dataset, 1))
    gamma_array = np.power(args.gamma, np.arange(args.horizon))

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), mininterval=300.0)

    for batch_id, batch_data in enumerate(train_loader):
        # Unpack based on whether concept is included
        if use_concept_guidance:
            state, s_T, clip, clip_T, selection, operation, reward, terminated, _, in_grid, out_grid, ex_in, ex_out, concept = batch_data
        else:
            state, s_T, clip, clip_T, selection, operation, reward, terminated, _, in_grid, out_grid, ex_in, ex_out = batch_data
            concept = None
        

        state = state.to(args.device)
        clip = clip.to(args.device)
        in_grid = in_grid.to(args.device)
        selection = selection.to(args.device)
        operation = operation.to(args.device)
        terminated = terminated.to(args.device)
        pair_in = ex_in.to(args.device)
        pair_out = ex_out.to(args.device)
        s_T = s_T.to(args.device)
        clip_T = clip_T.to(args.device)

        start_idx = batch_id * args.batch_size
        end_idx = start_idx + args.batch_size
        
        states_gt[start_idx : end_idx, 0] = state[:, 0, :, :].cpu().numpy()
        clip_gt[start_idx : end_idx, 0] = clip[:, 0, :, :].cpu().numpy()
        in_grid_gt[start_idx : end_idx, 0] = in_grid[:, 0, :, :].cpu().numpy()
        sT_gt[start_idx: end_idx, 0] = s_T[:, 0, :, :].cpu().numpy()
        clip_T_gt[start_idx: end_idx, 0] = clip_T[:, 0, :, :].cpu().numpy()
        
        pair_in_gt[start_idx: end_idx] = pair_in[:, :, :, :].cpu().numpy()
        pair_out_gt[start_idx: end_idx] = pair_out[:, :, :, :].cpu().numpy()
        
        rewards_gt[start_idx: end_idx, 0] = np.sum(reward.cpu().numpy() * gamma_array, axis=1)
        terminals_gt[start_idx: end_idx,0] = np.sum(terminated.cpu().numpy(), axis=1)
        
        if not args.do_diffusion:
            with torch.no_grad():
                prior_latent_mean, prior_latent_std = skill_model.prior(state[:, 0:1, :], clip[:, 0:1, :], in_grid, pair_in, pair_out)   # 이게 맞나????
                prior_latent_mean = prior_latent_mean.repeat_interleave(args.num_prior_samples, 0)
                prior_latent_std = prior_latent_std.repeat_interleave(args.num_prior_samples, 0)
                
                prior_latents_gt[start_idx : end_idx] = torch.stack(torch.distributions.normal.Normal(prior_latent_mean.squeeze(1), prior_latent_std.squeeze(1)).sample().chunk(state.shape[0])).cpu().numpy()
        else:
            diffusion_state = state[:, 0:1, :].repeat_interleave(args.num_diffusion_samples, 0)
            diffusion_clip = clip[:, 0:1, :].repeat_interleave(args.num_diffusion_samples, 0)
            diffusion_in_grid = in_grid[:, 0:1, :].repeat_interleave(args.num_diffusion_samples, 0)
            diffusion_pair_in = pair_in.repeat_interleave(args.num_diffusion_samples, 0)
            diffusion_pair_out = pair_out.repeat_interleave(args.num_diffusion_samples, 0)

            # print(pair_in.shape)
            # print(pair_out.shape)
            # print(diffusion_in_grid.shape)
            # print(diffusion_pair_in.shape)
            # print(diffusion_pair_out.shape)

            # Determine text_guide_emb: use direct output embedding OR concept encoding
            text_guide_emb = None

            # Priority 1: Direct output guidance (on-the-fly prediction)
            if getattr(args, 'use_direct_output_for_diffusion', False) and args.use_direct_output_predictor:
                with torch.no_grad():
                    # Predict direct output for current batch (non-repeated grids)
                    direct_output_logits = skill_model.direct_output_predictor(in_grid, pair_in, pair_out)
                    batch_size = direct_output_logits.shape[0]
                    grid_size = args.max_grid_size * args.max_grid_size
                    direct_logits_reshaped = direct_output_logits.reshape(batch_size, grid_size, 11)

                    # Get predicted output (argmax)
                    predicted_output = torch.argmax(direct_logits_reshaped, dim=-1)

                    # Embed predicted output using state_emb_layer
                    predicted_output_grid = predicted_output.reshape(batch_size, 1, args.max_grid_size, args.max_grid_size).float()
                    direct_output_emb = skill_model.decoder.state_emb_layer(predicted_output_grid)
                    direct_output_emb_flat = direct_output_emb.reshape(batch_size, -1)

                    # Project to z_dim (256) to match concept embedding dimension
                    if not hasattr(skill_model, 'direct_output_proj'):
                        skill_model.direct_output_proj = torch.nn.Linear(
                            direct_output_emb_flat.shape[1], args.z_dim
                        ).to(direct_output_emb_flat.device)

                    direct_output_emb_projected = skill_model.direct_output_proj(direct_output_emb_flat)

                    # Repeat for each diffusion sample
                    text_guide_emb = direct_output_emb_projected.repeat_interleave(args.num_diffusion_samples, 0)

            # Priority 2: Concept guidance (if direct output not used)
            elif use_concept_guidance and concept_encoder is not None:
                valid_concepts = [c for c in concept if c]  # Filter out empty concepts
                if valid_concepts:
                    with torch.no_grad():
                        # Encode concepts for this batch
                        batch_concept_emb = concept_encoder(valid_concepts, batch_size=1)
                        # Repeat for each diffusion sample
                        text_guide_emb = batch_concept_emb.repeat_interleave(args.num_diffusion_samples, 0)

            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.float16):
                    if args.use_ddim:
                        # Use DDIM sampling
                        generated_latents = diffusion_model.ddim_sample_extra(
                            diffusion_state, diffusion_clip, diffusion_in_grid,
                            diffusion_pair_in, diffusion_pair_out,
                            ddim_steps=args.ddim_steps,
                            ddim_eta=args.ddim_eta,
                            ddim_discr=args.ddim_discr,
                            extra_steps=args.extra_steps,
                            predict_noise=bool(args.predict_noise),
                            noise_temperature=args.noise_temperature,
                            text_guide_emb=text_guide_emb
                        )
                    else:
                        # Use original DDPM sampling
                        generated_latents = diffusion_model.sample_extra(
                            diffusion_state, diffusion_clip, diffusion_in_grid,
                            diffusion_pair_in, diffusion_pair_out,
                            predict_noise=args.predict_noise,
                            extra_steps=args.extra_steps,
                            text_guide_emb=text_guide_emb
                        )

                diffusion_latents_gt[start_idx : end_idx] = torch.stack(generated_latents.float().chunk(state.shape[0])).cpu().numpy()

        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.float16):
                output, output_std = skill_model.encoder(state, clip, in_grid, operation, selection, pair_in, pair_out)
        latent_gt[start_idx : end_idx] = output.detach().float().cpu().numpy().squeeze(1)
        if args.save_z_dist:
            latent_std_gt[start_idx : end_idx] = output_std.detach().float().cpu().numpy().squeeze(1)

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4]+ '_states.npy'), states_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_clip.npy'), clip_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_in_grid.npy'), in_grid_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_latents.npy'), latent_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_sT.npy'), sT_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_clip_T.npy'), clip_T_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_rewards.npy'), rewards_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_pair_in.npy'), pair_in_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_pair_out.npy'), pair_out_gt)
    
    if args.do_diffusion:
        np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_sample_latents.npy'), diffusion_latents_gt)
    else:
        np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_prior_latents.npy'), prior_latents_gt)
    if args.save_z_dist:
        np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_latents_std.npy'), latent_std_gt)
    if not 'maze' in args.env and not 'kitchen' in args.env:
        np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_terminals.npy'), terminals_gt)


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--solar_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=parent_folder+'/checkpoints/')
    parser.add_argument('--skill_model_filename', type=str)
    parser.add_argument('--diffusion_model_filename', type=str)
    
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--append_goals', type=int, default=0)
    parser.add_argument('--save_z_dist', type=int, default=0)
    parser.add_argument('--cum_rewards', type=int, default=0)

    parser.add_argument('--do_diffusion', type=int, default=1)
    parser.add_argument('--num_diffusion_samples', type=int, default=300)
    parser.add_argument('--num_prior_samples', type=int, default=300)
    parser.add_argument('--diffusion_steps', type=int, default=500)
    parser.add_argument('--cfg_weight', type=float, default=0.0)
    parser.add_argument('--extra_steps', type=int, default=4)
    parser.add_argument('--predict_noise', type=int, default=0)
    
    # DDIM sampling options
    parser.add_argument('--use_ddim', type=int, default=0, help='Use DDIM sampling instead of DDPM (0=DDPM, 1=DDIM)')
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of DDIM sampling steps (fewer = faster)')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM stochasticity (0=deterministic, 1=DDPM-like)')
    parser.add_argument('--ddim_discr', type=str, default='uniform', help='DDIM timestep discretization (uniform/quad)')
    parser.add_argument('--noise_temperature', type=float, default=1.0, help='Temperature scaling for sampling noise (higher = more diversity)')

    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--a_dist', type=str, default='normal')
    parser.add_argument('--encoder_type', type=str, default='gru', choices=['gru', 'transformer'], help='Encoder type (gru or transformer)')
    parser.add_argument('--state_decoder_type', type=str, default='mlp')
    parser.add_argument('--policy_decoder_type', type=str, default='mlp')       # 원래는 autoregressive
    parser.add_argument('--per_element_sigma', type=int, default=1)
    
    parser.add_argument('--train_diffusion_prior', type=int, default=1)
    parser.add_argument('--conditional_prior', type=int, default=1)
    parser.add_argument('--normalize_latent', type=int, default=0)
    
    parser.add_argument('--skill_model_diffusion_steps', type=int, default=100)

    parser.add_argument('--a_dim', type=int, default=36)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--max_grid_size', type=int, default=30)
    parser.add_argument('--use_in_out', type=int, default=0)  # 0: False, 1: True
    parser.add_argument('--use_enhanced_pair_encoding', type=int, default=0)  # 0: False, 1: True
    parser.add_argument('--use_shared_grid_embedding', type=int, default=0, help='Use shared grid embedding for pair encoding (0=disable, 1=enable)')
    parser.add_argument('--disable_pair_encoding', type=int, default=0)  # 0: False, 1: True
    parser.add_argument('--use_split_pair_trajectory_encoding', type=int, default=0, help='Split pair and trajectory encoding (0=disable, 1=enable)')
    parser.add_argument('--use_direct_output_predictor', type=int, default=0, help='Use direct output predictor (0=disable, 1=enable)')
    parser.add_argument('--use_direct_output_for_diffusion', type=int, default=0, help='Use direct output embedding for diffusion conditioning (0=disable, 1=enable)')
    parser.add_argument('--use_concept_guidance', type=int, default=0, help='Use concept text guidance for diffusion sampling (0=disable, 1=enable)')
    parser.add_argument('--use_discrete_concepts', type=int, default=0, help='Use discrete concept IDs instead of text embeddings (0=text-based, 1=discrete)')
    parser.add_argument('--use_cfg_for_concept', type=int, default=1, help='Use CFG for concept guidance (0=direct conditioning, 1=CFG)')
    parser.add_argument('--use_concept_in_encoder', type=int, default=0, help='Use concept in skill encoder (0=disable, 1=enable)')
    parser.add_argument('--num_concepts', type=int, default=0, help='Number of concepts (0=auto-detect from data)')
    parser.add_argument('--concept_scale', type=float, default=1.0, help='Scale factor for concept embeddings (default=1.0)')
    parser.add_argument('--use_compile', type=int, default=0, help='Use torch.compile for faster inference (0=disable, 1=enable, requires PyTorch 2.0+)')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead', choices=['default', 'reduce-overhead', 'max-autotune'], help='Compilation mode for torch.compile')

    args = parser.parse_args()

    collect_data(args)
