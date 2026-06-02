"""
Collect Q-learning dataset using VAE Prior Latents (No Diffusion)
Samples latents from VAE's learned prior p(z|s_0, context) instead of diffusion model.
"""
import os
import sys

curr_folder = os.path.abspath(__file__)
parent_folder = os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder)

from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.skill_model import SkillModel
from utils.utils import ARC_Segment_Dataset


def collect_data(args):
    state_dim = args.h_dim
    a_dim = args.a_dim

    skill_model_path = os.path.join(args.checkpoint_dir, args.skill_model_filename)
    checkpoint = torch.load(skill_model_path, weights_only=False)

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
        color_num=11,
        action_num=args.a_dim,
        max_grid_size=args.max_grid_size,
        use_in_out=args.use_in_out,
        use_enhanced_pair_encoding=args.use_enhanced_pair_encoding,
        disable_pair_encoding=args.disable_pair_encoding,
        use_shared_grid_embedding=args.use_shared_grid_embedding,
        use_split_pair_trajectory_encoding=args.use_split_pair_trajectory_encoding,
        use_direct_output_predictor=bool(args.use_direct_output_predictor),
        use_direct_output_for_diffusion=bool(args.use_direct_output_for_diffusion),
        use_concept_guidance=bool(args.use_concept_guidance),
        use_positional_encoding=bool(args.use_positional_encoding),
    ).to(args.device)

    skill_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    skill_model.eval()
    print(f"Loaded skill model from {skill_model_path}")

    # Dataset
    dataset = ARC_Segment_Dataset(
        data_path=args.solar_dir,
        return_concept=False
    )
    len_train_dataset = dataset.__len__()
    print(f"Dataset length: {len_train_dataset}")

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=8
    )

    # Pre-allocate arrays
    states_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    clip_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    in_grid_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    pair_in_gt = np.zeros((len_train_dataset, 3, args.max_grid_size, args.max_grid_size))
    pair_out_gt = np.zeros((len_train_dataset, 3, args.max_grid_size, args.max_grid_size))
    latent_gt = np.zeros((len_train_dataset, args.z_dim))
    sT_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    clip_T_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    rewards_gt = np.zeros((len_train_dataset, 1))
    terminals_gt = np.zeros((len_train_dataset, 1))

    # VAE prior latents (instead of diffusion latents)
    prior_latents_gt = np.zeros((len_train_dataset, args.num_prior_samples, args.z_dim))

    gamma_array = np.power(args.gamma, np.arange(args.horizon))

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Collecting VAE prior latents", mininterval=60.0)

    for batch_id, batch_data in pbar:
        state, s_T, clip, clip_T, selection, operation, reward, terminated, _, in_grid, out_grid, ex_in, ex_out = batch_data

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

        batch_size = state.shape[0]
        start_idx = batch_id * args.batch_size
        end_idx = min(start_idx + batch_size, len_train_dataset)
        actual_batch_size = end_idx - start_idx

        # Store ground truth data
        states_gt[start_idx:end_idx, 0] = state[:actual_batch_size, 0, :, :].cpu().numpy()
        clip_gt[start_idx:end_idx, 0] = clip[:actual_batch_size, 0, :, :].cpu().numpy()
        in_grid_gt[start_idx:end_idx, 0] = in_grid[:actual_batch_size, 0, :, :].cpu().numpy()
        sT_gt[start_idx:end_idx, 0] = s_T[:actual_batch_size, 0, :, :].cpu().numpy()
        clip_T_gt[start_idx:end_idx, 0] = clip_T[:actual_batch_size, 0, :, :].cpu().numpy()
        pair_in_gt[start_idx:end_idx] = pair_in[:actual_batch_size, :, :, :].cpu().numpy()
        pair_out_gt[start_idx:end_idx] = pair_out[:actual_batch_size, :, :, :].cpu().numpy()
        rewards_gt[start_idx:end_idx, 0] = np.sum(reward[:actual_batch_size].cpu().numpy() * gamma_array, axis=1)
        terminals_gt[start_idx:end_idx, 0] = np.sum(terminated[:actual_batch_size].cpu().numpy(), axis=1)

        with torch.no_grad():
            # Get VAE posterior (encoder) latent - this is z from q(z|trajectory)
            output, output_std = skill_model.encoder(
                state[:actual_batch_size], clip[:actual_batch_size], in_grid[:actual_batch_size],
                operation[:actual_batch_size], selection[:actual_batch_size],
                pair_in[:actual_batch_size], pair_out[:actual_batch_size]
            )
            latent_gt[start_idx:end_idx] = output.detach().cpu().numpy().squeeze(1)

            # Sample from VAE prior p(z|s_0, context)
            # Expand for multiple samples
            state_expanded = state[:actual_batch_size, 0:1, :].repeat_interleave(args.num_prior_samples, 0)
            clip_expanded = clip[:actual_batch_size, 0:1, :].repeat_interleave(args.num_prior_samples, 0)
            in_grid_expanded = in_grid[:actual_batch_size].repeat_interleave(args.num_prior_samples, 0)
            pair_in_expanded = pair_in[:actual_batch_size].repeat_interleave(args.num_prior_samples, 0)
            pair_out_expanded = pair_out[:actual_batch_size].repeat_interleave(args.num_prior_samples, 0)

            # Get prior distribution parameters
            prior_mean, prior_std = skill_model.prior(
                state_expanded, clip_expanded, in_grid_expanded, pair_in_expanded, pair_out_expanded
            )

            # Sample from prior using reparameterization trick
            eps = torch.randn_like(prior_mean)
            prior_samples = prior_mean + prior_std * eps
            prior_samples = prior_samples.squeeze(1)  # Remove seq dim

            # Reshape to (batch_size, num_samples, z_dim)
            prior_latents_gt[start_idx:end_idx] = torch.stack(
                prior_samples.chunk(actual_batch_size)
            ).cpu().numpy()

    # Save data
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    base_filename = args.skill_model_filename[:-4]

    np.save(os.path.join(args.data_dir, base_filename + '_states.npy'), states_gt)
    np.save(os.path.join(args.data_dir, base_filename + '_clip.npy'), clip_gt)
    np.save(os.path.join(args.data_dir, base_filename + '_in_grid.npy'), in_grid_gt)
    np.save(os.path.join(args.data_dir, base_filename + '_latents.npy'), latent_gt)
    np.save(os.path.join(args.data_dir, base_filename + '_sT.npy'), sT_gt)
    np.save(os.path.join(args.data_dir, base_filename + '_clip_T.npy'), clip_T_gt)
    np.save(os.path.join(args.data_dir, base_filename + '_rewards.npy'), rewards_gt)
    np.save(os.path.join(args.data_dir, base_filename + '_pair_in.npy'), pair_in_gt)
    np.save(os.path.join(args.data_dir, base_filename + '_pair_out.npy'), pair_out_gt)
    np.save(os.path.join(args.data_dir, base_filename + '_terminals.npy'), terminals_gt)
    np.save(os.path.join(args.data_dir, base_filename + '_prior_latents.npy'), prior_latents_gt)

    print(f"\nSaved data to {args.data_dir}")
    print(f"  - States: {states_gt.shape}")
    print(f"  - Latents (posterior): {latent_gt.shape}")
    print(f"  - Prior latents: {prior_latents_gt.shape}")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='ARCLE')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--solar_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default=parent_folder+'/checkpoints/')
    parser.add_argument('--skill_model_filename', type=str, required=True)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_prior_samples', type=int, default=300)

    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--a_dist', type=str, default='normal')
    parser.add_argument('--encoder_type', type=str, default='gru')
    parser.add_argument('--state_decoder_type', type=str, default='mlp')
    parser.add_argument('--policy_decoder_type', type=str, default='mlp')
    parser.add_argument('--per_element_sigma', type=int, default=1)

    parser.add_argument('--train_diffusion_prior', type=int, default=1)
    parser.add_argument('--conditional_prior', type=int, default=1)
    parser.add_argument('--normalize_latent', type=int, default=0)

    parser.add_argument('--skill_model_diffusion_steps', type=int, default=100)

    parser.add_argument('--a_dim', type=int, default=36)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--max_grid_size', type=int, default=10)
    parser.add_argument('--use_in_out', type=int, default=0)
    parser.add_argument('--use_enhanced_pair_encoding', type=int, default=0)
    parser.add_argument('--use_shared_grid_embedding', type=int, default=0)
    parser.add_argument('--disable_pair_encoding', type=int, default=0)
    parser.add_argument('--use_split_pair_trajectory_encoding', type=int, default=0)
    parser.add_argument('--use_direct_output_predictor', type=int, default=0)
    parser.add_argument('--use_direct_output_for_diffusion', type=int, default=0)
    parser.add_argument('--use_concept_guidance', type=int, default=0)
    parser.add_argument('--use_positional_encoding', type=int, default=0)

    args = parser.parse_args()

    collect_data(args)
