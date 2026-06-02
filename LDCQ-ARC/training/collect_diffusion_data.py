import os
import sys

curr_folder=os.path.abspath(__file__)
parent_folder=os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder) 

from argparse import ArgumentParser

from tqdm import tqdm
# import gym
import numpy as np
import torch
torch.backends.cudnn.enabled = False  # cuDNN/CUDA 12.9 driver compatibility fix
from torch.utils.data import DataLoader

from models.skill_model import SkillModel
from utils.utils import get_dataset, ARC_Segment_Dataset

def collect_data(args):
    if 'ARCLE' in args.env:
        state_dim = args.s_dim
        a_dim = args.a_dim
    else:
        raise NotImplementedError
    
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

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
                            diffusion_steps=args.diffusion_steps,
                            normalize_latent=args.normalize_latent,
                            action_num=a_dim,
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

    dataset = ARC_Segment_Dataset(data_path=args.solar_dir, return_concept=bool(args.save_concepts))

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8)
    
    len_train_dataset = dataset.__len__()
    
    states_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    clip_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    in_grid_gt = np.zeros((len_train_dataset, 1, args.max_grid_size, args.max_grid_size))
    latent_gt = np.zeros((len_train_dataset, args.z_dim))
    pair_in_gt = np.zeros((len_train_dataset, args.num_ex_pair, args.max_grid_size, args.max_grid_size))
    pair_out_gt = np.zeros((len_train_dataset, args.num_ex_pair, args.max_grid_size, args.max_grid_size))

    if args.save_concepts:
        concepts_gt = []  # Store concept descriptions

    if args.save_z_dist:
        latent_std_gt = np.zeros((len_train_dataset, args.z_dim))

    # Direct output guidance: save direct output embeddings
    if args.use_direct_output_for_diffusion and args.use_direct_output_predictor:
        # Dimension: depends on skill_model's state_emb_layer output
        # We'll get it dynamically from the first batch
        direct_output_emb_gt = None


    pbar = tqdm(enumerate(train_loader), total=len(train_loader), mininterval=300.0)

    for batch_id, batch_data in enumerate(train_loader):
        # Unpack data based on whether concepts are being saved
        if args.save_concepts:
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
        
        states_gt[start_idx : end_idx, 0, :, :] = state[:, 0, :, :].cpu().numpy()
        clip_gt[start_idx : end_idx, 0, :, :] = clip[:, 0, :, :].cpu().numpy()
        in_grid_gt[start_idx : end_idx, 0, :, :] = in_grid[:, 0, :, :].cpu().numpy()
        pair_in_gt[start_idx: end_idx] = pair_in[:, :, :, :].cpu().numpy()
        pair_out_gt[start_idx: end_idx] = pair_out[:, :, :, :].cpu().numpy()

        output, output_std = skill_model.encoder(state, clip, in_grid, operation, selection, pair_in, pair_out)
        latent_gt[start_idx : end_idx] = output.detach().cpu().numpy().squeeze(1)

        # Store concepts (convert from list/batch to individual items)
        if args.save_concepts:
            concepts_gt.extend(concept)

        # Generate and store direct output embeddings for diffusion guidance
        if args.use_direct_output_for_diffusion and args.use_direct_output_predictor:
            with torch.no_grad():
                # Predict direct output
                direct_output_logits = skill_model.direct_output_predictor(in_grid, pair_in, pair_out)
                batch_size = direct_output_logits.shape[0]
                grid_size = args.max_grid_size * args.max_grid_size
                direct_logits_reshaped = direct_output_logits.reshape(batch_size, grid_size, 11)

                # Get predicted output (argmax)
                predicted_output = torch.argmax(direct_logits_reshaped, dim=-1)  # (batch, grid_size)

                # Embed predicted output using state_emb_layer (same as in skill_model.py:1943-1944)
                predicted_output_grid = predicted_output.reshape(batch_size, 1, args.max_grid_size, args.max_grid_size).float()
                direct_output_emb = skill_model.decoder.state_emb_layer(predicted_output_grid)
                # Flatten: (batch, s_dim)
                direct_output_emb_flat = direct_output_emb.reshape(batch_size, -1)

                # Project to z_dim (256) to match concept embedding dimension
                # Use a simple linear projection
                if not hasattr(skill_model, 'direct_output_proj'):
                    skill_model.direct_output_proj = torch.nn.Linear(
                        direct_output_emb_flat.shape[1], args.z_dim
                    ).to(direct_output_emb_flat.device)

                direct_output_emb_projected = skill_model.direct_output_proj(direct_output_emb_flat)

                # Initialize array on first batch
                if direct_output_emb_gt is None:
                    direct_output_emb_gt = np.zeros((len_train_dataset, args.z_dim))

                direct_output_emb_gt[start_idx : end_idx] = direct_output_emb_projected.detach().cpu().numpy()

    if not os.path.exists(args.data_dir):
        # 디렉토리가 없으면 새로 만듭니다
        os.makedirs(args.data_dir)
        print(f"디렉토리가 생성되었습니다: {args.data_dir}")

    np.save(os.path.join(args.data_dir,f'{args.skill_model_filename[:-4]}_states.npy'), states_gt)
    np.save(os.path.join(args.data_dir,f'{args.skill_model_filename[:-4]}_latents.npy'), latent_gt)
    np.save(os.path.join(args.data_dir,f'{args.skill_model_filename[:-4]}_clip.npy'), clip_gt)
    np.save(os.path.join(args.data_dir,f'{args.skill_model_filename[:-4]}_in_grid.npy'), in_grid_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_pair_in.npy'), pair_in_gt)
    np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_pair_out.npy'), pair_out_gt)

    if args.save_concepts:
        np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_concepts.npy'), np.array(concepts_gt, dtype=object))

    if args.save_z_dist:
        np.save(os.path.join(args.data_dir,f'{args.skill_model_filename[:-4]}_latents_std.npy'), latent_std_gt)

    # Save direct output embeddings for diffusion guidance
    if args.use_direct_output_for_diffusion and args.use_direct_output_predictor and direct_output_emb_gt is not None:
        np.save(os.path.join(args.data_dir, args.skill_model_filename[:-4] + '_direct_output_emb.npy'), direct_output_emb_gt)
        print(f"Saved direct output embeddings: shape {direct_output_emb_gt.shape}")

if __name__ == '__main__':

    parser = ArgumentParser()
     # #####해놓은 것들이 argument 잘못넣으면 안 돌아가는 것들, 돌리기 전 꼭 확인할 것
    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2') #####
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default=parent_folder+'/checkpoints')
    parser.add_argument('--data_dir', type=str, default=parent_folder+'/data')
    parser.add_argument('--solar_dir', type=str, default=None)
    parser.add_argument('--skill_model_filename', type=str) #####
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--append_goals', type=int, default=0) #####
    parser.add_argument('--save_z_dist', type=int, default=0)
    parser.add_argument('--save_concepts', type=int, default=1, help='Save concept descriptions from trajectory data (0=disable, 1=enable)')
    parser.add_argument('--get_rewards', type=int, default=1)
    
    parser.add_argument('--horizon', type=int, default=30)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--a_dist', type=str, default='normal')

    parser.add_argument('--encoder_type', type=str, default='gru', choices=['gru', 'transformer'], help='Encoder type (gru or transformer)')
    parser.add_argument('--state_decoder_type', type=str, default='mlp') #####
    parser.add_argument('--policy_decoder_type', type=str, default='autoregressive') #####
    parser.add_argument('--per_element_sigma', type=int, default=1)
    parser.add_argument('--conditional_prior', type=int, default=0)
    parser.add_argument('--train_diffusion_prior', type=int, default=0)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--a_dim', type=int, default=256)
    parser.add_argument('--s_dim', type=int, default=256)
    
    parser.add_argument('--normalize_latent', type=int, default=0)
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--max_grid_size', type=int, default=30)
    parser.add_argument('--num_ex_pair', type=int, default=3)
    parser.add_argument('--use_in_out', type=int, default=0)  # 0: False, 1: True
    parser.add_argument('--use_enhanced_pair_encoding', type=int, default=0)  # 0: False, 1: True
    parser.add_argument('--use_shared_grid_embedding', type=int, default=0, help='Use shared grid embedding for pair encoding (0=disable, 1=enable)')
    parser.add_argument('--disable_pair_encoding', type=int, default=0)  # 0: False, 1: True
    parser.add_argument('--use_split_pair_trajectory_encoding', type=int, default=0, help='Split pair and trajectory encoding (0=disable, 1=enable)')
    parser.add_argument('--use_direct_output_predictor', type=int, default=0, help='Use direct output predictor (0=disable, 1=enable)')
    parser.add_argument('--use_direct_output_for_diffusion', type=int, default=0, help='Use direct output embedding for diffusion conditioning (0=disable, 1=enable)')
    parser.add_argument('--use_concept_guidance', type=int, default=0, help='Use concept guidance for diffusion prior (0=disable, 1=enable)')
    parser.add_argument('--use_concept_in_encoder', type=int, default=0, help='Use concept in skill encoder (0=disable, 1=enable)')
    parser.add_argument('--num_concepts', type=int, default=0, help='Number of concepts (0=auto-detect from data)')
    parser.add_argument('--concept_scale', type=float, default=1.0, help='Scale factor for concept embeddings (default=1.0)')

    args = parser.parse_args()

    collect_data(args)
