import os
import sys
import datetime
curr_folder = os.path.abspath(__file__)
parent_folder = os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder)
from argparse import ArgumentParser
import pickle
import numpy as np
from tqdm import tqdm
import torch
from per_utils import FixedPrioritizedBuffer
from torch.utils.data import Dataset, DataLoader
from models.dqn import DDQN
from models.skill_model import SkillModel


class VAEPriorDDQN(DDQN):
    def __init__(self, skill_model, *args, **kwargs):
        kwargs['diffusion_prior'] = None
        super().__init__(*args, **kwargs)
        self.skill_model = skill_model
        self.skill_model.eval()

    @torch.no_grad()
    def get_max_skills(self, states, clip, in_grid, pair_in, pair_out, net=0, is_eval=False, sample_latents=None, concept_encoder=None, concept=None):
        if not is_eval:
            n_states = states.shape[0]
            states_expanded = states.repeat_interleave(self.num_prior_samples, 0)
            clip_expanded = clip.repeat_interleave(self.num_prior_samples, 0)
            in_grid_expanded = in_grid.repeat_interleave(self.num_prior_samples, 0)
            pair_in_expanded = pair_in.repeat_interleave(self.num_prior_samples, 0)
            pair_out_expanded = pair_out.repeat_interleave(self.num_prior_samples, 0)
        else:
            n_states = states.shape[0]
            states_expanded = states
            clip_expanded = clip
            in_grid_expanded = in_grid
            pair_in_expanded = pair_in
            pair_out_expanded = pair_out
        if sample_latents is not None:
            perm = torch.randperm(self.total_prior_samples)[:self.num_prior_samples]
            sample_latents = sample_latents[:, perm.cpu().numpy(), :]
            z_samples = torch.FloatTensor(sample_latents).to(self.device).reshape(
                sample_latents.shape[0] * self.num_prior_samples, sample_latents.shape[2]
            )
        else:
            prior_mean, prior_std = self.skill_model.prior(
                states_expanded, clip_expanded, in_grid_expanded, pair_in_expanded, pair_out_expanded
            )
            eps = torch.randn_like(prior_mean)
            z_samples = prior_mean + prior_std * eps
            z_samples = z_samples.squeeze(1)
        if is_eval:
            q_vals = torch.minimum(
                self.target_net_0(states_expanded, clip_expanded, in_grid_expanded, z_samples, pair_in_expanded, pair_out_expanded)[:, 0],
                self.target_net_1(states_expanded, clip_expanded, in_grid_expanded, z_samples, pair_in_expanded, pair_out_expanded)[:, 0]
            )
            return z_samples, q_vals
        else:
            if net == 0:
                q_vals = self.target_net_0(states_expanded, clip_expanded, in_grid_expanded, z_samples, pair_in_expanded, pair_out_expanded)[:, 0]
            else:
                q_vals = self.target_net_1(states_expanded, clip_expanded, in_grid_expanded, z_samples, pair_in_expanded, pair_out_expanded)[:, 0]
        q_vals = q_vals.reshape(n_states, self.num_prior_samples)
        max_vals = torch.max(q_vals, dim=1)
        max_q_vals = max_vals.values
        max_indices = max_vals.indices
        idx = torch.arange(n_states).cuda() * self.num_prior_samples + max_indices
        max_z = z_samples[idx]
        return max_z, max_q_vals


def PER_buffer_filler(data_dir, filename, test_prop=0.1, sample_z=False, sample_max_latents=False, alpha=0.6, args=None):
    state_all = np.load(os.path.join(data_dir, filename + "_states.npy"), allow_pickle=True)
    clip_all = np.load(os.path.join(data_dir, filename + "_clip.npy"), allow_pickle=True)
    in_grid_all = np.load(os.path.join(data_dir, filename + "_in_grid.npy"), allow_pickle=True)
    latent_all = np.load(os.path.join(data_dir, filename + "_latents.npy"), allow_pickle=True)
    sT_all = np.load(os.path.join(data_dir, filename + "_sT.npy"), allow_pickle=True)
    clip_T_all = np.load(os.path.join(data_dir, filename + "_clip_T.npy"), allow_pickle=True)
    pair_in_all = np.load(os.path.join(data_dir, filename + "_pair_in.npy"), allow_pickle=True)
    pair_out_all = np.load(os.path.join(data_dir, filename + "_pair_out.npy"), allow_pickle=True)
    rewards_all = np.load(os.path.join(data_dir, filename + "_rewards.npy"), allow_pickle=True)
    if sample_max_latents:
        max_latents = np.load(os.path.join(data_dir, filename + "_prior_latents.npy"), allow_pickle=True)
    terminals_all = np.load(os.path.join(data_dir, filename + "_terminals.npy"), allow_pickle=True)
    n_train = int(state_all.shape[0] * (1 - test_prop))
    state_all = state_all[:n_train]
    clip_all = clip_all[:n_train]
    in_grid_all = in_grid_all[:n_train]
    latent_all = latent_all[:n_train]
    sT_all = sT_all[:n_train]
    clip_T_all = clip_T_all[:n_train]
    rewards_all = rewards_all[:n_train]
    pair_in_all = pair_in_all[:n_train]
    pair_out_all = pair_out_all[:n_train]
    terminals_all = terminals_all[:n_train]
    if sample_max_latents:
        max_latents_all = max_latents[:n_train]
    else:
        max_latents_all = None
    replay_buffer = FixedPrioritizedBuffer(
        n_train,
        num_samples=args.total_prior_samples,
        z_dim=args.z_dim,
        max_grid_size=args.max_grid_size,
        prob_alpha=alpha
    )
    for i in tqdm(range(n_train), desc="Loading PER buffer", mininterval=300.0):
        replay_buffer.push(
            state_all[i],
            clip_all[i],
            in_grid_all[i],
            latent_all[i],
            rewards_all[i],
            sT_all[i],
            clip_T_all[i],
            terminals_all[i],
            pair_in_all[i],
            pair_out_all[i],
            max_latents_all[i] if max_latents_all is not None else None
        )
    return replay_buffer, args.h_dim, latent_all.shape[-1]


def train(args):
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
        train_diffusion_prior=False,
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
        use_direct_output_for_diffusion=False,
        use_concept_guidance=False,
        use_positional_encoding=bool(args.use_positional_encoding),
    ).to(args.device)
    skill_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    skill_model.eval()
    per_buffer, x_shape, y_dim = PER_buffer_filler(
        args.data_dir,
        args.skill_model_filename[:-4],
        test_prop=args.test_split,
        sample_z=args.sample_z,
        sample_max_latents=args.sample_max_latents,
        alpha=args.alpha,
        args=args
    )
    dqn_agent = VAEPriorDDQN(
        skill_model=skill_model,
        state_dim=x_shape,
        z_dim=y_dim,
        h_dim=args.h_dim,
        total_prior_samples=args.total_prior_samples,
        num_prior_samples=args.num_prior_samples,
        gamma=args.gamma,
        max_grid_size=args.max_grid_size,
        horizon=args.horizon,
        use_ddim=False,
        use_mlp_embed_q=args.use_mlp_embed_q,
        use_enhanced_pair_encoding=args.use_enhanced_pair_encoding,
        disable_pair_encoding=args.disable_pair_encoding,
        tau=args.tau,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        update_steps_multiplier=args.update_steps_multiplier,
        beta_increment=args.beta_increment,
        scheduler_type=args.scheduler_type,
        cosine_t_max=args.cosine_t_max,
        cosine_eta_min=args.cosine_eta_min,
        use_positional_encoding=args.use_positional_encoding,
    )
    task_name = args.solar_dir.split("/")[-1]
    dqn_agent.learn(
        dataload_train=per_buffer,
        n_epochs=args.n_epoch,
        diffusion_model_name=args.skill_model_filename[:-4] + '_vae_prior',
        cfg_weight=0.0,
        per_buffer=args.per_buffer,
        batch_size=args.batch_size,
        gpu_name=args.gpu_name,
        q_checkpoint_dir=args.q_checkpoint_dir,
        task_name=task_name,
        args=args
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='ARCLE')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_epoch', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_split', type=float, default=0.0)
    parser.add_argument('--sample_z', type=int, default=0)
    parser.add_argument('--per_buffer', type=int, default=1)
    parser.add_argument('--sample_max_latents', type=int, default=1)
    parser.add_argument('--total_prior_samples', type=int, default=300)
    parser.add_argument('--num_prior_samples', type=int, default=300)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--checkpoint_dir', type=str, default=parent_folder+'/checkpoints/')
    parser.add_argument('--q_checkpoint_dir', type=str, default=parent_folder+'/q_checkpoints/')
    parser.add_argument('--solar_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=parent_folder+'/data/')
    parser.add_argument('--skill_model_filename', type=str, required=True)
    parser.add_argument('--a_dim', type=int, default=36)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--s_dim', type=int, default=256)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--a_dist', type=str, default='normal')
    parser.add_argument('--encoder_type', type=str, default='gru')
    parser.add_argument('--state_decoder_type', type=str, default='mlp')
    parser.add_argument('--policy_decoder_type', type=str, default='mlp')
    parser.add_argument('--per_element_sigma', type=int, default=1)
    parser.add_argument('--conditional_prior', type=int, default=1)
    parser.add_argument('--normalize_latent', type=int, default=0)
    parser.add_argument('--skill_model_diffusion_steps', type=int, default=100)
    parser.add_argument('--gpu_name', type=str, required=True)
    parser.add_argument('--max_grid_size', type=int, default=10)
    parser.add_argument('--use_in_out', type=int, default=0)
    parser.add_argument('--use_mlp_embed_q', type=int, default=0)
    parser.add_argument('--use_enhanced_pair_encoding', type=int, default=0)
    parser.add_argument('--use_shared_grid_embedding', type=int, default=0)
    parser.add_argument('--disable_pair_encoding', type=int, default=0)
    parser.add_argument('--use_split_pair_trajectory_encoding', type=int, default=0)
    parser.add_argument('--use_positional_encoding', type=int, default=0)
    parser.add_argument('--use_direct_output_predictor', type=int, default=0)
    parser.add_argument('--tau', type=float, default=0.995)
    parser.add_argument('--lr_step_size', type=int, default=50)
    parser.add_argument('--lr_gamma', type=float, default=0.3)
    parser.add_argument('--update_steps_multiplier', type=int, default=1)
    parser.add_argument('--beta_increment', type=float, default=0.03)
    parser.add_argument('--scheduler_type', type=str, default='step', choices=['step', 'cosine'])
    parser.add_argument('--cosine_t_max', type=int, default=100)
    parser.add_argument('--cosine_eta_min', type=float, default=1e-6)
    args = parser.parse_args()
    train(args)
