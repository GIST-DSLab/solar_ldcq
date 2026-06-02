import os
import sys
import datetime

curr_folder=os.path.abspath(__file__)
parent_folder=os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder) 

from argparse import ArgumentParser
# from comet_ml import Experiment

# import gym
import pickle
import numpy as np
from tqdm import tqdm
import torch
from per_utils import NaivePrioritizedBuffer, FixedPrioritizedBuffer
from torch.utils.data import Dataset, DataLoader

from models.diffusion_models import (
    Model_mlp,
    Model_cnn_mlp,
    Model_Cond_Diffusion,
)
from models.dqn import DDQN
from models.skill_model import SkillModel


class QLearningDataset(Dataset):
    def __init__(
        self, data_dir, filename, train_or_test="train", test_prop=0.1, sample_z=False
    ):
        # just load it all into RAM
        self.state_all = np.load(os.path.join(data_dir, filename + "_states.npy"), allow_pickle=True)
        self.clip_all = np.load(os.path.join(data_dir, filename + "_clip.npy"), allow_pickle=True)
        self.in_grid_all = np.load(os.path.join(data_dir, filename + "_in_grid.npy"), allow_pickle=True)
        self.latent_all = np.load(os.path.join(data_dir, filename + "_latents.npy"), allow_pickle=True)
        self.sT_all = np.load(os.path.join(data_dir, filename + "_sT.npy"), allow_pickle=True)
        self.clip_T_all = np.load(os.path.join(data_dir, filename + "_clip_T.npy"), allow_pickle=True)
        self.pair_in_all = np.load(os.path.join(data_dir, filename + "_pair_in.npy"), allow_pickle=True)
        self.pair_out_all = np.load(os.path.join(data_dir, filename + "_pair_out.npy"), allow_pickle=True)
        self.rewards_all = np.load(os.path.join(data_dir, filename + "_rewards.npy"), allow_pickle=True)#(4*np.load(os.path.join(data_dir, filename + "_rewards.npy"), allow_pickle=True) - 30*4*0.5)/10 #zero-centering
        self.sample_z = sample_z
        if sample_z:
            self.latent_all_std = np.load(os.path.join(data_dir, filename + "_latents_std.npy"), allow_pickle=True)
        
        n_train = int(self.state_all.shape[0] * (1 - test_prop))
        if train_or_test == "train":
            self.state_all = self.state_all[:n_train]
            self.clip_all = self.clip_all[:n_train]
            self.in_grid_all = self.in_grid_all[:n_train]
            self.latent_all = self.latent_all[:n_train]
            self.sT_all = self.sT_all[:n_train]
            self.clip_T_all = self.clip_T_all[:n_train]
            self.pair_in_all = self.pair_in_all[:n_train]
            self.pair_out_all = self.pair_out_all[:n_train]
            self.rewards_all = self.rewards_all[:n_train]
        elif train_or_test == "test":
            self.state_all = self.state_all[n_train:]
            self.clip_all = self.clip_all[n_train:]
            self.in_gird_all = self.in_gird_all[n_train:]
            self.latent_all = self.latent_all[n_train:]
            self.sT_all = self.sT_all[n_train:]
            self.clip_T_all = self.clip_T_all[n_train:]
            self.pair_in_all = self.pair_in_all[n_train:]
            self.pair_out_all = self.pair_out_all[n_train:]
            self.rewards_all = self.rewards_all[n_train:]
        else:
            raise NotImplementedError

    def __len__(self):
        return self.state_all.shape[0]

    def __getitem__(self, index):
        state = self.state_all[index]
        clip = self.clip_all[index]
        in_grid = self.in_grid_all[index]
        latent = self.latent_all[index]
        if self.sample_z:
            latent_std = self.latent_all_std[index]
            latent = np.random.normal(latent,latent_std)
        sT = self.sT_all[index]
        clip_T = self.clip_T_all[index]
        reward = self.rewards_all[index]
        pair_in = self.pair_in_all[index]
        pair_out = self.pair_out_all[index]

        # return (state, clip, latent, reward, sT, clip_T, reward, pair_in, pair_out)
        return (state, clip, in_grid, latent, sT, clip_T, reward, pair_in, pair_out)
        # return (state, latent, sT, reward)

def PER_buffer_filler(data_dir, filename, test_prop=0.1, sample_z=False, sample_max_latents=False, alpha=0.6, do_diffusion=1):
    # just load it all into RAM
    state_all = np.load(os.path.join(data_dir, filename + "_states.npy"), allow_pickle=True)
    clip_all = np.load(os.path.join(data_dir, filename + "_clip.npy"), allow_pickle=True)
    in_grid_all = np.load(os.path.join(data_dir, filename + "_in_grid.npy"), allow_pickle=True)
    latent_all = np.load(os.path.join(data_dir, filename + "_latents.npy"), allow_pickle=True)
    sT_all = np.load(os.path.join(data_dir, filename + "_sT.npy"), allow_pickle=True)
    clip_T_all = np.load(os.path.join(data_dir, filename + "_clip_T.npy"), allow_pickle=True)
    pair_in_all = np.load(os.path.join(data_dir, filename + "_pair_in.npy"), allow_pickle=True)
    pair_out_all = np.load(os.path.join(data_dir, filename + "_pair_out.npy"), allow_pickle=True)
    rewards_all = np.load(os.path.join(data_dir, filename + "_rewards.npy"), allow_pickle=True)#(4*np.load(os.path.join(data_dir, filename + "_rewards.npy"), allow_pickle=True) - 30*4*0.5)/10 #zero-centering
    
    if sample_z:
        latent_all_std = np.load(os.path.join(data_dir, filename + "_latents_std.npy"), allow_pickle=True)
        
    if sample_max_latents:
        if do_diffusion:
            max_latents = np.load(os.path.join(data_dir, filename + "_sample_latents.npy"), allow_pickle=True)
        else:
            max_latents = np.load(os.path.join(data_dir, filename + "_prior_latents.npy"), allow_pickle=True)
            
    if not 'maze' in filename and not 'kitchen' in filename:
        terminals_all = np.load(os.path.join(data_dir, filename + "_terminals.npy"), allow_pickle=True)
        # rewards_all = rewards_all/10
    
    n_train = int(state_all.shape[0] * (1 - test_prop))
    
    # PER is only for training
    state_all = state_all[:n_train]
    clip_all = clip_all[:n_train]
    in_grid_all = in_grid_all[:n_train]
    latent_all = latent_all[:n_train]
    sT_all = sT_all[:n_train]
    clip_T_all = clip_T_all[:n_train]
    rewards_all = rewards_all[:n_train]
    pair_in_all = pair_in_all[:n_train]
    pair_out_all = pair_out_all[:n_train]
    
    if not 'maze' in filename and not 'kitchen' in filename:
        terminals_all = terminals_all[:n_train]
    if sample_max_latents:
        max_latents_all = max_latents[:n_train]
    else:
        max_latents_all = None
    
    # load into PER buffer
    # replay_buffer = NaivePrioritizedBuffer(n_train, prob_alpha=alpha)
    replay_buffer = FixedPrioritizedBuffer(n_train, num_samples=args.total_prior_samples, z_dim=args.z_dim, max_grid_size=args.max_grid_size, prob_alpha=alpha)
    
    for i in tqdm(range(n_train), mininterval=300.0):
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
            max_latents_all[i]
        )
        
    return replay_buffer, state_all.shape[-1], latent_all.shape[-1]

def train(args):
    # get datasets set up
    if args.per_buffer:
        # per_buffer, x_shape, y_dim = PER_buffer_filler(args.data_dir, args.skill_model_filename[:-4], test_prop=args.test_split, sample_z=args.sample_z, sample_max_latents=args.sample_max_latents, alpha=args.alpha, do_diffusion=args.do_diffusion)
        per_buffer, _, y_dim = PER_buffer_filler(args.data_dir, args.skill_model_filename[:-4], test_prop=args.test_split, sample_z=args.sample_z, sample_max_latents=args.sample_max_latents, alpha=args.alpha, do_diffusion=args.do_diffusion)
        x_shape = args.h_dim
    else:
        torch_data_train = QLearningDataset(
            args.data_dir, args.skill_model_filename[:-4], train_or_test="train", test_prop=args.test_split, sample_z=args.sample_z
        )
        x_shape = args.h_dim
        y_dim = torch_data_train.latent_all.shape[1]
        
        dataload_train = DataLoader(
            torch_data_train, batch_size=args.batch_size, shuffle=True, num_workers=8
        )
    '''

    torch_data_test = QLearningDataset(
        args.data_dir, args.skill_model_filename[:-4], train_or_test="test", test_prop=args.test_split, sample_z=args.sample_z
    )
    dataload_test = DataLoader(
        torch_data_test, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    '''
    # create model
    model = None
    if args.do_diffusion:
        # diffusion_nn_model = torch.load(os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4] + '_diffusion_prior_best.pt')).to(args.device)
        diffusion_nn_model = torch.load(os.path.join(args.checkpoint_dir, args.diffusion_model_filename),weights_only=False).to(args.device)
        if not hasattr(diffusion_nn_model, 'use_in_out'):
            diffusion_nn_model.use_in_out = args.use_in_out
        if hasattr(diffusion_nn_model, 'nn') and not hasattr(diffusion_nn_model.nn, 'use_in_out'):
            diffusion_nn_model.nn.use_in_out = args.use_in_out

        model = Model_Cond_Diffusion(
            diffusion_nn_model,
            betas=(1e-4, 0.02),
            n_T=args.diffusion_steps,
            device=args.device,
            x_dim=x_shape,
            y_dim=y_dim,
            drop_prob=args.drop_prob,
            guide_w=args.cfg_weight,
            use_in_out=args.use_in_out,  # 0: False, 1: True
            normalize_latent=args.normalize_latent
        ).to(args.device)
        model.eval()

    # Load skill model if using direct output for diffusion guidance
    skill_model = None
    if getattr(args, 'use_direct_output_for_diffusion', False) and args.use_direct_output_predictor:
        skill_model = torch.load(os.path.join(args.checkpoint_dir, args.skill_model_filename), weights_only=False).to(args.device)
        skill_model.eval()

    dqn_agent = DDQN(state_dim = x_shape, z_dim=y_dim, h_dim=args.h_dim, diffusion_prior=model, total_prior_samples=args.total_prior_samples, num_prior_samples=args.num_prior_samples, gamma=args.gamma,max_grid_size=args.max_grid_size,horizon=args.horizon,
                     use_ddim=args.use_ddim, ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta, ddim_discr=args.ddim_discr, use_mlp_embed_q=args.use_mlp_embed_q, use_enhanced_pair_encoding=args.use_enhanced_pair_encoding, disable_pair_encoding=args.disable_pair_encoding,
                     tau=args.tau, lr_step_size=args.lr_step_size, lr_gamma=args.lr_gamma, update_steps_multiplier=args.update_steps_multiplier, beta_increment=args.beta_increment,
                     scheduler_type=args.scheduler_type, cosine_t_max=args.cosine_t_max, cosine_eta_min=args.cosine_eta_min, use_positional_encoding=args.use_positional_encoding,
                     skill_model=skill_model, use_direct_output_for_diffusion=getattr(args, 'use_direct_output_for_diffusion', False))

    # Apply torch.compile to DQN agent if enabled
    use_compile = bool(args.use_compile)
    if use_compile:
        print(f"Compiling DQN Q-networks with mode='{args.compile_mode}'...")
        dqn_agent.q_net_0 = torch.compile(dqn_agent.q_net_0, mode=args.compile_mode)
        dqn_agent.q_net_1 = torch.compile(dqn_agent.q_net_1, mode=args.compile_mode)
        print("DQN Q-networks compiled!")

    # dqn_agent.learn(dataload_train=per_buffer if args.per_buffer else dataload_train, n_epochs=args.n_epoch,
    #     diffusion_model_name=args.skill_model_filename[:-4], cfg_weight=args.cfg_weight, per_buffer = args.per_buffer, batch_size = args.batch_size, gpu_name=args.gpu_name)
    task_name = args.solar_dir.split("/")[-1]
    
    # Determine eval script path
    eval_script_path = os.path.join(parent_folder, 'eval', 'plan_skills_diffusion_ARCLE.py')

    dqn_agent.learn(dataload_train=per_buffer, n_epochs=args.n_epoch,
        diffusion_model_name=args.skill_model_filename[:-4], cfg_weight=args.cfg_weight, per_buffer = args.per_buffer, batch_size = args.batch_size, gpu_name=args.gpu_name,q_checkpoint_dir=args.q_checkpoint_dir,task_name=task_name,args=args,
        # Evaluation during training options
        eval_during_training=bool(args.eval_during_training),
        eval_interval=args.eval_interval,
        eval_script_path=eval_script_path,
        test_solar_dir=args.test_solar_dir,
        checkpoint_dir=args.checkpoint_dir,
        skill_model_filename=args.skill_model_filename,
        diffusion_model_filename=args.diffusion_model_filename,
        eval_num_episodes=args.eval_num_episodes,
        max_q_steps=args.max_q_steps)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_epoch', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--net_type', type=str, default='unet')
    parser.add_argument('--n_hidden', type=int, default=512)
    parser.add_argument('--test_split', type=float, default=0.0)
    parser.add_argument('--sample_z', type=int, default=0)
    parser.add_argument('--per_buffer', type=int, default=1)
    parser.add_argument('--sample_max_latents', type=int, default=1)
    parser.add_argument('--total_prior_samples', type=int, default=1000)
    parser.add_argument('--num_prior_samples', type=int, default=1000)

    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--alpha', type=float, default=0.7)

    parser.add_argument('--checkpoint_dir', type=str, default=parent_folder+'/checkpoints/')
    parser.add_argument('--q_checkpoint_dir', type=str, default=parent_folder+'/q_checkpoints/')
    parser.add_argument('--solar_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=parent_folder+'/data/')
    parser.add_argument('--skill_model_filename', type=str)
    parser.add_argument('--diffusion_model_filename', type=str)

    parser.add_argument('--do_diffusion', type=int, default=1)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--skill_model_diffusion_steps', type=int, default=100)
    parser.add_argument('--cfg_weight', type=float, default=0.0)
    parser.add_argument('--use_cfg_for_concept', type=int, default=1, help='Use CFG for concept guidance (0=direct conditioning, 1=CFG)')
    parser.add_argument('--predict_noise', type=int, default=0)
    parser.add_argument('--num_diffusion_samples', type=int, default=1, help='Number of diffusion samples for evaluation')
    
    # DDIM sampling options
    parser.add_argument('--use_ddim', type=int, default=0, help='Use DDIM sampling instead of DDPM (0=DDPM, 1=DDIM)')
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of DDIM sampling steps (fewer = faster)')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM stochasticity (0=deterministic, 1=DDPM-like)')
    parser.add_argument('--ddim_discr', type=str, default='uniform', help='DDIM timestep discretization (uniform/quad)')
    
    parser.add_argument('--a_dim', type=int, default=36)
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--s_dim', type=int, default=256)
    parser.add_argument('--horizon',type=int, default=5)
    parser.add_argument('--gpu_name', type=str, required=True)
    parser.add_argument('--max_grid_size', type=int, default=30)
    parser.add_argument('--use_in_out', type=int, default=0)  # 0: False, 1: True
    parser.add_argument('--use_mlp_embed_q', type=int, default=0)  # 0: False, 1: True
    parser.add_argument('--use_enhanced_pair_encoding', type=int, default=0)  # 0: False, 1: True
    parser.add_argument('--use_shared_grid_embedding', type=int, default=0, help='Use shared grid embedding for pair encoding (0=disable, 1=enable)')
    parser.add_argument('--disable_pair_encoding', type=int, default=0)  # 0: False, 1: True
    parser.add_argument('--use_split_pair_trajectory_encoding', type=int, default=0, help='Split pair and trajectory encoding (0=disable, 1=enable)')
    parser.add_argument('--use_positional_encoding', type=int, default=0, help='Use 2D positional encoding for state embeddings (0=disable, 1=enable)')
    parser.add_argument('--normalize_latent', type=int, default=0)
    
    # Q-learning hyperparameters
    parser.add_argument('--tau', type=float, default=0.995, help='Target network update rate (lower=faster update)')
    parser.add_argument('--lr_step_size', type=int, default=50, help='Learning rate scheduler step size')
    parser.add_argument('--lr_gamma', type=float, default=0.3, help='Learning rate scheduler gamma')
    parser.add_argument('--update_steps_multiplier', type=int, default=1, help='Multiplier for update_steps interval (higher=less frequent LR updates)')
    parser.add_argument('--beta_increment', type=float, default=0.03, help='PER beta increment per update')

    # Scheduler options
    parser.add_argument('--scheduler_type', type=str, default='step', choices=['step', 'cosine'], help='Type of learning rate scheduler (step/cosine)')
    parser.add_argument('--cosine_t_max', type=int, default=100, help='Maximum number of iterations for cosine scheduler')
    parser.add_argument('--cosine_eta_min', type=float, default=1e-6, help='Minimum learning rate for cosine scheduler')

    # Direct output guidance options
    parser.add_argument('--use_direct_output_predictor', type=int, default=0, help='Use direct output predictor in skill model (0=disable, 1=enable)')
    parser.add_argument('--use_direct_output_for_diffusion', type=int, default=0, help='Use direct output embeddings as diffusion guidance (0=disable, 1=enable)')

    # Concept guidance options (for skill model loading)
    parser.add_argument('--use_concept_in_encoder', type=int, default=0, help='Use concept in skill encoder (0=disable, 1=enable)')
    parser.add_argument('--num_concepts', type=int, default=0, help='Number of concepts (0=auto-detect from data)')
    parser.add_argument('--concept_scale', type=float, default=1.0, help='Scale factor for concept embeddings (default=1.0)')

    # Evaluation during training options
    parser.add_argument('--eval_during_training', type=int, default=0, help='Run evaluation during training (0=disable, 1=enable)')
    parser.add_argument('--eval_interval', type=int, default=50, help='Checkpoint save and eval interval (multiplier for update_steps, default=50)')
    parser.add_argument('--test_solar_dir', type=str, default=None, help='Path to test data for evaluation')
    parser.add_argument('--eval_num_episodes', type=int, default=100, help='Number of episodes per evaluation')
    parser.add_argument('--max_q_steps', type=int, default=0, help='Maximum Q-learning steps before termination (0=no limit, terminates after first checkpoint at this step)')

    # torch.compile options
    parser.add_argument('--use_compile', type=int, default=0, help='Use torch.compile for faster training (0=disable, 1=enable, requires PyTorch 2.0+)')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead', choices=['default', 'reduce-overhead', 'max-autotune'], help='Compilation mode for torch.compile')

    args = parser.parse_args()
    
    train(args)