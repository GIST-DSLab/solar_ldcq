import os
import sys
import datetime
import subprocess
import json
import re

curr_folder=os.path.abspath(__file__)
parent_folder=os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random
from models.q_learning_models import MLP_Q
# from comet_ml import Experiment
import copy
try:
    import wandb
    if not hasattr(wandb, 'init'):
        wandb = None
except Exception:
    wandb = None
from tqdm import tqdm

class DDQN(nn.Module):
    def __init__(self, state_dim, z_dim, h_dim=256, gamma=0.995, tau=0.995, lr=1e-3, num_prior_samples=100, total_prior_samples=100, extra_steps=0, horizon=10,device='cuda', diffusion_prior=None,max_grid_size=30, use_ddim=False, ddim_steps=50, ddim_eta=0.0, ddim_discr='uniform', use_mlp_embed_q=False, use_enhanced_pair_encoding=False, disable_pair_encoding=False, lr_step_size=50, lr_gamma=0.3, update_steps_multiplier=1, beta_increment=0.03, scheduler_type='step', cosine_t_max=100, cosine_eta_min=1e-6, use_positional_encoding=False, skill_model=None, use_direct_output_for_diffusion=False):
        super(DDQN,self).__init__()

        self.state_dim = state_dim
        self.z_dim = z_dim
        self.gamma = gamma
        self.lr = lr
        self.num_prior_samples = num_prior_samples
        self.total_prior_samples = total_prior_samples
        self.extra_steps = extra_steps
        self.device = device
        self.tau = tau
        self.diffusion_prior = diffusion_prior
        self.horizon = horizon
        self.max_grid_size = max_grid_size

        # DDIM options
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.ddim_discr = ddim_discr

        # Direct output guidance options
        self.skill_model = skill_model
        self.use_direct_output_for_diffusion = use_direct_output_for_diffusion

        # Store scheduling parameters
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.update_steps_multiplier = update_steps_multiplier
        self.beta_increment = beta_increment
        self.scheduler_type = scheduler_type
        self.cosine_t_max = cosine_t_max
        self.cosine_eta_min = cosine_eta_min

        self.q_net_0 = MLP_Q(state_dim=state_dim,z_dim=z_dim,h_dim=h_dim,max_grid_size=max_grid_size,use_mlp_embed_q=use_mlp_embed_q,use_enhanced_pair_encoding=use_enhanced_pair_encoding,disable_pair_encoding=disable_pair_encoding,use_positional_encoding=use_positional_encoding).to(self.device)
        self.q_net_1 = MLP_Q(state_dim=state_dim,z_dim=z_dim,h_dim=h_dim,max_grid_size=max_grid_size,use_mlp_embed_q=use_mlp_embed_q,use_enhanced_pair_encoding=use_enhanced_pair_encoding,disable_pair_encoding=disable_pair_encoding,use_positional_encoding=use_positional_encoding).to(self.device)
        self.target_net_0 = None
        self.target_net_1 = None

        # self.optimizer_0 = optim.Adam(params=self.q_net_0.parameters(), lr=lr)
        # self.optimizer_1 = optim.Adam(params=self.q_net_1.parameters(), lr=lr)
        self.optimizer_0 = optim.AdamW(params=self.q_net_0.parameters(), lr=lr)
        self.optimizer_1 = optim.AdamW(params=self.q_net_1.parameters(), lr=lr)

        # Initialize schedulers based on scheduler_type
        if self.scheduler_type == 'cosine':
            self.scheduler_0 = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_0, T_max=cosine_t_max, eta_min=cosine_eta_min)
            self.scheduler_1 = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_1, T_max=cosine_t_max, eta_min=cosine_eta_min)
        else:  # default to step scheduler
            self.scheduler_0 = optim.lr_scheduler.StepLR(self.optimizer_0, step_size=lr_step_size, gamma=lr_gamma)
            self.scheduler_1 = optim.lr_scheduler.StepLR(self.optimizer_1, step_size=lr_step_size, gamma=lr_gamma)


    @torch.no_grad()
    def get_q(self, states, clip, in_grid, sample_latents=None, pair_in=None, pair_out=None, n_samples=1000):
        if sample_latents is not None:
            perm = torch.randperm(self.total_prior_samples)[:n_samples]
            z_samples = torch.FloatTensor(sample_latents).to(self.device).reshape(sample_latents.shape[0]*n_samples,sample_latents.shape[2])
        else:
            if self.use_ddim:
                z_samples = self.diffusion_prior.ddim_sample_extra(
                    states, clip, in_grid, pair_in, pair_out, 
                    ddim_steps=self.ddim_steps, ddim_eta=self.ddim_eta, 
                    ddim_discr=self.ddim_discr, extra_steps=self.extra_steps, 
                    predict_noise=False
                )
            else:
                z_samples = self.diffusion_prior.sample_extra(states, clip, in_grid, pair_in, pair_out, predict_noise=0, extra_steps=self.extra_steps)

        # q_vals_0 = self.q_net_0(states, z_samples)[:,0]
        # q_vals_1 = self.q_net_1(states, z_samples)[:,0]
        q_vals_0 = self.q_net_0(states, clip, in_grid, z_samples, pair_in, pair_out)[:,0]
        q_vals_1 = self.q_net_1(states, clip, in_grid, z_samples, pair_in, pair_out)[:,0]
        q_vals = torch.minimum(q_vals_0, q_vals_1)
        return z_samples, q_vals


    @torch.no_grad()
    def get_max_skills(self, states, clip, in_grid, pair_in, pair_out, net=0, is_eval=False, sample_latents=None, concept_encoder=None, concept=None):
        '''
        INPUTS:
            states: batch_size x state_dim
        OUTPUTS:
            max_z: batch_size x z_dim
        '''
        if not is_eval:
            n_states = states.shape[0]
            states = states.repeat_interleave(self.num_prior_samples, 0)
            clip = clip.repeat_interleave(self.num_prior_samples, 0)
            in_grid = in_grid.repeat_interleave(self.num_prior_samples, 0)
            pair_in = pair_in.repeat_interleave(self.num_prior_samples, 0)
            pair_out = pair_out.repeat_interleave(self.num_prior_samples, 0)

        if sample_latents is not None:
            perm = torch.randperm(self.total_prior_samples)[:self.num_prior_samples]
            sample_latents = sample_latents[:,perm.cpu().numpy(),:]
            z_samples = torch.FloatTensor(sample_latents).to(self.device).reshape(sample_latents.shape[0]*self.num_prior_samples, sample_latents.shape[2])
        else:
            # Determine text_guide_emb: use direct output embedding OR concept encoding
            text_guide_emb = None

            # Priority 1: Direct output guidance (on-the-fly prediction)
            if getattr(self, 'use_direct_output_for_diffusion', False) and getattr(self, 'skill_model', None) is not None:
                # Get the original batch size before repeat_interleave
                if is_eval:
                    original_batch_size = states.shape[0]
                    original_in_grid = in_grid
                    original_pair_in = pair_in
                    original_pair_out = pair_out
                else:
                    original_batch_size = states.shape[0] // self.num_prior_samples
                    original_in_grid = in_grid[::self.num_prior_samples]
                    original_pair_in = pair_in[::self.num_prior_samples]
                    original_pair_out = pair_out[::self.num_prior_samples]

                # Predict direct output for original batch (before repeating)
                direct_output_logits = self.skill_model.direct_output_predictor(
                    original_in_grid, original_pair_in, original_pair_out
                )
                grid_size = self.max_grid_size * self.max_grid_size
                direct_logits_reshaped = direct_output_logits.reshape(original_batch_size, grid_size, 11)

                # Get predicted output (argmax)
                predicted_output = torch.argmax(direct_logits_reshaped, dim=-1)

                # Embed predicted output using state_emb_layer
                predicted_output_grid = predicted_output.reshape(original_batch_size, 1, self.max_grid_size, self.max_grid_size).float()
                direct_output_emb = self.skill_model.decoder.state_emb_layer(predicted_output_grid)
                direct_output_emb_flat = direct_output_emb.reshape(original_batch_size, -1)

                # Project to z_dim (256) to match concept embedding dimension
                if not hasattr(self.skill_model, 'direct_output_proj'):
                    self.skill_model.direct_output_proj = nn.Linear(
                        direct_output_emb_flat.shape[1], self.z_dim
                    ).to(direct_output_emb_flat.device)

                direct_output_emb_projected = self.skill_model.direct_output_proj(direct_output_emb_flat)

                # Repeat for each diffusion sample
                if is_eval:
                    text_guide_emb = direct_output_emb_projected
                else:
                    text_guide_emb = direct_output_emb_projected.repeat_interleave(self.num_prior_samples, 0)

            # Priority 2: Concept guidance (if direct output not used)
            elif concept_encoder is not None and concept:
                text_guide_emb = concept_encoder(concept, batch_size=states.shape[0])

            if self.use_ddim:
                z_samples = self.diffusion_prior.ddim_sample_extra(
                    states, clip, in_grid, pair_in, pair_out,
                    ddim_steps=self.ddim_steps, ddim_eta=self.ddim_eta,
                    ddim_discr=self.ddim_discr, extra_steps=self.extra_steps,
                    predict_noise=False,
                    text_guide_emb=text_guide_emb
                )
            else:
                z_samples = self.diffusion_prior.sample_extra(states, clip, in_grid, pair_in, pair_out, predict_noise=0, extra_steps=self.extra_steps, text_guide_emb=text_guide_emb)

        if is_eval:
            q_vals = torch.minimum(self.target_net_0(states, clip, in_grid, z_samples, pair_in, pair_out)[:, 0], self.target_net_1(states, clip, in_grid, z_samples, pair_in, pair_out)[:, 0])
        else:
            if net==0:
                q_vals = self.target_net_0(states, clip, in_grid, z_samples, pair_in, pair_out)[:,0]#self.q_net_0(states,z_samples)[:,0]
            else:
                q_vals = self.target_net_1(states, clip, in_grid, z_samples, pair_in, pair_out)[:,0]#self.q_net_1(states,z_samples)[:,0]

        if is_eval:
            return z_samples, q_vals
        q_vals = q_vals.reshape(n_states, self.num_prior_samples)
        max_vals = torch.max(q_vals, dim=1)
        max_q_vals = max_vals.values
        max_indices = max_vals.indices
        idx = torch.arange(n_states).cuda()*self.num_prior_samples + max_indices 
        max_z = z_samples[idx]

        return max_z, max_q_vals


    def learn(self, dataload_train, dataload_test=None, n_epochs=10000, update_frequency=1, diffusion_model_name='',q_checkpoint_dir = '', cfg_weight=0.0, per_buffer = 0.0, batch_size = 128, gpu_name=None ,task_name='',args=None,
              # Evaluation during training options
              eval_during_training=False, eval_interval=50, eval_script_path=None, test_solar_dir=None,
              checkpoint_dir=None, skill_model_filename=None, diffusion_model_filename=None, eval_num_episodes=100, eval_log_dir=None, max_q_steps=0):
        # assert self.diffusion_prior is not None,
        
        beta = 0.3
        update_steps = 2000
        
        d = datetime.datetime.now()
        # Handle different task name formats (e.g., "train.task_name.s10.H5" vs "five_task")
        if "." in task_name:
            task = task_name.split(".")[1]
        else:
            task = task_name
        config = {
                'task':task_name,
                'diffusion_prior':diffusion_model_name,
                'cfg_weight':cfg_weight,
                'per_buffer': per_buffer,
                'beta': beta,
                'update_steps': update_steps,
        }
        
        if args is not None:
            base_config = vars(args).copy()  # args의 모든 속성을 dict로 변환
            config = {**config, **base_config}

        import os
        run = None
        if wandb is not None:
            try:
                run = wandb.init(
                    entity='dbsgh797210',
                    project='LDCQ_single',
                    name='LDCQ_'+gpu_name+'_'+'Q'+'_'+task+'_'+str(d.month)+'.'+str(d.day)+'_'+str(d.hour)+'.'+str(d.minute),
                    config=config,
                    mode='offline',
                )
            except Exception as e:
                print(f"[WARN] wandb.init failed: {e}")
                run = None
        if run:
            print("WandB run initialized with name:", run.name)
        steps_net_0, steps_net_1, steps_total = 0, 0, 0
        self.target_net_0 = copy.deepcopy(self.q_net_0)
        self.target_net_1 = copy.deepcopy(self.q_net_1)
        self.target_net_0.eval()
        self.target_net_1.eval()
        loss_net_0, loss_net_1, loss_total = 0, 0, 0 
        epoch = 0
        update = 0
        
        if not os.path.exists(q_checkpoint_dir) :
            os.makedirs(q_checkpoint_dir)

        # Setup eval log directory
        if eval_during_training and eval_log_dir is None:
            # Auto-generate log dir based on q_checkpoint_dir
            # e.g., eval/log/gpu5_01.26
            import pathlib
            eval_base = pathlib.Path(__file__).parent.parent / 'eval' / 'log'
            q_checkpoint_basename = os.path.basename(q_checkpoint_dir)
            eval_log_dir = eval_base / q_checkpoint_basename
            eval_log_dir.mkdir(parents=True, exist_ok=True)
            eval_log_dir = str(eval_log_dir)
            print(f"[Eval] Log directory: {eval_log_dir}")

        # Run initial evaluation at step 0 (before training)
        if eval_during_training and eval_script_path and test_solar_dir:
            print("\n[Eval] Running initial evaluation at step 0 (before training)...")
            # Save initial checkpoint
            q_step = 0
            checkpoint_filename = diffusion_model_name+'_dqn_agent_'+str(q_step)+'_cfg_weight_'+str(cfg_weight)+'{}.pt'.format('_PERbuffer' if per_buffer == 1 else '')
            checkpoint_path = os.path.join(q_checkpoint_dir, checkpoint_filename)
            torch.save(self, checkpoint_path)
            print(f"[Checkpoint] Saved initial: {checkpoint_filename}")

            eval_results = self._run_evaluation(
                eval_script_path=eval_script_path,
                checkpoint_dir=checkpoint_dir,
                q_checkpoint_dir=q_checkpoint_dir,
                q_checkpoint_steps=q_step,
                test_solar_dir=test_solar_dir,
                skill_model_filename=skill_model_filename,
                diffusion_model_filename=diffusion_model_filename,
                num_episodes=eval_num_episodes,
                eval_log_dir=eval_log_dir,
                gpu_name=gpu_name,
                args=args
            )
            if run and eval_results:
                self._log_eval_results(eval_results, q_step)

        for ep in tqdm(range(n_epochs), desc="Epoch", mininterval=600.0):
            n_batch = 0
            loss_ep = 0
            self.q_net_0.train()
            self.q_net_1.train()
            
            if per_buffer:
                pbar = tqdm(range(len(dataload_train) // batch_size), mininterval=600.0)
                for _ in pbar: # same num_iters as w/o PER
                    # s0, z, reward, sT, dones, indices, weights, max_latents = dataload_train.sample(batch_size, beta)
                    s0, clip0, in_grid, z, reward, sT, clip_T, dones, pair_in, pair_out, indices, weights, max_latents = dataload_train.sample(batch_size, beta)
                    
                    
                    s0 = torch.FloatTensor(s0).to(self.device)
                    clip0 = torch.FloatTensor(clip0).to(self.device)
                    in_grid = torch.FloatTensor(in_grid).to(self.device)
                    z = torch.FloatTensor(z).to(self.device)
                    sT = torch.FloatTensor(sT).to(self.device)
                    clip_T = torch.FloatTensor(clip_T).to(self.device)
                    reward = torch.FloatTensor(reward)[...,None].to(self.device)
                    weights = torch.FloatTensor(weights).to(self.device)
                    dones = torch.FloatTensor(dones).to(self.device)
                    pair_in = torch.FloatTensor(pair_in).to(self.device)
                    pair_out = torch.FloatTensor(pair_out).to(self.device)
                    #net_id = np.random.binomial(n=1, p=0.5, size=(1,))
                    net_id = 0
                    #if net_id==0:
                    self.optimizer_0.zero_grad()

                    q_s0z = self.q_net_0(s0, clip0, in_grid, z, pair_in, pair_out)
                    max_sT_skills,_ = self.get_max_skills(sT, clip_T, in_grid, pair_in, pair_out, net=1-net_id, sample_latents=max_latents)
                    
                    with torch.no_grad():
                        q_sTz = torch.minimum(self.target_net_0(sT, clip_T, in_grid, max_sT_skills.detach(), pair_in, pair_out), self.target_net_1(sT, clip_T, in_grid, max_sT_skills.detach(), pair_in, pair_out),)
                    
                    if 'maze' in diffusion_model_name:
                        q_target = (reward + self.gamma*(reward==0.0)*q_sTz).detach()
                    elif 'kitchen' in diffusion_model_name:
                        q_target = (reward + self.gamma * q_sTz).detach()
                    else:
                        q_target = (reward + (self.gamma**self.horizon)*(dones==0.0)*q_sTz).detach()

                    bellman_loss  = (q_s0z - q_target).pow(2)
                    prios = bellman_loss[...,0] + 5e-6
                    bellman_loss = bellman_loss * weights
                    bellman_loss  = bellman_loss.mean()
                    
                    # bellman_loss = F.mse_loss(q_s0z, q_target)
                    bellman_loss.backward()
                    clip_grad_norm_(self.q_net_0.parameters(), 1)
                    self.optimizer_0.step()
                    loss_net_0 += bellman_loss.detach().item()
                    loss_total += bellman_loss.detach().item()
                    loss_ep += bellman_loss.detach().item()
                    steps_net_0 += 1
                    
                    net_id = 1
                    #else:
                    self.optimizer_1.zero_grad()

                    q_s0z = self.q_net_1(s0, clip0, in_grid, z, pair_in, pair_out)
                    max_sT_skills,_ = self.get_max_skills(sT, clip_T, in_grid, pair_in, pair_out, net=1-net_id, sample_latents=max_latents)

                    with torch.no_grad():
                        q_sTz = torch.minimum(self.target_net_0(sT, clip_T, in_grid, max_sT_skills.detach(), pair_in, pair_out), self.target_net_1(sT, clip_T, in_grid, max_sT_skills.detach(), pair_in, pair_out),)
                    
                    q_target = (reward + (self.gamma**self.horizon)*(dones==0.0)*q_sTz).detach()

                    bellman_loss  = (q_s0z - q_target).pow(2)
                    prios += bellman_loss[...,0] + 5e-6
                    bellman_loss = bellman_loss * weights
                    bellman_loss  = bellman_loss.mean()
                    
                    bellman_loss.backward()
                    clip_grad_norm_(self.q_net_1.parameters(), 1)
                    self.optimizer_1.step()
                    loss_net_1 += bellman_loss.detach().item()
                    loss_total += bellman_loss.detach().item()
                    loss_ep += bellman_loss.detach().item()
                    steps_net_1 += 1

                    dataload_train.update_priorities(indices, prios.data.cpu().numpy()/2)
                    n_batch += 1
                    steps_total += 1

                    # Update progress bar description only every 500 batches to reduce output frequency  
                    if n_batch % 500 == 0:
                        pbar.set_description(f"train loss: {loss_ep/n_batch:.4f}")
                    
                    if steps_total%update_frequency == 0:
                        loss_net_0 /= (steps_net_0+1e-4)
                        loss_net_1 /= (steps_net_1+1e-4)
                        loss_total /= 2*update_frequency
                        
                        update += 1
                        if run:
                            wandb.log({"train_Q/train_loss_0": loss_net_0})
                            wandb.log({"train_Q/train_loss_1": loss_net_1})
                            wandb.log({"train_Q/train_loss": loss_total})
                            wandb.log({"train_Q/step_per_update": update/update_steps,"train_Q/steps": steps_total})
                            wandb.log({"train_Q/epoches": ep, "train_Q/steps": steps_total})
                            
                        loss_net_0, loss_net_1, loss_total = 0,0,0
                        steps_net_0, steps_net_1 = 0,0
                        #self.target_net_0 = copy.deepcopy(self.q_net_0)
                        #self.target_net_1 = copy.deepcopy(self.q_net_1)
                        for target_param, local_param in zip(self.target_net_0.parameters(), self.q_net_0.parameters()):
                            target_param.data.copy_((1.0-self.tau)*local_param.data + (self.tau)*target_param.data)
                        for target_param, local_param in zip(self.target_net_1.parameters(), self.q_net_1.parameters()):
                            target_param.data.copy_((1.0-self.tau)*local_param.data + (self.tau)*target_param.data)
                        self.target_net_0.eval()
                        self.target_net_1.eval()
                        
                    if steps_total % (update_steps * self.update_steps_multiplier) == 0:
                        beta = np.min((beta+self.beta_increment,1))
                        self.scheduler_0.step()
                        self.scheduler_1.step()
                        
                    if steps_total % (update_steps * eval_interval) == 0:
                        q_step = steps_total // update_steps
                        checkpoint_filename = diffusion_model_name+'_dqn_agent_'+str(q_step)+'_cfg_weight_'+str(cfg_weight)+'{}.pt'.format('_PERbuffer' if per_buffer == 1 else '')
                        checkpoint_path = os.path.join(q_checkpoint_dir, checkpoint_filename)
                        torch.save(self, checkpoint_path)
                        print(f"\n[Checkpoint] Saved: {checkpoint_filename} (step {q_step})")

                        # Run evaluation if enabled
                        if eval_during_training and eval_script_path and test_solar_dir:
                            eval_results = self._run_evaluation(
                                eval_script_path=eval_script_path,
                                checkpoint_dir=checkpoint_dir,
                                q_checkpoint_dir=q_checkpoint_dir,
                                q_checkpoint_steps=q_step,
                                test_solar_dir=test_solar_dir,
                                skill_model_filename=skill_model_filename,
                                diffusion_model_filename=diffusion_model_filename,
                                num_episodes=eval_num_episodes,
                                eval_log_dir=eval_log_dir,
                                gpu_name=gpu_name,
                                args=args
                            )

                            # Log eval results to wandb
                            if run and eval_results:
                                self._log_eval_results(eval_results, q_step)

                        # Terminate training if max_q_steps is reached
                        if max_q_steps > 0 and q_step >= max_q_steps:
                            print(f"\n[Termination] Reached max_q_steps={max_q_steps}. Training completed.")
                            return

                # self.scheduler_0.step()
                # self.scheduler_1.step()
            # experiment.log_metric("train_loss_episode", loss_ep/n_batch, step=epoch)
            if run:
                wandb.log({"train_Q/train_loss_episode": loss_ep/n_batch, "train_Q/udates": update})
            epoch += 1

    def _run_evaluation(self, eval_script_path, checkpoint_dir, q_checkpoint_dir, q_checkpoint_steps,
                        test_solar_dir, skill_model_filename, diffusion_model_filename, num_episodes,
                        eval_log_dir, gpu_name, args):
        """
        Run evaluation script, save log file, and parse results.

        Returns dict with task-group results:
        {
            'task_name': {'submit': 0.83, 'reach': 0.97, 'avg_steps': 8.5, 'total': 100, 'submit_count': 83, 'reach_count': 97},
            ...
            'overall': {'submit': 0.65, 'reach': 0.72, 'avg_steps': 10.2, 'total': 500, 'submit_count': 325, 'reach_count': 360}
        }
        """
        # Create log filename: gpu5_q150_1.log
        num_diffusion_samples = getattr(args, 'num_diffusion_samples', 1) if args else 1
        log_filename = f"{gpu_name}_q{q_checkpoint_steps}_{num_diffusion_samples}.log"
        log_file = os.path.join(eval_log_dir, log_filename)

        print(f"\n[Eval] Running evaluation at step {q_checkpoint_steps}...")
        print(f"[Eval] Log file: {log_filename}")

        try:
            # Build evaluation command with all parameters
            # For evaluation, always use the full environment ID
            env_id = 'ARCLE/O2ARCv2Env-v0' if args and 'ARCLE' in getattr(args, 'env', '') else 'ARCLE/O2ARCv2Env-v0'
            cmd = [
                'python', eval_script_path,
                '--env', env_id,
                '--checkpoint_dir', checkpoint_dir,
                '--q_checkpoint_dir', q_checkpoint_dir,
                '--q_checkpoint_steps', str(q_checkpoint_steps),
                '--test_solar_dir', test_solar_dir,
                '--skill_model_filename', skill_model_filename,
                '--diffusion_model_filename', diffusion_model_filename,
                '--num_evals', str(num_episodes),
                '--policy', 'q',
                '--num_parallel_envs', '1',
            ]

            # Add all model parameters if available
            if args:
                cmd.extend(['--policy_decoder_type', getattr(args, 'policy_decoder_type', 'mlp')])
                cmd.extend(['--num_diffusion_samples', str(num_diffusion_samples)])
                cmd.extend(['--diffusion_steps', str(getattr(args, 'diffusion_steps', 500))])
                cmd.extend(['--skill_model_diffusion_steps', str(getattr(args, 'skill_model_diffusion_steps', 100))])
                cmd.extend(['--a_dim', str(getattr(args, 'a_dim', 36))])
                cmd.extend(['--z_dim', str(getattr(args, 'z_dim', 256))])
                cmd.extend(['--h_dim', str(getattr(args, 'h_dim', 512))])
                cmd.extend(['--s_dim', str(getattr(args, 's_dim', 512))])
                cmd.extend(['--train_diffusion_prior', str(getattr(args, 'train_diffusion_prior', 1))])
                cmd.extend(['--conditional_prior', str(getattr(args, 'conditional_prior', 1))])
                cmd.extend(['--normalize_latent', str(getattr(args, 'normalize_latent', 0))])
                cmd.extend(['--exec_horizon', str(getattr(args, 'exec_horizon', 1))])
                cmd.extend(['--horizon', str(getattr(args, 'horizon', 5))])
                cmd.extend(['--beta', str(getattr(args, 'beta', 0.1))])
                cmd.extend(['--max_grid_size', str(getattr(args, 'max_grid_size', 10))])
                cmd.extend(['--use_in_out', str(getattr(args, 'use_in_out', 1))])
                cmd.extend(['--max_episode_steps', str(getattr(args, 'max_episode_steps', 30))])
                cmd.extend(['--use_ddim', str(getattr(args, 'use_ddim', 1))])
                cmd.extend(['--ddim_steps', str(getattr(args, 'ddim_steps', 100))])
                cmd.extend(['--ddim_eta', str(getattr(args, 'ddim_eta', 0.0))])
                cmd.extend(['--ddim_discr', 'uniform'])
                cmd.extend(['--noise_temperature', str(getattr(args, 'noise_temperature', 1.0))])
                cmd.extend(['--encoder_type', getattr(args, 'encoder_type', 'gru')])
                cmd.extend(['--update_in_grid_on_fail', str(getattr(args, 'update_in_grid_on_fail', 1))])
                cmd.extend(['--repetition_threshold', str(getattr(args, 'repetition_threshold', 5))])
                cmd.extend(['--cfg_weight', str(getattr(args, 'cfg_weight', 0.0))])

            # Run evaluation and save output to log file
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=7200,  # 2 hour timeout
                    env=os.environ.copy()
                )

            if result.returncode != 0:
                print(f"[Eval] Warning: Eval script returned non-zero exit code")
                print(f"[Eval] stderr: {result.stderr[:1000]}")

            # Parse the log file
            eval_results = self._parse_log_file(log_file)

            if eval_results:
                overall = eval_results.get('overall', {})
                print(f"[Eval] Completed. Overall Submit: {overall.get('submit_count', 0)}/{overall.get('total', 0)} ({overall.get('submit', 0)*100:.1f}%)")
            else:
                print(f"[Eval] Failed to parse results from log file")

            return eval_results

        except subprocess.TimeoutExpired:
            print(f"[Eval] Timeout after 2 hours")
            return None
        except Exception as e:
            print(f"[Eval] Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _parse_log_file(self, log_file):
        """
        Parse evaluation log file to extract task group results.

        Returns dict with task-group results:
        {
            'task_name': {'submit': 0.83, 'reach': 0.97, 'avg_steps': 8.5, 'total': 100, 'submit_count': 83, 'reach_count': 97},
            ...
            'overall': {'submit': 0.65, 'reach': 0.72, 'avg_steps': 10.2, 'total': 500, 'submit_count': 325, 'reach_count': 360}
        }
        """
        if not os.path.exists(log_file):
            return None

        with open(log_file, 'r') as f:
            content = f.read()

        return self._parse_eval_output(content)

    def _parse_eval_output(self, output):
        """
        Parse evaluation output to extract task group results.

        Expected output format:
        Task Group: 4258a5f9-gold_standard
          Submit: 47/100 (47.0%)
          Reach:  47/100 (47.0%)
          Avg Steps: 17.8
        """
        results = {}
        current_task = None

        total_submit = 0
        total_reach = 0
        total_steps = 0
        total_count = 0

        for line in output.split('\n'):
            line = line.strip()

            # Match task group line
            task_match = re.match(r'Task Group:\s*(.+)', line)
            if task_match:
                current_task = task_match.group(1).strip()
                results[current_task] = {}
                continue

            if current_task:
                # Match Submit line: "Submit: 47/100 (47.0%)"
                submit_match = re.match(r'Submit:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)', line)
                if submit_match:
                    submit_count = int(submit_match.group(1))
                    total_task = int(submit_match.group(2))
                    submit_rate = float(submit_match.group(3)) / 100.0
                    results[current_task]['submit'] = submit_rate
                    results[current_task]['submit_count'] = submit_count
                    results[current_task]['total'] = total_task
                    total_submit += submit_count
                    total_count += total_task
                    continue

                # Match Reach line: "Reach:  47/100 (47.0%)"
                reach_match = re.match(r'Reach:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)', line)
                if reach_match:
                    reach_count = int(reach_match.group(1))
                    reach_rate = float(reach_match.group(3)) / 100.0
                    results[current_task]['reach'] = reach_rate
                    results[current_task]['reach_count'] = reach_count
                    total_reach += reach_count
                    continue

                # Match Avg Steps line: "Avg Steps: 17.8"
                steps_match = re.match(r'Avg Steps:\s*([\d.]+)', line)
                if steps_match:
                    avg_steps = float(steps_match.group(1))
                    results[current_task]['avg_steps'] = avg_steps
                    total_steps += avg_steps * results[current_task].get('total', 1)
                    current_task = None  # Done with this task
                    continue

        # Calculate overall
        if total_count > 0:
            results['overall'] = {
                'submit': total_submit / total_count,
                'reach': total_reach / total_count,
                'avg_steps': total_steps / total_count,
                'total': total_count,
                'submit_count': total_submit,
                'reach_count': total_reach
            }

        return results if results else None

    def _log_eval_results(self, eval_results, step):
        """Log evaluation results to wandb."""
        if not eval_results:
            return

        # Prepare all metrics in a single dict
        log_data = {'q_step': step}

        # Log overall metrics
        if 'overall' in eval_results:
            overall = eval_results['overall']
            log_data.update({
                'eval/overall_submit_rate': overall.get('submit', 0),
                'eval/overall_reach_rate': overall.get('reach', 0),
                'eval/overall_avg_steps': overall.get('avg_steps', 0),
                'eval/overall_submit_count': overall.get('submit_count', 0),
                'eval/overall_reach_count': overall.get('reach_count', 0),
                'eval/overall_total': overall.get('total', 0),
            })

        # Log per-task metrics
        for task_name, metrics in eval_results.items():
            if task_name == 'overall':
                continue

            # Sanitize task name for wandb (replace special chars)
            safe_name = task_name.replace('-', '_').replace('.', '_').replace(':', '_')
            log_data.update({
                f'eval_task/{safe_name}_submit_rate': metrics.get('submit', 0),
                f'eval_task/{safe_name}_reach_rate': metrics.get('reach', 0),
                f'eval_task/{safe_name}_submit_count': metrics.get('submit_count', 0),
                f'eval_task/{safe_name}_reach_count': metrics.get('reach_count', 0),
                f'eval_task/{safe_name}_total': metrics.get('total', 0),
                f'eval_task/{safe_name}_avg_steps': metrics.get('avg_steps', 0),
            })

        # Log all metrics at once
        if wandb is not None:
            try: wandb.log(log_data, step=step)
            except Exception: pass