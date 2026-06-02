import os
import sys

curr_folder=os.path.abspath(__file__)
parent_folder=os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder) 
from argparse import ArgumentParser

import numpy as np
import torch
import random
import json
import gymnasium as gym
from functools import partial
# import d4rl
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import colors
# matplotlib.use('TkAgg')
# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from models.diffusion_models import (
    Model_mlp,
    Model_cnn_mlp,
    Model_Cond_Diffusion,
)
from models.skill_model import SkillModel
# ConceptEncoder는 use_concept_guidance=1일 때만 import (sentence_transformers 의존성)
# from models.concept_encoder import ConceptEncoder
from models.discrete_concept_encoder import DiscreteConceptEncoder

# Import VAEPriorDDQN for q_vae policy
sys.path.insert(0, os.path.join(parent_folder, 'training'))
from train_q_net_vae_prior import VAEPriorDDQN

import multiprocessing as mp
from arcle.loaders import Loader
from typing import Dict, List, Tuple
from numpy.typing import NDArray
from pathlib import Path
import json
import time
import subprocess
import pickle
from tqdm import tqdm

# ARC color map for visualization
cmap = colors.ListedColormap([
    '#000000',  # 0 black
    '#0074D9',  # 1 blue
    '#FF4136',  # 2 red
    '#2ECC40',  # 3 green
    '#FFDC00',  # 4 yellow
    '#AAAAAA',  # 5 gray
    '#F012BE',  # 6 pink
    '#FF851B',  # 7 orange
    '#7FDBFF',  # 8 skyblue
    '#870C25',  # 9 maroon
    '#FFFFFF',  # 10 white (for padding)
])

norm = colors.Normalize(vmin=0, vmax=10)

def mapping_operation(operation_num):
    """Map operation number to operation name - from SOLAR-Generator utils.py"""
    action_names = [
        "Color0", "Color1", "Color2", "Color3", "Color4", "Color5", "Color6", "Color7", "Color8", "Color9",
        "FloodFill0", "FloodFill1", "FloodFill2", "FloodFill3", "FloodFill4", "FloodFill5", "FloodFill6", 
        "FloodFill7", "FloodFill8", "FloodFill9", "MoveU", "MoveD", "MoveR", "MoveL", "Rotate90", "Rotate270",
        "FlipH", "FlipV", "CopyI", "CopyO", "Paste", "CopyInput", "ResetGrid", "ResizeGrid", "Submit", "None"
    ]
    try:
        return action_names[operation_num]
    except:
        return f"Unknown({operation_num})"

def plot_one_step(ax, grid, step_info, is_input=False, is_target=False):
    """Plot one step following SOLAR-Generator style"""
    local_norm = colors.Normalize(vmin=0, vmax=10)
    ax.pcolormesh(np.flip(grid, 0), cmap=cmap, norm=local_norm, 
                 edgecolors='lightgrey', linewidth=0.05)
    ax.set_aspect('equal')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    
    if is_input:
        ax.set_title("test input", fontsize=20)
    elif is_target:
        ax.set_title("target output", fontsize=20)
    else:
        operation_num = step_info['operation']
        operation_name = mapping_operation(operation_num)
        ax.set_title(f"step {step_info['step']}", fontsize=20)
        ax.text(0.5, -0.05, f'{operation_num}  {operation_name}', 
               ha='center', transform=ax.transAxes, fontsize=16)

def create_trajectory_visualization(trajectory_data, task_id, is_successful, save_path):
    """Create visualization for a single trajectory during evaluation"""
    steps = trajectory_data['steps']
    initial_grid = trajectory_data['initial_grid']
    target_grid = trajectory_data['target_grid']
    examples = trajectory_data.get('examples', [])
    
    # Calculate layout
    n_steps = len(steps)
    n_examples = len(examples)
    
    cols = max(6, n_steps + 2)  # input + steps + target + padding
    rows = n_examples + 1  # examples + test trajectory
    
    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    gs = GridSpec(nrows=rows, ncols=cols, figure=fig)
    
    # Title
    success_status = "Success" if is_successful else "Failed"
    fig.suptitle(f'Task: {task_id} ({success_status})', fontsize=24, fontweight='bold')
    
    # Plot demonstration examples if available
    if n_examples > 0:
        for ex_idx in range(min(n_examples, 3)):  # Show max 3 examples
            if ex_idx < len(examples):
                ex_data = examples[ex_idx]
                
                # Example input
                ex_in_ax = fig.add_subplot(gs[ex_idx, 0])
                plot_one_step(ex_in_ax, ex_data['input'], None, is_input=True)
                ex_in_ax.set_title(f'demonstration input {ex_idx+1}', fontsize=16)
                
                # Example output
                ex_out_ax = fig.add_subplot(gs[ex_idx, 1])
                plot_one_step(ex_out_ax, ex_data['output'], None, is_target=True)
                ex_out_ax.set_title(f'demonstration output {ex_idx+1}', fontsize=16)
    
    # Plot test trajectory on the last row
    test_row = rows - 1
    
    # Test input
    test_input_ax = fig.add_subplot(gs[test_row, 0])
    plot_one_step(test_input_ax, initial_grid, None, is_input=True)
    
    # Each step
    for i, step in enumerate(steps):
        if i + 1 < cols - 1:  # Leave space for target
            step_ax = fig.add_subplot(gs[test_row, i + 1])
            plot_one_step(step_ax, step['grid'], step)
    
    # Target output
    if cols > n_steps + 1:
        target_ax = fig.add_subplot(gs[test_row, cols - 1])
        plot_one_step(target_ax, target_grid, None, is_target=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

class ARC_Dataloader(Loader):
    def __init__(self, data_path, train=True) -> None:
        self.data_path = data_path
        super().__init__(train=train)
        
    
    def get_path(self, **kwargs) -> List[str]:
        data_path = Path(self.data_path)

        pathlist = []

        for path, _, files in os.walk(data_path, followlinks=True):
            for name in files:
                if 'expert' in name or 'gold_standard' in name or 'golden-standard' in name:
                    pathlist.append(os.path.join(path, name))   

        self.num_dataset = len(pathlist)
        
        if(self.num_dataset == 0):
            raise ValueError("Wrong data path or empty folder. Please check the data path.")
        else:
            print("Number of episodes: {0}".format(self.num_dataset))
        
        # pathlist.sort()
        return pathlist

    def parse(self, **kwargs) -> List[Tuple[List[NDArray], List[NDArray], List[NDArray], List[NDArray], Dict]]:
        dat = []

        for p in self._pathlist:
            with open(p) as fp:
                trajectory = json.load(fp)

                ti: List[NDArray] = []
                to: List[NDArray] = []
                ei: List[NDArray] = []
                eo: List[NDArray] = []
                
                ti_h, ti_w = trajectory['grid_dim'][0]
                to_h, to_w = trajectory['grid_dim'][-1]
                
                # Single Task 경우
                ti.append(np.array(trajectory['in_grid'], dtype=np.int8)[:ti_h, :ti_w])
                to.append(np.array(trajectory['out_grid'], dtype=np.int8)[:to_h, :to_w])


                for i in range(len(trajectory['ex_in'])):
                    ei_h, ei_w = trajectory['ex_in_grid_dim'][i]
                    eo_h, eo_w = trajectory['ex_out_grid_dim'][i]
                    
                    ei.append(np.array(trajectory['ex_in'][i], dtype=np.int8)[:ei_h, :ei_w])
                    eo.append(np.array(trajectory['ex_out'][i], dtype=np.int8)[:eo_h, :eo_w])

                desc = {'id': trajectory['desc']['id'],
                        'ex_in_grid_dim': trajectory['ex_in_grid_dim'],
                        'ex_out_grid_dim' : trajectory['ex_out_grid_dim'],
                        'concept': trajectory['desc'].get('concept', ''),  # Extract concept from desc
                        # Sequential chain extra fields (present only in full-chain dataset)
                        'phase2_ex_in': trajectory['desc'].get('phase2_ex_in', []),
                        'phase2_ex_out': trajectory['desc'].get('phase2_ex_out', []),
                        'phase2_ex_in_grid_dim': trajectory['desc'].get('phase2_ex_in_grid_dim', []),
                        'phase2_ex_out_grid_dim': trajectory['desc'].get('phase2_ex_out_grid_dim', []),
                        'intermediate_grid': trajectory['desc'].get('intermediate_grid', None),
                        'intermediate_grid_dim': trajectory['desc'].get('intermediate_grid_dim', None),
                    }

                dat.append((ei,eo,ti,to,desc))  # ARCLE 순서

        return self.convert_grid_to_uint8(dat)
        # return dat

    def convert_grid_to_uint8(self, item):
            if isinstance(item, tuple):
                return tuple(self.convert_grid_to_uint8(elem) for elem in item)
            elif isinstance(item, list):
                return [self.convert_grid_to_uint8(elem) for elem in item]
            elif isinstance(item, np.ndarray):
                return np.array([self.convert_grid_to_uint8(elem) for elem in item])
            elif isinstance(item, np.integer):
                return np.uint8(item)
            elif isinstance(item, np.floating):
                return np.uint8(item)
            else:
                return item

def sel_bbox_to_mask(selection_bbox, max_grid_size):
    x, y, h, w = selection_bbox
    sel_mask = np.zeros(max_grid_size, dtype=np.int8)
    sel_mask[x:x+h+1, y:y+w+1] = 1
    return sel_mask

def q_policy(
        diffusion_model,
        skill_model,
        state_0,
        clip_0,
        in_grid,
        pair_in,
        pair_out,
        state_mean,
        state_std,
        latent_mean,
        latent_std,
        num_parallel_envs,
        num_diffusion_samples,
        extra_steps,
        planning_depth,
        predict_noise,
        append_goals,
        dqn_agent,
        use_ddim=False,
        ddim_steps=50,
        ddim_eta=0.0,
        ddim_discr='uniform',
        concept_encoder=None,
        concept=None,
        use_vae_prior_for_latent=False,
        q_threshold=None,
        q_stats_collector=None,
    ):

    # state_dim = state_0.shape[1]
    state = state_0.repeat_interleave(num_diffusion_samples, 0).unsqueeze(1)
    clip = clip_0.repeat_interleave(num_diffusion_samples, 0).unsqueeze(1)
    in_grid_inter = in_grid.repeat_interleave(num_diffusion_samples, 0).unsqueeze(1)
    pair_in_repeated = pair_in.repeat_interleave(num_diffusion_samples, 0)
    pair_out_repeated = pair_out.repeat_interleave(num_diffusion_samples, 0)

    if use_vae_prior_for_latent:
        # 후보 latent를 VAE prior에서 샘플링
        with torch.no_grad():
            latent_prior_mean, latent_prior_std = skill_model.prior(state, clip, in_grid_inter, pair_in_repeated, pair_out_repeated)
            eps = torch.normal(torch.zeros(latent_prior_mean.size()).cuda(), torch.ones(latent_prior_mean.size()).cuda())
            latent = latent_prior_mean + latent_prior_std * eps
            latent = latent.squeeze(1)  # shape: (batch, z_dim)

        # Q값 계산 (기존 Q network 사용)
        q_vals = torch.minimum(
            dqn_agent.q_net_0(state.float(), clip.float(), in_grid_inter.float(), latent, pair_in_repeated.float(), pair_out_repeated.float())[:, 0],
            dqn_agent.q_net_1(state.float(), clip.float(), in_grid_inter.float(), latent, pair_in_repeated.float(), pair_out_repeated.float())[:, 0]
        )
    else:
        # 기존 방식: diffusion에서 샘플링
        latent, q_vals = dqn_agent.get_max_skills(
            state.float(), clip.float(), in_grid_inter.float(), pair_in_repeated.float(), pair_out_repeated.float(),
            is_eval=True, concept_encoder=concept_encoder, concept=concept
        )

    best_latent = torch.zeros((num_parallel_envs, latent.shape[-1])).to(args.device)

    for env_idx in range(num_parallel_envs):
        start_idx = env_idx * num_diffusion_samples
        end_idx = start_idx + num_diffusion_samples

        # top10_values, top10_indices = torch.topk(q_vals[start_idx:end_idx], k=5)
        # top_z = latent[start_idx + top10_indices].clone()
        # print(np.round(top10_values.clone().cpu().numpy().astype(np.float64), 4).tolist())
        # for z in top_z:
        #     operation, x, y, h, w = skill_model.decoder.ll_policy.tensor_policy(
        #                             state_0[env_idx].unsqueeze(0), clip_0[env_idx].unsqueeze(0), in_grid[env_idx].unsqueeze(0), z, pair_in[env_idx].unsqueeze(0), pair_out[env_idx].unsqueeze(0))

        #     operation = torch.argmax(operation.clone()).cpu().numpy()  # argmax 후 CPU로 옮겨 NumPy로 변환하고 스칼라로 만듦
        #     x = torch.argmax(x.clone()).cpu().numpy()  # argmax 후 NumPy 배열로 변환
        #     y = torch.argmax(y.clone()).cpu().numpy()  # argmax 후 NumPy 배열로 변환
        #     h = torch.argmax(h.clone()).cpu().numpy()  # argmax 후 NumPy 배열로 변환
        #     w = torch.argmax(w.clone()).cpu().numpy()

        #     print("op: {0}, x: {1}, y: {2}, h : {3}, w : {4}".format(operation, x, y, h, w))

        local_q = q_vals[start_idx:end_idx]

        # [Exp 1] Q-threshold gating: discard proposals below threshold
        if q_threshold is not None:
            mask = local_q > q_threshold
            if mask.any():
                gated_q = local_q.clone()
                gated_q[~mask] = float('-inf')
            else:
                gated_q = local_q  # all below threshold: fallback to argmax over all
        else:
            gated_q = local_q

        # [Exp 3] Q-value stats logging
        if q_stats_collector is not None:
            q_np = local_q.float().cpu().numpy()
            n_valid = int((local_q > q_threshold).sum().item()) if q_threshold is not None else len(q_np)
            q_stats_collector.append({
                'mean': float(q_np.mean()),
                'std': float(q_np.std()),
                'max': float(q_np.max()),
                'min': float(q_np.min()),
                'n_valid': n_valid,
                'n_total': len(q_np),
            })

        max_idx = torch.argmax(gated_q)
        best_latent[env_idx] = latent[start_idx + max_idx]

    return best_latent

def q_policy_with_skill_prior(
        diffusion_model,
        skill_model,
        state_0,
        clip_0,
        in_grid,
        pair_in,
        pair_out,
        state_mean,
        state_std,
        latent_mean,
        latent_std,
        num_parallel_envs,
        num_diffusion_samples,
        extra_steps,
        planning_depth,
        predict_noise,
        append_goals,
        dqn_agent,
        use_ddim=False,
        ddim_steps=50,
        ddim_eta=0.0,
        ddim_discr='uniform',
        concept_encoder=None,
        concept=None,
        use_vae_prior_for_latent=False,  # 이 함수는 항상 VAE prior 사용, 인자는 호환성을 위해 추가
        q_threshold=None,
        q_stats_collector=None,
    ):

    # state_dim = state_0.shape[1]
    state = state_0.repeat_interleave(num_diffusion_samples, 0).unsqueeze(1)
    clip = clip_0.repeat_interleave(num_diffusion_samples, 0).unsqueeze(1)
    in_grid_inter = in_grid.repeat_interleave(num_diffusion_samples, 0).unsqueeze(1)
    pair_in_repeated = pair_in.repeat_interleave(num_diffusion_samples, 0)
    pair_out_repeated = pair_out.repeat_interleave(num_diffusion_samples, 0)

    # skill model의 prior를 사용하여 latent 샘플링
    with torch.no_grad():
        latent_prior_mean, latent_prior_std = skill_model.prior(state, clip, in_grid_inter, pair_in_repeated, pair_out_repeated)
        eps = torch.normal(torch.zeros(latent_prior_mean.size()).cuda(), torch.ones(latent_prior_mean.size()).cuda())
        latent = latent_prior_mean + latent_prior_std * eps

    # Q값 계산
    q_vals = torch.minimum(
        dqn_agent.q_net_0(state.float(), clip.float(), in_grid_inter.float(), latent.squeeze(1), pair_in_repeated.float(), pair_out_repeated.float())[:, 0],
        dqn_agent.q_net_1(state.float(), clip.float(), in_grid_inter.float(), latent.squeeze(1), pair_in_repeated.float(), pair_out_repeated.float())[:, 0]
    )

    best_latent = torch.zeros((num_parallel_envs, latent.shape[-1])).to(state_0.device)

    for env_idx in range(num_parallel_envs):
        start_idx = env_idx * num_diffusion_samples
        end_idx = start_idx + num_diffusion_samples

        local_q = q_vals[start_idx:end_idx]

        # [Exp 1] Q-threshold gating
        if q_threshold is not None:
            mask = local_q > q_threshold
            gated_q = local_q.clone()
            gated_q[~mask] = float('-inf') if mask.any() else local_q
        else:
            gated_q = local_q

        # [Exp 3] Q-value stats logging
        if q_stats_collector is not None:
            q_np = local_q.float().cpu().numpy()
            n_valid = int((local_q > q_threshold).sum().item()) if q_threshold is not None else len(q_np)
            q_stats_collector.append({
                'mean': float(q_np.mean()), 'std': float(q_np.std()),
                'max': float(q_np.max()), 'min': float(q_np.min()),
                'n_valid': n_valid, 'n_total': len(q_np),
            })

        max_idx = torch.argmax(gated_q)
        best_latent[env_idx] = latent[start_idx + max_idx].squeeze()

    return best_latent

def diffusion_prior_policy(
        diffusion_model,
        skill_model,
        state_0,
        clip_0,
        in_grid,
        pair_in,
        pair_out,
        state_mean,
        state_std,
        latent_mean,
        latent_std,
        num_parallel_envs,
        num_diffusion_samples,
        extra_steps,
        planning_depth,
        predict_noise,
        append_goals,
        dqn_agent,
        use_ddim=False,
        ddim_steps=50,
        ddim_eta=0.0,
        ddim_discr='uniform',
        concept_encoder=None,
        concept=None,
        use_vae_prior_for_latent=False  # 미사용, 호환성을 위해 추가
    ):

    state_dim = state_0.shape[1]
    state_0 = state_0.unsqueeze(1)

    clip_dim = clip_0.shape[1]
    clip_0 = clip_0.unsqueeze(1)
    
    in_grid_dim = in_grid.shape[1]
    in_grid_unsq = in_grid.unsqueeze(1)
    
    if use_ddim:
        latent = diffusion_model.ddim_sample_extra(
            (state_0 - state_mean) / state_std, 
            (clip_0 - state_mean) / state_std, 
            (in_grid_unsq - state_mean) / state_std, 
            pair_in, pair_out, 
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
            ddim_discr=ddim_discr,
            extra_steps=extra_steps,
            predict_noise=bool(predict_noise)
        ) * latent_std + latent_mean
    else:
        latent = diffusion_model.sample_extra(
            (state_0 - state_mean) / state_std, 
            (clip_0 - state_mean) / state_std, 
            (in_grid_unsq - state_mean) / state_std, 
            pair_in, pair_out, 
            predict_noise=predict_noise, 
            extra_steps=extra_steps
        ) * latent_std + latent_mean

    
    return latent


def prior_policy(
        diffusion_model,
        skill_model,
        state_0,
        clip_0,
        in_grid,
        pair_in,
        pair_out,
        state_mean,
        state_std,
        latent_mean,
        latent_std,
        num_parallel_envs,
        num_diffusion_samples,
        extra_steps,
        planning_depth,
        predict_noise,
        append_goals,
        dqn_agent,
        use_ddim=False,
        ddim_steps=50,
        ddim_eta=0.0,
        ddim_discr='uniform',
        concept_encoder=None,
        concept=None,
        use_vae_prior_for_latent=False  # 미사용, 호환성을 위해 추가
    ):

    state_dim = state_0.shape[1]
    state_0 = state_0.unsqueeze(1)

    clip_dim = clip_0.shape[1]
    clip_0 = clip_0.unsqueeze(1)

    in_grid_dim = in_grid.shape[1]
    in_grid_unsq = in_grid.unsqueeze(1)

    latent, latent_prior_std = skill_model.prior(state_0, clip_0, in_grid_unsq, pair_in, pair_out)
    eps = torch.normal(torch.zeros(latent.size()).cuda(), torch.ones(latent.size()).cuda())

    return latent + latent_prior_std * eps


def vae_diffusion_prior_policy(
        diffusion_model,
        skill_model,
        state_0,
        clip_0,
        in_grid,
        pair_in,
        pair_out,
        state_mean,
        state_std,
        latent_mean,
        latent_std,
        num_parallel_envs,
        num_diffusion_samples,
        extra_steps,
        planning_depth,
        predict_noise,
        append_goals,
        dqn_agent,
        use_ddim=False,
        ddim_steps=50,
        ddim_eta=0.0,
        ddim_discr='uniform',
        concept_encoder=None,
        concept=None,
        use_vae_prior_for_latent=False  # 미사용, 호환성을 위해 추가
    ):
    """
    Use VAE's built-in diffusion prior (trained jointly with VAE) for sampling.
    This diffusion prior was trained end-to-end with the VAE encoder/decoder.
    """
    state_dim = state_0.shape[1]
    state_0 = state_0.unsqueeze(1)

    clip_dim = clip_0.shape[1]
    clip_0 = clip_0.unsqueeze(1)

    in_grid_dim = in_grid.shape[1]
    in_grid_unsq = in_grid.unsqueeze(1)

    # Use the VAE's internal diffusion prior (skill_model.diffusion_prior)
    # diffusion_model here is actually skill_model.diffusion_prior
    if use_ddim:
        latent = diffusion_model.ddim_sample_extra(
            (state_0 - state_mean) / state_std,
            (clip_0 - state_mean) / state_std,
            (in_grid_unsq - state_mean) / state_std,
            pair_in, pair_out,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
            ddim_discr=ddim_discr,
            extra_steps=extra_steps,
            predict_noise=bool(predict_noise)
        ) * latent_std + latent_mean
    else:
        latent = diffusion_model.sample_extra(
            (state_0 - state_mean) / state_std,
            (clip_0 - state_mean) / state_std,
            (in_grid_unsq - state_mean) / state_std,
            pair_in, pair_out,
            predict_noise=predict_noise,
            extra_steps=extra_steps
        ) * latent_std + latent_mean

    return latent
        
def eval_func(diffusion_model,
              skill_model,
              policy,
              envs,
              state_dim,
              state_mean,
              state_std,
              latent_mean,
              latent_std,
              num_evals,
              num_parallel_envs,
              num_diffusion_samples,
              extra_steps,
              planning_depth,
              exec_horizon,
              predict_noise,
              render,
              append_goals,
              dqn_agent=None,
              env_name=None,
              skill_latent_data=None,
              loader=None,
              use_ddim=False,
              ddim_steps=50,
              ddim_eta=0.0,
              ddim_discr='uniform',
              concept_encoder=None):
    
    print("Render mode : None")
    print(f"test_data: {args.test_solar_dir}")
    print(f"exec_horizon: {args.exec_horizon}")
    print(f"q_checkpoint_dir: {args.q_checkpoint_dir}")
    print(f"q_checkpoint_steps: {args.q_checkpoint_steps}")
    print(f"checkpoint_dir :{args.checkpoint_dir}")
    print(f"skill_model_filename :{args.skill_model_filename}")
    print(f"use_vae_prior_for_latent: {args.use_vae_prior_for_latent}")

    
    with torch.no_grad():
        assert num_evals % num_parallel_envs == 0
        num_evals = num_evals // num_parallel_envs

        # Check dataset size and adjust num_evals if necessary
        try:
            # Try to get dataset size from loader
            dataset_size = getattr(loader, 'problem_size', None)
            if dataset_size is None:
                # Fallback: try to access the length of the loader's problem data
                if hasattr(loader, 'problem_data'):
                    dataset_size = len(loader.problem_data)
                else:
                    # Last resort: assume at least as many as num_evals
                    dataset_size = num_evals
                    print(f"Warning: Could not determine dataset size, assuming {num_evals}")
            
            print(f"Dataset size: {dataset_size}, Requested num_evals: {num_evals}")
            
            # Adjust num_evals to not exceed dataset size
            if num_evals > dataset_size:
                print(f"Warning: num_evals ({num_evals}) exceeds dataset size ({dataset_size}). Adjusting num_evals to {dataset_size}")
                num_evals = dataset_size
                
        except Exception as e:
            print(f"Warning: Error checking dataset size: {e}. Proceeding with num_evals={num_evals}")

        score_submit = 0
        score_reach = 0

        # Task-specific score tracking
        task_scores = {}  # {task_id: {'submit': bool, 'reach': bool, 'steps': int}}
        task_group_scores = {}  # {task_group: {'total': int, 'submit': int, 'reach': int, 'total_steps': int}}

        # [Exp 4] buffer for online data collection
        _online_data_buf = {k: [] for k in
                            ['states', 'clips', 'in_grids', 'latents',
                             'sTs', 'clip_Ts', 'pair_ins', 'pair_outs']}
        
        # Trajectory collection for visualization
        trajectories = []
        
        # pbar = tqdm(range(num_evals))
        
        for eval_step in range(num_evals):
        # for eval_step in pbar:
            # Flag to break out of outer loop
            data_access_error = False
            
            state_0 = torch.full((num_parallel_envs, args.max_grid_size, args.max_grid_size), 10).to(args.device)
            clip_0 = torch.full((num_parallel_envs, args.max_grid_size, args.max_grid_size), 10).to(args.device)
            in_grid = torch.full((num_parallel_envs, args.max_grid_size, args.max_grid_size), 10).to(args.device)
            
            pair_in = torch.full((num_parallel_envs, 3, args.max_grid_size, args.max_grid_size), 10).to(args.device)
            pair_out = torch.full((num_parallel_envs, 3, args.max_grid_size, args.max_grid_size), 10).to(args.device)
            
            done = [False] * num_parallel_envs
            reach_ans = [False] * num_parallel_envs
            count_none = [0] * num_parallel_envs
            # Sequential chain: track phase per env (1=Phase1, 2=Phase2)
            seq_phase = [1] * num_parallel_envs
            for env_idx in range(len(envs)):
                # input-output pair 추출 (safe data access)
                try:
                    ex_in, ex_out, tt_in, tt_out, desc = loader.pick(data_index=eval_step)
                except (AssertionError, IndexError) as e:
                    print(f"Warning: Cannot access data at index {eval_step}. Error: {e}")
                    print(f"Breaking evaluation loop at step {eval_step}")
                    data_access_error = True
                    break
            
            # Break outer loop if data access failed
            if data_access_error:
                break
                
            for i in range(3):
                ei_h, ei_w = desc['ex_in_grid_dim'][i]
                eo_h, eo_w = desc['ex_out_grid_dim'][i]
            
                pair_in[env_idx, i, :ei_h, :ei_w] = torch.from_numpy(np.array(ex_in[i])).to(args.device)
                pair_out[env_idx, i, :eo_h, :eo_w] = torch.from_numpy(np.array(ex_out[i])).to(args.device)

            # test input 추출 (safe environment reset)
            try:
                obs, info = envs[env_idx].reset(options={'prob_index': eval_step, 'subprob_index': 0, 'adaptation':False})
            except (AssertionError, IndexError) as e:
                print(f"Warning: Cannot reset environment at index {eval_step}. Error: {e}")
                print(f"Breaking evaluation loop at step {eval_step}")
                break

            obs_x, obs_y = obs['grid_dim']
            clip_x, clip_y = obs['clip_dim']
            
            state_0[env_idx, :obs_x, :obs_y] = torch.from_numpy(obs['grid'][:obs_x, :obs_y].copy()).to(args.device)
            clip_0[env_idx, :clip_x, :clip_y] = torch.from_numpy(obs['clip'][:clip_x, :clip_y].copy()).to(args.device)
            in_grid[env_idx, :obs_x, :obs_y] = torch.from_numpy(obs['grid'][:obs_x, :obs_y].copy()).to(args.device)

            # DETAILED DEBUG: Print exact model inputs on first episode
            if eval_step == 0 and env_idx == 0:
                print(f"\n{'='*80}")
                print("ARCLE MODEL INPUTS (First Episode)")
                print(f"{'='*80}")
                print(f"state_0[0] shape: {state_0[0].shape}, dtype: {state_0[0].dtype}")
                print(f"clip_0[0] shape: {clip_0[0].shape}, dtype: {clip_0[0].dtype}")
                print(f"in_grid[0] shape: {in_grid[0].shape}, dtype: {in_grid[0].dtype}")
                print(f"pair_in[0] shape: {pair_in[0].shape}, dtype: {pair_in[0].dtype}")
                print(f"pair_out[0] shape: {pair_out[0].shape}, dtype: {pair_out[0].dtype}")
                print(f"obs['grid_dim']: {obs['grid_dim']}, obs['clip_dim']: {obs['clip_dim']}")

                # Show actual values for valid regions
                print(f"\nstate_0[0] valid region [0:{obs_x}, 0:{obs_y}]:")
                print(state_0[0, :obs_x, :obs_y].cpu().numpy())

                if clip_x > 0 and clip_y > 0:
                    clip_non_pad = clip_0[0, :clip_x, :clip_y] != 10
                    if clip_non_pad.any():
                        print(f"\nclip_0[0] valid region [0:{clip_x}, 0:{clip_y}]:")
                        print(clip_0[0, :clip_x, :clip_y].cpu().numpy())
                    else:
                        print(f"\nclip_0[0]: Valid region [0:{clip_x}, 0:{clip_y}] but all zeros (empty)")
                else:
                    print(f"\nclip_0[0]: Empty (clip_dim = 0)")

                print(f"\npair_in[0] examples (non-padding regions):")
                for i in range(pair_in.shape[1]):
                    pair_non_pad = pair_in[0, i] != 10
                    if pair_non_pad.any():
                        rows = torch.where(pair_non_pad.any(dim=1))[0]
                        cols = torch.where(pair_non_pad.any(dim=0))[0]
                        h = rows[-1].item() - rows[0].item() + 1
                        w = cols[-1].item() - cols[0].item() + 1
                        print(f"  Pair {i} valid region: [{h}x{w}]")
                        if h <= 10 and w <= 10:
                            print(f"    Input:\n{pair_in[0, i, :h, :w].cpu().numpy()}")
                            print(f"    Output:\n{pair_out[0, i, :h, :w].cpu().numpy()}")
                print(f"{'='*80}\n")

            # state_0[env_idx] = torch.from_numpy(envs[env_idx].reset())

            env_step = 0
                
            if 'ARCLE/O2ARCv2Env-v0' in env_name:
                total_steps = args.max_episode_steps if args.max_episode_steps > 0 else 20  # Use 20 as default if unlimited
            else:
                ValueError("Only ARCLE!")

            task_id = desc['id']
            # Extract task group by removing trailing _number suffixes
            import re
            # Remove one or more trailing _number patterns (e.g., _1, _1_2, _10_20)
            task_group = re.sub(r'(_\d+)+$', '', task_id)

            # Extract concept for concept-guided diffusion
            concept = desc.get('concept', '')

            print(f"id: {task_id} (group: {task_group})")
            if concept and concept_encoder is not None:
                print(f"concept: {concept}")
            
            # Initialize task score tracking
            if task_id not in task_scores:
                task_scores[task_id] = {'submit': False, 'reach': False, 'steps': 0}
            
            # Initialize task group score tracking
            if task_group not in task_group_scores:
                task_group_scores[task_group] = {'total': 0, 'submit': 0, 'reach': 0, 'total_steps': 0}
            
            # Initialize trajectory data for this episode
            current_trajectory = {
                'task_id': desc['id'],
                'initial_grid': obs['grid'][:obs_x, :obs_y].copy(),
                'target_grid': tt_out[0].copy(),
                'examples': [{'input': ex_in[i], 'output': ex_out[i]} for i in range(len(ex_in))],
                'steps': [],
                'is_successful': False
            }
                
            while env_step < total_steps:
                # s0, clip0, pair_in, pair_out,
                best_latent = policy(
                                diffusion_model,
                                skill_model,
                                state_0,
                                clip_0,
                                in_grid,
                                pair_in,
                                pair_out,
                                state_mean,
                                state_std,
                                latent_mean,
                                latent_std,
                                num_parallel_envs,
                                num_diffusion_samples,
                                extra_steps,
                                planning_depth,
                                predict_noise,
                                append_goals,
                                dqn_agent,
                                use_ddim,
                                ddim_steps,
                                ddim_eta,
                                ddim_discr,
                                concept_encoder,
                                concept,
                                args.use_vae_prior_for_latent
                )
                
                # Save skill latents for analysis
                for env_idx in range(len(envs)):
                    if not done[env_idx]:  # Only save for active environments
                        skill_latent_data['latents'].append(best_latent[env_idx].cpu().numpy().copy())
                        skill_latent_data['task_ids'].append(desc['id'])
                        skill_latent_data['task_groups'].append(desc['id'].split('-')[0])  # Extract task group
                        skill_latent_data['step_numbers'].append(env_step)
                        skill_latent_data['trajectory_ids'].append(f"{desc['id']}_env{env_idx}")
                
                # best_latent = torch.randn(1, 1, 1, args.z_dim).to(args.device)

                for _ in range(exec_horizon):
                    for env_idx in range(len(envs)):
                        
                        # for sample_num in range(args.num_diffusion_samples):
                        if not done[env_idx]:
                            # print(state_0[env_idx].shape)
                            # print(clip_0[env_idx].shape)
                            # print(in_grid[env_idx].shape)
                            # print(best_latent[env_idx].shape)
                            # print(pair_in[env_idx].shape)
                            # print(pair_out[env_idx].shape)
                            operation, x, y, h, w = skill_model.decoder.ll_policy.tensor_policy(
                                state_0[env_idx].unsqueeze(0), clip_0[env_idx].unsqueeze(0), in_grid[env_idx].unsqueeze(0), best_latent[env_idx].unsqueeze(0), pair_in[env_idx].unsqueeze(0), pair_out[env_idx].unsqueeze(0))
                            
                            operation = torch.argmax(operation.clone(), dim=-1).cpu().numpy().item()  # argmax 후 CPU로 옮겨 NumPy로 변환하고 스칼라로 만듦
                            x = torch.argmax(x.clone(), dim=-1).cpu().numpy().item()  # argmax 후 NumPy 배열로 변환
                            y = torch.argmax(y.clone(), dim=-1).cpu().numpy().item()  # argmax 후 NumPy 배열로 변환
                            h = torch.argmax(h.clone(), dim=-1).cpu().numpy().item()  # argmax 후 NumPy 배열로 변환
                            w = torch.argmax(w.clone(), dim=-1).cpu().numpy().item() 
                            
                            # operation = np.argmax(operation)
                            # x = np.argmax(x)
                            # y = np.argmax(y)
                            # h = np.argmax(h)
                            # w = np.argmax(w)
                            
                            if(render != "ansi"):
                                if operation == 35:
                                    print("None!!")
                                    # count_none[env_idx] +=1
                                    # if count_none[env_idx] >= 10:
                                    #     break
                                    done[env_idx] = 1
                                    
                                    
                                print("Step: {0}| op: {1}, x: {2}, y: {3}, h : {4}, w : {5}".format(env_step, operation, x, y, h, w))
                                
                                # Collect step data for visualization
                                if env_idx == 0:  # Only collect for first environment to avoid duplicates
                                    # Get current grid state (before action)
                                    current_grid = state_0[env_idx][:args.max_grid_size, :args.max_grid_size].cpu().numpy()
                                    # Remove padding (value 10)
                                    valid_rows = np.where(np.any(current_grid != 10, axis=1))[0]
                                    valid_cols = np.where(np.any(current_grid != 10, axis=0))[0]
                                    if len(valid_rows) > 0 and len(valid_cols) > 0:
                                        current_grid = current_grid[valid_rows[0]:valid_rows[-1]+1, valid_cols[0]:valid_cols[-1]+1]
                                    
                                    step_data = {
                                        'step': env_step,
                                        'operation': int(operation),
                                        'x': int(x), 'y': int(y), 'h': int(h), 'w': int(w),
                                        'grid': current_grid.copy()
                                    }
                                    current_trajectory['steps'].append(step_data)
                            
                            # Operation에서 None 나오면 env에 안넣고 패스
                            if(operation == 35):
                                done[env_idx] = 1
                            else:
                                # time.sleep(1.0)
                                select = sel_bbox_to_mask((x, y, h, w), (args.max_grid_size, args.max_grid_size))
                                action = {'selection': select.astype(bool), 'operation': operation}
                                
                                try:
                                    obs, reward, done[env_idx], _, _ = envs[env_idx].step(action)
                                
                                    # time.sleep(2.0)
                                    if reward:
                                        score_submit += 1
                                        task_scores[task_id]['submit'] = True
                                        task_group_scores[task_group]['submit'] += 1

                                        # [Exp 4] collect successful (state, latent, sT) for online fine-tuning
                                        if getattr(args, 'save_online_data', 0):
                                            _obs_x, _obs_y = obs['grid_dim']
                                            _clip_x, _clip_y = obs['clip_dim']
                                            _sT = torch.full((args.max_grid_size, args.max_grid_size), 10.0)
                                            _sT[:_obs_x, :_obs_y] = torch.from_numpy(obs['grid'][:_obs_x, :_obs_y].copy())
                                            _clipT = torch.full((args.max_grid_size, args.max_grid_size), 10.0)
                                            _clipT[:_clip_x, :_clip_y] = torch.from_numpy(obs['clip'][:_clip_x, :_clip_y].copy())
                                            _online_data_buf['states'].append(state_0[env_idx].cpu().numpy().copy())
                                            _online_data_buf['clips'].append(clip_0[env_idx].cpu().numpy().copy())
                                            _online_data_buf['in_grids'].append(in_grid[env_idx].cpu().numpy().copy())
                                            _online_data_buf['latents'].append(best_latent[env_idx].cpu().numpy().copy())
                                            _online_data_buf['sTs'].append(_sT.numpy())
                                            _online_data_buf['clip_Ts'].append(_clipT.numpy())
                                            _online_data_buf['pair_ins'].append(pair_in[env_idx].cpu().numpy().copy())
                                            _online_data_buf['pair_outs'].append(pair_out[env_idx].cpu().numpy().copy())

                                    obs_x, obs_y  = obs['grid_dim']
                                    state_0[env_idx].fill_(10)
                                    state_0[env_idx, :obs_x, :obs_y] = torch.from_numpy(obs['grid'][:obs_x, :obs_y].copy())

                                    clip_x, clip_y  = obs['clip_dim']
                                    clip_0[env_idx].fill_(10)
                                    clip_0[env_idx, :clip_x, :clip_y] = torch.from_numpy(obs['clip'][:clip_x, :clip_y].copy())

                                    if np.array_equal(obs['grid'][:obs_x, :obs_y], tt_out[0]):
                                        reach_ans[env_idx] = True

                                    # ── Sequential chain: phase switch check ──────────────────
                                    if getattr(args, 'sequential_chain', 0) and seq_phase[env_idx] == 1:
                                        inter_grid = desc.get('intermediate_grid')
                                        inter_dim = desc.get('intermediate_grid_dim')
                                        if inter_grid is not None and inter_dim is not None:
                                            ih, iw = int(inter_dim[0]), int(inter_dim[1])
                                            curr_np = obs['grid'][:obs_x, :obs_y]
                                            inter_np = np.array(inter_grid)[:ih, :iw]
                                            if obs_x == ih and obs_y == iw and np.array_equal(curr_np, inter_np):
                                                seq_phase[env_idx] = 2
                                                print(f"[SeqChain] Phase 1→2 switch at step {env_step} (env {env_idx})")
                                                # Update in_grid to current state (a+b) so Phase 2 model
                                                # sees the correct "task input" reference
                                                in_grid[env_idx].fill_(10)
                                                in_grid[env_idx, :obs_x, :obs_y] = state_0[env_idx, :obs_x, :obs_y]
                                                ph2_ex_in = desc.get('phase2_ex_in', [])
                                                ph2_ex_out = desc.get('phase2_ex_out', [])
                                                ph2_dims_in = desc.get('phase2_ex_in_grid_dim', [])
                                                ph2_dims_out = desc.get('phase2_ex_out_grid_dim', [])
                                                pair_in[env_idx].fill_(10)
                                                pair_out[env_idx].fill_(10)
                                                for pi in range(min(3, len(ph2_ex_in))):
                                                    p2h_i, p2w_i = int(ph2_dims_in[pi][0]), int(ph2_dims_in[pi][1])
                                                    p2h_o, p2w_o = int(ph2_dims_out[pi][0]), int(ph2_dims_out[pi][1])
                                                    pair_in[env_idx, pi, :p2h_i, :p2w_i] = torch.from_numpy(
                                                        np.array(ph2_ex_in[pi], dtype=np.uint8)[:p2h_i, :p2w_i]).to(args.device)
                                                    pair_out[env_idx, pi, :p2h_o, :p2w_o] = torch.from_numpy(
                                                        np.array(ph2_ex_out[pi], dtype=np.uint8)[:p2h_o, :p2w_o]).to(args.device)
                                    # ─────────────────────────────────────────────────────────

                                except Exception as e:
                                    print("ARCLE execution error")
                                    continue
                                    
                            if render and env_idx == 0:
                                envs[env_idx].render()
                            
                            if(done[env_idx]):
                                print("Terminal!!")
                                print("================================================================")
                                # time.sleep(2.0)
                                break
                        
                    
                    env_step += 1
                    #print(env_step, score_submit)
                    if env_step > total_steps:
                        break
                if sum(done) == num_parallel_envs:
                    break
                
            for reach in reach_ans:
                if reach :
                    score_reach += 1 
                    task_scores[task_id]['reach'] = True
                    task_group_scores[task_group]['reach'] += 1
            
            # Record steps taken for this task
            task_scores[task_id]['steps'] = env_step
            task_group_scores[task_group]['total'] += 1
            task_group_scores[task_group]['total_steps'] += env_step
            
            # Mark trajectory as successful and save it
            current_trajectory['is_successful'] = reach_ans[0]  # Use first env result
            trajectories.append(current_trajectory)
                
            print("env_step : {0}".format(env_step))
            
            total_runs = (eval_step + 1) * num_parallel_envs
            
            print(f'Total score: {score_submit} out of {total_runs} i.e. Acc: {(score_submit / total_runs) * 100}%')
            print(f'Reach answer: {score_reach} out of {total_runs} i.e. Acc: {(score_reach / total_runs) * 100}%')

        # Print task group results at the end
        print("\n" + "="*80)
        print("TASK GROUP RESULTS")
        print("="*80)
        
        for task_group, scores in sorted(task_group_scores.items()):
            submit_rate = (scores['submit'] / scores['total']) * 100 if scores['total'] > 0 else 0
            reach_rate = (scores['reach'] / scores['total']) * 100 if scores['total'] > 0 else 0
            avg_steps = scores['total_steps'] / scores['total'] if scores['total'] > 0 else 0
            
            print(f"Task Group: {task_group}")
            print(f"  Submit: {scores['submit']}/{scores['total']} ({submit_rate:.1f}%)")
            print(f"  Reach:  {scores['reach']}/{scores['total']} ({reach_rate:.1f}%)")
            print(f"  Avg Steps: {avg_steps:.1f}")
            print()
        
        # Calculate overall average scores
        total_tasks = sum(scores['total'] for scores in task_group_scores.values())
        total_submit = sum(scores['submit'] for scores in task_group_scores.values())
        total_reach = sum(scores['reach'] for scores in task_group_scores.values())
        total_steps = sum(scores['total_steps'] for scores in task_group_scores.values())

        overall_submit_rate = (total_submit / total_tasks) * 100 if total_tasks > 0 else 0
        overall_reach_rate = (total_reach / total_tasks) * 100 if total_tasks > 0 else 0
        overall_avg_steps = total_steps / total_tasks if total_tasks > 0 else 0

        print("OVERALL AVERAGE SCORE")
        print("="*80)
        print(f"Total Tasks: {total_tasks}")
        print(f"Total Groups: {len(task_group_scores)}")
        print(f"Submit Rate: {total_submit}/{total_tasks} ({overall_submit_rate:.2f}%)")
        print(f"Reach Rate:  {total_reach}/{total_tasks} ({overall_reach_rate:.2f}%)")
        print(f"Average Steps: {overall_avg_steps:.2f}")
        print("="*80)

        # # Print individual task results
        # print("\n" + "="*80)
        # print("INDIVIDUAL TASK RESULTS")
        # print("="*80)

        # for task_id, scores in sorted(task_scores.items()):
        #     submit_status = "✓" if scores['submit'] else "✗"
        #     reach_status = "✓" if scores['reach'] else "✗"
        #     print(f"Task: {task_id}")
        #     print(f"  Submit: {submit_status} | Reach: {reach_status} | Steps: {scores['steps']}")

        # print("="*80)

        # # Save skill latent data for analysis
        # skill_latent_filename = f"skill_latents_{args.policy}_{args.skill_model_filename[:-4]}"
        # if hasattr(args, 'q_checkpoint_steps') and args.q_checkpoint_steps:
        #     skill_latent_filename += f"_q{args.q_checkpoint_steps}"
        # skill_latent_filename += ".pkl"
        
        # skill_latent_path = os.path.join(os.path.dirname(__file__), skill_latent_filename)
        
        # print(f"\nSaving {len(skill_latent_data['latents'])} skill latents to: {skill_latent_path}")
        # with open(skill_latent_path, 'wb') as f:
        #     pickle.dump(skill_latent_data, f)
        
        # print(f"Skill latent data saved with {len(set(skill_latent_data['task_ids']))} unique tasks")
        
        # [Exp 4] save online data collected during eval
        if getattr(args, 'save_online_data', 0) and _online_data_buf['states']:
            n_online = len(_online_data_buf['states'])
            save_dir = getattr(args, 'online_data_dir', args.q_checkpoint_dir)
            os.makedirs(save_dir, exist_ok=True)
            prefix = os.path.join(save_dir, args.skill_model_filename[:-4] + '_online')
            np.save(prefix + '_states.npy',   np.stack(_online_data_buf['states']))
            np.save(prefix + '_clips.npy',    np.stack(_online_data_buf['clips']))
            np.save(prefix + '_in_grids.npy', np.stack(_online_data_buf['in_grids']))
            np.save(prefix + '_latents.npy',  np.stack(_online_data_buf['latents']))
            np.save(prefix + '_sTs.npy',      np.stack(_online_data_buf['sTs']))
            np.save(prefix + '_clip_Ts.npy',  np.stack(_online_data_buf['clip_Ts']))
            np.save(prefix + '_pair_ins.npy', np.stack(_online_data_buf['pair_ins']))
            np.save(prefix + '_pair_outs.npy',np.stack(_online_data_buf['pair_outs']))
            np.save(prefix + '_rewards.npy',  np.ones((n_online, 1), dtype=np.float32))
            print(f"[Exp 4] Saved {n_online} online success samples → {prefix}_*.npy")

        return trajectories


def evaluate(args, loader):
    # env = gym.make(args.env, render_mode=None, data_loader=loader,
    #                max_grid_size=args.max_grid_size, colors=10, max_episode_steps=None, max_trial=3)
    
    # dataset = env.get_dataset()
    # state_dim = dataset['observations'].shape[1]
    # a_dim = dataset['actions'].shape[1]
    state_dim = args.s_dim
    a_dim = args.a_dim
    
    # Initialize skill latent collection
    skill_latent_data = {
        'latents': [],
        'task_ids': [],
        'task_groups': [],
        'step_numbers': [],
        'trajectory_ids': [],
        'metadata': {
            'policy': args.policy,
            'q_checkpoint_steps': getattr(args, 'q_checkpoint_steps', None),
            'z_dim': args.z_dim,
            'skill_model_filename': args.skill_model_filename
        }
    }
    
    skill_model = SkillModel(state_dim,
                            a_dim,
                            args.z_dim,
                            args.h_dim,
                            horizon=args.horizon,
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
                            max_grid_size=args.max_grid_size,
                            use_in_out=args.use_in_out,
                            disable_pair_encoding=args.disable_pair_encoding_skill,
                            use_concept_guidance=bool(getattr(args, 'use_concept_guidance', False)),
                            use_cfg_for_concept=bool(getattr(args, 'use_cfg_for_concept', True)),
                            ).to(args.device)

    skill_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.skill_model_filename))['model_state_dict'], strict=False)
    skill_model.eval()

    diffusion_model = None
    if args.policy == 'vae_diffusion_prior':
        # Use VAE's built-in diffusion prior (trained jointly with VAE)
        if skill_model.diffusion_prior is not None:
            diffusion_model = skill_model.diffusion_prior
            diffusion_model.eval()
            print("Using VAE's built-in diffusion prior (trained jointly with VAE)")
        else:
            raise ValueError("VAE model does not have a diffusion prior. Make sure train_diffusion_prior=1 was used during training.")
    elif not args.policy == 'prior':
        # if args.append_goals:
        #   diffusion_nn_model = torch.load(os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4] + '_diffusion_prior_gc_best.pt')).to(args.device)
        # else:
        #   diffusion_nn_model = torch.load(os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4] + '_diffusion_prior_best.pt')).to(args.device)
        diffusion_nn_model = torch.load(os.path.join(args.checkpoint_dir, args.diffusion_model_filename),weights_only=False).to(args.device)
        if not hasattr(diffusion_nn_model, 'use_in_out'):
            diffusion_nn_model.use_in_out = args.use_in_out  # 또는 원하는 기본값
        if hasattr(diffusion_nn_model, 'nn') and not hasattr(diffusion_nn_model.nn, 'use_in_out'):
            diffusion_nn_model.nn.use_in_out = args.use_in_out

        diffusion_model = Model_Cond_Diffusion(
            diffusion_nn_model,
            betas=(1e-4, 0.02),
            n_T=args.diffusion_steps,
            device=args.device,
            x_dim=state_dim + args.append_goals*2,
            y_dim=args.z_dim,
            drop_prob=None,
            guide_w=args.cfg_weight,
        )
        diffusion_model.eval()

        # Load z-score normalization stats if available (for denormalization during sampling)
        base_filename = args.skill_model_filename[:-4]
        latent_stats_path = os.path.join(args.checkpoint_dir, base_filename + '_latent_zscore_stats.npz')

        if os.path.exists(latent_stats_path):
            latent_stats = np.load(latent_stats_path)
            diffusion_model.set_latent_stats(latent_stats['mean'], latent_stats['std'])
            print(f"[Inference] Loaded z-score stats from: {latent_stats_path}")
        elif getattr(args, 'apply_zscore_denorm', 0) == 1:
            # Manual denormalization using data directory stats
            # Compute from saved latent data if exists
            # Try multiple possible paths
            latent_data_paths = [
                # 1. Use explicit path if provided
                getattr(args, 'latent_data_path', None),
                # 2. Look in data directory with same structure as checkpoint
                os.path.join(os.path.dirname(args.checkpoint_dir).replace('checkpoints', 'data'),
                            os.path.basename(args.checkpoint_dir), base_filename + '_latents.npy'),
                # 3. Look in parent/data directory
                os.path.join(parent_folder, 'data', os.path.basename(args.checkpoint_dir), base_filename + '_latents.npy'),
            ]

            latent_data = None
            for path in latent_data_paths:
                if path and os.path.exists(path):
                    print(f"[Inference] Computing z-score stats from: {path}")
                    latent_data = np.load(path, allow_pickle=True)
                    break

            if latent_data is not None:
                latent_mean = latent_data.mean(axis=0)
                latent_std = latent_data.std(axis=0)
                latent_std = np.where(latent_std < 1e-6, 1.0, latent_std)
                diffusion_model.set_latent_stats(latent_mean, latent_std)
                print(f"[Inference] Applied z-score denorm. Mean range: [{latent_mean.min():.4f}, {latent_mean.max():.4f}], Std range: [{latent_std.min():.4f}, {latent_std.max():.4f}]")
            else:
                print(f"[Inference] Warning: --apply_zscore_denorm=1 but no latent data found. Tried:")
                for path in latent_data_paths:
                    if path:
                        print(f"  - {path}")

        # Set concept_scale on the nn_model for concept guidance strength
        if hasattr(diffusion_nn_model, 'nn'):
            diffusion_nn_model.nn.concept_scale = args.concept_scale
        else:
            diffusion_nn_model.concept_scale = args.concept_scale

        # envs = [gym.make(args.env, render_mode=None, data_loader=loader, max_grid_size=args.max_grid_size, colors=10,
        #                  max_episode_steps=None, max_trial=3) for _ in range(args.num_parallel_envs)]

    # Initialize ConceptEncoder if concept guidance is enabled
    concept_encoder = None
    if args.use_concept_guidance:
        if args.use_discrete_concepts:
            # Use DiscreteConceptEncoder
            print(f"Initializing DiscreteConceptEncoder with {args.num_concepts} concepts...")
            concept_encoder = DiscreteConceptEncoder(
                num_concepts=args.num_concepts,
                embedding_dim=args.z_dim,  # Match latent dimension
                device=args.device
            )
            concept_encoder.eval()

            # Try to load saved weights (check multiple possible paths for compatibility)
            base_filename = args.skill_model_filename[:-4]
            if args.concept_encoder_weights:
                weights_path = args.concept_encoder_weights
                mappings_path = weights_path.replace('.pth', '_mappings.json')
            else:
                # Try new naming convention first, then old naming convention
                weights_path_new = os.path.join(args.checkpoint_dir, base_filename + '_discrete_concept_encoder.pth')
                weights_path_old = os.path.join(args.checkpoint_dir, base_filename + '_concept_weights.pth')
                mappings_path_new = os.path.join(args.checkpoint_dir, base_filename + '_discrete_concept_encoder_mappings.json')
                mappings_path_old = os.path.join(args.checkpoint_dir, base_filename + '_concept_mapping.json')

                weights_path = weights_path_new if os.path.exists(weights_path_new) else weights_path_old
                mappings_path = mappings_path_new if os.path.exists(mappings_path_new) else mappings_path_old

            if os.path.exists(weights_path):
                concept_encoder.load_weights(weights_path)
            else:
                print(f"Warning: DiscreteConceptEncoder weights not found at {weights_path}")

            if os.path.exists(mappings_path):
                concept_encoder.load_mappings(mappings_path)
            else:
                print(f"Warning: DiscreteConceptEncoder mappings not found at {mappings_path}")
        else:
            # Use text-based ConceptEncoder
            print("Initializing text-based ConceptEncoder...")
            from models.concept_encoder import ConceptEncoder
            concept_encoder = ConceptEncoder(
                model_name='all-MiniLM-L6-v2',
                projection_dim=args.z_dim,  # Match latent dimension
                device=args.device
            )
            concept_encoder.eval()

            # Try to load saved projection weights
            base_filename = args.skill_model_filename[:-4]
            weights_path = args.concept_encoder_weights or os.path.join(
                args.checkpoint_dir, base_filename + '_text_concept_encoder.pth'
            )

            if os.path.exists(weights_path):
                concept_encoder.load_projection(weights_path)
            else:
                print(f"Warning: ConceptEncoder projection weights not found at {weights_path}")
                print("Using random projection weights - concept guidance may not work correctly!")

    # Convert max_episode_steps to None if set to negative value for unlimited steps
    max_episode_steps = None if args.max_episode_steps < 0 else args.max_episode_steps
    
    if(args.render == 'ansi'):
        # envs = gym.make(args.env, data_loader=loader, render_mode='ansi', max_grid_size=(args.max_grid_size,args.max_grid_size), colors=10, max_trial=3)
        envs = [gym.make(args.env, data_loader=loader, render_mode='ansi', max_grid_size=(args.max_grid_size,args.max_grid_size), colors=10, max_trial=3)]
    else:
        envs = [gym.make(args.env, data_loader=loader, max_grid_size=(args.max_grid_size,args.max_grid_size), colors=10, max_trial=3)]
        
        
    if not args.append_goals:
        #state_all = np.load(os.path.join(args.test_solar_dir, args.skill_model_filename[:-4] + "_states.npy"), allow_pickle=True)
        state_mean = 0    #torch.from_numpy(state_all.mean(axis=0)).to(args.device).float()
        state_std = 1     #torch.from_numpy(state_all.std(axis=0)).to(args.device).float()

        #latent_all = np.load(os.path.join(args.test_solar_dir, args.skill_model_filename[:-4] + "_latents.npy"), allow_pickle=True)
        latent_mean = 0   #torch.from_numpy(latent_all.mean(axis=0)).to(args.device).float()
        latent_std = 1    #torch.from_numpy(latent_all.std(axis=0)).to(args.device).float()
    else:
        #state_all = np.load(os.path.join(args.test_solar_dir, args.skill_model_filename[:-4] + "_goals_states.npy"), allow_pickle=True)
        state_mean = 0    #torch.from_numpy(state_all.mean(axis=0)).to(args.device).float()
        state_std = 1     #torch.from_numpy(state_all.std(axis=0)).to(args.device).float()

        #latent_all = np.load(os.path.join(args.test_solar_dir, args.skill_model_filename[:-4] + "_goals_latents.npy"), allow_pickle=True)
        latent_mean = 0   #torch.from_numpy(latent_all.mean(axis=0)).to(args.device).float()
        latent_std = 1    #torch.from_numpy(latent_all.std(axis=0)).to(args.device).float()

    dqn_agent = None
    if args.policy == 'prior':
        policy_fn = prior_policy
    elif args.policy == 'diffusion_prior':
        policy_fn = diffusion_prior_policy
    elif args.policy == 'vae_diffusion_prior':
        policy_fn = vae_diffusion_prior_policy
    elif args.policy == 'q':
        # dqn_agent = torch.load(os.path.join(args.q_checkpoint_dir, args.skill_model_filename[:-4]+'_dqn_agent_'+str(args.q_checkpoint_steps)+'_cfg_weight_'+str(args.cfg_weight)+'_PERbuffer.pt')).to(args.device)
        dqn_agent = torch.load(os.path.join(args.q_checkpoint_dir, args.skill_model_filename[:-4]+'_dqn_agent_'+str(args.q_checkpoint_steps)+'_cfg_weight_'+str(args.cfg_weight)+'_PERbuffer.pt'),weights_only=False).to(args.device)
        dqn_agent.diffusion_prior = diffusion_model
        dqn_agent.extra_steps = args.extra_steps
        dqn_agent.target_net_0 = dqn_agent.q_net_0
        dqn_agent.target_net_1 = dqn_agent.q_net_1
        dqn_agent.eval()
        dqn_agent.num_prior_samples = args.num_diffusion_samples
        
        # Add DDIM attributes if missing (for backward compatibility with old models)
        if not hasattr(dqn_agent, 'use_ddim'):
            dqn_agent.use_ddim = args.use_ddim
        if not hasattr(dqn_agent, 'ddim_steps'):
            dqn_agent.ddim_steps = args.ddim_steps
        if not hasattr(dqn_agent, 'ddim_eta'):
            dqn_agent.ddim_eta = args.ddim_eta
        if not hasattr(dqn_agent, 'ddim_discr'):
            dqn_agent.ddim_discr = args.ddim_discr

        # Add disable_pair_encoding attribute if missing (for backward compatibility)
        if not hasattr(dqn_agent, 'disable_pair_encoding'):
            dqn_agent.disable_pair_encoding = args.disable_pair_encoding_q

        # [Exp 1 & 3] bind q_threshold and q_stats_collector via partial
        q_stats_collector = [] if args.log_q_stats else None
        policy_fn = partial(q_policy,
                            q_threshold=args.q_threshold,
                            q_stats_collector=q_stats_collector)
    elif args.policy == 'q_vae':
        dqn_agent = torch.load(os.path.join(args.q_checkpoint_dir, args.skill_model_filename[:-4]+'_dqn_agent_'+str(args.q_checkpoint_steps)+'_cfg_weight_'+str(args.cfg_weight)+'_PERbuffer.pt'),weights_only=False).to(args.device)
        dqn_agent.diffusion_prior = diffusion_model
        dqn_agent.extra_steps = args.extra_steps
        dqn_agent.target_net_0 = dqn_agent.q_net_0
        dqn_agent.target_net_1 = dqn_agent.q_net_1
        dqn_agent.eval()
        dqn_agent.num_prior_samples = args.num_diffusion_samples
        
        # Add DDIM attributes if missing (for backward compatibility with old models)
        if not hasattr(dqn_agent, 'use_ddim'):
            dqn_agent.use_ddim = args.use_ddim
        if not hasattr(dqn_agent, 'ddim_steps'):
            dqn_agent.ddim_steps = args.ddim_steps
        if not hasattr(dqn_agent, 'ddim_eta'):
            dqn_agent.ddim_eta = args.ddim_eta
        if not hasattr(dqn_agent, 'ddim_discr'):
            dqn_agent.ddim_discr = args.ddim_discr

        # Add disable_pair_encoding attribute if missing (for backward compatibility)
        if not hasattr(dqn_agent, 'disable_pair_encoding'):
            dqn_agent.disable_pair_encoding = args.disable_pair_encoding_q

        # [Exp 1 & 3] bind q_threshold and q_stats_collector via partial
        q_stats_collector = [] if args.log_q_stats else None
        policy_fn = partial(q_policy_with_skill_prior,
                            q_threshold=args.q_threshold,
                            q_stats_collector=q_stats_collector)
    else:
        q_stats_collector = None
        raise NotImplementedError

    trajectories = eval_func(diffusion_model,
                skill_model,
                policy_fn,
                envs,
                state_dim,
                state_mean,
                state_std,
                latent_mean,
                latent_std,
                args.num_evals,
                args.num_parallel_envs,
                args.num_diffusion_samples,
                args.extra_steps,
                args.planning_depth,
                args.exec_horizon,
                args.predict_noise,
                args.render,
                args.append_goals,
                dqn_agent,
                args.env,
                skill_latent_data,
                loader,
                bool(args.use_ddim),
                args.ddim_steps,
                args.ddim_eta,
                args.ddim_discr,
                concept_encoder
                )
    
    # Generate visualizations if enabled
    if args.save_visualizations and trajectories:
        print("\n" + "="*50)
        print("Generating trajectory visualizations...")
        print("="*50)
        
        # Determine save directory
        if args.viz_save_dir is None:
            checkpoint_name = os.path.basename(args.checkpoint_dir)
            policy_suffix = f"_{args.policy}" if args.policy != 'q' else ""
            q_suffix = f"_q{args.q_checkpoint_steps}" if args.q_checkpoint_steps > 0 else ""
            args.viz_save_dir = f"eval_visualize/{checkpoint_name}{policy_suffix}{q_suffix}"
        
        os.makedirs(args.viz_save_dir, exist_ok=True)
        
        # Create visualizations
        for trajectory in tqdm(trajectories, desc="Creating visualizations"):
            save_path = os.path.join(args.viz_save_dir, f"{trajectory['task_id']}.png")
            create_trajectory_visualization(trajectory, trajectory['task_id'], 
                                          trajectory['is_successful'], save_path)
        
        print(f"Visualizations saved to {args.viz_save_dir}")
        print("="*50)

    return q_stats_collector


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='ARCLE/O2ARCv2Env-v0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_evals', type=int, default=500)
    parser.add_argument('--num_parallel_envs', type=int, default=1)
    parser.add_argument('--checkpoint_dir', type=str, default=parent_folder+'/checkpoints')
    parser.add_argument('--q_checkpoint_dir', type=str, default=parent_folder+'/q_checkpoints')
    parser.add_argument('--q_checkpoint_steps', type=int, default=0)
    parser.add_argument('--test_solar_dir', type=str, default=parent_folder+'/data')
    parser.add_argument('--skill_model_filename', type=str)
    parser.add_argument('--diffusion_model_filename', type=str)
    parser.add_argument('--append_goals', type=int, default=0)

    parser.add_argument('--policy', type=str, default='q') #greedy/exhaustive/q
    parser.add_argument('--num_diffusion_samples', type=int, default=10)
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--cfg_weight', type=float, default=0.0)
    parser.add_argument('--planning_depth', type=int, default=3)    # 이게 필요한가?
    parser.add_argument('--extra_steps', type=int, default=4)
    parser.add_argument('--predict_noise', type=int, default=0)
    
    # DDIM sampling options
    parser.add_argument('--use_ddim', type=int, default=0, help='Use DDIM sampling instead of DDPM (0=DDPM, 1=DDIM)')
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of DDIM sampling steps (fewer = faster)')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM stochasticity (0=deterministic, 1=DDPM-like)')
    parser.add_argument('--ddim_discr', type=str, default='uniform', help='DDIM timestep discretization (uniform/quad)')
    
    parser.add_argument('--exec_horizon', type=int, default=1)

    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--a_dist', type=str, default='normal')
    parser.add_argument('--encoder_type', type=str, default='gru')
    parser.add_argument('--state_decoder_type', type=str, default='mlp')
    parser.add_argument('--policy_decoder_type', type=str, default='mlp')    # 원래는 'autoregressive'
    parser.add_argument('--per_element_sigma', type=int, default=1)
    parser.add_argument('--conditional_prior', type=int, default=1)
    parser.add_argument('--horizon', type=int, default=5)
    
    parser.add_argument('--normalize_latent', type=int, default=0)  # 원래는 0(바활성화)
    parser.add_argument('--train_diffusion_prior', type=int, default=0)
    parser.add_argument('--skill_model_diffusion_steps', type=int, default=500)
    
    parser.add_argument('--a_dim', type=int, default=36)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--s_dim', type=int, default=256)
    
    parser.add_argument('--render', type=str, default=None)
    parser.add_argument('--max_grid_size', type=int, default=30)
    parser.add_argument('--use_in_out', type=int, default=0)
    parser.add_argument('--save_visualizations', type=int, default=0, help='Save trajectory visualizations (1=True, 0=False)')
    parser.add_argument('--viz_save_dir', type=str, default=None, help='Directory to save visualizations (auto-generated if None)')
    parser.add_argument('--max_episode_steps', type=int, default=20, help='Maximum number of steps per episode (None for unlimited)')
    parser.add_argument('--update_in_grid_on_fail', type=int, default=1, help='Update in_grid when submit fails (0=disable, 1=enable)')
    parser.add_argument('--repetition_threshold', type=int, default=5, help='Maximum number of repeated actions before stopping')
    parser.add_argument('--use_mlp_embed_q', type=int, default=1, help='Use MLP embedding in Q-network (0=disable, 1=enable)')
    parser.add_argument('--disable_pair_encoding_q', type=int, default=0, help='Disable pair encoding in Q-network (0=use pair encoding, 1=disable)')
    parser.add_argument('--disable_pair_encoding_skill', type=int, default=0, help='Disable pair encoding in skill model (0=use pair encoding, 1=disable)')
    parser.add_argument('--use_enhanced_pair_encoding', type=int, default=0, help='Use enhanced pair encoding')
    parser.add_argument('--use_shared_grid_embedding', type=int, default=0, help='Use shared grid embedding')
    parser.add_argument('--use_split_pair_trajectory_encoding', type=int, default=0, help='Use split pair trajectory encoding')
    parser.add_argument('--use_direct_output_predictor', type=int, default=0, help='Use direct output predictor')
    parser.add_argument('--use_direct_output_for_diffusion', type=int, default=0, help='Use direct output for diffusion')
    parser.add_argument('--noise_temperature', type=float, default=1.0, help='Temperature for noise scaling in diffusion sampling')
    parser.add_argument('--use_concept_guidance', type=int, default=0, help='Use concept guidance for diffusion sampling (0=disable, 1=enable)')
    parser.add_argument('--use_cfg_for_concept', type=int, default=1, help='Use CFG for concept guidance (0=direct conditioning, 1=CFG)')
    parser.add_argument('--use_discrete_concepts', type=int, default=0, help='Use discrete concept encoder instead of text-based (0=text, 1=discrete)')
    parser.add_argument('--num_concepts', type=int, default=2, help='Number of discrete concepts (only used if use_discrete_concepts=1)')
    parser.add_argument('--concept_scale', type=float, default=1.0, help='Scale factor for concept embeddings')
    parser.add_argument('--concept_encoder_weights', type=str, default=None, help='Path to saved concept encoder weights (auto-detected if None)')
    parser.add_argument('--apply_zscore_denorm', type=int, default=0, help='Apply z-score denormalization to diffusion output using encoder latent stats (0=disable, 1=enable). Use this for models trained without --normalize_latent_zscore.')
    parser.add_argument('--latent_data_path', type=str, default=None, help='Path to _latents.npy file for computing z-score stats (only used with --apply_zscore_denorm=1)')
    parser.add_argument('--use_vae_prior_for_latent', type=int, default=0, help='Use VAE prior for latent sampling instead of diffusion in q policy (0=use diffusion, 1=use VAE prior)')
    parser.add_argument('--sequential_chain', type=int, default=0, help='Enable sequential chain eval: switch ex pairs from Phase1 to Phase2 when current state == intermediate_grid (0=disable, 1=enable)')

    # [Exp 1] Q-threshold gating
    parser.add_argument('--q_threshold', type=float, default=None,
                        help='Q-value threshold for proposal gating. Proposals with Q < threshold are discarded. (None = disabled)')
    # [Exp 3] Q-value stats logging
    parser.add_argument('--log_q_stats', type=int, default=0,
                        help='Log Q-value distribution stats per step to q_stats.json (0=disable, 1=enable)')
    # [Exp 4] Online data collection
    parser.add_argument('--save_online_data', type=int, default=0,
                        help='Save successful (state, latent, reward=1) tuples for Q-net fine-tuning (0=disable, 1=enable)')
    parser.add_argument('--online_data_dir', type=str, default=None,
                        help='Directory to save online data (default: q_checkpoint_dir)')

    args = parser.parse_args()

    loader = ARC_Dataloader(data_path=args.test_solar_dir)
    q_stats_collector = evaluate(args, loader)

    # [Exp 3] Save Q-value stats if collected
    if args.log_q_stats and args.policy in ('q', 'q_vae'):
        try:
            stats = q_stats_collector
            if stats:
                import numpy as np
                means = [s['mean'] for s in stats]
                stats_summary = {
                    'per_step': stats,
                    'summary': {
                        'global_mean': float(np.mean(means)),
                        'global_std': float(np.std(means)),
                        'mean_n_valid': float(np.mean([s['n_valid'] for s in stats])),
                        'mean_n_total': float(np.mean([s['n_total'] for s in stats])),
                        'q_threshold': args.q_threshold,
                        'num_diffusion_samples': args.num_diffusion_samples,
                    }
                }
                save_path = os.path.join(
                    args.q_checkpoint_dir,
                    f"q_stats_thr{args.q_threshold}_N{args.num_diffusion_samples}.json"
                )
                with open(save_path, 'w') as f:
                    json.dump(stats_summary, f, indent=2)
                print(f"[Exp 3] Q-value stats saved to {save_path}")
                print(f"  global Q mean={stats_summary['summary']['global_mean']:.4f}, "
                      f"mean valid proposals={stats_summary['summary']['mean_n_valid']:.1f}/{stats_summary['summary']['mean_n_total']:.1f}")
        except Exception as e:
            print(f"[Exp 3] Warning: could not save Q stats: {e}")
    
    # Auto-generate visualizations if enabled
    if args.save_visualizations:
        print("\n" + "="*50)
        print("Generating trajectory visualizations...")
        print("="*50)
        
        # Determine save directory
        if args.viz_save_dir is None:
            # Auto-generate directory name from checkpoint directory
            checkpoint_name = os.path.basename(args.checkpoint_dir)
            policy_suffix = f"_{args.policy}" if args.policy != 'q' else ""
            q_suffix = f"_q{args.q_checkpoint_steps}" if args.q_checkpoint_steps > 0 else ""
            args.viz_save_dir = f"eval_visualize/{checkpoint_name}{policy_suffix}{q_suffix}"
        
        # Find the log file (assume it's been redirected to a log file)
        # For now, we'll skip this and let user manually run visualization
        print(f"To generate visualizations, run:")
        print(f"python visualize_eval_log.py \\")
        print(f"  --log_file <path_to_log_file> \\")
        print(f"  --test_data_dir {args.test_solar_dir} \\")
        print(f"  --save_dir {args.viz_save_dir}")
        print("="*50)
