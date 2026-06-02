"""
Original BC (Behavior Cloning) Evaluation Script for ARCLE
Joint Action Space (360,000 actions) - No factorization
"""
import os
import sys

curr_folder = os.path.abspath(__file__)
parent_folder = os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder)
from argparse import ArgumentParser

import numpy as np
import torch
import random
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import colors
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from models.bc_original import OriginalBC

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
    """Map operation number to operation name"""
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
    """Plot one step"""
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
    """Create visualization for a single trajectory"""
    steps = trajectory_data['steps']
    initial_grid = trajectory_data['initial_grid']
    target_grid = trajectory_data['target_grid']
    examples = trajectory_data.get('examples', [])

    # Calculate layout
    n_steps = len(steps)
    n_examples = len(examples)

    cols = max(6, n_steps + 2)
    rows = n_examples + 1

    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    gs = GridSpec(nrows=rows, ncols=cols, figure=fig)

    # Title
    success_status = "Success" if is_successful else "Failed"
    fig.suptitle(f'Task: {task_id} ({success_status})', fontsize=24, fontweight='bold')

    # Plot demonstration examples
    if n_examples > 0:
        for ex_idx in range(min(n_examples, 3)):
            if ex_idx < len(examples):
                ex_data = examples[ex_idx]

                ex_in_ax = fig.add_subplot(gs[ex_idx, 0])
                plot_one_step(ex_in_ax, ex_data['input'], None, is_input=True)
                ex_in_ax.set_title(f'demonstration input {ex_idx+1}', fontsize=16)

                ex_out_ax = fig.add_subplot(gs[ex_idx, 1])
                plot_one_step(ex_out_ax, ex_data['output'], None, is_target=True)
                ex_out_ax.set_title(f'demonstration output {ex_idx+1}', fontsize=16)

    # Plot test trajectory on last row
    test_row = rows - 1

    test_input_ax = fig.add_subplot(gs[test_row, 0])
    plot_one_step(test_input_ax, initial_grid, None, is_input=True)

    for i, step in enumerate(steps):
        if i + 1 < cols - 1:
            step_ax = fig.add_subplot(gs[test_row, i + 1])
            plot_one_step(step_ax, step['grid'], step)

    if cols > n_steps + 1:
        target_ax = fig.add_subplot(gs[test_row, cols - 1])
        plot_one_step(target_ax, target_grid, None, is_target=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

class ARC_Dataloader(Loader):
    def __init__(self, data_path, train=True, sample_per_folder=None, seed=42) -> None:
        self.data_path = data_path
        self.sample_per_folder = sample_per_folder
        self.seed = seed
        super().__init__(train=train)

    def get_path(self, **kwargs) -> List[str]:
        data_path = Path(self.data_path)
        pathlist = []

        if self.sample_per_folder is not None:
            # Sample N files per subfolder
            random.seed(self.seed)
            folder_files = {}

            for path, _, files in os.walk(data_path):
                if path == str(data_path):  # Skip root directory
                    continue

                valid_files = []
                for name in files:
                    if 'expert' in name or 'gold_standard' in name or 'golden-standard' in name:
                        valid_files.append(os.path.join(path, name))

                if valid_files:
                    folder_name = os.path.basename(path)
                    # Sample up to sample_per_folder files from this folder
                    n_sample = min(self.sample_per_folder, len(valid_files))
                    sampled = random.sample(valid_files, n_sample)
                    folder_files[folder_name] = sampled
                    pathlist.extend(sampled)
                    print(f"Sampled {n_sample}/{len(valid_files)} files from {folder_name}")
        else:
            # Original behavior: load all files
            for path, _, files in os.walk(data_path):
                for name in files:
                    if 'expert' in name or 'gold_standard' in name or 'golden-standard' in name:
                        pathlist.append(os.path.join(path, name))

        self.num_dataset = len(pathlist)

        if self.num_dataset == 0:
            raise ValueError("Wrong data path or empty folder.")
        else:
            print("Number of episodes: {0}".format(self.num_dataset))

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

                ti.append(np.array(trajectory['in_grid'], dtype=np.int8)[:ti_h, :ti_w])
                to.append(np.array(trajectory['out_grid'], dtype=np.int8)[:to_h, :to_w])

                for i in range(len(trajectory['ex_in'])):
                    ei_h, ei_w = trajectory['ex_in_grid_dim'][i]
                    eo_h, eo_w = trajectory['ex_out_grid_dim'][i]

                    ei.append(np.array(trajectory['ex_in'][i], dtype=np.int8)[:ei_h, :ei_w])
                    eo.append(np.array(trajectory['ex_out'][i], dtype=np.int8)[:eo_h, :eo_w])

                desc = {
                    'id': trajectory['desc']['id'],
                    'ex_in_grid_dim': trajectory['ex_in_grid_dim'],
                    'ex_out_grid_dim': trajectory['ex_out_grid_dim'],
                    'concept': trajectory['desc'].get('concept', ''),
                }

                dat.append((ei, eo, ti, to, desc))

        return self.convert_grid_to_uint8(dat)

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

def pad_grid(grid, max_size=10):
    """Pad grid to max_size x max_size"""
    h, w = grid.shape
    padded = np.full((max_size, max_size), 10, dtype=np.float32)
    h_copy = min(h, max_size)
    w_copy = min(w, max_size)
    padded[:h_copy, :w_copy] = grid[:h_copy, :w_copy]
    return padded

def sel_bbox_to_mask(selection_bbox, max_grid_size):
    """Convert selection bounding box to mask"""
    x, y, h, w = selection_bbox
    sel_mask = np.zeros(max_grid_size, dtype=np.int8)
    sel_mask[x:x+h+1, y:y+w+1] = 1
    return sel_mask

def evaluate_task(env, bc_agent, loader, task_idx, max_grid_size=10, max_steps=50, device='cuda', success_count=0, reach_count=0, total_evaluated=0):
    """Evaluate one task with Original BC"""
    # Get example pairs from loader
    ex_in, ex_out, tt_in, tt_out, desc = loader.pick(data_index=task_idx)

    # Extract task group name from task_id
    task_id = desc.get('id', 'unknown')
    # Remove one or more trailing _number patterns (e.g., _1, _1_2, _10_20)
    import re
    task_group = re.sub(r'(_\d+)+$', '', task_id)

    # Print task header
    print("\n" + "="*80)
    print(f"Evaluating Task {task_idx} (ID: {task_id})")
    print("="*80)

    # Reset environment
    obs, info = env.reset(options={'prob_index': task_idx, 'subprob_index': 0, 'adaptation': False})

    # Get example pairs
    examples = []
    for i in range(len(ex_in)):
        ex_in_arr = np.array(ex_in[i]) if not isinstance(ex_in[i], np.ndarray) else ex_in[i]
        ex_out_arr = np.array(ex_out[i]) if not isinstance(ex_out[i], np.ndarray) else ex_out[i]
        examples.append({'input': ex_in_arr.copy(), 'output': ex_out_arr.copy()})

    obs_h, obs_w = obs['grid_dim']
    initial_grid = obs['grid'][:obs_h, :obs_w].copy()
    target_grid_arr = np.array(tt_out[0]) if not isinstance(tt_out[0], np.ndarray) else tt_out[0]
    target_grid = target_grid_arr.copy()

    # Pad example pairs
    pair_in_list = []
    pair_out_list = []
    for ex in examples[:3]:
        pair_in_list.append(pad_grid(ex['input'], max_grid_size))
        pair_out_list.append(pad_grid(ex['output'], max_grid_size))

    while len(pair_in_list) < 3:
        pair_in_list.append(np.full((max_grid_size, max_grid_size), 10, dtype=np.float32))
        pair_out_list.append(np.full((max_grid_size, max_grid_size), 10, dtype=np.float32))

    pair_in = torch.FloatTensor(np.stack(pair_in_list)).unsqueeze(0).to(device)  # (1, 3, H, W)
    pair_out = torch.FloatTensor(np.stack(pair_out_list)).unsqueeze(0).to(device)

    # Pad input grid (use initial_grid which is the original input)
    in_grid_padded = pad_grid(initial_grid, max_grid_size)
    in_grid = torch.FloatTensor(in_grid_padded).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

    steps = []
    done = False
    step_count = 0

    while not done and step_count < max_steps:
        # Get current state
        obs_h, obs_w = obs['grid_dim']
        current_grid = obs['grid'][:obs_h, :obs_w]
        state = torch.FloatTensor(pad_grid(current_grid, max_grid_size)).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

        # Get clip from observation (clipboard)
        clip_grid = obs.get('clip', np.full((max_grid_size, max_grid_size), 10, dtype=np.float32))
        if clip_grid is not None:
            clip = torch.FloatTensor(pad_grid(clip_grid, max_grid_size)).unsqueeze(0).unsqueeze(0).to(device)
        else:
            clip = torch.FloatTensor(np.full((1, 1, max_grid_size, max_grid_size), 10, dtype=np.float32)).to(device)

        # Get action from Original BC (operation, x, y, h, w)
        operation, x, y, h, w = bc_agent.get_action(state, clip, in_grid, pair_in, pair_out)

        # Log action details
        print("Step: {0}| op: {1}, x: {2}, y: {3}, h: {4}, w: {5}".format(step_count, operation, x, y, h, w))

        # Create action dict with selection mask
        select_mask = sel_bbox_to_mask((x, y, h, w), (max_grid_size, max_grid_size))
        action_dict = {'selection': select_mask.astype(bool), 'operation': int(operation)}

        obs, reward, terminated, truncated, info = env.step(action_dict)

        steps.append({
            'step': step_count,
            'grid': current_grid.copy(),
            'operation': operation,
            'selection': [x, y, h, w],
            'reward': reward
        })

        done = terminated or truncated
        step_count += 1

        # Early termination on Submit
        if operation == 34:
            break

    is_successful = reward > 0

    # Check if reached answer (grid matches target)
    obs_h, obs_w = obs['grid_dim']
    final_grid = obs['grid'][:obs_h, :obs_w]

    # Compare grids
    reach_answer = False
    if final_grid.shape == target_grid.shape:
        reach_answer = np.array_equal(final_grid, target_grid)

    # Print task result
    print("env_step : {0}".format(step_count))

    # Calculate cumulative statistics
    new_success_count = success_count + (1 if is_successful else 0)
    new_reach_count = reach_count + (1 if reach_answer else 0)
    new_total = total_evaluated + 1

    print(f'Total score: {new_success_count} out of {new_total} i.e. Acc: {(new_success_count / new_total) * 100:.2f}%')
    print(f'Reach answer: {new_reach_count} out of {new_total} i.e. Acc: {(new_reach_count / new_total) * 100:.2f}%')
    print("="*80 + "\n")

    trajectory_data = {
        'task_id': task_id,
        'task_group': task_group,
        'initial_grid': initial_grid,
        'target_grid': target_grid,
        'steps': steps,
        'examples': examples,
        'reward': reward,
        'success': is_successful,
        'reach_answer': reach_answer
    }

    return trajectory_data, is_successful, reach_answer

def main(args):
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Original BC model
    print("Loading Original BC model...")

    # Load checkpoint (full model)
    bc_agent = torch.load(args.checkpoint_path, map_location=device)
    bc_agent.to(device)
    bc_agent.policy.eval()

    print(f"Loaded checkpoint from {args.checkpoint_path}")
    print(f"BC model loaded successfully")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data_path}")
    sample_per_folder = getattr(args, 'sample_per_folder', None)
    loader = ARC_Dataloader(args.data_path, train=False, sample_per_folder=sample_per_folder, seed=args.seed)

    # Create environment
    print("Creating ARCLE environment...")
    env = gym.make('ARCLE/O2ARCv2Env-v0', data_loader=loader, max_grid_size=(args.max_grid_size, args.max_grid_size))

    # Evaluation
    results = []
    success_count = 0
    reach_count = 0
    total_tasks = loader.num_dataset
    task_group_scores = {}

    print(f"\nEvaluating {total_tasks} tasks...")
    for task_idx in tqdm(range(total_tasks)):
        try:
            trajectory_data, is_successful, reach_answer = evaluate_task(
                env, bc_agent, loader, task_idx,
                max_grid_size=args.max_grid_size,
                max_steps=args.max_steps,
                device=device,
                success_count=success_count,
                reach_count=reach_count,
                total_evaluated=task_idx
            )

            task_id = trajectory_data.get('task_id', f'task_{task_idx}')
            task_group = trajectory_data.get('task_group', task_id)

            # Update task group statistics
            if task_group not in task_group_scores:
                task_group_scores[task_group] = {
                    'submit': 0,
                    'reach': 0,
                    'total': 0,
                    'total_steps': 0
                }

            task_group_scores[task_group]['total'] += 1
            task_group_scores[task_group]['total_steps'] += len(trajectory_data['steps'])
            if is_successful:
                task_group_scores[task_group]['submit'] += 1
            if reach_answer:
                task_group_scores[task_group]['reach'] += 1

            # Save visualization
            if args.save_viz:
                vis_path = os.path.join(vis_dir, f'{task_id}.png')
                create_trajectory_visualization(trajectory_data, task_id, is_successful, vis_path)

            results.append({
                'task_id': task_id,
                'task_group': task_group,
                'success': is_successful,
                'reach_answer': reach_answer,
                'reward': trajectory_data['reward'],
                'steps': len(trajectory_data['steps'])
            })

            if is_successful:
                success_count += 1
            if reach_answer:
                reach_count += 1

        except Exception as e:
            print(f"\nError on task {task_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Calculate statistics
    success_rate = success_count / total_tasks if total_tasks > 0 else 0
    reach_rate = reach_count / total_tasks if total_tasks > 0 else 0
    avg_steps = np.mean([r['steps'] for r in results]) if results else 0

    # Print task group results
    print("\n" + "="*80)
    print("TASK GROUP RESULTS")
    print("="*80)

    for task_group, scores in sorted(task_group_scores.items()):
        submit_rate = (scores['submit'] / scores['total']) * 100 if scores['total'] > 0 else 0
        reach_rate_group = (scores['reach'] / scores['total']) * 100 if scores['total'] > 0 else 0
        avg_steps_group = scores['total_steps'] / scores['total'] if scores['total'] > 0 else 0

        print(f"Task Group: {task_group}")
        print(f"  Submit: {scores['submit']}/{scores['total']} ({submit_rate:.1f}%)")
        print(f"  Reach:  {scores['reach']}/{scores['total']} ({reach_rate_group:.1f}%)")
        print(f"  Avg Steps: {avg_steps_group:.1f}")
        print()

    print("="*80)

    print("\n" + "="*60)
    print("OVERALL EVALUATION RESULTS")
    print("="*60)
    print(f"Total tasks: {total_tasks}")
    print(f"Successful (Submit): {success_count} ({success_rate:.2%})")
    print(f"Reach answer: {reach_count} ({reach_count / total_tasks * 100:.2f}%)")
    print(f"Average steps: {avg_steps:.2f}")
    print("="*60)

    # Save results
    results_summary = {
        'checkpoint': args.checkpoint_path,
        'total_tasks': total_tasks,
        'success_count': success_count,
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'details': results
    }

    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to {results_path}")

if __name__ == '__main__':
    parser = ArgumentParser()

    # Data
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to Original BC checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')

    # Model
    parser.add_argument('--a_dim', type=int, default=36, help='Action dimension')
    parser.add_argument('--h_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--max_grid_size', type=int, default=10, help='Max grid size')
    parser.add_argument('--use_positional_encoding', type=int, default=0, help='Use positional encoding')

    # DQN params (for model initialization, not used during eval)
    parser.add_argument('--gamma', type=float, default=0.7, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.995, help='Target network update rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (not used in eval)')

    # Evaluation
    parser.add_argument('--max_steps', type=int, default=50, help='Max steps per task')
    parser.add_argument('--save_viz', type=int, default=1, help='Save visualizations (0=no, 1=yes)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--sample_per_folder', type=int, default=None, help='Sample N files per subfolder (default: all)')

    args = parser.parse_args()
    main(args)
