"""
Original CQL (Conservative Q-Learning) Training Script for ARC tasks
Joint action space (non-factorized): 36 × max_grid_size^4
Supports PER (Prioritized Experience Replay)
Uses same data loading as train_factorized_dqn.py
"""
import os
import sys
import json
import datetime
from argparse import ArgumentParser
from glob import glob
import random

curr_folder = os.path.abspath(__file__)
parent_folder = os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder)

import numpy as np
import torch
from tqdm import tqdm

from models.cql_original import OriginalCQL


class CQLPrioritizedBuffer:
    """Prioritized Experience Replay Buffer for CQL"""

    def __init__(self, capacity, max_grid_size=10, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.max_grid_size = max_grid_size
        self.pos = 0
        self.buffer_size = 0

        # Pre-allocate numpy arrays
        self.states = np.zeros((capacity, 1, max_grid_size, max_grid_size), dtype=np.float32)
        self.clips = np.zeros((capacity, 1, max_grid_size, max_grid_size), dtype=np.float32)
        self.in_grids = np.zeros((capacity, 1, max_grid_size, max_grid_size), dtype=np.float32)
        self.pair_ins = np.zeros((capacity, 3, max_grid_size, max_grid_size), dtype=np.float32)
        self.pair_outs = np.zeros((capacity, 3, max_grid_size, max_grid_size), dtype=np.float32)
        self.selections = np.zeros((capacity, 4), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, 1, max_grid_size, max_grid_size), dtype=np.float32)
        self.next_clips = np.zeros((capacity, 1, max_grid_size, max_grid_size), dtype=np.float32)
        self.next_selections = np.zeros((capacity, 4), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, clip, in_grid, pair_in, pair_out, selection, action, reward, next_state, next_clip, next_selection, done):
        max_prio = self.priorities.max() if self.buffer_size > 0 else 100.0

        self.states[self.pos] = state
        self.clips[self.pos] = clip
        self.in_grids[self.pos] = in_grid
        self.pair_ins[self.pos] = pair_in
        self.pair_outs[self.pos] = pair_out
        self.selections[self.pos] = selection
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.next_clips[self.pos] = next_clip
        self.next_selections[self.pos] = next_selection
        self.dones[self.pos] = done
        self.priorities[self.pos] = max_prio

        self.pos = (self.pos + 1) % self.capacity
        self.buffer_size = min(self.buffer_size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        if self.buffer_size == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.buffer_size]

        probs = np.power(prios, self.prob_alpha)
        probs /= np.sum(probs)

        indices = np.random.choice(self.buffer_size, batch_size, p=probs)
        weights = np.power((self.buffer_size * probs[indices]), -beta)
        weights /= weights.max()

        return (
            self.states[indices],
            self.clips[indices],
            self.in_grids[indices],
            self.pair_ins[indices],
            self.pair_outs[indices],
            self.selections[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.next_clips[indices],
            self.next_selections[indices],
            self.dones[indices],
            indices,
            weights
        )

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities

    def __len__(self):
        return self.buffer_size


class ARCTrajectoryDataset:
    """
    Dataset for loading ARC trajectories for CQL training
    Same as train_factorized_dqn.py
    """

    def __init__(self, data_dir, max_grid_size=10, train=True, test_ratio=0.1, reward_scale=1.0):
        self.max_grid_size = max_grid_size
        self.reward_scale = reward_scale
        self.transitions = []

        # Find all task folders OR directly load JSON files if data_dir is a task folder
        task_folders = glob(os.path.join(data_dir, "train.*")) if train else glob(os.path.join(data_dir, "test.*"))

        # If no task folders found, check if data_dir itself contains JSON files (single task mode)
        if len(task_folders) == 0:
            direct_json_files = glob(os.path.join(data_dir, "*.json"))
            if len(direct_json_files) > 0:
                task_folders = [data_dir]  # Treat data_dir as a single task folder
                print(f"Single task mode: found {len(direct_json_files)} JSON files in {data_dir}")

        print(f"Found {len(task_folders)} task folders")

        for task_folder in tqdm(task_folders, desc="Loading trajectories"):
            json_files = glob(os.path.join(task_folder, "*.json"))

            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        traj = json.load(f)

                    self._process_trajectory(traj)
                except Exception as e:
                    # Silently skip errors
                    continue

        print(f"Total transitions: {len(self.transitions)}")

        # Train/test split
        if train:
            n_test = int(len(self.transitions) * test_ratio)
            self.transitions = self.transitions[:-n_test] if n_test > 0 else self.transitions
        else:
            n_test = int(len(self.transitions) * test_ratio)
            self.transitions = self.transitions[-n_test:] if n_test > 0 else self.transitions

    def _pad_grid(self, grid, max_size):
        """Pad grid to max_size x max_size with 10 (padding value)"""
        h, w = len(grid), len(grid[0]) if len(grid) > 0 else 0
        padded = np.full((max_size, max_size), 10, dtype=np.float32)
        for i in range(min(h, max_size)):
            for j in range(min(w, max_size)):
                padded[i, j] = grid[i][j]
        return padded

    def _process_trajectory(self, traj):
        """Process a single trajectory into transitions"""
        # 'grid' contains the state sequence (list of grids)
        states = traj.get('grid', [])
        clips = traj.get('clip', [])  # Add clip loading
        actions = traj.get('operation', [])
        rewards = traj.get('reward', [])
        selections = traj.get('selection', [])

        # Get example pairs (input-output examples)
        # JSON uses 'ex_in' and 'ex_out' keys
        ex_in = traj.get('ex_in', [])
        ex_out = traj.get('ex_out', [])

        # Get input grid (the target)
        in_grid = traj.get('in_grid', [])

        if len(states) < 2 or len(actions) < 1:
            return

        # Pad example pairs
        pair_in = []
        pair_out = []
        for i in range(min(3, len(ex_in))):
            pair_in.append(self._pad_grid(ex_in[i], self.max_grid_size))
            pair_out.append(self._pad_grid(ex_out[i], self.max_grid_size))

        # Pad to 3 pairs if less
        while len(pair_in) < 3:
            pair_in.append(np.full((self.max_grid_size, self.max_grid_size), 10, dtype=np.float32))
            pair_out.append(np.full((self.max_grid_size, self.max_grid_size), 10, dtype=np.float32))

        pair_in = np.stack(pair_in)  # (3, H, W)
        pair_out = np.stack(pair_out)  # (3, H, W)

        # Pad input grid
        in_grid_padded = self._pad_grid(in_grid, self.max_grid_size) if in_grid else np.full((self.max_grid_size, self.max_grid_size), 10, dtype=np.float32)

        # Create transitions
        for t in range(len(actions)):
            state = self._pad_grid(states[t], self.max_grid_size)
            # Load clip for current timestep
            clip = self._pad_grid(clips[t], self.max_grid_size) if t < len(clips) and clips[t] else np.full((self.max_grid_size, self.max_grid_size), 10, dtype=np.float32)
            action = actions[t]
            reward = (rewards[t] if t < len(rewards) else 0) * self.reward_scale

            # Selection (x1, y1, x2, y2)
            selection = selections[t] if t < len(selections) else [0, 0, 0, 0]

            # Done flag (last action or submit action=34)
            done = 1.0 if (t == len(actions) - 1 or action == 34) else 0.0

            # For terminal actions (Submit or last action), use current state as next_state
            # This is important for DQN to learn the Submit action correctly
            if done == 1.0 or t >= len(states) - 1:
                next_state = state  # Terminal state: next_state = current state
                next_clip = clip
                next_selection = selection
            else:
                next_state = self._pad_grid(states[t + 1], self.max_grid_size)
                next_clip = self._pad_grid(clips[t + 1], self.max_grid_size) if t + 1 < len(clips) and clips[t + 1] else np.full((self.max_grid_size, self.max_grid_size), 10, dtype=np.float32)
                next_selection = selections[t + 1] if t + 1 < len(selections) else [0, 0, 0, 0]

            # Normalize selection to [0, 1]
            selection = np.array(selection, dtype=np.float32) / self.max_grid_size
            next_selection = np.array(next_selection, dtype=np.float32) / self.max_grid_size

            self.transitions.append({
                'state': state,
                'clip': clip,
                'in_grid': in_grid_padded,
                'pair_in': pair_in,
                'pair_out': pair_out,
                'selection': selection,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'next_clip': next_clip,
                'next_selection': next_selection,
                'done': done
            })

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        t = self.transitions[idx]
        return t  # Return raw dict for PER buffer


def train(args):
    """Main training function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load dataset
    print("Loading dataset...")
    print(f"Reward scale: {args.reward_scale}")
    train_dataset = ARCTrajectoryDataset(
        args.data_dir,
        max_grid_size=args.max_grid_size,
        train=True,
        test_ratio=args.test_ratio,
        reward_scale=args.reward_scale
    )

    # Create CQL agent
    print(f"\nInitializing Original CQL Agent:")
    print(f"  Max grid size: {args.max_grid_size} (actions: 0~{args.max_grid_size-1})")
    print(f"  Total joint actions: {36 * (args.max_grid_size ** 4):,}")
    print(f"  CQL alpha: {args.cql_alpha}")
    print(f"  CQL temperature: {args.cql_temperature}")
    print(f"  Use PER: {bool(args.use_per)}")
    print(f"  Positional encoding: {bool(args.use_positional_encoding)}\n")

    cql_agent = OriginalCQL(
        a_dim=args.a_dim,
        h_dim=args.h_dim,
        max_grid_size=args.max_grid_size,
        gamma=args.gamma,
        tau=args.tau,
        lr=args.lr,
        cql_alpha=args.cql_alpha,
        cql_temperature=args.cql_temperature,
        use_positional_encoding=bool(args.use_positional_encoding),
        device=device,
        scheduler_type=args.scheduler_type,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        cosine_t_max=args.cosine_t_max,
        cosine_eta_min=args.cosine_eta_min,
    )

    # Create PER buffer
    use_per = bool(args.use_per)
    if use_per:
        print("Using Prioritized Experience Replay")
        per_buffer = CQLPrioritizedBuffer(
            capacity=len(train_dataset),
            max_grid_size=args.max_grid_size,
            prob_alpha=args.per_alpha
        )
        # Fill buffer with all transitions
        print("Filling PER buffer...")
        for t in tqdm(train_dataset.transitions, desc="Loading to PER buffer"):
            per_buffer.push(
                state=np.expand_dims(t['state'], 0),
                clip=np.expand_dims(t['clip'], 0),
                in_grid=np.expand_dims(t['in_grid'], 0),
                pair_in=t['pair_in'],
                pair_out=t['pair_out'],
                selection=t['selection'],
                action=t['action'],
                reward=t['reward'],
                next_state=np.expand_dims(t['next_state'], 0),
                next_clip=np.expand_dims(t['next_clip'], 0),
                next_selection=t['next_selection'],
                done=t['done']
            )
        print(f"PER buffer size: {len(per_buffer)}")

    # Extract task name
    task_name = os.path.basename(os.path.normpath(args.data_dir))

    # Start training
    print(f"\n{'='*60}")
    print(f"Starting Original CQL Training (Joint Action Space)")
    print(f"Task: {task_name}")
    print(f"Date: {args.date}")
    print(f"GPU: {args.gpu_name}")
    print(f"{'='*60}\n")

    cql_agent.learn(
        replay_buffer=per_buffer,
        n_epochs=args.n_epoch,
        update_frequency=args.target_update_freq,
        checkpoint_dir=args.checkpoint_dir,
        gpu_name=args.gpu_name,
        task_name=task_name,
        args=args,
        use_per=use_per,
        per_beta_start=args.per_beta_start,
        per_beta_increment=args.per_beta_increment,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    # Environment
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    # Data
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing trajectory JSON files')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory to save checkpoints')
    parser.add_argument('--test_ratio', type=float, default=0.1)

    # Network architecture
    parser.add_argument('--a_dim', type=int, default=36, help='Number of operations')
    parser.add_argument('--h_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--max_grid_size', type=int, default=10, help='Max grid size (10 means 0~9)')
    parser.add_argument('--use_positional_encoding', type=int, default=0, help='Use 2D positional encoding')

    # Training
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.7, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.995, help='Target network update rate')
    parser.add_argument('--target_update_freq', type=int, default=1, help='Target network update frequency')
    parser.add_argument('--save_cycle', type=int, default=50, help='Save checkpoint every N update cycles')
    parser.add_argument('--reward_scale', type=float, default=1.0, help='Reward scaling factor')

    # CQL specific
    parser.add_argument('--cql_alpha', type=float, default=1.0, help='CQL loss weight')
    parser.add_argument('--cql_temperature', type=float, default=1.0, help='CQL temperature for logsumexp')

    # PER (Prioritized Experience Replay)
    parser.add_argument('--use_per', type=int, default=1, help='Use PER (0=no, 1=yes)')
    parser.add_argument('--per_alpha', type=float, default=0.6, help='PER alpha (prioritization exponent)')
    parser.add_argument('--per_beta_start', type=float, default=0.4, help='PER beta start value')
    parser.add_argument('--per_beta_increment', type=float, default=0.001, help='PER beta increment per step')

    # Scheduler
    parser.add_argument('--scheduler_type', type=str, default='step', choices=['step', 'cosine'])
    parser.add_argument('--lr_step_size', type=int, default=50)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--cosine_t_max', type=int, default=100)
    parser.add_argument('--cosine_eta_min', type=float, default=1e-6)

    # Logging
    parser.add_argument('--gpu_name', type=str, required=True)
    parser.add_argument('--date', type=str, default='01.27')

    args = parser.parse_args()

    train(args)
