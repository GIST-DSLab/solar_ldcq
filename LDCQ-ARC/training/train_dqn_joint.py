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
from models.dqn_joint import JointDDQN


class JointDQNPrioritizedBuffer:
    def __init__(self, capacity, max_grid_size=10, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.max_grid_size = max_grid_size
        self.pos = 0
        self.buffer_size = 0
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
    def __init__(self, data_dir, max_grid_size=10, train=True, test_ratio=0.1, reward_scale=1.0):
        self.max_grid_size = max_grid_size
        self.reward_scale = reward_scale
        self.transitions = []
        task_folders = glob(os.path.join(data_dir, "train.*")) if train else glob(os.path.join(data_dir, "test.*"))
        if len(task_folders) == 0:
            direct_json_files = glob(os.path.join(data_dir, "*.json"))
            if len(direct_json_files) > 0:
                task_folders = [data_dir]
        for task_folder in tqdm(task_folders, desc="Loading trajectories"):
            json_files = glob(os.path.join(task_folder, "*.json"))
            if len(json_files) == 0:
                json_files = glob(os.path.join(task_folder, "*", "*.json"))
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        traj = json.load(f)
                    self._process_trajectory(traj)
                except Exception as e:
                    continue
        if train:
            n_test = int(len(self.transitions) * test_ratio)
            self.transitions = self.transitions[:-n_test] if n_test > 0 else self.transitions
        else:
            n_test = int(len(self.transitions) * test_ratio)
            self.transitions = self.transitions[-n_test:] if n_test > 0 else self.transitions

    def _pad_grid(self, grid, max_size):
        h, w = len(grid), len(grid[0]) if len(grid) > 0 else 0
        padded = np.full((max_size, max_size), 10, dtype=np.float32)
        for i in range(min(h, max_size)):
            for j in range(min(w, max_size)):
                padded[i, j] = grid[i][j]
        return padded

    def _process_trajectory(self, traj):
        states = traj.get('grid', [])
        clips = traj.get('clip', [])
        actions = traj.get('operation', [])
        rewards = traj.get('reward', [])
        selections = traj.get('selection', [])
        ex_in = traj.get('ex_in', [])
        ex_out = traj.get('ex_out', [])
        in_grid = traj.get('in_grid', [])
        if len(states) < 2 or len(actions) < 1:
            return
        pair_in = []
        pair_out = []
        for i in range(min(3, len(ex_in))):
            pair_in.append(self._pad_grid(ex_in[i], self.max_grid_size))
            pair_out.append(self._pad_grid(ex_out[i], self.max_grid_size))
        while len(pair_in) < 3:
            pair_in.append(np.full((self.max_grid_size, self.max_grid_size), 10, dtype=np.float32))
            pair_out.append(np.full((self.max_grid_size, self.max_grid_size), 10, dtype=np.float32))
        pair_in = np.stack(pair_in)
        pair_out = np.stack(pair_out)
        if in_grid:
            in_grid_padded = self._pad_grid(in_grid, self.max_grid_size)
        else:
            in_grid_padded = np.full((self.max_grid_size, self.max_grid_size), 10, dtype=np.float32)
        for t in range(len(actions)):
            state = self._pad_grid(states[t], self.max_grid_size)
            if t < len(clips) and clips[t]:
                clip = self._pad_grid(clips[t], self.max_grid_size)
            else:
                clip = np.full((self.max_grid_size, self.max_grid_size), 10, dtype=np.float32)
            action = actions[t]
            reward = (rewards[t] if t < len(rewards) else 0) * self.reward_scale
            selection = selections[t] if t < len(selections) else [0, 0, 0, 0]
            done = 1.0 if (t == len(actions) - 1 or action == 34) else 0.0
            if done == 1.0 or t >= len(states) - 1:
                next_state = state
                next_clip = clip
                next_selection = selection
            else:
                next_state = self._pad_grid(states[t + 1], self.max_grid_size)
                if t + 1 < len(clips) and clips[t + 1]:
                    next_clip = self._pad_grid(clips[t + 1], self.max_grid_size)
                else:
                    next_clip = np.full((self.max_grid_size, self.max_grid_size), 10, dtype=np.float32)
                next_selection = selections[t + 1] if t + 1 < len(selections) else [0, 0, 0, 0]
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
        return self.transitions[idx]


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train_dataset = ARCTrajectoryDataset(
        args.data_dir,
        max_grid_size=args.max_grid_size,
        train=True,
        test_ratio=args.test_ratio,
        reward_scale=args.reward_scale
    )
    dqn_agent = JointDDQN(
        a_dim=args.a_dim,
        h_dim=args.h_dim,
        max_grid_size=args.max_grid_size,
        gamma=args.gamma,
        tau=args.tau,
        lr=args.lr,
        use_positional_encoding=bool(args.use_positional_encoding),
        device=device,
        scheduler_type=args.scheduler_type,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        cosine_t_max=args.cosine_t_max,
        cosine_eta_min=args.cosine_eta_min,
    )
    use_per = bool(args.use_per)
    if use_per:
        per_buffer = JointDQNPrioritizedBuffer(
            capacity=len(train_dataset),
            max_grid_size=args.max_grid_size,
            prob_alpha=args.per_alpha
        )
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
    task_name = os.path.basename(os.path.normpath(args.data_dir))
    dqn_agent.learn(
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
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--a_dim', type=int, default=36)
    parser.add_argument('--h_dim', type=int, default=512)
    parser.add_argument('--max_grid_size', type=int, default=10)
    parser.add_argument('--use_positional_encoding', type=int, default=0)
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--tau', type=float, default=0.995)
    parser.add_argument('--target_update_freq', type=int, default=1)
    parser.add_argument('--save_cycle', type=int, default=50)
    parser.add_argument('--reward_scale', type=float, default=1.0)
    parser.add_argument('--use_per', type=int, default=1)
    parser.add_argument('--per_alpha', type=float, default=0.6)
    parser.add_argument('--per_beta_start', type=float, default=0.4)
    parser.add_argument('--per_beta_increment', type=float, default=0.001)
    parser.add_argument('--scheduler_type', type=str, default='step', choices=['step', 'cosine'])
    parser.add_argument('--lr_step_size', type=int, default=50)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--cosine_t_max', type=int, default=100)
    parser.add_argument('--cosine_eta_min', type=float, default=1e-6)
    parser.add_argument('--gpu_name', type=str, required=True)
    parser.add_argument('--date', type=str, default='02.03')
    args = parser.parse_args()
    train(args)
