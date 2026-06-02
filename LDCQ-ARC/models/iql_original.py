"""
Original Implicit Q-Learning (IQL) for ARC tasks
Direct state -> Q(s,a), V(s), π(a|s) mapping with joint action space (non-factorized)
Based on Kostrikov et al., ICLR 2022 "Offline Reinforcement Learning with Implicit Q-Learning"
"""
import os
import sys

curr_folder = os.path.abspath(__file__)
parent_folder = os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import copy
import datetime
from tqdm import tqdm

from models.skill_model import StateEmbeddingWithPositionalEncoding


def encode_action(op, x, y, h, w, max_grid_size):
    """
    Convert (op, x, y, h, w) to single action index

    Args:
        op: operation index [0, 35]
        x, y, h, w: coordinates [0, max_grid_size-1]
        max_grid_size: grid size (e.g., 10 means 0~9)

    Returns:
        action_idx: single integer [0, 36 * max_grid_size^4 - 1]
    """
    idx = (op * (max_grid_size ** 4) +
           x * (max_grid_size ** 3) +
           y * (max_grid_size ** 2) +
           h * max_grid_size +
           w)
    return idx


def decode_action(idx, max_grid_size):
    """
    Convert action index to (op, x, y, h, w)

    Args:
        idx: single action index
        max_grid_size: grid size (e.g., 10 means 0~9)

    Returns:
        op, x, y, h, w: action components
    """
    w = idx % max_grid_size
    idx = idx // max_grid_size
    h = idx % max_grid_size
    idx = idx // max_grid_size
    y = idx % max_grid_size
    idx = idx // max_grid_size
    x = idx % max_grid_size
    op = idx // max_grid_size
    return op, x, y, h, w


class StateEncoder(nn.Module):
    """
    Shared state encoder for IQL networks
    Encodes (state, clip, in_grid, pair_in, pair_out) -> combined embedding
    """

    def __init__(self, h_dim=512, max_grid_size=10, use_positional_encoding=True):
        super(StateEncoder, self).__init__()

        self.h_dim = h_dim
        self.max_grid_size = max_grid_size
        self.use_positional_encoding = use_positional_encoding

        # State embedding with CNN
        state_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        state_flatten_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * max_grid_size * max_grid_size, h_dim),
            nn.ReLU()
        )
        self.state_emb_layer = StateEmbeddingWithPositionalEncoding(
            base_cnn_layers=state_cnn,
            flatten_and_linear=state_flatten_linear,
            use_positional_encoding=use_positional_encoding,
            pos_encoding_channels=64,
            max_grid_size=max_grid_size
        )

        # Pair embedding (input-output examples)
        pair_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        pair_flatten_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * max_grid_size * max_grid_size, h_dim),
            nn.ReLU()
        )
        self.pair_emb_layer = StateEmbeddingWithPositionalEncoding(
            base_cnn_layers=pair_cnn,
            flatten_and_linear=pair_flatten_linear,
            use_positional_encoding=use_positional_encoding,
            pos_encoding_channels=32,
            max_grid_size=max_grid_size
        )

        # Combine pairs
        self.pair_combiner = nn.Sequential(
            nn.Linear(6 * h_dim, h_dim),  # 3 input + 3 output grids
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )

        # Clip embedding (same as state)
        clip_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        clip_flatten_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * max_grid_size * max_grid_size, h_dim),
            nn.ReLU()
        )
        self.clip_emb_layer = StateEmbeddingWithPositionalEncoding(
            base_cnn_layers=clip_cnn,
            flatten_and_linear=clip_flatten_linear,
            use_positional_encoding=use_positional_encoding,
            pos_encoding_channels=64,
            max_grid_size=max_grid_size
        )

    def forward(self, state, clip, in_grid, pair_in, pair_out):
        """
        INPUTS:
            state: (batch, 1, H, W) - current grid state
            clip: (batch, 1, H, W) - clipboard state
            in_grid: (batch, 1, H, W) - input grid (target)
            pair_in: (batch, 3, H, W) - example input grids
            pair_out: (batch, 3, H, W) - example output grids
        OUTPUTS:
            combined: (batch, 4*h_dim) - combined state embedding
        """
        # State embedding
        state_emb = self.state_emb_layer(state.float())
        clip_emb = self.clip_emb_layer(clip.float())
        in_grid_emb = self.state_emb_layer(in_grid.float())

        # Pair embedding
        pair_embs = []
        for i in range(3):
            pin = pair_in[:, i:i+1].float()
            pout = pair_out[:, i:i+1].float()
            pin_emb = self.pair_emb_layer(pin)
            pout_emb = self.pair_emb_layer(pout)
            pair_embs.extend([pin_emb, pout_emb])

        pair_concat = torch.cat(pair_embs, dim=-1)
        pair_emb = self.pair_combiner(pair_concat)

        # Combine all (state + clip + in_grid + pair)
        combined = torch.cat([state_emb, clip_emb, in_grid_emb, pair_emb], dim=-1)

        return combined


class OriginalIQL_Q(nn.Module):
    """
    Q(s,a) network for IQL
    Takes state (grid) and outputs Q-values for ALL joint actions
    """

    def __init__(self, a_dim=36, h_dim=512, max_grid_size=10, use_positional_encoding=True):
        super(OriginalIQL_Q, self).__init__()

        self.h_dim = h_dim
        self.a_dim = a_dim  # number of operations (36)
        self.max_grid_size = max_grid_size
        self.num_actions = a_dim * (max_grid_size ** 4)  # 36 × max_grid_size^4
        self.use_positional_encoding = use_positional_encoding

        print(f"OriginalIQL_Q initialized with {self.num_actions:,} joint actions")
        print(f"  - Operations: {a_dim}")
        print(f"  - Grid size: [0, {max_grid_size-1}]")
        print(f"  - Total actions: {a_dim} × {max_grid_size}^4 = {self.num_actions:,}")

        # State encoder
        self.state_encoder = StateEncoder(
            h_dim=h_dim,
            max_grid_size=max_grid_size,
            use_positional_encoding=use_positional_encoding
        )

        # Q-value head for ALL joint actions
        self.q_head = nn.Sequential(
            nn.Linear(h_dim * 4, h_dim),  # state + clip + in_grid + pair
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, self.num_actions)  # Output Q-value for each joint action
        )

    def forward(self, state, clip, in_grid, pair_in, pair_out):
        """
        INPUTS:
            state: (batch, 1, H, W) - current grid state
            clip: (batch, 1, H, W) - clipboard state
            in_grid: (batch, 1, H, W) - input grid (target)
            pair_in: (batch, 3, H, W) - example input grids
            pair_out: (batch, 3, H, W) - example output grids
        OUTPUTS:
            q_values: (batch, num_actions) - Q-values for all joint actions
        """
        combined = self.state_encoder(state, clip, in_grid, pair_in, pair_out)
        q_values = self.q_head(combined)
        return q_values


class OriginalIQL_V(nn.Module):
    """
    V(s) network for IQL
    Takes state (grid) and outputs scalar value
    """

    def __init__(self, h_dim=512, max_grid_size=10, use_positional_encoding=True):
        super(OriginalIQL_V, self).__init__()

        self.h_dim = h_dim
        self.max_grid_size = max_grid_size
        self.use_positional_encoding = use_positional_encoding

        # State encoder
        self.state_encoder = StateEncoder(
            h_dim=h_dim,
            max_grid_size=max_grid_size,
            use_positional_encoding=use_positional_encoding
        )

        # Value head
        self.v_head = nn.Sequential(
            nn.Linear(h_dim * 4, h_dim),  # state + clip + in_grid + pair
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)  # Output scalar value
        )

    def forward(self, state, clip, in_grid, pair_in, pair_out):
        """
        INPUTS:
            state: (batch, 1, H, W) - current grid state
            clip: (batch, 1, H, W) - clipboard state
            in_grid: (batch, 1, H, W) - input grid (target)
            pair_in: (batch, 3, H, W) - example input grids
            pair_out: (batch, 3, H, W) - example output grids
        OUTPUTS:
            value: (batch, 1) - state value
        """
        combined = self.state_encoder(state, clip, in_grid, pair_in, pair_out)
        value = self.v_head(combined)
        return value


class OriginalIQL_Policy(nn.Module):
    """
    π(a|s) network for IQL
    Takes state (grid) and outputs action logits for ALL joint actions
    """

    def __init__(self, a_dim=36, h_dim=512, max_grid_size=10, use_positional_encoding=True):
        super(OriginalIQL_Policy, self).__init__()

        self.h_dim = h_dim
        self.a_dim = a_dim
        self.max_grid_size = max_grid_size
        self.num_actions = a_dim * (max_grid_size ** 4)
        self.use_positional_encoding = use_positional_encoding

        # State encoder
        self.state_encoder = StateEncoder(
            h_dim=h_dim,
            max_grid_size=max_grid_size,
            use_positional_encoding=use_positional_encoding
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(h_dim * 4, h_dim),  # state + clip + in_grid + pair
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, self.num_actions)  # Output logits for each joint action
        )

    def forward(self, state, clip, in_grid, pair_in, pair_out):
        """
        INPUTS:
            state: (batch, 1, H, W) - current grid state
            clip: (batch, 1, H, W) - clipboard state
            in_grid: (batch, 1, H, W) - input grid (target)
            pair_in: (batch, 3, H, W) - example input grids
            pair_out: (batch, 3, H, W) - example output grids
        OUTPUTS:
            logits: (batch, num_actions) - action logits
        """
        combined = self.state_encoder(state, clip, in_grid, pair_in, pair_out)
        logits = self.policy_head(combined)
        return logits


class OriginalIQL(nn.Module):
    """
    Original Implicit Q-Learning agent for ARC tasks
    Uses joint action space (non-factorized) with expectile regression

    Key differences from CQL:
    1. Uses V(s) network with expectile regression to approximate in-sample max
    2. Q-function targets use V(s') instead of max_a' Q(s', a')
    3. Policy is trained with advantage-weighted regression (AWR)
    """

    def __init__(self, a_dim=36, h_dim=512, max_grid_size=10,
                 gamma=0.99, tau=0.995, lr=1e-4,
                 iql_tau=0.7, beta=3.0,
                 use_positional_encoding=True, device='cuda',
                 scheduler_type='step', lr_step_size=50, lr_gamma=0.5,
                 cosine_t_max=100, cosine_eta_min=1e-6):
        super(OriginalIQL, self).__init__()

        self.a_dim = a_dim
        self.gamma = gamma
        self.tau = tau  # Target network EMA coefficient
        self.lr = lr
        self.device = device
        self.max_grid_size = max_grid_size
        self.num_actions = a_dim * (max_grid_size ** 4)

        # IQL specific parameters
        self.iql_tau = iql_tau  # Expectile for V training (0.5 = mean, 1.0 = max)
        self.beta = beta  # Temperature for AWR policy extraction

        print(f"OriginalIQL initialized:")
        print(f"  - Total joint actions: {self.num_actions:,}")
        print(f"  - IQL tau (expectile): {iql_tau}")
        print(f"  - Beta (AWR temperature): {beta}")

        # Q networks (Double Q-learning)
        self.q_net_0 = OriginalIQL_Q(
            a_dim=a_dim, h_dim=h_dim, max_grid_size=max_grid_size,
            use_positional_encoding=use_positional_encoding
        ).to(device)

        self.q_net_1 = OriginalIQL_Q(
            a_dim=a_dim, h_dim=h_dim, max_grid_size=max_grid_size,
            use_positional_encoding=use_positional_encoding
        ).to(device)

        # Value network (for expectile regression)
        self.v_net = OriginalIQL_V(
            h_dim=h_dim, max_grid_size=max_grid_size,
            use_positional_encoding=use_positional_encoding
        ).to(device)

        # Policy network
        self.policy_net = OriginalIQL_Policy(
            a_dim=a_dim, h_dim=h_dim, max_grid_size=max_grid_size,
            use_positional_encoding=use_positional_encoding
        ).to(device)

        self.target_q_net_0 = None
        self.target_q_net_1 = None

        # Optimizers
        self.optimizer_q0 = optim.AdamW(self.q_net_0.parameters(), lr=lr)
        self.optimizer_q1 = optim.AdamW(self.q_net_1.parameters(), lr=lr)
        self.optimizer_v = optim.AdamW(self.v_net.parameters(), lr=lr)
        self.optimizer_policy = optim.AdamW(self.policy_net.parameters(), lr=lr)

        # Scheduler based on type
        if scheduler_type == 'cosine':
            self.scheduler_q0 = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_q0, T_max=cosine_t_max, eta_min=cosine_eta_min
            )
            self.scheduler_q1 = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_q1, T_max=cosine_t_max, eta_min=cosine_eta_min
            )
            self.scheduler_v = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_v, T_max=cosine_t_max, eta_min=cosine_eta_min
            )
            self.scheduler_policy = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_policy, T_max=cosine_t_max, eta_min=cosine_eta_min
            )
        else:  # 'step' scheduler
            self.scheduler_q0 = optim.lr_scheduler.StepLR(
                self.optimizer_q0, step_size=lr_step_size, gamma=lr_gamma
            )
            self.scheduler_q1 = optim.lr_scheduler.StepLR(
                self.optimizer_q1, step_size=lr_step_size, gamma=lr_gamma
            )
            self.scheduler_v = optim.lr_scheduler.StepLR(
                self.optimizer_v, step_size=lr_step_size, gamma=lr_gamma
            )
            self.scheduler_policy = optim.lr_scheduler.StepLR(
                self.optimizer_policy, step_size=lr_step_size, gamma=lr_gamma
            )

    def encode_action_batch(self, op, x, y, h, w):
        """Batch version of encode_action"""
        action_indices = (op * (self.max_grid_size ** 4) +
                         x * (self.max_grid_size ** 3) +
                         y * (self.max_grid_size ** 2) +
                         h * self.max_grid_size +
                         w)
        return action_indices

    def decode_action_batch(self, action_indices):
        """Batch version of decode_action"""
        w = action_indices % self.max_grid_size
        action_indices = action_indices // self.max_grid_size
        h = action_indices % self.max_grid_size
        action_indices = action_indices // self.max_grid_size
        y = action_indices % self.max_grid_size
        action_indices = action_indices // self.max_grid_size
        x = action_indices % self.max_grid_size
        op = action_indices // self.max_grid_size
        return op, x, y, h, w

    def get_action(self, state, clip, in_grid, pair_in, pair_out, deterministic=True):
        """Get action (operation, x, y, h, w) from policy network"""
        with torch.no_grad():
            logits = self.policy_net(state, clip, in_grid, pair_in, pair_out)

            if deterministic:
                # Greedy action selection
                action_idx = logits.argmax(dim=-1).item()
            else:
                # Sample from policy distribution
                probs = F.softmax(logits, dim=-1)
                action_idx = torch.multinomial(probs, 1).item()

            # Decode to (op, x, y, h, w)
            op, x, y, h, w = decode_action(action_idx, self.max_grid_size)

            return op, x, y, h, w

    def get_action_from_q(self, state, clip, in_grid, pair_in, pair_out):
        """Get action using minimum of two Q-networks (for comparison)"""
        with torch.no_grad():
            q_values_0 = self.q_net_0(state, clip, in_grid, pair_in, pair_out)
            q_values_1 = self.q_net_1(state, clip, in_grid, pair_in, pair_out)

            # Take minimum of two Q-networks for pessimistic estimation
            q_values = torch.minimum(q_values_0, q_values_1)

            # Get argmax action
            action_idx = q_values.argmax(dim=-1).item()

            # Decode to (op, x, y, h, w)
            op, x, y, h, w = decode_action(action_idx, self.max_grid_size)

            return op, x, y, h, w

    def update_target(self):
        """Soft update target Q networks"""
        if self.target_q_net_0 is None:
            self.target_q_net_0 = copy.deepcopy(self.q_net_0)
            self.target_q_net_1 = copy.deepcopy(self.q_net_1)
            self.target_q_net_0.eval()
            self.target_q_net_1.eval()
        else:
            for target_param, param in zip(self.target_q_net_0.parameters(), self.q_net_0.parameters()):
                target_param.data.copy_(self.tau * target_param.data + (1 - self.tau) * param.data)
            for target_param, param in zip(self.target_q_net_1.parameters(), self.q_net_1.parameters()):
                target_param.data.copy_(self.tau * target_param.data + (1 - self.tau) * param.data)

    def expectile_loss(self, diff, expectile=0.7):
        """
        Asymmetric squared loss for expectile regression

        L_tau(u) = |tau - 1(u < 0)| * u^2

        When tau > 0.5, this loss penalizes underestimation more than overestimation,
        effectively approximating the maximum.
        """
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return (weight * (diff ** 2)).mean()

    def compute_v_loss(self, state, clip, in_grid, pair_in, pair_out, action_indices):
        """
        Compute value loss using expectile regression

        L_V = E[L_τ(Q(s,a) - V(s))]

        Args:
            state, clip, in_grid, pair_in, pair_out: state inputs
            action_indices: taken action indices

        Returns:
            v_loss: scalar
        """
        # Get Q-values for taken actions (use target Q networks)
        with torch.no_grad():
            q_values_0 = self.target_q_net_0(state, clip, in_grid, pair_in, pair_out)
            q_values_1 = self.target_q_net_1(state, clip, in_grid, pair_in, pair_out)

            # Take minimum of two Q-networks
            q_values = torch.minimum(q_values_0, q_values_1)
            q_taken = q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        # Get value prediction
        v_pred = self.v_net(state, clip, in_grid, pair_in, pair_out).squeeze(1)

        # Expectile regression loss
        diff = q_taken - v_pred
        v_loss = self.expectile_loss(diff, self.iql_tau)

        return v_loss

    def compute_q_loss(self, state, clip, in_grid, pair_in, pair_out, action_indices,
                       reward, next_state, next_clip, done, net_id=0, weights=None):
        """
        Compute Q-function loss

        L_Q = E[(r + γV(s') - Q(s,a))^2]

        Key difference from CQL: Uses V(s') instead of max_a' Q(s', a')

        Args:
            state, clip, in_grid, pair_in, pair_out: current state inputs
            action_indices: taken action indices
            reward: rewards
            next_state, next_clip: next state inputs
            done: done flags
            net_id: which Q-network to train (0 or 1)
            weights: importance sampling weights for PER

        Returns:
            q_loss, td_error
        """
        # Select Q network
        if net_id == 0:
            q_net = self.q_net_0
            optimizer = self.optimizer_q0
        else:
            q_net = self.q_net_1
            optimizer = self.optimizer_q1

        # Current Q-values for taken actions
        q_values = q_net(state, clip, in_grid, pair_in, pair_out)
        q_taken = q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        # Target: r + γV(s')
        with torch.no_grad():
            next_v = self.v_net(next_state, next_clip, in_grid, pair_in, pair_out).squeeze(1)
            target = reward + self.gamma * next_v * (1 - done)

        # TD error for PER priority update
        td_error = (q_taken - target).abs()

        # Q loss (with importance sampling weights if using PER)
        if weights is not None:
            weights_tensor = torch.FloatTensor(weights).to(self.device)
            q_loss = ((q_taken - target).pow(2) * weights_tensor).mean()
        else:
            q_loss = F.mse_loss(q_taken, target)

        return q_loss, td_error

    def compute_policy_loss(self, state, clip, in_grid, pair_in, pair_out, action_indices):
        """
        Compute policy loss using Advantage Weighted Regression (AWR)

        L_π = -E[exp(β * A(s,a)) * log π(a|s)]

        where A(s,a) = Q(s,a) - V(s)

        Args:
            state, clip, in_grid, pair_in, pair_out: state inputs
            action_indices: taken action indices

        Returns:
            policy_loss: scalar
        """
        # Get advantage
        with torch.no_grad():
            q_values_0 = self.target_q_net_0(state, clip, in_grid, pair_in, pair_out)
            q_values_1 = self.target_q_net_1(state, clip, in_grid, pair_in, pair_out)
            q_values = torch.minimum(q_values_0, q_values_1)
            q_taken = q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)

            v_pred = self.v_net(state, clip, in_grid, pair_in, pair_out).squeeze(1)

            # Advantage
            advantage = q_taken - v_pred

            # Compute advantage weights with temperature
            # Clip advantage for numerical stability
            advantage = torch.clamp(advantage, -10.0, 10.0)
            weights = torch.exp(self.beta * advantage)
            weights = torch.clamp(weights, max=100.0)  # Clip weights for stability

        # Get policy logits
        logits = self.policy_net(state, clip, in_grid, pair_in, pair_out)

        # Cross-entropy loss weighted by advantages
        log_probs = F.log_softmax(logits, dim=-1)
        log_prob_actions = log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        policy_loss = -(weights * log_prob_actions).mean()

        return policy_loss

    def learn_step(self, batch, weights=None):
        """
        Single learning step for IQL

        Args:
            batch: tuple of (state, clip, in_grid, pair_in, pair_out, selection, action, reward, next_state, next_clip, next_selection, done)
            weights: importance sampling weights for PER (optional)

        Returns:
            dict of losses and td_error
        """
        state, clip, in_grid, pair_in, pair_out, selection, action, reward, next_state, next_clip, next_selection, done = batch

        # Extract ground truth actions and encode to joint action index
        gt_operation = action.long()
        selection_unnorm = (selection * self.max_grid_size).long()
        gt_x = selection_unnorm[:, 0].clamp(0, self.max_grid_size - 1)
        gt_y = selection_unnorm[:, 1].clamp(0, self.max_grid_size - 1)
        x2 = selection_unnorm[:, 2].clamp(0, self.max_grid_size - 1)
        y2 = selection_unnorm[:, 3].clamp(0, self.max_grid_size - 1)
        gt_h = (x2 - gt_x).clamp(min=0, max=self.max_grid_size-1)
        gt_w = (y2 - gt_y).clamp(min=0, max=self.max_grid_size-1)

        # Encode to joint action index
        action_indices = self.encode_action_batch(gt_operation, gt_x, gt_y, gt_h, gt_w)

        # 1. Update Value network
        v_loss = self.compute_v_loss(state, clip, in_grid, pair_in, pair_out, action_indices)

        self.optimizer_v.zero_grad()
        v_loss.backward()
        clip_grad_norm_(self.v_net.parameters(), 1.0)
        self.optimizer_v.step()

        # 2. Update Q networks
        q_loss_0, td_error_0 = self.compute_q_loss(
            state, clip, in_grid, pair_in, pair_out, action_indices,
            reward, next_state, next_clip, done, net_id=0, weights=weights
        )

        self.optimizer_q0.zero_grad()
        q_loss_0.backward()
        clip_grad_norm_(self.q_net_0.parameters(), 1.0)
        self.optimizer_q0.step()

        q_loss_1, td_error_1 = self.compute_q_loss(
            state, clip, in_grid, pair_in, pair_out, action_indices,
            reward, next_state, next_clip, done, net_id=1, weights=weights
        )

        self.optimizer_q1.zero_grad()
        q_loss_1.backward()
        clip_grad_norm_(self.q_net_1.parameters(), 1.0)
        self.optimizer_q1.step()

        # 3. Update Policy network
        policy_loss = self.compute_policy_loss(state, clip, in_grid, pair_in, pair_out, action_indices)

        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer_policy.step()

        # Average TD error for PER
        td_error = (td_error_0 + td_error_1) / 2.0

        return {
            'v_loss': v_loss.item(),
            'q_loss_0': q_loss_0.item(),
            'q_loss_1': q_loss_1.item(),
            'policy_loss': policy_loss.item(),
            'total_loss': v_loss.item() + q_loss_0.item() + q_loss_1.item() + policy_loss.item(),
            'td_error': td_error.detach().cpu().numpy()
        }

    def learn(self, dataloader=None, replay_buffer=None, n_epochs=100, update_frequency=1,
              checkpoint_dir='', gpu_name=None, task_name='', args=None, use_per=False,
              per_beta_start=0.4, per_beta_increment=0.001, batch_size=128):
        """
        Main training loop

        Args:
            dataloader: standard dataloader (if not using PER)
            replay_buffer: PER buffer (if using PER)
            use_per: whether to use PER
        """
        assert (dataloader is not None) or (replay_buffer is not None), "Must provide either dataloader or replay_buffer"
        d = datetime.datetime.now()

        # Handle different task name formats
        if "." in task_name:
            task = task_name.split(".")[1]
        else:
            task = task_name

        # WandB setup
        from wandb_config import setup_wandb_api_key
        setup_wandb_api_key()

        config = {
            'task': task_name,
            'iql_tau': self.iql_tau,
            'beta': self.beta,
            'gamma': self.gamma,
            'tau': self.tau,
            'lr': self.lr,
            'max_grid_size': self.max_grid_size,
            'num_actions': self.num_actions,
            'iql_type': 'original_joint_action',
        }

        if args is not None:
            base_config = vars(args).copy()
            config = {**config, **base_config}

        import wandb

        wandb_config = {
            'entity': 'dbsgh797210',
            'project': 'IQL_original',
            'api_key': '391af36b1546e19e6e1eb483f69c989abf5d202a'
        }

        os.environ["WANDB_API_KEY"] = wandb_config['api_key']

        try:
            wandb.login(key=wandb_config['api_key'])
            print("WandB login successful")
        except Exception as e:
            print(f"WandB login failed: {e}")

        os.environ['WANDB_DISABLE_CODE'] = 'true'
        os.environ['WANDB_DISABLE_GIT'] = 'true'
        _wandb_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training', 'wandb')
        os.makedirs(_wandb_dir, exist_ok=True)
        os.environ['WANDB_DIR'] = _wandb_dir

        try:
            run = wandb.init(
                entity=wandb_config['entity'],
                project=wandb_config['project'],
                name='OriginalIQL_' + gpu_name + '_' + task + '_' + str(d.month) + '.' + str(d.day) + '_' + str(d.hour) + '.' + str(d.minute),
                config=config,
                mode='offline',
                dir=_wandb_dir,
                save_code=False,
            )
            print("WandB initialized")
        except Exception as e:
            print(f"WandB failed ({e}), continuing without wandb")
            run = None

        # Initialize target networks
        self.target_q_net_0 = copy.deepcopy(self.q_net_0)
        self.target_q_net_1 = copy.deepcopy(self.q_net_1)
        self.target_q_net_0.eval()
        self.target_q_net_1.eval()

        steps_total = 0
        update_steps = 2000
        beta = per_beta_start  # PER beta
        best_loss = float('inf')  # Track best loss for saving best checkpoint

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        for epoch in tqdm(range(n_epochs), desc="Epoch", mininterval=600.0):
            self.q_net_0.train()
            self.q_net_1.train()
            self.v_net.train()
            self.policy_net.train()

            v_loss_ep, q_loss_0_ep, q_loss_1_ep, policy_loss_ep = 0, 0, 0, 0
            n_batch = 0

            if use_per:
                # PER training loop
                num_updates = len(replay_buffer) // batch_size if hasattr(replay_buffer, '__len__') else 1000
                pbar = tqdm(range(num_updates), mininterval=600.0)

                for _ in pbar:
                    # Sample from PER buffer
                    (state, clip, in_grid, pair_in, pair_out, selection, action,
                     reward, next_state, next_clip, next_selection, done,
                     indices, weights) = replay_buffer.sample(batch_size, beta)

                    # Convert to tensors
                    state = torch.FloatTensor(state).to(self.device)
                    clip = torch.FloatTensor(clip).to(self.device)
                    in_grid = torch.FloatTensor(in_grid).to(self.device)
                    pair_in = torch.FloatTensor(pair_in).to(self.device)
                    pair_out = torch.FloatTensor(pair_out).to(self.device)
                    selection = torch.FloatTensor(selection).to(self.device)
                    action = torch.LongTensor(action).to(self.device)
                    reward = torch.FloatTensor(reward).to(self.device)
                    next_state = torch.FloatTensor(next_state).to(self.device)
                    next_clip = torch.FloatTensor(next_clip).to(self.device)
                    next_selection = torch.FloatTensor(next_selection).to(self.device)
                    done = torch.FloatTensor(done).to(self.device)

                    batch_data = (state, clip, in_grid, pair_in, pair_out, selection, action,
                                 reward, next_state, next_clip, next_selection, done)

                    # Train all networks
                    losses = self.learn_step(batch_data, weights=weights)

                    # Update priorities
                    replay_buffer.update_priorities(indices, losses['td_error'] + 1e-6)

                    v_loss_ep += losses['v_loss']
                    q_loss_0_ep += losses['q_loss_0']
                    q_loss_1_ep += losses['q_loss_1']
                    policy_loss_ep += losses['policy_loss']
                    n_batch += 1
                    steps_total += 1

                    # Update beta for PER
                    if steps_total % update_steps == 0:
                        beta = min(beta + per_beta_increment, 1.0)

                    # Update progress bar
                    if n_batch % 100 == 0:
                        pbar.set_description(
                            f"v: {v_loss_ep/n_batch:.4f}, q: {(q_loss_0_ep+q_loss_1_ep)/(2*n_batch):.4f}, "
                            f"pi: {policy_loss_ep/n_batch:.4f}, beta: {beta:.3f}"
                        )

                    # Soft update target networks
                    if steps_total % update_frequency == 0:
                        self.update_target()

                        if run:
                            try:
                                wandb.log({
                                    "train_IQL/v_loss": v_loss_ep / n_batch,
                                    "train_IQL/q_loss_0": q_loss_0_ep / n_batch,
                                    "train_IQL/q_loss_1": q_loss_1_ep / n_batch,
                                    "train_IQL/policy_loss": policy_loss_ep / n_batch,
                                    "train_IQL/beta": beta,
                                    "train_IQL/steps": steps_total,
                                })
                            except Exception:
                                run = None

                    # Step scheduler
                    if steps_total % update_steps == 0:
                        self.scheduler_q0.step()
                        self.scheduler_q1.step()
                        self.scheduler_v.step()
                        self.scheduler_policy.step()

                    # Save checkpoint
                    if steps_total % (update_steps * 50) == 0:
                        save_path = os.path.join(
                            checkpoint_dir,
                            f'joint_iql_agent_{steps_total // update_steps}_tau_{self.iql_tau}_beta_{self.beta}.pt'
                        )
                        torch.save(self, save_path)
                        print(f"Saved checkpoint: {save_path}")
            else:
                # Standard dataloader training loop
                pbar = tqdm(dataloader, mininterval=600.0)
                for batch_idx, batch in enumerate(pbar):
                    # Move batch to device
                    state, in_grid, pair_in, pair_out, selection, action, reward, next_state, next_selection, done = batch

                    state = state.to(self.device).unsqueeze(1).float()
                    clip = state.clone()  # Use state as clip for now (modify if needed)
                    in_grid = in_grid.to(self.device).unsqueeze(1).float()
                    pair_in = pair_in.to(self.device).float()
                    pair_out = pair_out.to(self.device).float()
                    selection = selection.to(self.device).float()
                    action = action.to(self.device).squeeze(-1)
                    reward = reward.to(self.device).float().squeeze(-1)
                    next_state = next_state.to(self.device).unsqueeze(1).float()
                    next_clip = next_state.clone()
                    next_selection = next_selection.to(self.device).float()
                    done = done.to(self.device).float().squeeze(-1)

                    batch_data = (state, clip, in_grid, pair_in, pair_out, selection, action, reward, next_state, next_clip, next_selection, done)

                    # Train all networks
                    losses = self.learn_step(batch_data)

                    v_loss_ep += losses['v_loss']
                    q_loss_0_ep += losses['q_loss_0']
                    q_loss_1_ep += losses['q_loss_1']
                    policy_loss_ep += losses['policy_loss']
                    n_batch += 1
                    steps_total += 1

                    # Update progress bar
                    if n_batch % 100 == 0:
                        pbar.set_description(
                            f"v: {v_loss_ep/n_batch:.4f}, q: {(q_loss_0_ep+q_loss_1_ep)/(2*n_batch):.4f}, "
                            f"pi: {policy_loss_ep/n_batch:.4f}"
                        )

                    # Soft update target networks
                    if steps_total % update_frequency == 0:
                        self.update_target()

                        if run:
                            try:
                                wandb.log({
                                    "train_IQL/v_loss": v_loss_ep / n_batch,
                                    "train_IQL/q_loss_0": q_loss_0_ep / n_batch,
                                    "train_IQL/q_loss_1": q_loss_1_ep / n_batch,
                                    "train_IQL/policy_loss": policy_loss_ep / n_batch,
                                    "train_IQL/steps": steps_total,
                                })
                            except Exception:
                                run = None

                    # Step scheduler
                    if steps_total % update_steps == 0:
                        self.scheduler_q0.step()
                        self.scheduler_q1.step()
                        self.scheduler_v.step()
                        self.scheduler_policy.step()

                    # Save checkpoint
                    if steps_total % (update_steps * 50) == 0:
                        save_path = os.path.join(
                            checkpoint_dir,
                            f'joint_iql_agent_{steps_total // update_steps}_tau_{self.iql_tau}_beta_{self.beta}.pt'
                        )
                        torch.save(self, save_path)
                        print(f"Saved checkpoint: {save_path}")

            # End of epoch logging
            epoch_loss = (v_loss_ep + q_loss_0_ep + q_loss_1_ep + policy_loss_ep) / (4 * n_batch) if n_batch > 0 else float('inf')

            if run:
                try:
                    wandb.log({
                        "train_IQL/epoch_loss": epoch_loss,
                        "train_IQL/epoch": epoch,
                    })
                except Exception:
                    run = None

            # Save best checkpoint
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_path = os.path.join(checkpoint_dir, f'joint_iql_agent_best_tau_{self.iql_tau}_beta_{self.beta}.pt')
                torch.save(self, best_path)
                print(f"\nNew best model! Loss: {best_loss:.6f} -> Saved to {best_path}")

        # Save final model
        final_path = os.path.join(checkpoint_dir, f'joint_iql_agent_final_tau_{self.iql_tau}_beta_{self.beta}.pt')
        torch.save(self, final_path)
        print(f"Saved final model: {final_path}")

        if run:
            wandb.finish()

        return self
