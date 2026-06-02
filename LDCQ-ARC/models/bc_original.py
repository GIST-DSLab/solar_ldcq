"""
Original Behavior Cloning (BC) for ARC tasks
Direct state -> action mapping with joint action space (non-factorized)
Supervised learning approach
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


class OriginalBC_Policy(nn.Module):
    """
    Policy network for original BC
    Takes state (grid) and outputs action logits for ALL joint actions
    """

    def __init__(self, a_dim=36, h_dim=512, max_grid_size=10, use_positional_encoding=True):
        super(OriginalBC_Policy, self).__init__()

        self.h_dim = h_dim
        self.a_dim = a_dim  # number of operations (36)
        self.max_grid_size = max_grid_size
        self.num_actions = a_dim * (max_grid_size ** 4)  # 36 × max_grid_size^4
        self.use_positional_encoding = use_positional_encoding

        print(f"OriginalBC_Policy initialized with {self.num_actions:,} joint actions")
        print(f"  - Operations: {a_dim}")
        print(f"  - Grid size: [0, {max_grid_size-1}]")
        print(f"  - Total actions: {a_dim} × {max_grid_size}^4 = {self.num_actions:,}")

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

        # Policy head for ALL joint actions
        # This outputs logits for each possible action
        self.policy_head = nn.Sequential(
            nn.Linear(h_dim + h_dim + h_dim + h_dim, h_dim),  # state + clip + in_grid + pair
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, self.num_actions)  # Output logit for each joint action
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
            action_logits: (batch, num_actions) - logits for all joint actions
        """
        batch_size = state.shape[0]

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

        # Predict action logits for ALL joint actions
        action_logits = self.policy_head(combined)  # (batch, num_actions)

        return action_logits


class OriginalBC(nn.Module):
    """
    Original Behavior Cloning agent for ARC tasks
    Uses joint action space (non-factorized) with supervised learning
    """

    def __init__(self, a_dim=36, h_dim=512, max_grid_size=10,
                 lr=1e-4, use_positional_encoding=True, device='cuda',
                 scheduler_type='step', lr_step_size=50, lr_gamma=0.5,
                 cosine_t_max=100, cosine_eta_min=1e-6):
        super(OriginalBC, self).__init__()

        self.a_dim = a_dim
        self.lr = lr
        self.device = device
        self.max_grid_size = max_grid_size
        self.num_actions = a_dim * (max_grid_size ** 4)

        print(f"OriginalBC initialized:")
        print(f"  - Total joint actions: {self.num_actions:,}")
        print(f"  - Learning rate: {lr}")

        # Policy network
        self.policy = OriginalBC_Policy(
            a_dim=a_dim, h_dim=h_dim, max_grid_size=max_grid_size,
            use_positional_encoding=use_positional_encoding
        ).to(device)

        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr)

        # Scheduler based on type
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cosine_t_max, eta_min=cosine_eta_min
            )
        else:  # 'step' scheduler
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=lr_step_size, gamma=lr_gamma
            )

    def encode_action_batch(self, op, x, y, h, w):
        """Batch version of encode_action"""
        batch_size = op.shape[0]
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
        """
        Get action from policy

        Args:
            deterministic: if True, return argmax; if False, sample from distribution
        """
        with torch.no_grad():
            action_logits = self.policy(state, clip, in_grid, pair_in, pair_out)

            if deterministic:
                # Greedy action selection
                action_idx = action_logits.argmax(dim=-1).item()
            else:
                # Sample from distribution
                action_probs = F.softmax(action_logits, dim=-1)
                action_idx = torch.multinomial(action_probs, num_samples=1).item()

            # Decode to (op, x, y, h, w)
            op, x, y, h, w = decode_action(action_idx, self.max_grid_size)

            return op, x, y, h, w

    def learn_step(self, batch):
        """
        Single learning step with cross-entropy loss

        Args:
            batch: tuple of (state, clip, in_grid, pair_in, pair_out, selection, action)

        Returns:
            loss: cross-entropy loss value
        """
        state, clip, in_grid, pair_in, pair_out, selection, action = batch

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

        # Forward pass
        action_logits = self.policy(state, clip, in_grid, pair_in, pair_out)

        # Cross-entropy loss
        loss = F.cross_entropy(action_logits, action_indices)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        # Compute accuracy
        predicted_actions = action_logits.argmax(dim=-1)
        accuracy = (predicted_actions == action_indices).float().mean()

        return loss.item(), accuracy.item()

    def learn(self, dataloader, n_epochs=100,
              checkpoint_dir='', gpu_name=None, task_name='', args=None):
        """
        Main training loop
        """
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
            'lr': self.lr,
            'max_grid_size': self.max_grid_size,
            'num_actions': self.num_actions,
            'bc_type': 'original_joint_action',
        }

        if args is not None:
            base_config = vars(args).copy()
            config = {**config, **base_config}

        import wandb

        wandb_config = {
            'entity': 'dbsgh797210',
            'project': 'BC_original',
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

        try:
            run = wandb.init(
                entity=wandb_config['entity'],
                project=wandb_config['project'],
                name='OriginalBC_' + gpu_name + '_' + task + '_' + str(d.month) + '.' + str(d.day) + '_' + str(d.hour) + '.' + str(d.minute),
                config=config,
                mode='online',
                save_code=False,
            )
            print("WandB initialized")
        except Exception as e:
            print(f"WandB failed ({e}), continuing without wandb")
            run = None

        steps_total = 0
        update_steps = 2000
        best_loss = float('inf')  # Track best loss for saving best checkpoint

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        for epoch in tqdm(range(n_epochs), desc="Epoch", mininterval=600.0):
            self.policy.train()

            loss_ep = 0
            acc_ep = 0
            n_batch = 0

            pbar = tqdm(dataloader, mininterval=600.0)
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                state, clip, in_grid, pair_in, pair_out, selection, action = batch

                state = state.to(self.device).float()
                clip = clip.to(self.device).float()
                in_grid = in_grid.to(self.device).float()
                pair_in = pair_in.to(self.device).float()
                pair_out = pair_out.to(self.device).float()
                selection = selection.to(self.device).float()
                action = action.to(self.device).squeeze(-1)

                batch_data = (state, clip, in_grid, pair_in, pair_out, selection, action)

                # Train
                loss, accuracy = self.learn_step(batch_data)

                loss_ep += loss
                acc_ep += accuracy
                n_batch += 1
                steps_total += 1

                # Update progress bar
                if n_batch % 100 == 0:
                    pbar.set_description(f"loss: {loss_ep / n_batch:.4f}, acc: {acc_ep / n_batch:.4f}")

                # Log to wandb
                if steps_total % 100 == 0 and run:
                    wandb.log({
                        "train_BC/loss": loss_ep / n_batch,
                        "train_BC/accuracy": acc_ep / n_batch,
                        "train_BC/steps": steps_total,
                    })

                # Step scheduler
                if steps_total % update_steps == 0:
                    self.scheduler.step()

                # Save checkpoint
                if steps_total % (update_steps * 50) == 0:
                    save_path = os.path.join(
                        checkpoint_dir,
                        f'original_bc_agent_{steps_total // update_steps}.pt'
                    )
                    torch.save(self, save_path)
                    print(f"Saved checkpoint: {save_path}")

            # End of epoch logging
            epoch_loss = loss_ep / n_batch if n_batch > 0 else float('inf')

            if run:
                wandb.log({
                    "train_BC/epoch_loss": epoch_loss,
                    "train_BC/epoch_accuracy": acc_ep / n_batch,
                    "train_BC/epoch": epoch,
                })

            # Save best checkpoint
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_path = os.path.join(checkpoint_dir, f'original_bc_agent_best.pt')
                torch.save(self, best_path)
                print(f"\nNew best model! Loss: {best_loss:.6f} → Saved to {best_path}")

        # Save final model
        final_path = os.path.join(checkpoint_dir, f'original_bc_agent_final.pt')
        torch.save(self, final_path)
        print(f"Saved final model: {final_path}")

        if run:
            wandb.finish()

        return self
