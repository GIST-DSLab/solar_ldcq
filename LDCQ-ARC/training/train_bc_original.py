import os
import sys
import datetime

curr_folder = os.path.abspath(__file__)
parent_folder = os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder)

from argparse import ArgumentParser
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from models.bc_original import OriginalBC


class BCDataset(Dataset):
    """Dataset for BC training - only needs states and actions"""

    def __init__(self, data_dir, filename, train_or_test="train", test_prop=0.1):
        # Load all data into RAM
        self.state_all = np.load(os.path.join(data_dir, filename + "_states.npy"), allow_pickle=True)
        self.in_grid_all = np.load(os.path.join(data_dir, filename + "_in_grid.npy"), allow_pickle=True)
        self.pair_in_all = np.load(os.path.join(data_dir, filename + "_pair_in.npy"), allow_pickle=True)
        self.pair_out_all = np.load(os.path.join(data_dir, filename + "_pair_out.npy"), allow_pickle=True)

        # Load action data (latents contain action information)
        self.latent_all = np.load(os.path.join(data_dir, filename + "_latents.npy"), allow_pickle=True)

        # BC doesn't need rewards or next states, only (state, action) pairs
        print(f"Loaded BC dataset:")
        print(f"  - States: {self.state_all.shape}")
        print(f"  - Actions: {self.latent_all.shape}")

        # Split train/test
        n_train = int(self.state_all.shape[0] * (1 - test_prop))
        if train_or_test == "train":
            self.state_all = self.state_all[:n_train]
            self.in_grid_all = self.in_grid_all[:n_train]
            self.pair_in_all = self.pair_in_all[:n_train]
            self.pair_out_all = self.pair_out_all[:n_train]
            self.latent_all = self.latent_all[:n_train]
        elif train_or_test == "test":
            self.state_all = self.state_all[n_train:]
            self.in_grid_all = self.in_grid_all[n_train:]
            self.pair_in_all = self.pair_in_all[n_train:]
            self.pair_out_all = self.pair_out_all[n_train:]
            self.latent_all = self.latent_all[n_train:]
        else:
            raise NotImplementedError

    def __len__(self):
        return self.state_all.shape[0]

    def __getitem__(self, index):
        state = self.state_all[index]
        in_grid = self.in_grid_all[index]
        pair_in = self.pair_in_all[index]
        pair_out = self.pair_out_all[index]
        action = self.latent_all[index]

        # Create dummy selection vector (will need to extract from actual data)
        # selection format: [x1, y1, x2, y2] normalized by max_grid_size
        selection = np.zeros(4, dtype=np.float32)

        return (state, in_grid, pair_in, pair_out, selection, action)


def train(args):
    """Main training function"""

    # Create dataset
    torch_data_train = BCDataset(
        args.data_dir,
        args.skill_model_filename[:-4] if args.skill_model_filename.endswith('.npy') else args.skill_model_filename,
        train_or_test="train",
        test_prop=args.test_split
    )

    dataload_train = DataLoader(
        torch_data_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8
    )

    # Create BC agent
    bc_agent = OriginalBC(
        a_dim=args.a_dim,
        h_dim=args.h_dim,
        max_grid_size=args.max_grid_size,
        lr=args.lr,
        use_positional_encoding=bool(args.use_positional_encoding),
        device=args.device,
        scheduler_type=args.scheduler_type,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        cosine_t_max=args.cosine_t_max,
        cosine_eta_min=args.cosine_eta_min,
    )

    # Extract task name
    task_name = args.solar_dir.split("/")[-1] if args.solar_dir else "default_task"

    # Start training
    print(f"\n{'='*60}")
    print(f"Starting Original BC training")
    print(f"Task: {task_name}")
    print(f"Max grid size: {args.max_grid_size} (actions: 0~{args.max_grid_size-1})")
    print(f"Total actions: {36 * (args.max_grid_size ** 4):,}")
    print(f"{'='*60}\n")

    bc_agent.learn(
        dataloader=dataload_train,
        n_epochs=args.n_epoch,
        checkpoint_dir=args.q_checkpoint_dir,
        gpu_name=args.gpu_name,
        task_name=task_name,
        args=args
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    # Environment
    parser.add_argument('--env', type=str, default='arc-task')
    parser.add_argument('--device', type=str, default='cuda')

    # Training
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_split', type=float, default=0.1)

    # Network architecture
    parser.add_argument('--a_dim', type=int, default=36, help='Number of operations')
    parser.add_argument('--h_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--max_grid_size', type=int, default=10, help='Max grid size (10 means 0~9)')
    parser.add_argument('--use_positional_encoding', type=int, default=1, help='Use 2D positional encoding')

    # Scheduler
    parser.add_argument('--scheduler_type', type=str, default='step', choices=['step', 'cosine'])
    parser.add_argument('--lr_step_size', type=int, default=50)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--cosine_t_max', type=int, default=100)
    parser.add_argument('--cosine_eta_min', type=float, default=1e-6)

    # Data and checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default=parent_folder+'/checkpoints/')
    parser.add_argument('--q_checkpoint_dir', type=str, default=parent_folder+'/q_checkpoints/')
    parser.add_argument('--solar_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=parent_folder+'/data/')
    parser.add_argument('--skill_model_filename', type=str, required=True)

    # GPU
    parser.add_argument('--gpu_name', type=str, required=True)

    args = parser.parse_args()

    train(args)
