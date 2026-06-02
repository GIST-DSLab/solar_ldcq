# LDCQ-ARC: Latent Diffusion Constrained Q-Learning for ARC-AGI

Implementation of [Reasoning with Latent Diffusion in Offline Reinforcement Learning (ICLR 2024)](https://arxiv.org/abs/2309.06599) adapted for [ARC-AGI](https://github.com/fchollet/ARC) tasks using the [ARCLE](https://github.com/ConfeitoHS/arcle) environment.

---

## Installation

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Requirements:**
```
torch >= 2.0
arcle == 0.2.5
gymnasium
wandb
tqdm
matplotlib
ipdb
```

---

## Dataset

LDCQ-ARC is trained on the **SOLAR** dataset — synthesized expert and suboptimal trajectories for ARC-AGI tasks. See [../SOLAR-Generator/README.md](../SOLAR-Generator/README.md) for generation instructions.

Dataset format (per episode):
```
states.npy         # (N, H, W) working grid at each step
clips.npy          # (N, H, W) clipboard state
in_grids.npy       # (N, H, W) input grid (constant per episode)
pair_ins.npy       # (N, n_ex, H, W) example input grids
pair_outs.npy      # (N, n_ex, H, W) example output grids
latents.npy        # (N, z_dim) β-VAE encoded latent (filled after Stage 1)
```

---

## Training Pipeline

LDCQ uses a 5-stage training pipeline. Scripts are in `training/1_0/`.

### Stage 1: Train β-VAE Skill Model

Trains a β-Variational Autoencoder to encode action segments `(s_t, ..., s_{t+H})` into latent vectors `z`.

```bash
cd training/1_0
bash gpu0_train_1_skill_model.sh
```

Key arguments:
```
--horizon 5          # action segment length
--z_dim 256          # latent dimension
--h_dim 512          # hidden dimension
--beta 0.1           # β-VAE weight
--num_epochs 400
--encoder_type gru
```

### Stage 2: Collect Diffusion Training Data

Encodes all training trajectories into latent space using the trained skill model.

```bash
bash gpu0_train_2_collect_diffusion_data.sh
```

### Stage 3: Train Diffusion Prior

Trains a diffusion model to learn `p(z | state, task_context)` — the conditional distribution over latent behaviors given the current state and example pairs.

```bash
bash gpu0_train_3_diffusion.sh
```

Key arguments:
```
--diffusion_steps 500   # number of diffusion steps
--s_dim 512             # state embedding dimension
--batch_size 64
```

### Stage 4: Collect Q-Learning Data

Generates rollout data using the diffusion prior for Q-network training.

```bash
bash gpu0_train_4_collect_q_learning.sh
```

### Stage 5: Train Q-Network

Trains a Q-network over the latent space to select among diffusion proposals.

```bash
bash gpu0_train_5_q_learning.sh
```

Key arguments:
```
--total_prior_samples 100   # number of diffusion proposals per state
--n_epoch 2000
--gamma 0.7
--use_ddim 1
--ddim_steps 100
```

---

## Evaluation

```bash
cd eval
bash gpu0_test_ARCLE.sh
```

Key arguments for `plan_skills_diffusion_ARCLE.py`:
```
--num_diffusion_samples 100   # number of latent proposals to sample
--q_checkpoint_steps 150      # which Q-network checkpoint to load
--ddim_steps 100              # DDIM inference steps
--max_episode_steps 30        # maximum episode length
```

---

## Model Architecture

### Skill Model (`models/skill_model.py`)
- CNN encoder for grid observations
- β-VAE with encoder/decoder over action segments
- Positional encoding for grid spatial structure

### Diffusion Prior (`models/diffusion_models.py`)
- DDPM/DDIM over latent skill vectors
- Conditioned on: current grid, clipboard, input grid, example pairs
- Supports Classifier-Free Guidance (CFG)

### Q-Network (`models/dqn.py`)
- CNN grid encoder + pair encoder
- Input: `(state, clip, in_grid, pair_in, pair_out, latent z)`
- Output: Q-value scalar
- Double Q-network
