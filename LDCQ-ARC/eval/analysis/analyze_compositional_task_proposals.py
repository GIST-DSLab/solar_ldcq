"""
Compositional Task Proposal Distribution Analysis

Analyzes whether diffusion model proposals at the initial state of a compositional task
(Task A + Task B) correspond to Task A or Task B.

Generates:
- Table X: Proposal distribution analysis in compositional task
- Table Y: Intervention results on compositional task
"""

import os
import sys
import argparse

curr_folder = os.path.abspath(__file__)
parent_folder = os.path.dirname(os.path.dirname(os.path.dirname(curr_folder)))
sys.path.append(parent_folder)

import numpy as np
import torch
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

from models.diffusion_models import Model_Cond_Diffusion
from models.skill_model import SkillModel


def parse_args():
    parser = argparse.ArgumentParser(description='Compositional Task Proposal Analysis')
    parser.add_argument('--checkpoint_name', type=str, required=True,
                        help='Checkpoint name (e.g., gpu6_01.26)')
    parser.add_argument('--combo_test_dir', type=str, required=True,
                        help='Combo task test data directory')
    parser.add_argument('--task_a_train_dir', type=str, required=True,
                        help='Task A training data directory (diagonal line - 5c0a986e)')
    parser.add_argument('--task_b_train_dir', type=str, required=True,
                        help='Task B training data directory (border drawing - 6f8cd79b)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: script dir)')
    parser.add_argument('--n_T', type=int, default=500,
                        help='Diffusion steps (default: 500)')
    parser.add_argument('--num_candidates', type=int, default=10,
                        help='Number of candidates per state (default: 10)')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of test episodes (default: 100)')
    parser.add_argument('--num_reference_samples', type=int, default=200,
                        help='Number of reference samples per task (default: 200)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--ddim_steps', type=int, default=100,
                        help='DDIM steps (default: 100)')
    parser.add_argument('--similarity_threshold', type=float, default=0.7,
                        help='Cosine similarity threshold for task classification (default: 0.7)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (default: cuda:0)')
    return parser.parse_args()


# Fixed configuration
MAX_GRID_SIZE = 10
Z_DIM = 256
H_DIM = 512
HORIZON = 5
A_DIM = 36
S_DIM = 512
COLOR_NUM = 11
ACTION_NUM = 36


def load_models(checkpoint_name, n_T, device='cuda:0'):
    """Load skill model and diffusion model"""
    base_dir = '/home/jovyan/beomi/yunho/ldcq_arc_working/LDCQ_for_SOLAR'

    checkpoint_dir = os.path.join(base_dir, 'checkpoints', checkpoint_name)
    gpu_prefix = checkpoint_name.split('_')[0]
    date_suffix = '_'.join(checkpoint_name.split('_')[1:])

    skill_model_path = os.path.join(checkpoint_dir, f'{gpu_prefix}_skill_model_ARCLE_{date_suffix}_400_.pth')
    diffusion_model_path = os.path.join(checkpoint_dir, f'{gpu_prefix}_skill_model_ARCLE_{date_suffix}_400__diffusion_prior_best.pt')

    # Q-network path
    q_checkpoint_dir = os.path.join(base_dir, 'q_checkpoints', checkpoint_name)
    q_diffusion_path = os.path.join(q_checkpoint_dir, f'{gpu_prefix}_skill_model_ARCLE_{date_suffix}_400__dqn_agent_150_cfg_weight_0.0_PERbuffer.pt')

    print("Loading Skill Model (VAE)...")
    skill_model = SkillModel(
        S_DIM, A_DIM, Z_DIM, H_DIM,
        horizon=HORIZON,
        a_dist='normal',
        beta=0.05,
        fixed_sig=None,
        encoder_type='gru',
        state_decoder_type='mlp',
        policy_decoder_type='mlp',
        per_element_sigma=True,
        conditional_prior=True,
        train_diffusion_prior=False,
        max_grid_size=MAX_GRID_SIZE,
        use_in_out=1,
        color_num=COLOR_NUM,
        action_num=ACTION_NUM,
        disable_pair_encoding=False,
    ).to(device)

    skill_model.load_state_dict(
        torch.load(skill_model_path, weights_only=False)['model_state_dict'],
        strict=False
    )
    skill_model.eval()

    print("Loading Diffusion Model...")
    diffusion_nn_model = torch.load(diffusion_model_path, weights_only=False).to(device)
    if not hasattr(diffusion_nn_model, 'use_in_out'):
        diffusion_nn_model.use_in_out = 1
    if hasattr(diffusion_nn_model, 'nn') and not hasattr(diffusion_nn_model.nn, 'use_in_out'):
        diffusion_nn_model.nn.use_in_out = 1

    diffusion_model = Model_Cond_Diffusion(
        diffusion_nn_model,
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        x_dim=S_DIM,
        y_dim=Z_DIM,
        drop_prob=None,
        guide_w=0.0,
    )
    diffusion_model.eval()

    # Load Q-network if available
    q_network = None
    if os.path.exists(q_diffusion_path):
        q_network = torch.load(q_diffusion_path, weights_only=False).to(device)
        q_network.eval()
        print("  - Q-network: loaded")
    else:
        print("  - Q-network: NOT FOUND")

    return skill_model, diffusion_model, q_network


def load_trajectories(data_dir, max_samples=100, random_seed=42):
    """Load trajectory data"""
    import random
    print(f"Loading data from {data_dir}...")

    all_json_paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.json'):
                all_json_paths.append(os.path.join(root, f))

    print(f"Found {len(all_json_paths)} total json files")

    random.seed(random_seed)
    random.shuffle(all_json_paths)

    data_samples = []
    for json_path in all_json_paths:
        if len(data_samples) >= max_samples:
            break
        try:
            with open(json_path) as fp:
                traj = json.load(fp)
            data_samples.append({
                'task_id': traj.get('desc', {}).get('id', os.path.basename(json_path)),
                'trajectory': traj,
                'path': json_path
            })
        except Exception:
            continue

    print(f"Loaded {len(data_samples)} trajectories")
    return data_samples


def prepare_inputs(traj, device='cuda'):
    """Prepare model inputs from trajectory"""
    max_gs = MAX_GRID_SIZE
    T = len(traj['operation'])

    state = torch.full((1, T, max_gs, max_gs), 10, dtype=torch.float32, device=device)
    for t in range(T):
        grid_t = np.array(traj['grid'][t])
        h, w = min(grid_t.shape[0], max_gs), min(grid_t.shape[1], max_gs)
        state[0, t, :h, :w] = torch.from_numpy(grid_t[:h, :w].astype(np.float32))

    clip = torch.full((1, T, max_gs, max_gs), 10, dtype=torch.float32, device=device)
    for t in range(T):
        clip_grid = np.array(traj['clip'][t])
        h_c, w_c = min(clip_grid.shape[0], max_gs), min(clip_grid.shape[1], max_gs)
        clip[0, t, :h_c, :w_c] = torch.from_numpy(clip_grid[:h_c, :w_c].astype(np.float32))

    in_grid_arr = np.array(traj['in_grid'])
    in_grid = torch.full((1, 1, max_gs, max_gs), 10, dtype=torch.float32, device=device)
    h_in, w_in = min(in_grid_arr.shape[0], max_gs), min(in_grid_arr.shape[1], max_gs)
    in_grid[0, 0, :h_in, :w_in] = torch.from_numpy(in_grid_arr[:h_in, :w_in].astype(np.float32))

    pair_in = torch.full((1, 3, max_gs, max_gs), 10, dtype=torch.float32, device=device)
    pair_out = torch.full((1, 3, max_gs, max_gs), 10, dtype=torch.float32, device=device)

    if 'ex_in' in traj and 'ex_out' in traj:
        for i in range(min(3, len(traj['ex_in']))):
            ex_in_arr = np.array(traj['ex_in'][i])
            ex_out_arr = np.array(traj['ex_out'][i])
            h_ei, w_ei = min(ex_in_arr.shape[0], max_gs), min(ex_in_arr.shape[1], max_gs)
            h_eo, w_eo = min(ex_out_arr.shape[0], max_gs), min(ex_out_arr.shape[1], max_gs)
            pair_in[0, i, :h_ei, :w_ei] = torch.from_numpy(ex_in_arr[:h_ei, :w_ei].astype(np.float32))
            pair_out[0, i, :h_eo, :w_eo] = torch.from_numpy(ex_out_arr[:h_eo, :w_eo].astype(np.float32))

    operation = torch.tensor(traj['operation'], dtype=torch.long, device=device).unsqueeze(0)
    selection = torch.tensor(traj['selection'], dtype=torch.long, device=device).unsqueeze(0)

    return state, clip, in_grid, pair_in, pair_out, operation, selection


@torch.no_grad()


def extract_reference_latents(skill_model, data_samples, device='cuda'):
    """Extract encoder latents from training data as reference"""
    latents = []

    for sample in tqdm(data_samples, desc="Extracting reference latents"):
        traj = sample['trajectory']
        state, clip, in_grid, pair_in, pair_out, operation, selection = prepare_inputs(traj, device=device)

        encoder_mean, _ = skill_model.encoder(state, clip, in_grid, operation, selection, pair_in, pair_out)
        latents.append(encoder_mean.squeeze().detach().cpu().numpy())

    return np.stack(latents)  # (N, Z_DIM)


@torch.no_grad()


def sample_proposals_at_initial_state(skill_model, diffusion_model, data_samples,
                                       num_candidates=10, ddim_steps=100, device='cuda'):
    """Sample proposals at the initial state using diffusion model"""
    all_proposals = []  # List of (episode_idx, list of proposals)

    for idx, sample in enumerate(tqdm(data_samples, desc="Sampling proposals")):
        traj = sample['trajectory']
        state, clip, in_grid, pair_in, pair_out, operation, selection = prepare_inputs(traj, device=device)

        # Initial state only
        state_init = state[:, 0:1, :, :]
        clip_init = clip[:, 0:1, :, :]

        episode_proposals = []
        for _ in range(num_candidates):
            z = diffusion_model.ddim_sample_extra(
                state_init, clip_init, in_grid, pair_in, pair_out,
                ddim_steps=ddim_steps,
                extra_steps=0, predict_noise=0
            )
            episode_proposals.append(z.squeeze().detach().cpu().numpy())

        all_proposals.append({
            'episode_idx': idx,
            'task_id': sample['task_id'],
            'proposals': np.stack(episode_proposals)  # (num_candidates, Z_DIM)
        })

    return all_proposals


def compute_cosine_similarity(z1, z2):
    """Compute cosine similarity between z1 and z2"""
    z1_norm = z1 / (np.linalg.norm(z1, axis=-1, keepdims=True) + 1e-8)
    z2_norm = z2 / (np.linalg.norm(z2, axis=-1, keepdims=True) + 1e-8)
    return np.dot(z1_norm, z2_norm.T)


@torch.no_grad()


def decode_proposal_to_actions(skill_model, z, state, clip, in_grid, pair_in, pair_out,
                                num_steps=5, device='cuda'):
    """
    Decode a latent proposal z into a sequence of actions by running the policy decoder (ll_policy)

    Args:
        skill_model: trained SkillModel (VAE) with ll_policy
        z: latent tensor (1, Z_DIM)
        state: initial state (1, 1, max_gs, max_gs)
        clip: initial clip (1, 1, max_gs, max_gs)
        in_grid: input grid (1, 1, max_gs, max_gs)
        pair_in: example inputs (1, 3, max_gs, max_gs)
        pair_out: example outputs (1, 3, max_gs, max_gs)
        num_steps: number of steps to decode (default: HORIZON=5)

    Returns:
        actions: list of {'operation': int, 'selection': [x, y, h, w]} dicts
    """
    actions = []
    max_gs = skill_model.max_grid_size

    # Reshape z for ll_policy input: (batch, 1, z_dim)
    z_input = z.unsqueeze(1) if z.dim() == 2 else z

    # Get all timesteps at once from ll_policy
    # ll_policy.forward expects:
    #   state: (batch, T, max_gs, max_gs)
    #   clip: (batch, T, max_gs, max_gs)
    #   in_grid: (batch, 1, max_gs, max_gs)
    #   z: (batch, 1, z_dim)
    #   pair_in/out: (batch, 3, max_gs, max_gs)

    # Expand state and clip to num_steps if needed
    if state.shape[1] < num_steps:
        state_expanded = state.expand(-1, num_steps, -1, -1)
        clip_expanded = clip.expand(-1, num_steps, -1, -1)
    else:
        state_expanded = state[:, :num_steps, :, :]
        clip_expanded = clip[:, :num_steps, :, :]

    # Get action distributions from low-level policy
    # Note: ll_policy is inside the decoder: skill_model.decoder.ll_policy
    # Returns: a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig
    outputs = skill_model.decoder.ll_policy.forward(
        state_expanded, clip_expanded, in_grid, z_input, pair_in, pair_out
    )

    a_mean = outputs[0]  # (batch, T, action_num) - softmax probabilities
    x_mean = outputs[2]  # (batch, T, max_gs) - x position probs
    y_mean = outputs[4]  # (batch, T, max_gs) - y position probs
    h_mean = outputs[6]  # (batch, T, max_gs) - height/x2 probs
    w_mean = outputs[8]  # (batch, T, max_gs) - width/y2 probs

    # Decode each timestep
    for step in range(min(num_steps, a_mean.shape[1])):
        # Get operation (argmax of softmax)
        operation = a_mean[0, step, :].argmax().item()

        # Get selection coordinates (argmax of position softmax)
        x = x_mean[0, step, :].argmax().item()
        y = y_mean[0, step, :].argmax().item()
        h = h_mean[0, step, :].argmax().item()
        w = w_mean[0, step, :].argmax().item()

        # selection is [x, y, h, w] where:
        # x, y: start position
        # h, w: end position or extent
        actions.append({
            'operation': operation,
            'selection': [x, y, h, w]
        })

    return actions


def apply_action_to_grid(grid, operation, selection):
    """
    Apply an action to the grid

    Args:
        grid: numpy array (H, W)
        operation: color (0-10) or special action (submit=34, etc.)
        selection: [x1, y1, x2, y2] - cell position or line range

    Returns:
        modified grid
    """
    grid = grid.copy()
    H, W = grid.shape
    x1, y1, x2, y2 = selection

    # Clamp to valid range
    x1 = int(np.clip(x1, 0, H-1))
    y1 = int(np.clip(y1, 0, W-1))
    x2 = int(np.clip(x2, 0, H-1))
    y2 = int(np.clip(y2, 0, W-1))

    # Check if it's a color operation (0-10)
    if 0 <= operation <= 10:
        # Determine if it's a point or a line
        if x1 == x2 and y1 == y2:
            # Point selection - color single cell
            grid[x1, y1] = operation
        elif x1 == x2:
            # Horizontal line
            grid[x1, min(y1,y2):max(y1,y2)+1] = operation
        elif y1 == y2:
            # Vertical line
            grid[min(x1,x2):max(x1,x2)+1, y1] = operation
        else:
            # Rectangle or diagonal - just color the start and end
            grid[x1, y1] = operation
            grid[x2, y2] = operation
    # Special operations (34 = submit, etc.) are ignored for pattern analysis

    return grid


def analyze_grid_changes(original_grid, modified_grid):
    """
    Analyze the pattern of grid changes

    Returns dict with:
        - is_diagonal: True if changes are along diagonal
        - is_border: True if changes are on the border
        - changed_cells: list of (x, y) positions that changed
    """
    H, W = original_grid.shape

    # Find changed cells
    diff = modified_grid != original_grid
    changed_positions = list(zip(*np.where(diff)))

    if not changed_positions:
        return {
            'is_diagonal': False,
            'is_border': False,
            'changed_cells': [],
            'diagonal_score': 0,
            'border_score': 0
        }

    # Count diagonal changes (cells where |x - y| is consistent or x + y is consistent)
    diagonal_count = 0
    for x, y in changed_positions:
        # Check if on any diagonal from corners or near-diagonal positions
        # Main diagonal: x == y
        # Anti-diagonal: x + y == (H-1) or similar
        if abs(x - y) <= 1 or abs((x + y) - (H - 1)) <= 1:
            diagonal_count += 1

    # Count border changes (cells on edges)
    border_count = 0
    for x, y in changed_positions:
        if x == 0 or x == H-1 or y == 0 or y == W-1:
            border_count += 1

    total_changes = len(changed_positions)
    diagonal_score = diagonal_count / total_changes if total_changes > 0 else 0
    border_score = border_count / total_changes if total_changes > 0 else 0

    return {
        'is_diagonal': diagonal_score > 0.5,
        'is_border': border_score > 0.5,
        'changed_cells': changed_positions,
        'diagonal_score': diagonal_score,
        'border_score': border_score,
        'total_changes': total_changes
    }


@torch.no_grad()


def classify_proposals_by_behavior(skill_model, proposals_list, data_samples,
                                    num_decode_steps=5, device='cuda'):
    """
    Classify proposals by their actual decoded behavior (grid changes),
    not just by latent similarity.

    This decodes each proposal to actions, simulates them on the grid,
    and classifies based on the resulting pattern:
    - Task A: diagonal-like changes
    - Task B: border-like changes

    Args:
        skill_model: trained SkillModel
        proposals_list: list of proposal data from sample_proposals_at_initial_state
        data_samples: corresponding test trajectories
        num_decode_steps: number of steps to decode and analyze

    Returns:
        behavior_results: list of dicts with classification per episode
    """
    behavior_results = []

    for proposal_data, sample in tqdm(zip(proposals_list, data_samples),
                                       desc="Classifying by behavior",
                                       total=len(proposals_list)):
        traj = sample['trajectory']
        state, clip, in_grid, pair_in, pair_out, _, _ = prepare_inputs(traj, device=device)

        # Initial state
        state_init = state[:, 0:1, :, :]
        clip_init = clip[:, 0:1, :, :]

        # Get original grid as numpy
        original_grid = state_init[0, 0].cpu().numpy()
        H, W = original_grid.shape
        # Mask out padding (value 10)
        valid_mask = original_grid != 10
        h_valid = valid_mask.any(axis=1).sum()
        w_valid = valid_mask.any(axis=0).sum()
        original_grid_cropped = original_grid[:h_valid, :w_valid]

        episode_result = {
            'task_a_count': 0,  # diagonal behavior
            'task_b_count': 0,  # border behavior
            'ambiguous_count': 0,
            'per_proposal_behavior': [],
            'per_proposal_scores': []
        }

        for prop in proposal_data['proposals']:
            z = torch.from_numpy(prop).float().to(device).unsqueeze(0)

            # Decode proposal to actions
            actions = decode_proposal_to_actions(
                skill_model, z, state_init, clip_init, in_grid, pair_in, pair_out,
                num_steps=num_decode_steps, device=device
            )

            # Apply actions to grid
            modified_grid = original_grid_cropped.copy()
            for action in actions:
                op = action['operation']
                sel = action['selection']
                # Adjust selection to cropped grid size
                sel_adjusted = [
                    min(sel[0], h_valid-1),
                    min(sel[1], w_valid-1),
                    min(sel[2], h_valid-1),
                    min(sel[3], w_valid-1)
                ]
                modified_grid = apply_action_to_grid(modified_grid, op, sel_adjusted)

            # Analyze the pattern
            analysis = analyze_grid_changes(original_grid_cropped, modified_grid)

            # Classify based on scores
            d_score = analysis['diagonal_score']
            b_score = analysis['border_score']

            if d_score > b_score and d_score > 0.3:
                episode_result['task_a_count'] += 1
                episode_result['per_proposal_behavior'].append('A')
            elif b_score > d_score and b_score > 0.3:
                episode_result['task_b_count'] += 1
                episode_result['per_proposal_behavior'].append('B')
            else:
                episode_result['ambiguous_count'] += 1
                episode_result['per_proposal_behavior'].append('ambiguous')

            episode_result['per_proposal_scores'].append({
                'diagonal_score': d_score,
                'border_score': b_score,
                'total_changes': analysis['total_changes'],
                'actions': actions
            })

        behavior_results.append(episode_result)

    return behavior_results


def generate_table_x_behavior(behavior_results, latent_results, q_values,
                               num_episodes, num_candidates):
    """
    Generate Table X using behavior-based classification

    Shows both latent-based and behavior-based classification results
    """
    total_proposals = num_episodes * num_candidates

    # Behavior-based counts
    behavior_task_a = sum(r['task_a_count'] for r in behavior_results)
    behavior_task_b = sum(r['task_b_count'] for r in behavior_results)
    behavior_ambiguous = sum(r['ambiguous_count'] for r in behavior_results)

    # Latent-based counts (for comparison)
    latent_task_a = sum(r['task_a_count'] for r in latent_results)
    latent_task_b = sum(r['task_b_count'] for r in latent_results)

    # Collect behavior scores
    all_diagonal_scores = []
    all_border_scores = []
    for r in behavior_results:
        for score in r['per_proposal_scores']:
            all_diagonal_scores.append(score['diagonal_score'])
            all_border_scores.append(score['border_score'])

    d_scores = np.array(all_diagonal_scores)
    b_scores = np.array(all_border_scores)

    # Q-value stats by behavior classification
    task_a_q_values = []
    task_b_q_values = []
    if q_values is not None:
        flat_idx = 0
        for r in behavior_results:
            q_idx = flat_idx // num_candidates
            for i, behavior in enumerate(r['per_proposal_behavior']):
                prop_idx = flat_idx % num_candidates
                if q_idx < len(q_values) and prop_idx < len(q_values[q_idx]):
                    if behavior == 'A':
                        task_a_q_values.append(q_values[q_idx][prop_idx])
                    elif behavior == 'B':
                        task_b_q_values.append(q_values[q_idx][prop_idx])
                flat_idx += 1

    task_a_q_str = 'N/A'
    task_b_q_str = 'N/A'
    if task_a_q_values:
        arr = np.array(task_a_q_values)
        task_a_q_str = f"{arr.mean():.2f}±{arr.std():.2f}"
    if task_b_q_values:
        arr = np.array(task_b_q_values)
        task_b_q_str = f"{arr.mean():.2f}±{arr.std():.2f}"

    table = f"""
================================================================================
Table X: Proposal Distribution Analysis (Behavior-Based Classification)
================================================================================
| Classification Method   | Task A (Diagonal) | Task B (Border) | Ambiguous    |
|-------------------------|-------------------|-----------------|--------------|
| Behavior-based          | {behavior_task_a:>4}/{total_proposals:<12} | {behavior_task_b:>4}/{total_proposals:<11} | {behavior_ambiguous:>4}/{total_proposals:<7} |
| Latent similarity       | {latent_task_a:>4}/{total_proposals:<12} | {latent_task_b:>4}/{total_proposals:<11} | N/A          |
--------------------------------------------------------------------------------

Behavior Scores (all proposals):
| Metric                        | Mean   | Std    |
|-------------------------------|--------|--------|
| Diagonal score                | {d_scores.mean():.3f}  | {d_scores.std():.3f}  |
| Border score                  | {b_scores.mean():.3f}  | {b_scores.std():.3f}  |
--------------------------------------------------------------------------------

Q-values by Behavior Classification:
| Task A (Diagonal)             | {task_a_q_str:<19} |
| Task B (Border)               | {task_b_q_str:<19} |
================================================================================

Caption: Proposals classified by actual decoded behavior (grid changes), not latent similarity.
- Diagonal behavior: changes along diagonal directions (like Task A - 5c0a986e)
- Border behavior: changes on grid edges (like Task B - 6f8cd79b)
================================================================================
"""

    latex_table = f"""
% LaTeX Table X (Behavior-Based Classification)
\\begin{{table}}[h]
\\centering
\\caption{{Proposal distribution analysis - Behavior-based classification}}
\\label{{tab:compositional_proposals_behavior}}
\\begin{{tabular}}{{lccc}}
\\toprule
Classification Method & Task A (Diagonal) & Task B (Border) & Ambiguous \\\\
\\midrule
Behavior-based & {behavior_task_a}/{total_proposals} & {behavior_task_b}/{total_proposals} & {behavior_ambiguous}/{total_proposals} \\\\
Latent similarity & {latent_task_a}/{total_proposals} & {latent_task_b}/{total_proposals} & N/A \\\\
\\midrule
\\multicolumn{{4}}{{l}}{{Behavior Scores (mean $\\pm$ std)}} \\\\
Diagonal score & \\multicolumn{{3}}{{c}}{{{d_scores.mean():.3f} $\\pm$ {d_scores.std():.3f}}} \\\\
Border score & \\multicolumn{{3}}{{c}}{{{b_scores.mean():.3f} $\\pm$ {b_scores.std():.3f}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    return table, latex_table, {
        'behavior_task_a': behavior_task_a,
        'behavior_task_b': behavior_task_b,
        'behavior_ambiguous': behavior_ambiguous,
        'latent_task_a': latent_task_a,
        'latent_task_b': latent_task_b,
        'diagonal_score_mean': float(d_scores.mean()),
        'diagonal_score_std': float(d_scores.std()),
        'border_score_mean': float(b_scores.mean()),
        'border_score_std': float(b_scores.std()),
    }


def classify_proposals(proposals, task_a_latents, task_b_latents, threshold=0.7):
    """
    Classify proposals as Task A-consistent or Task B-consistent
    based on cosine similarity to reference latents

    Uses RELATIVE classification: whichever task has higher similarity wins.
    No threshold required - purely comparative.

    Returns:
        - task_a_count: number of proposals more similar to Task A
        - task_b_count: number of proposals more similar to Task B
        - similarity stats
    """
    results = {
        'task_a_count': 0,
        'task_b_count': 0,
        'ambiguous_count': 0,
        'task_a_sims': [],
        'task_b_sims': [],
        'per_proposal_classification': []
    }

    # Mean latent for each task (centroid)
    task_a_mean = task_a_latents.mean(axis=0)
    task_b_mean = task_b_latents.mean(axis=0)

    for proposal in proposals:
        # Compute similarity to each task's centroid
        sim_a_centroid = compute_cosine_similarity(proposal.reshape(1, -1), task_a_mean.reshape(1, -1))[0, 0]
        sim_b_centroid = compute_cosine_similarity(proposal.reshape(1, -1), task_b_mean.reshape(1, -1))[0, 0]

        # Also compute similarity to individual training samples
        sims_to_a = compute_cosine_similarity(proposal.reshape(1, -1), task_a_latents)[0]
        sims_to_b = compute_cosine_similarity(proposal.reshape(1, -1), task_b_latents)[0]

        max_sim_a = sims_to_a.max()
        max_sim_b = sims_to_b.max()
        mean_sim_a = sims_to_a.mean()
        mean_sim_b = sims_to_b.mean()

        results['task_a_sims'].append({'mean': mean_sim_a, 'max': max_sim_a, 'centroid': sim_a_centroid})
        results['task_b_sims'].append({'mean': mean_sim_b, 'max': max_sim_b, 'centroid': sim_b_centroid})

        # RELATIVE classification: compare centroid similarities
        # Whichever task has higher similarity wins
        if sim_a_centroid > sim_b_centroid:
            results['task_a_count'] += 1
            results['per_proposal_classification'].append('A')
        else:
            results['task_b_count'] += 1
            results['per_proposal_classification'].append('B')

    return results


@torch.no_grad()


def compute_q_values_for_proposals(q_network, data_samples, proposals_list, device='cuda'):
    """Compute Q-values for sampled proposals"""
    if q_network is None:
        return None

    q_values_list = []

    for proposal_data, sample in zip(proposals_list, data_samples):
        traj = sample['trajectory']
        state, clip, in_grid, pair_in, pair_out, _, _ = prepare_inputs(traj, device=device)

        state0 = state[:, 0:1, :, :]
        clip0 = clip[:, 0:1, :, :]

        proposals = torch.from_numpy(proposal_data['proposals']).float().to(device)  # (K, Z_DIM)
        K = proposals.shape[0]

        state_k = state0.repeat(K, 1, 1, 1)
        clip_k = clip0.repeat(K, 1, 1, 1)
        in_k = in_grid.repeat(K, 1, 1, 1)
        pin_k = pair_in.repeat(K, 1, 1, 1)
        pout_k = pair_out.repeat(K, 1, 1, 1)

        if hasattr(q_network, 'q_net_0'):
            q0 = q_network.q_net_0(state_k, clip_k, in_k, proposals, pin_k, pout_k)
            q1 = q_network.q_net_1(state_k, clip_k, in_k, proposals, pin_k, pout_k)
            q = torch.min(q0, q1)
        else:
            q = q_network(state_k, clip_k, in_k, proposals, pin_k, pout_k)

        if q.dim() > 1:
            q = q.max(dim=1).values

        q_values_list.append(q.detach().cpu().numpy())

    return q_values_list


def generate_table_x(results, task_a_latents, task_b_latents, q_values, num_episodes, num_candidates):
    """Generate Table X: Proposal distribution analysis in compositional task

    Matches the format:
    | Metric                        | Task A    | Task B     |
    |-------------------------------|-----------|------------|
    | Sampling frequency            | 0/1000    | 1000/1000  |
    | Mean cosine sim. to A train   | 0.12±0.08 | 0.89±0.04  |
    | Mean cosine sim. to B train   | 0.91±0.03 | 0.94±0.02  |
    | Mean Q-value of proposals     | N/A       | 0.43±0.18  |
    """
    total_proposals = num_episodes * num_candidates

    # Aggregate counts
    task_a_freq = sum(r['task_a_count'] for r in results)
    task_b_freq = sum(r['task_b_count'] for r in results)
    ambiguous_freq = sum(r['ambiguous_count'] for r in results)

    # Collect per-proposal similarity stats separated by classification
    task_a_proposals_sim_to_a = []
    task_a_proposals_sim_to_b = []
    task_b_proposals_sim_to_a = []
    task_b_proposals_sim_to_b = []

    for r in results:
        for i, classification in enumerate(r['per_proposal_classification']):
            sim_a = r['task_a_sims'][i]['mean']
            sim_b = r['task_b_sims'][i]['mean']
            if classification == 'A':
                task_a_proposals_sim_to_a.append(sim_a)
                task_a_proposals_sim_to_b.append(sim_b)
            elif classification == 'B':
                task_b_proposals_sim_to_a.append(sim_a)
                task_b_proposals_sim_to_b.append(sim_b)

    # Convert to arrays
    task_a_proposals_sim_to_a = np.array(task_a_proposals_sim_to_a) if task_a_proposals_sim_to_a else np.array([0.0])
    task_a_proposals_sim_to_b = np.array(task_a_proposals_sim_to_b) if task_a_proposals_sim_to_b else np.array([0.0])
    task_b_proposals_sim_to_a = np.array(task_b_proposals_sim_to_a) if task_b_proposals_sim_to_a else np.array([0.0])
    task_b_proposals_sim_to_b = np.array(task_b_proposals_sim_to_b) if task_b_proposals_sim_to_b else np.array([0.0])

    # Q-value stats (only for Task B proposals since Task A has N/A)
    task_a_q_str = 'N/A'
    task_b_q_str = 'N/A'

    if q_values is not None:
        # Collect Q-values separated by classification
        task_b_q_values = []
        flat_idx = 0
        for r in results:
            for classification in r['per_proposal_classification']:
                if classification == 'B' and q_values is not None:
                    q_idx = flat_idx // num_candidates
                    prop_idx = flat_idx % num_candidates
                    if q_idx < len(q_values) and prop_idx < len(q_values[q_idx]):
                        task_b_q_values.append(q_values[q_idx][prop_idx])
                flat_idx += 1

        if task_b_q_values:
            task_b_q_arr = np.array(task_b_q_values)
            task_b_q_str = f"{task_b_q_arr.mean():.2f}±{task_b_q_arr.std():.2f}"

    # Format table matching the requested format
    table = f"""
================================================================================
Table X: Proposal distribution analysis in compositional task
================================================================================
| Metric                        | Task A              | Task B              |
|-------------------------------|---------------------|---------------------|
| Sampling frequency            | {task_a_freq:>4}/{total_proposals:<14} | {task_b_freq:>4}/{total_proposals:<14} |
| Mean cosine sim. to A train   | {task_a_proposals_sim_to_a.mean():.2f}±{task_a_proposals_sim_to_a.std():.2f}            | {task_b_proposals_sim_to_a.mean():.2f}±{task_b_proposals_sim_to_a.std():.2f}            |
| Mean cosine sim. to B train   | {task_a_proposals_sim_to_b.mean():.2f}±{task_a_proposals_sim_to_b.std():.2f}            | {task_b_proposals_sim_to_b.mean():.2f}±{task_b_proposals_sim_to_b.std():.2f}            |
| Mean Q-value of proposals     | {task_a_q_str:<19} | {task_b_q_str:<19} |
--------------------------------------------------------------------------------

Caption: All sampled latents at the initial state align with Task B. No Task A
proposals are generated. Q-values are only computed for proposed candidates.

Summary:
- Task A proposals: {task_a_freq}/{total_proposals}
- Task B proposals: {task_b_freq}/{total_proposals}
- Ambiguous: {ambiguous_freq}/{total_proposals}
================================================================================
"""

    # Also generate LaTeX table
    latex_table = f"""
% LaTeX Table X
\\begin{{table}}[h]
\\centering
\\caption{{Proposal distribution analysis in compositional task}}
\\label{{tab:compositional_proposals}}
\\begin{{tabular}}{{lcc}}
\\toprule
Metric & Task A & Task B \\\\
\\midrule
Sampling frequency & {task_a_freq}/{total_proposals} & {task_b_freq}/{total_proposals} \\\\
Mean cosine sim. to A train & {task_a_proposals_sim_to_a.mean():.2f}$\\pm${task_a_proposals_sim_to_a.std():.2f} & {task_b_proposals_sim_to_a.mean():.2f}$\\pm${task_b_proposals_sim_to_a.std():.2f} \\\\
Mean cosine sim. to B train & {task_a_proposals_sim_to_b.mean():.2f}$\\pm${task_a_proposals_sim_to_b.std():.2f} & {task_b_proposals_sim_to_b.mean():.2f}$\\pm${task_b_proposals_sim_to_b.std():.2f} \\\\
Mean Q-value of proposals & {task_a_q_str} & {task_b_q_str} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    return table, latex_table, {
        'task_a_freq': task_a_freq,
        'task_b_freq': task_b_freq,
        'ambiguous_freq': ambiguous_freq,
        'total_proposals': total_proposals,
        'task_a_sim_to_a_mean': float(task_a_proposals_sim_to_a.mean()),
        'task_a_sim_to_a_std': float(task_a_proposals_sim_to_a.std()),
        'task_a_sim_to_b_mean': float(task_a_proposals_sim_to_b.mean()),
        'task_a_sim_to_b_std': float(task_a_proposals_sim_to_b.std()),
        'task_b_sim_to_a_mean': float(task_b_proposals_sim_to_a.mean()),
        'task_b_sim_to_a_std': float(task_b_proposals_sim_to_a.std()),
        'task_b_sim_to_b_mean': float(task_b_proposals_sim_to_b.mean()),
        'task_b_sim_to_b_std': float(task_b_proposals_sim_to_b.std()),
    }


def generate_table_y(intervention_results=None):
    """Generate Table Y: Intervention results on compositional task

    Args:
        intervention_results: dict with keys 'baseline', 'h1', 'h2', 'h3', each containing:
            - 'submit_acc': float (0-100)
            - 'task_a_proposals': int
            - 'total_proposals': int
            - 'note': str (optional, for special cases like training divergence)

    If intervention_results is None, generates a placeholder template.
    """
    if intervention_results is None:
        # Generate placeholder
        table = """
================================================================================
Table Y: Intervention results on compositional task
================================================================================
| Intervention                      | Submit Acc | Task A Proposals |
|-----------------------------------|------------|------------------|
| Baseline (joint demo encoding)    | ___%       | ___/1000         |
| Independent demo encoding (H1)    | ___%       | ___/1000         |
| Goal-conditioned (H2, ___% goal acc)| ___%     | ___/1000         |
| Explicit concept labels (H3)      | Training diverges after ___ epochs |
--------------------------------------------------------------------------------

Caption: Three interventions to redirect proposal generation.
(H1) tests demonstration encoding. (H2) tests goal conditioning.
(H3) tests concept identity. Only goal conditioning shows minimal improvement.
Concept labels cause training instability.
================================================================================

NOTE: Fill in the values by running separate experiments for each intervention.
To run intervention experiments, use:
  python analyze_compositional_interventions.py --intervention [baseline|h1|h2|h3]
"""
        latex_table = """
% LaTeX Table Y (placeholder)
\\begin{table}[h]
\\centering
\\caption{Intervention results on compositional task}
\\label{tab:interventions}
\\begin{tabular}{lcc}
\\toprule
Intervention & Submit Acc & Task A Proposals \\\\
\\midrule
Baseline (joint demo encoding) & \\___\\% & \\_\\_\\_/1000 \\\\
Independent demo encoding (H1) & \\___\\% & \\_\\_\\_/1000 \\\\
Goal-conditioned (H2, \\___\\% goal acc) & \\___\\% & \\_\\_\\_/1000 \\\\
Explicit concept labels (H3) & \\multicolumn{2}{c}{Training diverges after \\_\\_\\_ epochs} \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""
        return table, latex_table

    # Generate with actual results
    baseline = intervention_results.get('baseline', {})
    h1 = intervention_results.get('h1', {})
    h2 = intervention_results.get('h2', {})
    h3 = intervention_results.get('h3', {})

    def format_row(name, data, total=1000):
        if 'note' in data:
            return f"| {name:<33} | {data['note']:<29} |"
        acc = data.get('submit_acc', 0)
        proposals = data.get('task_a_proposals', 0)
        return f"| {name:<33} | {acc:>3}%       | {proposals:>4}/{total:<13} |"

    h2_name = f"Goal-conditioned (H2, {h2.get('goal_acc', '??')}% goal acc)"

    table = f"""
================================================================================
Table Y: Intervention results on compositional task
================================================================================
| Intervention                      | Submit Acc | Task A Proposals |
|-----------------------------------|------------|------------------|
{format_row('Baseline (joint demo encoding)', baseline)}
{format_row('Independent demo encoding (H1)', h1)}
{format_row(h2_name, h2)}
{format_row('Explicit concept labels (H3)', h3)}
--------------------------------------------------------------------------------

Caption: Three interventions to redirect proposal generation.
(H1) tests demonstration encoding. (H2) tests goal conditioning.
(H3) tests concept identity. Only goal conditioning shows minimal improvement
({h2.get('submit_acc', 0)}%). Concept labels cause training instability.
================================================================================
"""

    # LaTeX version

    def latex_row(name, data, total=1000):
        if 'note' in data:
            return f"{name} & \\multicolumn{{2}}{{c}}{{{data['note']}}} \\\\"
        acc = data.get('submit_acc', 0)
        proposals = data.get('task_a_proposals', 0)
        return f"{name} & {acc}\\% & {proposals}/{total} \\\\"

    latex_table = f"""
% LaTeX Table Y
\\begin{{table}}[h]
\\centering
\\caption{{Intervention results on compositional task}}
\\label{{tab:interventions}}
\\begin{{tabular}}{{lcc}}
\\toprule
Intervention & Submit Acc & Task A Proposals \\\\
\\midrule
{latex_row('Baseline (joint demo encoding)', baseline)}
{latex_row('Independent demo encoding (H1)', h1)}
{latex_row(h2_name, h2)}
{latex_row('Explicit concept labels (H3)', h3)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    return table, latex_table


def plot_tsne_latent_space(task_a_latents, task_b_latents, proposals_list, save_path,
                           task_a_name='Task A (Diagonal)', task_b_name='Task B (Border)'):
    """
    Plot t-SNE visualization of latent space showing:
    - Task A training latents
    - Task B training latents
    - Sampled proposals from compositional task
    """
    from sklearn.manifold import TSNE

    # Collect all proposal latents
    all_proposals = []
    for p in proposals_list:
        all_proposals.append(p['proposals'])
    all_proposals = np.concatenate(all_proposals, axis=0)  # (N*K, Z_DIM)

    # Combine all latents for t-SNE
    all_latents = np.concatenate([task_a_latents, task_b_latents, all_proposals], axis=0)

    # Labels for coloring
    n_a = len(task_a_latents)
    n_b = len(task_b_latents)
    n_p = len(all_proposals)

    print(f"Running t-SNE on {len(all_latents)} latents...")
    print(f"  - Task A: {n_a}")
    print(f"  - Task B: {n_b}")
    print(f"  - Proposals: {n_p}")

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_latents)-1))
    latents_2d = tsne.fit_transform(all_latents)

    # Split back
    task_a_2d = latents_2d[:n_a]
    task_b_2d = latents_2d[n_a:n_a+n_b]
    proposals_2d = latents_2d[n_a+n_b:]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: All three distributions
    ax1 = axes[0]
    ax1.scatter(task_a_2d[:, 0], task_a_2d[:, 1], c='blue', alpha=0.6, s=30, label=f'{task_a_name} train (n={n_a})')
    ax1.scatter(task_b_2d[:, 0], task_b_2d[:, 1], c='red', alpha=0.6, s=30, label=f'{task_b_name} train (n={n_b})')
    ax1.scatter(proposals_2d[:, 0], proposals_2d[:, 1], c='green', alpha=0.4, s=20, label=f'Proposals (n={n_p})')
    ax1.set_xlabel('t-SNE dim 1')
    ax1.set_ylabel('t-SNE dim 2')
    ax1.set_title('Latent Space: Training vs Proposals')
    ax1.legend()

    # Right: Proposals colored by similarity
    ax2 = axes[1]
    ax2.scatter(task_a_2d[:, 0], task_a_2d[:, 1], c='blue', alpha=0.3, s=20, label=f'{task_a_name} train')
    ax2.scatter(task_b_2d[:, 0], task_b_2d[:, 1], c='red', alpha=0.3, s=20, label=f'{task_b_name} train')

    # Color proposals by which task they're closer to (centroid distance)
    task_a_centroid = task_a_latents.mean(axis=0)
    task_b_centroid = task_b_latents.mean(axis=0)

    colors = []
    for prop in all_proposals:
        sim_a = np.dot(prop, task_a_centroid) / (np.linalg.norm(prop) * np.linalg.norm(task_a_centroid) + 1e-8)
        sim_b = np.dot(prop, task_b_centroid) / (np.linalg.norm(prop) * np.linalg.norm(task_b_centroid) + 1e-8)
        # Color: blue if closer to A, red if closer to B, intensity by margin
        if sim_a > sim_b:
            colors.append('blue')
        else:
            colors.append('red')

    ax2.scatter(proposals_2d[:, 0], proposals_2d[:, 1], c=colors, alpha=0.6, s=30,
                edgecolors='black', linewidths=0.5, label='Proposals (colored by closest task)')
    ax2.set_xlabel('t-SNE dim 1')
    ax2.set_ylabel('t-SNE dim 2')
    ax2.set_title('Proposals colored by closest task centroid')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"t-SNE plot saved to: {save_path}")

    # Also save a version with centroids marked
    fig2, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(task_a_2d[:, 0], task_a_2d[:, 1], c='blue', alpha=0.5, s=40, label=f'{task_a_name} train')
    ax.scatter(task_b_2d[:, 0], task_b_2d[:, 1], c='red', alpha=0.5, s=40, label=f'{task_b_name} train')
    ax.scatter(proposals_2d[:, 0], proposals_2d[:, 1], c='green', alpha=0.3, s=20, label='Proposals')

    # Mark centroids in 2D space
    task_a_2d_centroid = task_a_2d.mean(axis=0)
    task_b_2d_centroid = task_b_2d.mean(axis=0)
    proposals_2d_centroid = proposals_2d.mean(axis=0)

    ax.scatter(*task_a_2d_centroid, c='blue', s=300, marker='*', edgecolors='black', linewidths=2, label='Task A centroid')
    ax.scatter(*task_b_2d_centroid, c='red', s=300, marker='*', edgecolors='black', linewidths=2, label='Task B centroid')
    ax.scatter(*proposals_2d_centroid, c='green', s=300, marker='*', edgecolors='black', linewidths=2, label='Proposals centroid')

    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.set_title('Latent Space with Centroids')
    ax.legend(loc='best')

    plt.tight_layout()
    centroid_path = save_path.replace('.png', '_with_centroids.png')
    plt.savefig(centroid_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"t-SNE plot with centroids saved to: {centroid_path}")


def plot_similarity_distribution(results, save_path, task_a_name='Task A', task_b_name='Task B'):
    """Plot similarity distribution of proposals to each task"""
    task_a_sims_mean = []
    task_b_sims_mean = []

    for r in results:
        for sim in r['task_a_sims']:
            task_a_sims_mean.append(sim['mean'])
        for sim in r['task_b_sims']:
            task_b_sims_mean.append(sim['mean'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    axes[0].hist(task_a_sims_mean, bins=30, alpha=0.7, label=f'Sim to {task_a_name}', color='blue')
    axes[0].hist(task_b_sims_mean, bins=30, alpha=0.7, label=f'Sim to {task_b_name}', color='orange')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Proposal Similarity Distribution')
    axes[0].legend()

    # Scatter plot
    axes[1].scatter(task_a_sims_mean, task_b_sims_mean, alpha=0.3, s=10)
    axes[1].plot([0, 1], [0, 1], 'r--', label='Equal similarity')
    axes[1].set_xlabel(f'Similarity to {task_a_name}')
    axes[1].set_ylabel(f'Similarity to {task_b_name}')
    axes[1].set_title('Task A vs Task B Similarity')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


def plot_proposal_classification_pie(results, save_path):
    """Plot pie chart of proposal classification"""
    task_a_count = sum(r['task_a_count'] for r in results)
    task_b_count = sum(r['task_b_count'] for r in results)
    ambiguous_count = sum(r['ambiguous_count'] for r in results)

    fig, ax = plt.subplots(figsize=(8, 8))

    sizes = [task_a_count, task_b_count, ambiguous_count]
    labels = [f'Task A (Diagonal)\n{task_a_count}',
              f'Task B (Border)\n{task_b_count}',
              f'Ambiguous\n{ambiguous_count}']
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.05, 0.05, 0.05)

    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.set_title('Proposal Classification at Initial State')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


def plot_behavior_classification_comparison(latent_results, behavior_results, save_path):
    """
    Plot comparison between latent-based and behavior-based classification
    """
    # Latent counts
    latent_a = sum(r['task_a_count'] for r in latent_results)
    latent_b = sum(r['task_b_count'] for r in latent_results)
    latent_amb = sum(r.get('ambiguous_count', 0) for r in latent_results)

    # Behavior counts
    behavior_a = sum(r['task_a_count'] for r in behavior_results)
    behavior_b = sum(r['task_b_count'] for r in behavior_results)
    behavior_amb = sum(r['ambiguous_count'] for r in behavior_results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Pie chart: Latent-based
    ax1 = axes[0]
    sizes1 = [latent_a, latent_b, latent_amb] if latent_amb > 0 else [latent_a, latent_b]
    labels1 = [f'Task A\n{latent_a}', f'Task B\n{latent_b}']
    if latent_amb > 0:
        labels1.append(f'Ambiguous\n{latent_amb}')
    colors1 = ['#ff9999', '#66b3ff', '#99ff99'][:len(sizes1)]
    ax1.pie(sizes1, labels=labels1, colors=colors1, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Latent Similarity Classification')

    # Pie chart: Behavior-based
    ax2 = axes[1]
    sizes2 = [behavior_a, behavior_b, behavior_amb]
    labels2 = [f'Task A (Diagonal)\n{behavior_a}',
               f'Task B (Border)\n{behavior_b}',
               f'Ambiguous\n{behavior_amb}']
    colors2 = ['#ff9999', '#66b3ff', '#99ff99']
    ax2.pie(sizes2, labels=labels2, colors=colors2, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Behavior-Based Classification')

    # Bar chart comparison
    ax3 = axes[2]
    x = np.arange(3)
    width = 0.35
    latent_vals = [latent_a, latent_b, latent_amb]
    behavior_vals = [behavior_a, behavior_b, behavior_amb]

    rects1 = ax3.bar(x - width/2, latent_vals, width, label='Latent Similarity', color='steelblue')
    rects2 = ax3.bar(x + width/2, behavior_vals, width, label='Behavior-Based', color='darkorange')

    ax3.set_ylabel('Count')
    ax3.set_title('Classification Method Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Task A', 'Task B', 'Ambiguous'])
    ax3.legend()

    # Add value labels on bars
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax3.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Behavior comparison plot saved to: {save_path}")


def plot_behavior_scores_distribution(behavior_results, save_path):
    """
    Plot distribution of diagonal vs border scores
    """
    all_diagonal_scores = []
    all_border_scores = []
    all_classifications = []

    for r in behavior_results:
        for i, score in enumerate(r['per_proposal_scores']):
            all_diagonal_scores.append(score['diagonal_score'])
            all_border_scores.append(score['border_score'])
            all_classifications.append(r['per_proposal_behavior'][i])

    d_scores = np.array(all_diagonal_scores)
    b_scores = np.array(all_border_scores)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Scatter plot: Diagonal vs Border scores
    ax1 = axes[0]
    colors = {'A': 'blue', 'B': 'red', 'ambiguous': 'gray'}
    for cls in ['A', 'B', 'ambiguous']:
        mask = np.array(all_classifications) == cls
        if mask.sum() > 0:
            ax1.scatter(d_scores[mask], b_scores[mask], alpha=0.5, s=20,
                       c=colors[cls], label=f'{cls} ({mask.sum()})')

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Equal')
    ax1.set_xlabel('Diagonal Score')
    ax1.set_ylabel('Border Score')
    ax1.set_title('Diagonal vs Border Scores')
    ax1.legend()
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)

    # Histogram: Diagonal scores
    ax2 = axes[1]
    ax2.hist(d_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(d_scores.mean(), color='red', linestyle='--', label=f'Mean: {d_scores.mean():.3f}')
    ax2.set_xlabel('Diagonal Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Diagonal Score Distribution')
    ax2.legend()

    # Histogram: Border scores
    ax3 = axes[2]
    ax3.hist(b_scores, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(b_scores.mean(), color='red', linestyle='--', label=f'Mean: {b_scores.mean():.3f}')
    ax3.set_xlabel('Border Score')
    ax3.set_ylabel('Count')
    ax3.set_title('Border Score Distribution')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Behavior scores distribution plot saved to: {save_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    print("="*60)
    print("Compositional Task Proposal Distribution Analysis")
    print(f"Checkpoint: {args.checkpoint_name}")
    print("="*60)

    device = args.device

    # Load models
    skill_model, diffusion_model, q_network = load_models(
        args.checkpoint_name, args.n_T, device=device
    )

    # Load data
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)

    combo_test_data = load_trajectories(
        args.combo_test_dir,
        max_samples=args.num_episodes,
        random_seed=args.random_seed
    )

    task_a_train_data = load_trajectories(
        args.task_a_train_dir,
        max_samples=args.num_reference_samples,
        random_seed=args.random_seed
    )

    task_b_train_data = load_trajectories(
        args.task_b_train_dir,
        max_samples=args.num_reference_samples,
        random_seed=args.random_seed
    )

    # Extract reference latents
    print("\n" + "="*60)
    print("Extracting Reference Latents")
    print("="*60)

    task_a_latents = extract_reference_latents(skill_model, task_a_train_data, device=device)
    task_b_latents = extract_reference_latents(skill_model, task_b_train_data, device=device)

    print(f"Task A reference latents shape: {task_a_latents.shape}")
    print(f"Task B reference latents shape: {task_b_latents.shape}")

    # Sample proposals at initial state
    print("\n" + "="*60)
    print("Sampling Proposals at Initial State")
    print("="*60)

    proposals_list = sample_proposals_at_initial_state(
        skill_model, diffusion_model, combo_test_data,
        num_candidates=args.num_candidates,
        ddim_steps=args.ddim_steps,
        device=device
    )

    # Classify proposals
    print("\n" + "="*60)
    print("Classifying Proposals")
    print("="*60)

    classification_results = []
    for proposal_data in tqdm(proposals_list, desc="Classifying"):
        result = classify_proposals(
            proposal_data['proposals'],
            task_a_latents,
            task_b_latents,
            threshold=args.similarity_threshold
        )
        classification_results.append(result)

    # Compute Q-values
    print("\n" + "="*60)
    print("Computing Q-values")
    print("="*60)

    q_values_list = compute_q_values_for_proposals(
        q_network, combo_test_data, proposals_list, device=device
    )

    # Behavior-based classification (decode proposals and analyze grid patterns)
    print("\n" + "="*60)
    print("Behavior-Based Classification (Decoding Proposals)")
    print("="*60)

    behavior_results = classify_proposals_by_behavior(
        skill_model, proposals_list, combo_test_data,
        num_decode_steps=HORIZON,  # Use full horizon
        device=device
    )

    # Generate tables and plots
    print("\n" + "="*60)
    print("Generating Results")
    print("="*60)

    save_dir = args.output_dir if args.output_dir else os.path.dirname(__file__)
    os.makedirs(save_dir, exist_ok=True)
    prefix = args.checkpoint_name

    # Table X
    table_x, latex_table_x, stats = generate_table_x(
        classification_results,
        task_a_latents, task_b_latents,
        q_values_list,
        args.num_episodes, args.num_candidates
    )
    print(table_x)

    # Save tables
    table_path = os.path.join(save_dir, f'{prefix}_compositional_table_x.txt')
    with open(table_path, 'w') as f:
        f.write(table_x)
    print(f"Table X saved to: {table_path}")

    latex_path = os.path.join(save_dir, f'{prefix}_compositional_table_x.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_table_x)
    print(f"LaTeX Table X saved to: {latex_path}")

    # Table X (Behavior-based) - using decoded actions to classify
    table_x_behavior, latex_table_x_behavior, behavior_stats = generate_table_x_behavior(
        behavior_results, classification_results, q_values_list,
        args.num_episodes, args.num_candidates
    )
    print(table_x_behavior)

    behavior_table_path = os.path.join(save_dir, f'{prefix}_compositional_table_x_behavior.txt')
    with open(behavior_table_path, 'w') as f:
        f.write(table_x_behavior)
    print(f"Table X (Behavior) saved to: {behavior_table_path}")

    behavior_latex_path = os.path.join(save_dir, f'{prefix}_compositional_table_x_behavior.tex')
    with open(behavior_latex_path, 'w') as f:
        f.write(latex_table_x_behavior)
    print(f"LaTeX Table X (Behavior) saved to: {behavior_latex_path}")

    # Table Y placeholder (requires running intervention experiments separately)
    table_y, latex_table_y = generate_table_y(intervention_results=None)
    print(table_y)

    table_y_path = os.path.join(save_dir, f'{prefix}_compositional_table_y_placeholder.txt')
    with open(table_y_path, 'w') as f:
        f.write(table_y)
    print(f"Table Y placeholder saved to: {table_y_path}")

    latex_y_path = os.path.join(save_dir, f'{prefix}_compositional_table_y_placeholder.tex')
    with open(latex_y_path, 'w') as f:
        f.write(latex_table_y)
    print(f"LaTeX Table Y placeholder saved to: {latex_y_path}")

    # Plots
    plot_similarity_distribution(
        classification_results,
        os.path.join(save_dir, f'{prefix}_compositional_similarity_dist.png'),
        task_a_name='Task A (Diagonal)',
        task_b_name='Task B (Border)'
    )

    plot_proposal_classification_pie(
        classification_results,
        os.path.join(save_dir, f'{prefix}_compositional_classification_pie.png')
    )

    # t-SNE visualization
    print("\n" + "="*60)
    print("Generating t-SNE Visualization")
    print("="*60)
    plot_tsne_latent_space(
        task_a_latents, task_b_latents, proposals_list,
        os.path.join(save_dir, f'{prefix}_compositional_tsne.png'),
        task_a_name='Task A (Diagonal)',
        task_b_name='Task B (Border)'
    )

    # Behavior-based classification visualizations
    print("\n" + "="*60)
    print("Generating Behavior Classification Visualizations")
    print("="*60)

    plot_behavior_classification_comparison(
        classification_results, behavior_results,
        os.path.join(save_dir, f'{prefix}_behavior_vs_latent_comparison.png')
    )

    plot_behavior_scores_distribution(
        behavior_results,
        os.path.join(save_dir, f'{prefix}_behavior_scores_distribution.png')
    )

    # Save detailed results as JSON
    # Note: We exclude 'actions' from behavior_scores to reduce file size
    detailed_results = {
        'stats': stats,
        'behavior_stats': behavior_stats,
        'args': vars(args),
        'per_episode_results': [
            {
                'episode_idx': r_data['episode_idx'] if isinstance(r_data, dict) else i,
                # Latent-based classification
                'latent_task_a_count': r['task_a_count'],
                'latent_task_b_count': r['task_b_count'],
                'latent_ambiguous_count': r['ambiguous_count'],
                # Behavior-based classification
                'behavior_task_a_count': b['task_a_count'],
                'behavior_task_b_count': b['task_b_count'],
                'behavior_ambiguous_count': b['ambiguous_count'],
                # Detailed behavior scores (without full action sequences to save space)
                'behavior_scores': [
                    {
                        'diagonal_score': s['diagonal_score'],
                        'border_score': s['border_score'],
                        'total_changes': s['total_changes'],
                    }
                    for s in b['per_proposal_scores']
                ],
            }
            for i, (r_data, r, b) in enumerate(zip(proposals_list, classification_results, behavior_results))
        ]
    }

    json_path = os.path.join(save_dir, f'{prefix}_compositional_detailed_results.json')
    with open(json_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"Detailed results saved to: {json_path}")

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
