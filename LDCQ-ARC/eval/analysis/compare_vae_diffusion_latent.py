"""
VAE latent vs Diffusion latent 통계량 + diversity/coverage + best-of-K Q 분석
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

from models.diffusion_models import Model_Cond_Diffusion
from models.skill_model import SkillModel

# VAEPriorDDQN import (for loading q_vae checkpoint)
sys.path.insert(0, os.path.join(parent_folder, 'training'))
try:
    from train_q_net_vae_prior import VAEPriorDDQN
except ImportError:
    VAEPriorDDQN = None


def parse_args():
    parser = argparse.ArgumentParser(description='VAE vs Diffusion latent 비교 분석')
    parser.add_argument('--checkpoint_name', type=str, required=True,
                        help='체크포인트 이름 (예: gpu6_01.26)')
    parser.add_argument('--test_data_dir', type=str, required=True,
                        help='테스트 데이터 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='출력 디렉토리 (기본: script dir)')
    parser.add_argument('--n_T', type=int, default=500,
                        help='Diffusion steps (기본: 500)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='state당 prior/diffusion 샘플 수 (기본: 5)')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='로드할 최대 trajectory 수 (기본: 100)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='랜덤 시드 (기본: 42)')
    parser.add_argument('--ddim_steps', type=int, default=100,
                        help='DDIM steps (기본: 100)')
    parser.add_argument('--num_compare_q', type=int, default=30,
                        help='best-of-K Q 분석에 사용할 trajectory 수 (기본: 30)')
    return parser.parse_args()


# 고정 설정
MAX_GRID_SIZE = 10
Z_DIM = 256
H_DIM = 512
HORIZON = 5
A_DIM = 36
S_DIM = 512
COLOR_NUM = 11
ACTION_NUM = 36


def load_models(checkpoint_name, n_T, device='cuda:0'):
    """모델들 로드"""
    base_dir = '/home/jovyan/beomi/yunho/ldcq_arc_working/LDCQ_for_SOLAR'

    checkpoint_dir = os.path.join(base_dir, 'checkpoints', checkpoint_name)
    gpu_prefix = checkpoint_name.split('_')[0]
    date_suffix = '_'.join(checkpoint_name.split('_')[1:])

    skill_model_path = os.path.join(checkpoint_dir, f'{gpu_prefix}_skill_model_ARCLE_{date_suffix}_400_.pth')
    diffusion_model_path = os.path.join(checkpoint_dir, f'{gpu_prefix}_skill_model_ARCLE_{date_suffix}_400__diffusion_prior_best.pt')

    q_checkpoint_dir = os.path.join(base_dir, 'q_checkpoints', checkpoint_name)
    q_diffusion_path = os.path.join(q_checkpoint_dir, f'{gpu_prefix}_skill_model_ARCLE_{date_suffix}_400__dqn_agent_150_cfg_weight_0.0_PERbuffer.pt')

    q_vae_checkpoint_dir = os.path.join(base_dir, 'q_checkpoints', f'{checkpoint_name}_vae_prior')
    q_vae_path = os.path.join(q_vae_checkpoint_dir, f'{gpu_prefix}_skill_model_ARCLE_{date_suffix}_400__dqn_agent_150_cfg_weight_0.0_PERbuffer.pt')

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

    print("Loading Q-networks...")
    q_diffusion = None
    q_vae = None

    if os.path.exists(q_diffusion_path):
        q_diffusion = torch.load(q_diffusion_path, weights_only=False).to(device)
        q_diffusion.eval()
        print("  - Q (diffusion trained): loaded")
    else:
        print("  - Q (diffusion trained): NOT FOUND")

    if os.path.exists(q_vae_path):
        q_vae = torch.load(q_vae_path, weights_only=False).to(device)
        q_vae.eval()
        print("  - Q (VAE trained): loaded")
    else:
        print("  - Q (VAE trained): NOT FOUND")

    return skill_model, diffusion_model, q_diffusion, q_vae


def load_test_data(test_data_dir, max_samples=50, random_seed=42):
    """테스트 데이터 로드 (랜덤 샘플링)"""
    import random
    print(f"Loading test data from {test_data_dir}...")

    all_json_paths = []
    for root, _, files in os.walk(test_data_dir):
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
                'trajectory': traj
            })
        except Exception:
            continue

    print(f"Loaded {len(data_samples)} trajectories (random sampled)")
    return data_samples


def prepare_inputs(traj, device='cuda'):
    """trajectory에서 모델 입력 준비 (encoder용 action 정보 포함)"""
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


def compare_latents(skill_model, diffusion_model, data_samples, num_samples_per_state=5, ddim_steps=100):
    """VAE Encoder, VAE Prior, Diffusion latent 비교 (prior/diffusion 모두 K샘플)"""
    encoder_latents_all = []   # (N, Z)
    prior_latents_all = []     # (N*K, Z)
    diff_latents_all = []      # (N*K, Z)

    print("\nComparing VAE Encoder vs VAE Prior vs Diffusion latents...")

    for sample in tqdm(data_samples):
        traj = sample['trajectory']
        state, clip, in_grid, pair_in, pair_out, operation, selection = prepare_inputs(traj)

        # Encoder target (전체 trajectory)
        encoder_mean, _ = skill_model.encoder(state, clip, in_grid, operation, selection, pair_in, pair_out)
        encoder_latents_all.append(encoder_mean.squeeze().detach().cpu().numpy().reshape(1, -1))

        # state init only
        state_init = state[:, 0:1, :, :]
        clip_init = clip[:, 0:1, :, :]

        # VAE Prior K samples (reparam)
        prior_mean, prior_std = skill_model.prior(state_init, clip_init, in_grid, pair_in, pair_out)
        for _ in range(num_samples_per_state):
            eps = torch.randn_like(prior_mean)
            z_prior = prior_mean + prior_std * eps
            prior_latents_all.append(z_prior.detach().cpu().numpy().reshape(1, -1))

        # Diffusion K samples
        for _ in range(num_samples_per_state):
            z_diff = diffusion_model.ddim_sample_extra(
                state_init, clip_init, in_grid, pair_in, pair_out,
                ddim_steps=ddim_steps,
                extra_steps=0, predict_noise=0
            )
            diff_latents_all.append(z_diff.detach().cpu().numpy().reshape(1, -1))

    encoder_latents = np.concatenate(encoder_latents_all, axis=0)
    prior_latents = np.concatenate(prior_latents_all, axis=0)
    diff_latents = np.concatenate(diff_latents_all, axis=0)

    return encoder_latents, prior_latents, diff_latents


def compute_statistics(latents, name):
    print(f"\n{'='*50}")
    print(f"{name} Latent Statistics")
    print(f"{'='*50}")
    print(f"Shape: {latents.shape}")
    print(f"Mean: {latents.mean():.6f}")
    print(f"Std:  {latents.std():.6f}")
    print(f"Min:  {latents.min():.6f}")
    print(f"Max:  {latents.max():.6f}")
    print(f"Abs Mean: {np.abs(latents).mean():.6f}")

    dim_means = latents.mean(axis=0)
    dim_stds = latents.std(axis=0)
    print(f"\nPer-dimension Mean range: [{dim_means.min():.4f}, {dim_means.max():.4f}]")
    print(f"Per-dimension Std range:  [{dim_stds.min():.4f}, {dim_stds.max():.4f}]")

    return {
        'mean': float(latents.mean()),
        'std': float(latents.std()),
        'min': float(latents.min()),
        'max': float(latents.max()),
        'dim_means': dim_means,
        'dim_stds': dim_stds
    }


def mean_pairwise_l2(Z):
    """Z: (K, ZDIM)"""
    K = Z.shape[0]
    diffs = Z[:, None, :] - Z[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)  # (K,K)
    return float(dists[np.triu_indices(K, k=1)].mean())


def trace_cov(Z):
    """Z: (K, ZDIM)"""
    if Z.shape[0] < 2:
        return 0.0
    C = np.cov(Z, rowvar=False)
    return float(np.trace(C))


def analyze_diversity(prior_latents, diff_latents, num_states, K):
    """state별 diversity metric 계산"""
    prior_pw, diff_pw = [], []
    prior_tc, diff_tc = [], []

    for i in range(num_states):
        Zp = prior_latents[i*K:(i+1)*K]
        Zd = diff_latents[i*K:(i+1)*K]
        prior_pw.append(mean_pairwise_l2(Zp))
        diff_pw.append(mean_pairwise_l2(Zd))
        prior_tc.append(trace_cov(Zp))
        diff_tc.append(trace_cov(Zd))

    prior_pw = np.array(prior_pw)
    diff_pw = np.array(diff_pw)
    prior_tc = np.array(prior_tc)
    diff_tc = np.array(diff_tc)

    return {
        'prior_pairwise_l2': prior_pw,
        'diff_pairwise_l2': diff_pw,
        'prior_trace_cov': prior_tc,
        'diff_trace_cov': diff_tc
    }


def q_values_for_candidates(q_network, state, clip, in_grid, pair_in, pair_out, Zcand):
    """
    Zcand: torch (K, ZDIM)
    returns: numpy (K,) of max_a Q(s,a|z)

    Q network forward signature: (s, clip, in_grid, z, pair_in, pair_out)
    """
    K = Zcand.shape[0]
    state_k = state.repeat(K, 1, 1, 1)
    clip_k = clip.repeat(K, 1, 1, 1)
    in_k = in_grid.repeat(K, 1, 1, 1)
    pin_k = pair_in.repeat(K, 1, 1, 1)
    pout_k = pair_out.repeat(K, 1, 1, 1)

    # NOTE: Q network forward order is (s, clip, in_grid, z, pair_in, pair_out)
    if hasattr(q_network, 'q_net_0'):
        q0 = q_network.q_net_0(state_k, clip_k, in_k, Zcand, pin_k, pout_k)
        q1 = q_network.q_net_1(state_k, clip_k, in_k, Zcand, pin_k, pout_k)
        q = torch.min(q0, q1)
    else:
        q = q_network(state_k, clip_k, in_k, Zcand, pin_k, pout_k)

    if q.dim() == 1:
        # (K,) 형태면 그대로
        q_max = q
    else:
        # (K, A)라 가정
        q_max = q.max(dim=1).values

    return q_max.detach().cpu().numpy()


@torch.no_grad()


def analyze_best_of_k_q(q_network, data_samples, prior_latents, diff_latents, K, num_compare=30, device='cuda'):
    """state별 best-of-K Q metric (prior vs diffusion)"""
    if q_network is None:
        print("\nQ network not available, skipping best-of-K Q analysis")
        return None

    num_compare = min(num_compare, len(data_samples))
    print(f"\nRunning best-of-K Q analysis on {num_compare} samples (K={K})...")

    prior_max, diff_max = [], []
    prior_mean, diff_mean = [], []
    prior_var, diff_var = [], []
    prior_margin, diff_margin = [], []

    for i in tqdm(range(num_compare)):
        traj = data_samples[i]['trajectory']
        state, clip, in_grid, pair_in, pair_out, operation, selection = prepare_inputs(traj, device=device)

        # init-only로 Q 입력 맞춤 (학습 코드 기준이 init-only면 이게 안전)
        state0 = state[:, 0:1, :, :]
        clip0 = clip[:, 0:1, :, :]

        Zp = torch.from_numpy(prior_latents[i*K:(i+1)*K]).float().to(device)
        Zd = torch.from_numpy(diff_latents[i*K:(i+1)*K]).float().to(device)

        q_p = q_values_for_candidates(q_network, state0, clip0, in_grid, pair_in, pair_out, Zp)  # (K,)
        q_d = q_values_for_candidates(q_network, state0, clip0, in_grid, pair_in, pair_out, Zd)  # (K,)

        prior_max.append(q_p.max())
        diff_max.append(q_d.max())

        prior_mean.append(q_p.mean())
        diff_mean.append(q_d.mean())

        prior_var.append(q_p.var())
        diff_var.append(q_d.var())

        # margin: top1 - top2
        q_p_sorted = np.sort(q_p)[::-1]
        q_d_sorted = np.sort(q_d)[::-1]
        prior_margin.append(q_p_sorted[0] - q_p_sorted[1] if len(q_p_sorted) > 1 else 0.0)
        diff_margin.append(q_d_sorted[0] - q_d_sorted[1] if len(q_d_sorted) > 1 else 0.0)

    prior_max = np.array(prior_max)
    diff_max = np.array(diff_max)
    prior_mean = np.array(prior_mean)
    diff_mean = np.array(diff_mean)
    prior_var = np.array(prior_var)
    diff_var = np.array(diff_var)
    prior_margin = np.array(prior_margin)
    diff_margin = np.array(diff_margin)

    print("\nBest-of-K Q summary")
    print(f"  maxQ   prior: {prior_max.mean():.4f} ± {prior_max.std():.4f} | diff: {diff_max.mean():.4f} ± {diff_max.std():.4f}")
    print(f"  meanQ  prior: {prior_mean.mean():.4f} ± {prior_mean.std():.4f} | diff: {diff_mean.mean():.4f} ± {diff_mean.std():.4f}")
    print(f"  varQ   prior: {prior_var.mean():.4f} ± {prior_var.std():.4f} | diff: {diff_var.mean():.4f} ± {diff_var.std():.4f}")
    print(f"  margin prior: {prior_margin.mean():.4f} ± {prior_margin.std():.4f} | diff: {diff_margin.mean():.4f} ± {diff_margin.std():.4f}")

    return {
        'prior_maxQ': prior_max, 'diff_maxQ': diff_max,
        'prior_meanQ': prior_mean, 'diff_meanQ': diff_mean,
        'prior_varQ': prior_var, 'diff_varQ': diff_var,
        'prior_margin': prior_margin, 'diff_margin': diff_margin
    }


def plot_hist_two(arr1, arr2, label1, label2, title, xlabel, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(arr1, bins=30, alpha=0.7, label=label1, density=True)
    ax.hist(arr2, bins=30, alpha=0.7, label=label2, density=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


def plot_best_of_k_curve(q_stats, save_path):
    """maxQ의 누적 best-of-k 형태를 보여주고 싶으면 K를 sweep해야 하지만,
    여기선 K 고정이라 간단히 prior/diff maxQ 분포만 그림."""
    # placeholder: 분포로 대체
    plot_hist_two(
        q_stats['prior_maxQ'], q_stats['diff_maxQ'],
        "Prior", "Diffusion",
        "Best-of-K MaxQ Distribution",
        "maxQ over K candidates",
        save_path
    )


def main():
    args = parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    print("="*60)
    print("VAE Encoder vs VAE Prior vs Diffusion Latent Comparison (+ diversity + best-of-K Q)")
    print(f"Checkpoint: {args.checkpoint_name}")
    print("="*60)

    device = 'cuda:0'

    # 모델 로드
    skill_model, diffusion_model, q_diffusion, q_vae = load_models(
        args.checkpoint_name, args.n_T, device=device
    )

    # 테스트 데이터 로드
    data_samples = load_test_data(
        args.test_data_dir,
        max_samples=args.max_samples,
        random_seed=args.random_seed
    )
    N = len(data_samples)
    K = args.num_samples

    # Latent 비교
    encoder_latents, prior_latents, diff_latents = compare_latents(
        skill_model, diffusion_model, data_samples,
        num_samples_per_state=K,
        ddim_steps=args.ddim_steps
    )

    # 출력 디렉토리
    save_dir = args.output_dir if args.output_dir else os.path.dirname(__file__)
    os.makedirs(save_dir, exist_ok=True)
    prefix = args.checkpoint_name

    # 통계량 출력
    print("\n" + "="*60)
    print("STATISTICS COMPARISON")
    print("="*60)
    _ = compute_statistics(encoder_latents, "VAE Encoder (Diffusion Target)")
    _ = compute_statistics(prior_latents, "VAE Prior (sampled)")
    _ = compute_statistics(diff_latents, "Diffusion Output (sampled)")

    # Distance 분석
    print(f"\n{'='*60}")
    print("DISTANCE ANALYSIS")
    print(f"{'='*60}")

    encoder_repeated = np.repeat(encoder_latents, K, axis=0)  # (N*K, Z)

    # Encoder vs Diffusion
    l2_enc_diff = np.linalg.norm(encoder_repeated - diff_latents, axis=1)
    enc_norm = encoder_repeated / (np.linalg.norm(encoder_repeated, axis=1, keepdims=True) + 1e-8)
    diff_norm = diff_latents / (np.linalg.norm(diff_latents, axis=1, keepdims=True) + 1e-8)
    cos_enc_diff = (enc_norm * diff_norm).sum(axis=1)

    print(f"\n[Encoder vs Diffusion]")
    print(f"  L2 Distance - Mean: {l2_enc_diff.mean():.4f}, Std: {l2_enc_diff.std():.4f}")
    print(f"  Cosine Sim  - Mean: {cos_enc_diff.mean():.4f}, Std: {cos_enc_diff.std():.4f}")

    # Encoder vs Prior  (FIXED: use encoder_repeated)
    l2_enc_prior = np.linalg.norm(encoder_repeated - prior_latents, axis=1)
    prior_norm = prior_latents / (np.linalg.norm(prior_latents, axis=1, keepdims=True) + 1e-8)
    cos_enc_prior = (enc_norm * prior_norm).sum(axis=1)

    print(f"\n[Encoder vs Prior]")
    print(f"  L2 Distance - Mean: {l2_enc_prior.mean():.4f}, Std: {l2_enc_prior.std():.4f}")
    print(f"  Cosine Sim  - Mean: {cos_enc_prior.mean():.4f}, Std: {cos_enc_prior.std():.4f}")

    # Prior vs Diffusion (same shape N*K)
    l2_prior_diff = np.linalg.norm(prior_latents - diff_latents, axis=1)
    cos_prior_diff = (prior_norm * diff_norm).sum(axis=1)

    print(f"\n[Prior vs Diffusion]")
    print(f"  L2 Distance - Mean: {l2_prior_diff.mean():.4f}, Std: {l2_prior_diff.std():.4f}")
    print(f"  Cosine Sim  - Mean: {cos_prior_diff.mean():.4f}, Std: {cos_prior_diff.std():.4f}")

    # 기존 plot 저장 (latent distribution)
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    # 동일한 bin edge 사용 (x축 범위 -10~10에서 30개 bin)
    bin_edges = np.linspace(-10, 10, 31)
    ax1.hist(encoder_latents.flatten(), bins=bin_edges, alpha=0.5, label='Encoder', density=True)
    ax1.hist(prior_latents.flatten(), bins=bin_edges, alpha=0.5, label='VAE Prior', density=True)
    ax1.hist(diff_latents.flatten(), bins=bin_edges, alpha=0.5, label='Diffusion', density=True)
    ax1.set_xlim(-10, 10)
    ax1.set_xlabel('Latent Value')
    ax1.set_ylabel('Density')
    ax1.legend()
    plt.tight_layout()
    save_path1 = os.path.join(save_dir, f'{prefix}_latent_distribution.png')
    plt.savefig(save_path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"\nPlot saved to: {save_path1}")

    # distance histogram 저장
    plot_hist_two(
        l2_enc_diff, l2_enc_prior,
        "Enc-Diff", "Enc-Prior",
        f"L2 Distance Distribution (mean±std)\nEnc-Diff: {l2_enc_diff.mean():.2f}±{l2_enc_diff.std():.2f} | Enc-Prior: {l2_enc_prior.mean():.2f}±{l2_enc_prior.std():.2f}",
        "L2 Distance",
        os.path.join(save_dir, f'{prefix}_l2_distance.png')
    )
    plot_hist_two(
        cos_enc_diff, cos_enc_prior,
        "Enc-Diff", "Enc-Prior",
        f"Cosine Similarity Distribution (mean±std)\nEnc-Diff: {cos_enc_diff.mean():.3f}±{cos_enc_diff.std():.3f} | Enc-Prior: {cos_enc_prior.mean():.3f}±{cos_enc_prior.std():.3f}",
        "Cosine Similarity",
        os.path.join(save_dir, f'{prefix}_cosine_similarity.png')
    )

    # ------------------------------------------------------------
    # NEW: diversity / coverage analysis
    # ------------------------------------------------------------
    print(f"\n{'='*60}")
    print("DIVERSITY / COVERAGE ANALYSIS (state-wise)")
    print(f"{'='*60}")

    div = analyze_diversity(prior_latents, diff_latents, num_states=N, K=K)

    prior_pw = div['prior_pairwise_l2']
    diff_pw = div['diff_pairwise_l2']
    prior_tc = div['prior_trace_cov']
    diff_tc = div['diff_trace_cov']

    print(f"Pairwise L2 (within-state)  prior: {prior_pw.mean():.4f} ± {prior_pw.std():.4f} | diff: {diff_pw.mean():.4f} ± {diff_pw.std():.4f}")
    print(f"Trace(Cov) (within-state)   prior: {prior_tc.mean():.4f} ± {prior_tc.std():.4f} | diff: {diff_tc.mean():.4f} ± {diff_tc.std():.4f}")

    plot_hist_two(
        prior_pw, diff_pw,
        "Prior", "Diffusion",
        "Within-state Diversity (Mean Pairwise L2)",
        "Mean pairwise L2 over K candidates",
        os.path.join(save_dir, f'{prefix}_diversity_pairwise_l2.png')
    )
    plot_hist_two(
        prior_tc, diff_tc,
        "Prior", "Diffusion",
        "Within-state Diversity (Trace of Covariance)",
        "trace(Cov) over K candidates",
        os.path.join(save_dir, f'{prefix}_diversity_trace_cov.png')
    )

    # ------------------------------------------------------------
    # NEW: best-of-K Q analysis
    # ------------------------------------------------------------
    # 어떤 Q로 볼지 선택: diffusion trained Q가 기본
    q_net = q_diffusion if q_diffusion is not None else q_vae

    q_stats = analyze_best_of_k_q(
        q_net,
        data_samples=data_samples,
        prior_latents=prior_latents,
        diff_latents=diff_latents,
        K=K,
        num_compare=args.num_compare_q,
        device=device
    )

    if q_stats is not None:
        plot_hist_two(
            q_stats['prior_maxQ'], q_stats['diff_maxQ'],
            "Prior", "Diffusion",
            "Best-of-K MaxQ Distribution",
            "maxQ over K candidates",
            os.path.join(save_dir, f'{prefix}_bestofK_maxQ.png')
        )
        plot_hist_two(
            q_stats['prior_margin'], q_stats['diff_margin'],
            "Prior", "Diffusion",
            "Selection Margin Distribution (top1 - top2)",
            "margin",
            os.path.join(save_dir, f'{prefix}_bestofK_margin.png')
        )
        plot_hist_two(
            q_stats['prior_varQ'], q_stats['diff_varQ'],
            "Prior", "Diffusion",
            "Q Variance Across Candidates",
            "var(Q) over K candidates",
            os.path.join(save_dir, f'{prefix}_bestofK_varQ.png')
        )

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
