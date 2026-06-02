import numpy as np


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, clip, in_grid, latent, reward, sT, clip_T, done, pair_in, pair_out, max_latent=None):
        assert state.ndim == sT.ndim
        state       = np.expand_dims(state, 0)
        clip        = np.expand_dims(clip, 0)
        in_grid        = np.expand_dims(in_grid, 0)
        sT          = np.expand_dims(sT, 0)
        clip_T      = np.expand_dims(clip_T, 0)
        pair_in     = np.expand_dims(pair_in, 0)
        pair_out    = np.expand_dims(pair_out, 0)
        if max_latent is not None:
            max_latent = np.expand_dims(max_latent, 0)
        max_prio = self.priorities.max() if self.buffer else 100.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, clip, in_grid, latent, reward, sT, clip_T, done, pair_in, pair_out, max_latent))
        else:
            self.buffer[self.pos] = (state, clip, in_grid, latent, reward, sT, clip_T, done, pair_in, pair_out, max_latent)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities.copy()
        else:
            prios = self.priorities[:self.pos].copy()
        eps = 1e-8
        prios = np.where(np.isnan(prios) | np.isinf(prios) | (prios <= 0), eps, prios)
        probs  = prios ** self.prob_alpha
        prob_sum = probs.sum()
        if prob_sum <= 0 or np.isnan(prob_sum):
            probs = np.ones(len(prios)) / len(prios)
        else:
            probs /= prob_sum
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        batch       = list(zip(*samples))
        states      = np.concatenate(batch[0])
        clip        = np.concatenate(batch[1])
        in_grid     = np.concatenate(batch[2])
        latents     = np.stack(batch[3])
        rewards     = np.concatenate(batch[4])
        sT          = np.concatenate(batch[5])
        clip_T      = np.concatenate(batch[6])
        dones       = np.array(batch[7])
        pair_in     = np.concatenate(batch[8])
        pair_out    = np.concatenate(batch[9])
        max_latents = np.concatenate(batch[10])
        return states, clip, in_grid, latents, rewards, sT, clip_T, dones, pair_in, pair_out, indices, weights, max_latents

    def update_priorities(self, batch_indices, batch_priorities):
        eps = 1e-8
        for idx, prio in zip(batch_indices, batch_priorities):
            if np.isnan(prio) or np.isinf(prio) or prio <= 0:
                prio = eps
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


class FixedPrioritizedBuffer(object):
    def __init__(self, capacity, num_samples, z_dim, max_grid_size, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.pos = 0
        self.buffer_size = 0
        self.num_samples = num_samples
        self.z_dim = z_dim
        self.max_grid_size = max_grid_size
        self.states = np.zeros((capacity, self.max_grid_size, self.max_grid_size), dtype=np.float32)
        self.clips = np.zeros((capacity, self.max_grid_size, self.max_grid_size), dtype=np.float32)
        self.in_grids = np.zeros((capacity, self.max_grid_size, self.max_grid_size), dtype=np.float32)
        self.latents = np.zeros((capacity, self.z_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.s_T = np.zeros((capacity, self.max_grid_size, self.max_grid_size), dtype=np.float32)
        self.clip_T = np.zeros((capacity, self.max_grid_size, self.max_grid_size), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.pair_in = np.zeros((capacity, 3, self.max_grid_size, self.max_grid_size), dtype=np.float32)
        self.pair_out = np.zeros((capacity, 3, self.max_grid_size, self.max_grid_size), dtype=np.float32)
        self.max_latents = np.zeros((capacity, self.num_samples, self.z_dim), dtype=np.float32)
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, clip, in_grid, latent, reward, s_T, clip_T, done, pair_in, pair_out, max_latent=None):
        self.states[self.pos] = state
        self.clips[self.pos] = clip
        self.in_grids[self.pos] = in_grid
        self.latents[self.pos] = latent
        # Convert reward to scalar if it's an array
        self.rewards[self.pos] = float(np.squeeze(reward)) if hasattr(reward, '__len__') else float(reward)
        self.s_T[self.pos] = s_T
        self.clip_T[self.pos] = clip_T
        # Convert done to scalar if it's an array
        self.dones[self.pos] = float(np.squeeze(done)) if hasattr(done, '__len__') else float(done)
        self.pair_in[self.pos] = pair_in
        self.pair_out[self.pos] = pair_out
        if max_latent is not None:
            self.max_latents[self.pos] = max_latent
        max_prio = self.priorities.max() if self.buffer_size > 0 else 100.0
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        self.buffer_size = min(self.buffer_size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        if self.buffer_size == self.capacity:
            prios = self.priorities.copy()
        else:
            prios = self.priorities[:self.buffer_size].copy()
        eps = 1e-8
        prios = np.where(np.isnan(prios) | np.isinf(prios) | (prios <= 0), eps, prios)
        probs = np.power(prios, self.prob_alpha)
        prob_sum = np.sum(probs)
        if prob_sum <= 0 or np.isnan(prob_sum):
            probs = np.ones(len(prios)) / len(prios)
        else:
            probs /= prob_sum
        indices = np.random.choice(self.buffer_size, batch_size, p=probs)
        weights = np.power((self.buffer_size * probs[indices]), -beta)
        weights /= weights.max()
        states = np.expand_dims(self.states[indices], axis=1)
        clips = np.expand_dims(self.clips[indices], axis=1)
        in_grids = np.expand_dims(self.in_grids[indices], axis=1)
        latents = self.latents[indices]
        rewards = self.rewards[indices]
        s_T = np.expand_dims(self.s_T[indices], axis=1)
        clip_T = np.expand_dims(self.clip_T[indices], axis=1)
        dones = self.dones[indices]
        pair_in = self.pair_in[indices]
        pair_out = self.pair_out[indices]
        max_latents = self.max_latents[indices]
        return states, clips, in_grids, latents, rewards, s_T, clip_T, dones, pair_in, pair_out, indices, weights, max_latents

    def update_priorities(self, batch_indices, batch_priorities):
        batch_priorities = np.array(batch_priorities, dtype=np.float32)
        eps = 1e-8
        batch_priorities = np.where(
            np.isnan(batch_priorities) | np.isinf(batch_priorities) | (batch_priorities <= 0),
            eps,
            batch_priorities
        )
        self.priorities[batch_indices] = batch_priorities

    def __len__(self):
        return self.buffer_size
