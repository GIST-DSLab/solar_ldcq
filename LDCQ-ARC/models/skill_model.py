import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.distributions.transformed_distribution import TransformedDistribution
import torch.distributions.normal as Normal
import torch.distributions.kl as KL
from utils.utils import reparameterize
from models.diffusion_models import (
    Model_mlp,
    Model_cnn_mlp,
    Model_Cond_Diffusion,
    Grid2DPositionalEncoding,
)


class StateEmbeddingWithPositionalEncoding(nn.Module):
    """
    Wrapper for state embedding layer that optionally applies 2D positional encoding.

    The positional encoding is applied AFTER the first Conv2d layer (which outputs 32 channels),
    adding spatial position information to the CNN features.

    Args:
        base_cnn_layers: List of CNN layers (Conv2d, ReLU, etc.)
        flatten_and_linear: The flatten and linear layers after CNN
        use_positional_encoding: Whether to add 2D positional encoding
        pos_encoding_channels: Number of channels for positional encoding (default 32)
        max_grid_size: Maximum grid size (height and width)
    """

    def __init__(self, base_cnn_layers, flatten_and_linear, use_positional_encoding=False,
                 pos_encoding_channels=32, max_grid_size=30):
        super().__init__()
        self.base_cnn_layers = base_cnn_layers
        self.flatten_and_linear = flatten_and_linear
        self.use_positional_encoding = use_positional_encoding

        if use_positional_encoding:
            self.pos_encoder = Grid2DPositionalEncoding(
                channels=pos_encoding_channels,
                max_h=max_grid_size,
                max_w=max_grid_size
            )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 1, height, width)
        Returns:
            Embedded tensor after CNN, positional encoding (if enabled), flatten, and linear layers
        """
        # Apply CNN layers
        x = self.base_cnn_layers(x)

        # Apply positional encoding after CNN (x is now [batch, 32, H, W])
        if self.use_positional_encoding:
            x = self.pos_encoder(x)

        # Apply flatten and linear layers
        x = self.flatten_and_linear(x)

        return x


class AbstractDynamics(nn.Module):
    '''
    P(s_T|s_0,z) is our "abstract dynamics model", because it predicts the resulting state transition over T timesteps given a skill 
    (so similar to regular dynamics model, but in skill space and also temporally extended)
    See Encoder and Decoder for more description
    '''

    def __init__(self,state_dim,z_dim,h_dim,per_element_sigma=True):

        super(AbstractDynamics,self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim+z_dim,h_dim),
            nn.ReLU(),
            nn.Linear(h_dim,h_dim),
            nn.ReLU())
        self.mean_layer = nn.Sequential(
            nn.Linear(h_dim,h_dim),
            nn.ReLU(),
            nn.Linear(h_dim,state_dim))
        if per_element_sigma:
            self.sig_layer  = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,state_dim),nn.Softplus())
        else:
            self.sig_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,1),nn.Softplus())

        self.state_dim = state_dim
        self.per_element_sigma = per_element_sigma

    def forward(self,s0,z):

        '''
        INPUTS:
            s0: batch_size x 1 x state_dim initial state (first state in execution of skill)
            z:  batch_size x 1 x z_dim "skill"/z
        OUTPUTS: 
            sT_mean: batch_size x 1 x state_dim tensor of terminal (time=T) state means
            sT_sig:  batch_size x 1 x state_dim tensor of terminal (time=T) state standard devs
        '''

        # concatenate s0 and z
        s0_z = torch.cat([s0,z],dim=-1)
        # pass s0_z through layers
        feats = self.layers(s0_z)
        # get mean and stand dev of action distribution
        sT_mean = self.mean_layer(feats)
        sT_sig  = self.sig_layer(feats)

        if not self.per_element_sigma:
            sT_sig = torch.cat(self.state_dim*[sT_sig],dim=-1)

        return sT_mean, sT_sig


class ARCAbstractDynamics(nn.Module):
    '''
    ARC-specific abstract dynamics model: P(s_T|s_0,clip,in_grid,z,pair)
    Predicts the terminal grid state given initial state, context, and skill latent.
    Uses residual connection: s_T = s_0 + delta(s_0, clip, in_grid, z, pair)

    This is suitable for ARC tasks where the output is often a transformation of the input.
    '''

    def __init__(self, z_dim, h_dim, max_grid_size=10,
                 state_emb_layer=None, pair_emb_layer=None,
                 use_enhanced_pair_encoding=False, enhanced_layers=None,
                 disable_pair_encoding=False, predict_residual=True):

        super(ARCAbstractDynamics, self).__init__()

        self.max_grid_size = max_grid_size
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.state_dim = max_grid_size * max_grid_size
        self.predict_residual = predict_residual

        # Reuse existing embedding layers from skill model
        self.state_emb_layer = state_emb_layer
        self.pair_emb_layer = pair_emb_layer
        self.use_enhanced_pair_encoding = use_enhanced_pair_encoding
        self.enhanced_layers = enhanced_layers
        self.disable_pair_encoding = disable_pair_encoding

        # Input dimension: s0_emb + clip_emb + in_grid_emb + z (+ pair_emb if enabled)
        input_dim = h_dim + h_dim + h_dim + z_dim
        if not disable_pair_encoding:
            input_dim += h_dim

        # Feature extraction layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )

        # Predict delta (change from s0 to sT)
        self.delta_predictor = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, self.state_dim)
        )

        # Uncertainty estimation (per-pixel sigma)
        self.sig_layer = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, self.state_dim),
            nn.Softplus()
        )

        # Color classifier for cross entropy loss
        # Predicts color class (0-10) for each grid cell
        self.color_classifier = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, self.state_dim * 11)  # 11 color classes per cell
        )

    def forward(self, s0, clip0, in_grid, z, pair_in, pair_out):
        '''
        INPUTS:
            s0: batch_size x 1 x state_dim - initial grid (flattened)
            clip0: batch_size x 1 x state_dim - clip grid (flattened)
            in_grid: batch_size x 1 x state_dim - input grid reference (flattened)
            z: batch_size x 1 x z_dim - skill latent variable
            pair_in: batch_size x 3 x max_grid_size x max_grid_size - input examples
            pair_out: batch_size x 3 x max_grid_size x max_grid_size - output examples
        OUTPUTS:
            sT_mean: batch_size x 1 x state_dim - predicted terminal state mean
            sT_sig: batch_size x 1 x state_dim - predicted terminal state std dev
            sT_logits: batch_size x 1 x (state_dim * 11) - color class logits for cross entropy
        '''
        batch_size = s0.shape[0]

        # Embed s0 using CNN encoder
        s0_emb = self.state_emb_layer(
            s0.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous()
        )
        s0_emb = s0_emb.reshape(batch_size, 1, self.h_dim)

        # Embed clip
        clip_emb = self.state_emb_layer(
            clip0.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous()
        )
        clip_emb = clip_emb.reshape(batch_size, 1, self.h_dim)

        # Embed in_grid
        in_grid_emb = self.state_emb_layer(
            in_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous()
        )
        in_grid_emb = in_grid_emb.reshape(batch_size, 1, self.h_dim)

        # Encode input-output pairs (if not disabled)
        if not self.disable_pair_encoding:
            if self.use_enhanced_pair_encoding and self.enhanced_layers:
                # Enhanced pair encoding: process each pair separately
                pair_transforms = []
                for i in range(3):
                    input_grid = pair_in[:, i:i+1, :, :]
                    output_grid = pair_out[:, i:i+1, :, :]

                    input_emb = self.enhanced_layers['pair_input_emb_layer'](
                        input_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous()
                    )
                    output_emb = self.enhanced_layers['pair_output_emb_layer'](
                        output_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous()
                    )

                    # Learn transformation from input to output
                    pair_transform = self.enhanced_layers['pair_transform_layer'](
                        torch.cat([input_emb, output_emb], dim=-1)
                    )
                    pair_transforms.append(pair_transform)

                # Combine all pairs
                combined_pairs = torch.stack(pair_transforms, dim=1)
                combined_pairs = combined_pairs.reshape(batch_size, -1)
                pair_emb = self.enhanced_layers['enhanced_pair_combiner'](combined_pairs)
                pair_emb = pair_emb.reshape(batch_size, 1, self.h_dim)
            else:
                # Original pair encoding: concatenate all pairs
                pair = torch.cat([pair_in, pair_out], dim=1)
                pair_shape = pair.shape

                pair_emb = self.state_emb_layer(
                    pair.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous()
                )
                pair_emb = pair_emb.reshape(pair_shape[0], 1, pair_shape[1] * self.h_dim)
                pair_emb = self.pair_emb_layer(pair_emb)

            # Concatenate all features
            combined = torch.cat([s0_emb, clip_emb, in_grid_emb, z, pair_emb], dim=-1)
        else:
            # Exclude pair information
            combined = torch.cat([s0_emb, clip_emb, in_grid_emb, z], dim=-1)

        # Extract features
        feats = self.layers(combined)

        # Predict delta (change from s0)
        delta = self.delta_predictor(feats)

        # Compute final state
        if self.predict_residual:
            # Residual connection: s_T = s_0 + delta
            s0_flat = s0.reshape(batch_size, 1, -1)
            sT_mean = s0_flat + delta
        else:
            # Direct prediction
            sT_mean = delta

        # Predict uncertainty
        sT_sig = self.sig_layer(feats)

        # Predict color class logits for cross entropy loss
        sT_logits = self.color_classifier(feats)

        return sT_mean, sT_sig, sT_logits


class AutoregressiveStateDecoder(nn.Module):
    '''
    P(s_T|s_0,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    See Encoder and Decoder for more description
    '''

    def __init__(self,state_dim,z_dim,h_dim,per_element_sigma=True):

        super(AutoregressiveStateDecoder,self).__init__()
        self.decoder_components = nn.ModuleList([LowLevelPolicy(state_dim+i,1,z_dim,h_dim,a_dist='normal') for i in range(state_dim)])
        self.state_dim = state_dim

    def forward(self,state,s_T,z, evaluation=False):
        '''
        INPUTS:
            state: batch_size x 1 x state_dim tensor of states 
            action: batch_size x 1 x a_dim tensor of actions
            z:     batch_size x 1 x z_dim tensor of states
        OUTPUTS:
            a_mean: batch_size x T x a_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  batch_size x T x a_dim tensor of action standard devs for each t in {0.,,,.T}
        
        Iterate through each low level policy component.
        The ith element gets to condition on all elements up to but NOT including a_i
        '''
        s_means = []
        s_sigs = []

        s_means_tensor = torch.zeros_like(state)
        s_sigs_tensor = torch.zeros_like(state)

        for i in range(self.state_dim):
            # Concat state, and a up to i.  state_a takes place of state in orginary policy.
            if not evaluation:
                state_a = torch.cat([state, s_T[:,:,:i]],dim=-1)
            else:
                state_a = torch.cat([state, s_means_tensor[:, :, :i].detach()], dim=-1)
            # pass through ith policy component
            s_T_mean_i,s_T_sig_i = self.decoder_components[i](state_a,z) # these are batch_size x T x 1
            # add to growing list of policy elements
            s_means.append(s_T_mean_i)
            s_sigs.append(s_T_sig_i)

            if evaluation:
                s_means_tensor = torch.cat(s_means, dim=-1)
                s_sigs_tensor = torch.cat(s_sigs, dim=-1)

        s_means = torch.cat(s_means,dim=-1)
        s_sigs  = torch.cat(s_sigs, dim=-1)
        return s_means, s_sigs

    def sample(self,state,z):
        states = []
        for i in range(self.state_dim):
            # Concat state, a up to i, and z_tiled
            state_a = torch.cat([state]+states,dim=-1)
            # pass through ith policy component
            s_T_mean_i,s_T_sig_i = self.decoder_components[i](state_a,z)  # these are batch_size x T x 1
            s_i = reparameterize(s_T_mean_i,s_T_sig_i)
            states.append(s_i)

        return torch.cat(states,dim=-1)

    def numpy_dynamics(self,state,z):
        '''
        maps state as a numpy array and z as a pytorch tensor to a numpy action
        '''
        state = torch.reshape(torch.tensor(state,device=torch.device('cuda:0'),dtype=torch.float32),(1,1,-1))
        
        s_T = self.sample(state,z)
        s_T = s_T.detach().cpu().numpy()
        
        return s_T.reshape([self.state_dim,])


class LowLevelPolicy(nn.Module):
    '''
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    See Encoder and Decoder for more description
    '''

    def __init__(self,state_dim,a_dim,z_dim,h_dim,a_dist,action_num,max_grid_size,fixed_sig=None,state_emb_layer=None,pair_emb_layer=None,use_enhanced_pair_encoding=False,enhanced_layers=None,disable_pair_encoding=False):

        super(LowLevelPolicy, self).__init__()

        self.state_emb_layer = state_emb_layer
        self.pair_emb_layer = pair_emb_layer
        self.use_enhanced_pair_encoding = use_enhanced_pair_encoding
        self.enhanced_layers = enhanced_layers
        self.disable_pair_encoding = disable_pair_encoding
        
        # Adjust input dimension based on whether pair encoding is disabled
        input_dim = h_dim+h_dim+h_dim+z_dim  # state, clip, in_grid, z
        if not disable_pair_encoding:
            input_dim += h_dim  # add pair dimension

        self.layers = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.h_dim = h_dim
        self.max_grid_size = max_grid_size
        
        self.a_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,action_num))
        self.a_act = nn.Softmax(dim=2)
        
        self.x_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,self.max_grid_size))
        self.x_act = nn.Softmax(dim=2)
        
        self.y_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,self.max_grid_size))
        self.y_act = nn.Softmax(dim=2)
        
        self.h_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,self.max_grid_size))
        self.h_act = nn.Softmax(dim=2)
        
        self.w_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,self.max_grid_size))
        self.w_act = nn.Softmax(dim=2)
        
        self.a_dist = a_dist
        self.a_dim = a_dim
        self.fixed_sig = fixed_sig

    def forward(self, state, clip, in_grid, z, pair_in, pair_out):
        '''
        INPUTS:
            state: batch_size x T x state_dim tensor of states 
            z:     batch_size x 1 x z_dim tensor of states
        OUTPUTS:
            a_mean: batch_size x T x a_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  batch_size x T x a_dim tensor of action standard devs for each t in {0.,,,.T}
        '''
        # tile z along time axis so dimension matches state
        # z_tiled = z.tile([1, state.shape[-2], 1]) #not sure about this// state에 붙이려고 batch_size x T x state_dim형태로 변환

        # 원본 Concat state and z_tiled
        # state_z = torch.cat([state, z_tiled], dim=-1)
        
        # ARC 전용 Concat state and z_tiled
        s_emb = self.state_emb_layer(state.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())    # (batch * block_size, h_dim)
        s_emb = s_emb.reshape(state.shape[0], state.shape[1], self.h_dim) # (batch, block_size, n_embd)
        
        # clip 임베딩
        clip_emb = self.state_emb_layer(clip.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())    # (batch * block_size, h_dim)
        clip_emb = clip_emb.reshape(clip.shape[0], clip.shape[1], self.h_dim) # (batch, block_size, n_embd)
        
        # in_grid 임베딩
        in_grid_emb = self.state_emb_layer(in_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
        in_grid_emb = in_grid_emb.reshape(in_grid.shape[0], in_grid.shape[1], self.h_dim)
        in_grid_tiled = in_grid_emb.tile([1, s_emb.shape[-2], 1])       # 차원 맞춰주려고 N번 반복
        
        # z 반복 - 차원 맞춰주려고 N번 반복
        z_tiled = z.tile([1, s_emb.shape[-2], 1])

        # input-output pair 임베딩 - Only if pair encoding is not disabled
        if not self.disable_pair_encoding:
            if self.use_enhanced_pair_encoding and self.enhanced_layers:
                # Enhanced: Process each input-output pair separately and learn transformations
                pair_transforms = []
                for i in range(3):  # 3 example pairs
                    input_grid = pair_in[:, i:i+1, :, :]  # (batch, 1, max_grid_size, max_grid_size)
                    output_grid = pair_out[:, i:i+1, :, :]

                    # Individual embeddings
                    input_emb = self.enhanced_layers['pair_input_emb_layer'](input_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
                    output_emb = self.enhanced_layers['pair_output_emb_layer'](output_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())

                    # Learn input->output transformation
                    pair_transform = self.enhanced_layers['pair_transform_layer'](torch.cat([input_emb, output_emb], dim=-1))
                    pair_transforms.append(pair_transform)

                # Combine all pair transformations
                combined_pairs = torch.stack(pair_transforms, dim=1)  # (batch, 3, h_dim)
                combined_pairs = combined_pairs.reshape(combined_pairs.shape[0], -1)  # (batch, 3*h_dim)
                pair_emb = self.enhanced_layers['enhanced_pair_combiner'](combined_pairs)  # (batch, h_dim)
                pair_emb = pair_emb.reshape(pair_emb.shape[0], 1, self.h_dim)
                pair_tiled = pair_emb.tile([1, s_emb.shape[-2], 1])
            else:
                # Original method: concatenate all pairs
                pair = torch.cat([pair_in, pair_out], dim=1)
                pair_shape = pair.shape
                # print("LL policy - pair shape : {0}".format(pair.shape))

                pair_emb = self.state_emb_layer(pair.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
                # print("LL policy - Before pair_emb : {0}".format(pair_emb.shape))

                pair_emb = pair_emb.reshape(pair_shape[0], 1, pair_shape[1]*self.h_dim)
                # print("LL policy - After pair_emb : {0}".format(pair_emb.shape))

                pair_emb = self.pair_emb_layer(pair_emb)
                pair_tiled = pair_emb.tile([1, s_emb.shape[-2], 1])       # 차원 맞춰주려고 N번 반복

            state_z = torch.cat([s_emb, clip_emb, in_grid_tiled, z_tiled, pair_tiled], dim=-1)
        else:
            # Exclude pair information
            state_z = torch.cat([s_emb, clip_emb, in_grid_tiled, z_tiled], dim=-1)

        # pass z and state through layers
        feats = self.layers(state_z)
        
        # get mean and stand dev of action distribution
        a_mean = self.a_layer(feats)
        a_mean = self.a_act(a_mean)
        
        x_mean = self.x_layer(feats)
        x_mean = self.x_act(x_mean)
        
        y_mean = self.y_layer(feats)
        y_mean = self.y_act(y_mean)
        
        h_mean = self.h_layer(feats)
        h_mean = self.h_act(h_mean)
        
        w_mean = self.w_layer(feats)
        w_mean = self.w_act(w_mean)
        
        return a_mean, None, x_mean, None, y_mean, None, h_mean, None, w_mean, None

    def numpy_policy(self, state, clip, in_grid, z, pair_in, pair_out):
        '''
        maps state as a numpy array and z as a pytorch tensor to a numpy action
        '''
        state = torch.reshape(state.clone().detach().to(device=torch.device('cuda:0'), dtype=torch.float32), (1,1,-1))
        clip = torch.reshape(clip.clone().detach().to(device=torch.device('cuda:0'), dtype=torch.float32), (1,1,-1))
        in_grid = torch.reshape(in_grid.clone().detach().to(device=torch.device('cuda:0'), dtype=torch.float32), (1,1,-1))

        # state = torch.reshape(torch.tensor(state, device=torch.device('cuda:0'), dtype=torch.float32), (1,1,-1))
        # clip = torch.reshape(torch.tensor(clip, device=torch.device('cuda:0'), dtype=torch.float32), (1,1,-1))
        # in_grid = torch.reshape(torch.tensor(in_grid, device=torch.device('cuda:0'), dtype=torch.float32), (1,1,-1))
        
        a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig = self.forward(state, clip, in_grid, z, pair_in, pair_out)
        # a_mean, a_sig = self.forward(state, z)
        action = self.reparameterize(a_mean, a_sig, self.a_dim)
        
        x = self.reparameterize(x_mean, x_sig, self.max_grid_size)
        y = self.reparameterize(y_mean, y_sig, self.max_grid_size)
        h = self.reparameterize(h_mean, h_sig, self.max_grid_size)
        w = self.reparameterize(w_mean, w_sig, self.max_grid_size)
        
        if self.a_dist == 'tanh_normal':
            action = nn.Tanh()(action)
            x = nn.Tanh()(x)
            y = nn.Tanh()(y)
            h = nn.Tanh()(h)
            w = nn.Tanh()(w)
            
        action = action.detach().cpu().numpy()
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        h = h.detach().cpu().numpy()
        w = w.detach().cpu().numpy()
        
        return action, x, y, h, w
        # return action.reshape([self.a_dim,])

    def tensor_policy(self, state, clip, in_grid, z, pair_in, pair_out):
        '''
        maps state as a numpy array and z as a pytorch tensor to a numpy action
        '''
        state_shape = state.shape
        state = state.clone().detach().to(device=torch.device('cuda:0'), dtype=torch.float32).reshape(state_shape[0], 1, -1)

        clip_shape = clip.shape
        clip = clip.clone().detach().to(device=torch.device('cuda:0'), dtype=torch.float32).reshape(clip_shape[0], 1, -1)

        in_grid_shape = in_grid.shape
        in_grid = in_grid.clone().detach().to(device=torch.device('cuda:0'), dtype=torch.float32).reshape(in_grid_shape[0], 1, -1)
        
        a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig = self.forward(state, clip, in_grid, z, pair_in, pair_out)
        # a_mean, a_sig = self.forward(state, z)
        action = self.reparameterize(a_mean, a_sig, self.a_dim)
        
        x = self.reparameterize(x_mean, x_sig, self.max_grid_size)
        y = self.reparameterize(y_mean, y_sig, self.max_grid_size)
        h = self.reparameterize(h_mean, h_sig, self.max_grid_size)
        w = self.reparameterize(w_mean, w_sig, self.max_grid_size)
        
        if self.a_dist == 'tanh_normal':
            action = nn.Tanh()(action)
            x = nn.Tanh()(x)
            y = nn.Tanh()(y)
            h = nn.Tanh()(h)
            w = nn.Tanh()(w)
        
        return action, x, y, h, w

    def reparameterize(self, mean, std, dim):
        if self.a_dist=='softmax':
            intervals = torch.linspace(-1, 1, dim).cuda()
            max_idx = torch.argmax(mean, dim=2).unsqueeze(2)
            max_interval = intervals[max_idx]
            return max_interval
        
        if(std == None):
            return mean
        else:
            eps = torch.normal(torch.zeros(mean.size()).cuda(), torch.ones(mean.size()).cuda())
            return mean + std*eps


class AutoregressiveLowLevelPolicy(nn.Module):
    '''
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    See Encoder and Decoder for more description
    '''

    def __init__(self,state_dim,a_dim,z_dim,h_dim,a_dist,fixed_sig=None):

        super(AutoregressiveLowLevelPolicy,self).__init__()
        self.policy_components = nn.ModuleList([LowLevelPolicy(state_dim+i,1,z_dim,h_dim,a_dist=a_dist,fixed_sig=fixed_sig) for i in range(a_dim)])
        self.a_dim = a_dim
        self.a_dist = a_dist

    def forward(self,state,actions,z):
        '''
        INPUTS:
            state: batch_size x T x state_dim tensor of states
            action: batch_size x T x a_dim tensor of actions
            z:     batch_size x 1 x z_dim tensor of states
        OUTPUTS:
            a_mean: batch_size x T x a_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  batch_size x T x a_dim tensor of action standard devs for each t in {0.,,,.T}
        
        Iterate through each low level policy component.
        The ith element gets to condition on all elements up to but NOT including a_i
        '''
        a_means = []
        a_sigs = []
        for i in range(self.a_dim):
            # Concat state, and a up to i.  state_a takes place of state in orginary policy.
            state_a = torch.cat([state,actions[:,:,:i]],dim=-1)
            # pass through ith policy component
            a_mean_i,a_sig_i = self.policy_components[i](state_a,z)  # these are batch_size x T x 1
            if self.a_dist == 'softmax':
                a_mean_i = a_mean_i.unsqueeze(dim=2)
            # add to growing list of policy elements
            a_means.append(a_mean_i)
            if not self.a_dist == 'softmax':
                a_sigs.append(a_sig_i)
        if self.a_dist == 'softmax':
            a_means = torch.cat(a_means,dim=2)
            return a_means, None
        a_means = torch.cat(a_means,dim=-1)
        a_sigs  = torch.cat(a_sigs, dim=-1)
        return a_means, a_sigs

    def sample(self,state,z):
        actions = []
        for i in range(self.a_dim):
            # Concat state, a up to i, and z_tiled
            state_a = torch.cat([state]+actions,dim=-1)
            # pass through ith policy component
            a_mean_i,a_sig_i = self.policy_components[i](state_a,z)  # these are batch_size x T x 1

            a_i = self.reparameterize(a_mean_i,a_sig_i)
            #a_i = a_mean_i

            if self.a_dist == 'tanh_normal':
                a_i = nn.Tanh()(a_i)
            actions.append(a_i)

        return torch.cat(actions,dim=-1)

    def numpy_policy(self,state,z):
        '''
        maps state as a numpy array and z as a pytorch tensor to a numpy action
        '''
        state = torch.reshape(torch.tensor(state,device=torch.device('cuda:0'),dtype=torch.float32),(1,1,-1))
        
        action = self.sample(state,z)
        action = action.detach().cpu().numpy()
        
        return action.reshape([self.a_dim,])

    def reparameterize(self, mean, std):
        if self.a_dist=='softmax':
            intervals = torch.linspace(-1, 1, 21).cuda()
            # max_idx = torch.distributions.categorical.Categorical(mean).sample()
            max_idx = torch.argmax(mean, dim=2)
            max_interval = intervals[max_idx]
            return max_interval.unsqueeze(-1)
        eps = torch.normal(torch.zeros(mean.size()).cuda(), torch.ones(mean.size()).cuda())
        return mean + std*eps


class TransformEncoder(nn.Module):
    '''
    Transformer-based Encoder module for ARC tasks.
    Similar to GRUEncoder but uses multi-head attention instead of RNN.
    '''

    def __init__(self, state_dim, a_dim, z_dim, h_dim, horizon=5, n_transformer_layers=4, n_heads=8, dropout=0.1, normalize_latent=False,
                 color_num=11, action_num=36, max_grid_size=30, state_emb_layer=None, pair_emb_layer=None,
                 use_enhanced_pair_encoding=False, enhanced_layers=None, disable_pair_encoding=False,
                 use_split_pair_trajectory_encoding=False, use_concept_in_encoder=False):
        super(TransformEncoder, self).__init__()

        self.state_dim = state_dim
        self.a_dim = a_dim
        self.normalize_latent = normalize_latent
        self.h_dim = h_dim
        self.max_grid_size = max_grid_size
        self.z_dim = z_dim
        self.horizon = horizon
        self.use_split_pair_trajectory_encoding = use_split_pair_trajectory_encoding
        self.use_concept_in_encoder = use_concept_in_encoder

        self.state_emb_layer = state_emb_layer
        self.pair_emb_layer = pair_emb_layer
        self.use_enhanced_pair_encoding = use_enhanced_pair_encoding
        self.enhanced_layers = enhanced_layers
        self.disable_pair_encoding = disable_pair_encoding

        # Concept projection layer (if using concept in encoder)
        if self.use_concept_in_encoder:
            self.concept_proj = nn.Sequential(
                nn.Linear(z_dim, h_dim),  # Assume concept embedding is z_dim
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
            )
            # Concept classifier: classifies latent z back to concept
            # This serves as auxiliary task to ensure z contains concept information
            # num_concepts will be passed during model initialization
            self.concept_classifier = None  # Will be initialized in SkillModel with num_concepts

        # Positional encoding for timesteps (important for transformer!)
        # Calculate max sequence length: states + clip + in_grid + pair(if not disabled) + actions + selections
        # Each timestep has: state, clip, in_grid, (pair), action, x, y, h, w
        # For horizon=5: 5 timesteps * (2 state + 1 in_grid + 1 pair + 5 action/selection) = ~40-50 tokens
        max_seq_len = horizon * 10 + 10  # Conservative estimate
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, h_dim))
        self.embed_ln = nn.LayerNorm(h_dim)

        # Action and selection embeddings
        self.action_emb_layer = nn.Sequential(nn.Embedding(action_num, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU())
        self.x_emb_layer = nn.Sequential(nn.Embedding(self.max_grid_size, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU())
        self.y_emb_layer = nn.Sequential(nn.Embedding(self.max_grid_size, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU())
        self.h_emb_layer = nn.Sequential(nn.Embedding(self.max_grid_size, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU())
        self.w_emb_layer = nn.Sequential(nn.Embedding(self.max_grid_size, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU())

        if use_split_pair_trajectory_encoding:
            half_z_dim = z_dim // 2

            self.pair_concat_proj = nn.Linear(2 * h_dim, h_dim)
            self.pair_attention = nn.MultiheadAttention(h_dim, num_heads=4, batch_first=True)
            self.pair_proj = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU()
            )
            self.pair_mean_layer = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, half_z_dim)
            )
            self.pair_sig_layer = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, half_z_dim),
                nn.Softplus()
            )

            # Transformer for trajectory encoding
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=h_dim,
                nhead=n_heads,
                dim_feedforward=4*h_dim,
                dropout=dropout,
                batch_first=True
            )
            self.traj_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

            self.traj_mean_layer = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, half_z_dim)
            )
            self.traj_sig_layer = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, half_z_dim),
                nn.Softplus()
            )
        else:
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=h_dim,
                nhead=n_heads,
                dim_feedforward=4*h_dim,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

            self.mean_layer = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, z_dim)
            )
            self.sig_layer = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, z_dim),
                nn.Softplus()
            )

    def forward(self, states, clip, in_grid, actions, selection, pair_in, pair_out, concept_emb=None):
        '''
        Takes a sequence of states and actions, and infers the distribution over latent skill variable, z

        INPUTS:
            states: batch_size x T x state_dim state sequence tensor
            clip: batch_size x T x state_dim clip sequence tensor
            in_grid: batch_size x 1 x state_dim input grid tensor
            actions: batch_size x T x 1 action sequence tensor
            selection: batch_size x T x 4 selection sequence tensor
            pair_in: batch_size x 3 x max_grid_size x max_grid_size
            pair_out: batch_size x 3 x max_grid_size x max_grid_size
            concept_emb: (optional) batch_size x concept_dim tensor for concept conditioning
        OUTPUTS:
            z_mean: batch_size x 1 x z_dim tensor indicating mean of z distribution
            z_sig:  batch_size x 1 x z_dim tensor indicating standard deviation of z distribution
        '''
        # State embeddings (2D grids)
        s_emb = self.state_emb_layer(states.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
        s_emb = s_emb.reshape(states.shape[0], states.shape[1], 1, self.h_dim)

        # Clip embeddings
        clip_emb = self.state_emb_layer(clip.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
        clip_emb = clip_emb.reshape(clip.shape[0], clip.shape[1], 1, self.h_dim)

        # In_grid embeddings
        in_grid_emb = self.state_emb_layer(in_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
        in_grid_emb = in_grid_emb.reshape(in_grid.shape[0], in_grid.shape[1], 1, self.h_dim)
        in_grid_tiled = in_grid_emb.tile([1, s_emb.shape[1], 1, 1])

        # Pair embeddings - Only if pair encoding is not disabled
        if not self.disable_pair_encoding:
            if self.use_enhanced_pair_encoding and self.enhanced_layers:
                pair_transforms = []
                for i in range(3):
                    input_grid = pair_in[:, i:i+1, :, :]
                    output_grid = pair_out[:, i:i+1, :, :]

                    input_emb = self.enhanced_layers['pair_input_emb_layer'](input_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
                    input_emb = input_emb.reshape(input_grid.shape[0], -1)

                    output_emb = self.enhanced_layers['pair_output_emb_layer'](output_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
                    output_emb = output_emb.reshape(output_grid.shape[0], -1)

                    pair_transform = self.enhanced_layers['pair_transform_layer'](torch.cat([input_emb, output_emb], dim=-1))
                    pair_transforms.append(pair_transform)

                combined_pairs = torch.stack(pair_transforms, dim=1)
                combined_pairs = combined_pairs.reshape(combined_pairs.shape[0], -1)
                pair_emb = self.enhanced_layers['enhanced_pair_combiner'](combined_pairs)
                pair_emb = pair_emb.reshape(pair_emb.shape[0], 1, 1, self.h_dim)
                pair_tiled = pair_emb.tile([1, s_emb.shape[1], 1, 1])
            else:
                pair = torch.cat([pair_in, pair_out], dim=1)
                pair_emb = self.state_emb_layer(pair.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
                pair_emb = pair_emb.reshape(pair.shape[0], 1, 1, pair.shape[1]*self.h_dim)
                pair_emb = self.pair_emb_layer(pair_emb)
                pair_tiled = pair_emb.tile([1, s_emb.shape[1], 1, 1])

        # Action embeddings
        a_emb = self.action_emb_layer(actions)
        x_emb = self.x_emb_layer(selection[:, :, 0])
        y_emb = self.y_emb_layer(selection[:, :, 1])
        h_emb = self.h_emb_layer(selection[:, :, 2])
        w_emb = self.w_emb_layer(selection[:, :, 3])

        if self.use_split_pair_trajectory_encoding:
            # Convert to 4D if needed
            if a_emb.ndim == 3:
                a_emb = a_emb.unsqueeze(2)
            if x_emb.ndim == 3:
                x_emb = x_emb.unsqueeze(2)
                y_emb = y_emb.unsqueeze(2)
                h_emb = h_emb.unsqueeze(2)
                w_emb = w_emb.unsqueeze(2)

            if not self.disable_pair_encoding:
                if self.use_enhanced_pair_encoding and self.enhanced_layers:
                    pair_tokens = []
                    for i in range(3):
                        input_grid = pair_in[:, i:i+1, :, :]
                        output_grid = pair_out[:, i:i+1, :, :]

                        input_emb = self.enhanced_layers['pair_input_emb_layer'](input_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
                        input_emb = input_emb.reshape(input_grid.shape[0], -1)

                        output_emb = self.enhanced_layers['pair_output_emb_layer'](output_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
                        output_emb = output_emb.reshape(output_grid.shape[0], -1)

                        pair_token = torch.cat([input_emb, output_emb], dim=-1)
                        pair_token = self.enhanced_layers['pair_transform_layer'](pair_token)
                        pair_tokens.append(pair_token)

                    pair_seq = torch.stack(pair_tokens, dim=1)
                else:
                    pair_tokens = []
                    for i in range(3):
                        input_grid = pair_in[:, i:i+1, :, :]
                        output_grid = pair_out[:, i:i+1, :, :]

                        input_emb = self.state_emb_layer(input_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
                        input_emb = input_emb.reshape(input_grid.shape[0], -1)

                        output_emb = self.state_emb_layer(output_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
                        output_emb = output_emb.reshape(output_grid.shape[0], -1)

                        pair_token = torch.cat([input_emb, output_emb], dim=-1)
                        pair_token = self.pair_concat_proj(pair_token)
                        pair_tokens.append(pair_token)

                    pair_seq = torch.stack(pair_tokens, dim=1)

                attn_output, _ = self.pair_attention(pair_seq, pair_seq, pair_seq)
                pair_feats = attn_output.mean(dim=1, keepdim=True)
                pair_feats = self.pair_proj(pair_feats)

                pair_z_mean = self.pair_mean_layer(pair_feats)
                pair_z_sig = self.pair_sig_layer(pair_feats)
            else:
                half_z_dim = self.z_dim // 2
                pair_z_mean = torch.zeros(s_emb.shape[0], 1, half_z_dim).to(s_emb.device)
                pair_z_sig = torch.ones(s_emb.shape[0], 1, half_z_dim).to(s_emb.device)

            # Trajectory encoding with transformer
            traj_input = torch.cat([s_emb, clip_emb, in_grid_tiled, a_emb, x_emb, y_emb, h_emb, w_emb], dim=-2)
            traj_input_shape = traj_input.shape
            traj_input = traj_input.view(traj_input_shape[0], -1, traj_input_shape[-1]).contiguous()

            # Add positional encoding
            seq_len = traj_input.shape[1]
            traj_input = traj_input + self.positional_encoding[:, :seq_len, :]
            traj_input = self.embed_ln(traj_input)

            # Pass through transformer
            traj_feats = self.traj_transformer(traj_input)
            traj_hn = traj_feats[:, -1:, :]  # Take last token
            traj_z_mean = self.traj_mean_layer(traj_hn)
            traj_z_sig = self.traj_sig_layer(traj_hn)

            z_mean = torch.cat([pair_z_mean, traj_z_mean], dim=-1)
            z_sig = torch.cat([pair_z_sig, traj_z_sig], dim=-1)

            if self.normalize_latent:
                z_mean = z_mean / torch.norm(z_mean, dim=-1).unsqueeze(-1)
        else:
            # Concatenate all embeddings
            if not self.disable_pair_encoding:
                s_emb_a = torch.cat([s_emb, clip_emb, in_grid_tiled, pair_tiled, a_emb, x_emb, y_emb, h_emb, w_emb], dim=-2)
            else:
                s_emb_a = torch.cat([s_emb, clip_emb, in_grid_tiled, a_emb, x_emb, y_emb, h_emb, w_emb], dim=-2)

            s_emb_a_shape = s_emb_a.shape
            s_emb_a = s_emb_a.view(s_emb_a_shape[0], -1, s_emb_a_shape[-1]).contiguous()

            # Add concept embedding if provided and enabled
            if self.use_concept_in_encoder and concept_emb is not None:
                # Project concept embedding to h_dim and add as first token
                concept_token = self.concept_proj(concept_emb).unsqueeze(1)  # (batch, 1, h_dim)
                s_emb_a = torch.cat([concept_token, s_emb_a], dim=1)  # Prepend concept token

            # Add positional encoding
            seq_len = s_emb_a.shape[1]
            s_emb_a = s_emb_a + self.positional_encoding[:, :seq_len, :]
            s_emb_a = self.embed_ln(s_emb_a)

            # Pass through transformer
            feats = self.transformer(s_emb_a)
            hn = feats[:, -1:, :]  # Take last token as summary
            z_mean = self.mean_layer(hn)
            z_sig = self.sig_layer(hn)

            if self.normalize_latent:
                z_mean = z_mean / torch.norm(z_mean, dim=-1).unsqueeze(-1)

        return z_mean, z_sig


class GRUEncoder(nn.Module):
    '''
    Encoder module.
    -Concat states+actions
    -Pass through linear embedding
    -Pass through bidirectional RNN
    -Pass output of bidirectional RNN through 2 linear layers, one to get mean of z and one to get stand dev (we're estimating one z ("skill") for entire episode)
    '''

    def __init__(self,state_dim,a_dim,z_dim,h_dim,n_gru_layers=4,normalize_latent=False,
                 color_num=11,action_num=36,max_grid_size=30,state_emb_layer=None,pair_emb_layer=None,use_enhanced_pair_encoding=False,enhanced_layers=None,disable_pair_encoding=False,use_split_pair_trajectory_encoding=False,use_concept_in_encoder=False):
        super(GRUEncoder, self).__init__()

        self.state_dim = state_dim # state dimension
        self.a_dim = a_dim # action dimension
        self.normalize_latent = normalize_latent
        self.h_dim = h_dim
        self.max_grid_size = max_grid_size
        self.z_dim = z_dim
        self.use_split_pair_trajectory_encoding = use_split_pair_trajectory_encoding
        self.use_concept_in_encoder = use_concept_in_encoder
        # 아래 방법은 1차원으로 펴서 사용하는 방법
        # self.state_emb_layer  = nn.Sequential(
        #     nn.Embedding(color_num, h_dim),
        #     nn.ReLU(),
        #     nn.Linear(h_dim, h_dim),
        #     nn.ReLU()
        # )

        self.state_emb_layer = state_emb_layer
        self.pair_emb_layer = pair_emb_layer
        self.use_enhanced_pair_encoding = use_enhanced_pair_encoding
        self.enhanced_layers = enhanced_layers
        self.disable_pair_encoding = disable_pair_encoding

        self.action_emb_layer  = nn.Sequential(nn.Embedding(action_num, h_dim),nn.ReLU(),nn.Linear(h_dim, h_dim),nn.ReLU())
        self.x_emb_layer  = nn.Sequential(nn.Embedding(self.max_grid_size, h_dim),nn.ReLU(),nn.Linear(h_dim, h_dim),nn.ReLU())
        self.y_emb_layer  = nn.Sequential(nn.Embedding(self.max_grid_size, h_dim),nn.ReLU(),nn.Linear(h_dim, h_dim),nn.ReLU())
        self.h_emb_layer  = nn.Sequential(nn.Embedding(self.max_grid_size, h_dim),nn.ReLU(),nn.Linear(h_dim, h_dim),nn.ReLU())
        self.w_emb_layer  = nn.Sequential(nn.Embedding(self.max_grid_size, h_dim),nn.ReLU(),nn.Linear(h_dim, h_dim),nn.ReLU())

        # Concept projection layer (if using concept in encoder)
        if self.use_concept_in_encoder:
            self.concept_proj = nn.Sequential(
                nn.Linear(z_dim, h_dim),  # Assume concept embedding is z_dim
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
            )
            # Concept classifier will be initialized in SkillModel
            self.concept_classifier = None

        # self.rnn = nn.GRU(h_dim+a_dim, h_dim, batch_first=True, bidirectional=True, num_layers=n_gru_layers)
        if use_split_pair_trajectory_encoding:
            half_z_dim = z_dim // 2

            self.pair_concat_proj = nn.Linear(2 * h_dim, h_dim)
            self.pair_attention = nn.MultiheadAttention(h_dim, num_heads=4, batch_first=True)
            self.pair_proj = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU()
            )
            self.pair_mean_layer = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, half_z_dim)
            )
            self.pair_sig_layer = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, half_z_dim),
                nn.Softplus()
            )

            self.traj_rnn = nn.GRU(h_dim, h_dim, batch_first=True, bidirectional=True, num_layers=n_gru_layers)
            self.traj_mean_layer = nn.Sequential(
                nn.Linear(2*h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, half_z_dim)
            )
            self.traj_sig_layer = nn.Sequential(
                nn.Linear(2*h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, half_z_dim),
                nn.Softplus()
            )
        else:
            self.rnn = nn.GRU(h_dim, h_dim, batch_first=True, bidirectional=True, num_layers=n_gru_layers)

            #self.mean_layer = nn.Linear(h_dim,z_dim)
            self.mean_layer = nn.Sequential(
                nn.Linear(2*h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, z_dim)
            )
            #self.sig_layer  = nn.Sequential(nn.Linear(h_dim,z_dim),nn.Softplus())  # using softplus to ensure stand dev is positive
            self.sig_layer  = nn.Sequential(
                nn.Linear(2*h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, z_dim),
                nn.Softplus()
            )

    def forward(self, states, clip, in_grid, actions, selection, pair_in, pair_out, concept_emb=None):

        '''
        Takes a sequence of states and actions, and infers the distribution over latent skill variable, z

        INPUTS:
            states: batch_size x T x state_dim state sequence tensor
            actions: batch_size x T x a_dim action sequence tensor
            concept_emb: (optional) batch_size x concept_dim tensor for concept conditioning
        OUTPUTS:
            z_mean: batch_size x 1 x z_dim tensor indicating mean of z distribution
            z_sig:  batch_size x 1 x z_dim tensor indicating standard deviation of z distribution
        '''
        # State가 1차원인 경우
        # s_emb = self.state_emb_layer(states)
        
        # State가 2차원인 경우
        s_emb = self.state_emb_layer(states.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())    # (batch * block_size, h_dim)
        s_emb = s_emb.reshape(states.shape[0], states.shape[1], 1, self.h_dim)                          # (batch, block_size, n_embd)
        
        # clip 임베딩
        clip_emb = self.state_emb_layer(clip.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
        clip_emb = clip_emb.reshape(clip.shape[0], clip.shape[1], 1, self.h_dim)
        
        # in_grid 임베딩
        in_grid_emb = self.state_emb_layer(in_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
        in_grid_emb = in_grid_emb.reshape(in_grid.shape[0], in_grid.shape[1], 1, self.h_dim)
        in_grid_tiled = in_grid_emb.tile([1, s_emb.shape[1], 1, 1])       # 차원 맞춰주려고 N번 반복
        
        # input-output pair 임베딩 - Only if pair encoding is not disabled
        if not self.disable_pair_encoding:
            if self.use_enhanced_pair_encoding and self.enhanced_layers:
                # Enhanced: Process each input-output pair separately and learn transformations
                # Assuming pair_in and pair_out are (batch, 3, max_grid_size, max_grid_size)
                pair_transforms = []
                for i in range(3):  # 3 example pairs
                    # Get individual input and output grids
                    input_grid = pair_in[:, i:i+1, :, :]  # (batch, 1, max_grid_size, max_grid_size)
                    output_grid = pair_out[:, i:i+1, :, :]

                    # Embed input and output separately with dedicated layers
                    input_emb = self.enhanced_layers['pair_input_emb_layer'](input_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
                    input_emb = input_emb.reshape(input_grid.shape[0], -1)  # (batch, h_dim)

                    output_emb = self.enhanced_layers['pair_output_emb_layer'](output_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
                    output_emb = output_emb.reshape(output_grid.shape[0], -1)  # (batch, h_dim)

                    # Learn the transformation from input to output
                    pair_transform = self.enhanced_layers['pair_transform_layer'](torch.cat([input_emb, output_emb], dim=-1))
                    pair_transforms.append(pair_transform)

                # Combine all pair transformations
                combined_pairs = torch.stack(pair_transforms, dim=1)  # (batch, 3, h_dim)
                combined_pairs = combined_pairs.reshape(combined_pairs.shape[0], -1)  # (batch, 3*h_dim)
                pair_emb = self.enhanced_layers['enhanced_pair_combiner'](combined_pairs)  # (batch, h_dim)
                pair_emb = pair_emb.reshape(pair_emb.shape[0], 1, 1, self.h_dim)
                pair_tiled = pair_emb.tile([1, s_emb.shape[1], 1, 1])
            else:
                # Original method: concatenate all pairs
                pair = torch.cat([pair_in, pair_out], dim=1)
                pair_emb = self.state_emb_layer(pair.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
                pair_emb = pair_emb.reshape(pair.shape[0], 1, 1, pair.shape[1]*self.h_dim)
                pair_emb = self.pair_emb_layer(pair_emb)
                pair_tiled = pair_emb.tile([1, s_emb.shape[1], 1, 1])       # 차원 맞춰주려고 N번 반복

        # action 임베딩
        a_emb = self.action_emb_layer(actions)
        x_emb = self.x_emb_layer(selection[:, :, 0])
        y_emb = self.y_emb_layer(selection[:, :, 1])
        h_emb = self.h_emb_layer(selection[:, :, 2])
        w_emb = self.w_emb_layer(selection[:, :, 3])

        # action/selection embedding이 3D일 경우 4D로 변환 (s_emb와 맞추기 위해)
        if a_emb.ndim == 3:
            a_emb = a_emb.unsqueeze(2)
        if x_emb.ndim == 3:
            x_emb = x_emb.unsqueeze(2)
            y_emb = y_emb.unsqueeze(2)
            h_emb = h_emb.unsqueeze(2)
            w_emb = w_emb.unsqueeze(2)

        if self.use_split_pair_trajectory_encoding:
            if not self.disable_pair_encoding:
                if self.use_enhanced_pair_encoding and self.enhanced_layers:
                    pair_tokens = []
                    for i in range(3):
                        input_grid = pair_in[:, i:i+1, :, :]
                        output_grid = pair_out[:, i:i+1, :, :]

                        input_emb = self.enhanced_layers['pair_input_emb_layer'](input_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
                        input_emb = input_emb.reshape(input_grid.shape[0], -1)

                        output_emb = self.enhanced_layers['pair_output_emb_layer'](output_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
                        output_emb = output_emb.reshape(output_grid.shape[0], -1)

                        pair_token = torch.cat([input_emb, output_emb], dim=-1)
                        pair_token = self.enhanced_layers['pair_transform_layer'](pair_token)
                        pair_tokens.append(pair_token)

                    pair_seq = torch.stack(pair_tokens, dim=1)
                else:
                    pair_tokens = []
                    for i in range(3):
                        input_grid = pair_in[:, i:i+1, :, :]
                        output_grid = pair_out[:, i:i+1, :, :]

                        input_emb = self.state_emb_layer(input_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
                        input_emb = input_emb.reshape(input_grid.shape[0], -1)

                        output_emb = self.state_emb_layer(output_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
                        output_emb = output_emb.reshape(output_grid.shape[0], -1)

                        pair_token = torch.cat([input_emb, output_emb], dim=-1)
                        pair_token = self.pair_concat_proj(pair_token)
                        pair_tokens.append(pair_token)

                    pair_seq = torch.stack(pair_tokens, dim=1)

                attn_output, _ = self.pair_attention(pair_seq, pair_seq, pair_seq)
                pair_feats = attn_output.mean(dim=1, keepdim=True)
                pair_feats = self.pair_proj(pair_feats)

                pair_z_mean = self.pair_mean_layer(pair_feats)
                pair_z_sig = self.pair_sig_layer(pair_feats)
            else:
                half_z_dim = self.z_dim // 2
                pair_z_mean = torch.zeros(s_emb.shape[0], 1, half_z_dim).to(s_emb.device)
                pair_z_sig = torch.ones(s_emb.shape[0], 1, half_z_dim).to(s_emb.device)

            traj_input = torch.cat([s_emb, clip_emb, in_grid_tiled, a_emb, x_emb, y_emb, h_emb, w_emb], dim=-2)
            traj_input_shape = traj_input.shape
            traj_input = traj_input.view(traj_input_shape[0], -1, traj_input_shape[-1]).contiguous()

            # Add concept embedding if provided and enabled
            if self.use_concept_in_encoder and concept_emb is not None:
                # Project concept embedding to h_dim and add as first token
                concept_token = self.concept_proj(concept_emb).unsqueeze(1)  # (batch, 1, h_dim)
                traj_input = torch.cat([concept_token, traj_input], dim=1)  # Prepend concept token

            traj_feats, _ = self.traj_rnn(traj_input)
            traj_hn = traj_feats[:,-1:,:]
            traj_z_mean = self.traj_mean_layer(traj_hn)
            traj_z_sig = self.traj_sig_layer(traj_hn)

            z_mean = torch.cat([pair_z_mean, traj_z_mean], dim=-1)
            z_sig = torch.cat([pair_z_sig, traj_z_sig], dim=-1)

            if self.normalize_latent:
                z_mean = z_mean/torch.norm(z_mean, dim=-1).unsqueeze(-1)
        else:
            # through rnn - conditionally include pair information
            if not self.disable_pair_encoding:
                s_emb_a = torch.cat([s_emb, clip_emb, in_grid_tiled, pair_tiled, a_emb, x_emb, y_emb, h_emb, w_emb], dim=-2)
            else:
                s_emb_a = torch.cat([s_emb, clip_emb, in_grid_tiled, a_emb, x_emb, y_emb, h_emb, w_emb], dim=-2)
            s_emb_a_shape = s_emb_a.shape
            s_emb_a = s_emb_a.view(s_emb_a_shape[0], -1, s_emb_a_shape[-1]).contiguous()

            # Add concept embedding if provided and enabled
            if self.use_concept_in_encoder and concept_emb is not None:
                # Project concept embedding to h_dim and add as first token
                concept_token = self.concept_proj(concept_emb).unsqueeze(1)  # (batch, 1, h_dim)
                s_emb_a = torch.cat([concept_token, s_emb_a], dim=1)  # Prepend concept token

            feats, _ = self.rnn(s_emb_a)
            hn = feats[:,-1:,:]
            z_mean = self.mean_layer(hn)
            z_sig = self.sig_layer(hn)

            if self.normalize_latent:
                z_mean = z_mean/torch.norm(z_mean, dim=-1).unsqueeze(-1)

        return z_mean, z_sig


class Decoder(nn.Module):
    '''
    Decoder module.
    Decoder takes states, actions, and a sampled z and outputs parameters of P(s_T|s_0,z) and P(a_t|s_t,z) for all t in {0,...,T}
    P(s_T|s_0,z) is our "abstract dynamics model", because it predicts the resulting state transition over T timesteps given a skill 
    (so similar to regular dynamics model, but in skill space and also temporally extended)
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    We can try the following architecture:
    -embed z
    -Pass into fully connected network to get "state T features"
    '''

    def __init__(self,state_dim,a_dim,z_dim,h_dim,a_dist,fixed_sig,state_decoder_type,policy_decoder_type,per_element_sigma,
                 action_num,max_grid_size,state_emb_layer,pair_emb_layer,use_enhanced_pair_encoding=False,enhanced_layers=None,disable_pair_encoding=False):

        super(Decoder,self).__init__()

        self.state_emb_layer = state_emb_layer
        self.pair_emb_layer = pair_emb_layer
        self.use_enhanced_pair_encoding = use_enhanced_pair_encoding
        self.enhanced_layers = enhanced_layers
        self.max_grid_size = max_grid_size

        print('in decoder a_dist: ', a_dist)
        self.state_dim = state_dim
        self.a_dim = a_dim
        self.z_dim = z_dim

        if state_decoder_type == 'mlp':
            # Use old AbstractDynamics (simple MLP, no ARC context)
            self.abstract_dynamics = AbstractDynamics(state_dim,z_dim,h_dim,per_element_sigma=per_element_sigma)
        elif state_decoder_type == 'arc_residual':
            # Use new ARCAbstractDynamics (with residual connection and full ARC context)
            self.abstract_dynamics = ARCAbstractDynamics(
                z_dim=z_dim,
                h_dim=h_dim,
                max_grid_size=max_grid_size,
                state_emb_layer=self.state_emb_layer,
                pair_emb_layer=self.pair_emb_layer,
                use_enhanced_pair_encoding=self.use_enhanced_pair_encoding,
                enhanced_layers=self.enhanced_layers,
                disable_pair_encoding=disable_pair_encoding,
                predict_residual=True
            )
        elif state_decoder_type == 'arc_direct':
            # Use ARCAbstractDynamics without residual (direct prediction)
            self.abstract_dynamics = ARCAbstractDynamics(
                z_dim=z_dim,
                h_dim=h_dim,
                max_grid_size=max_grid_size,
                state_emb_layer=self.state_emb_layer,
                pair_emb_layer=self.pair_emb_layer,
                use_enhanced_pair_encoding=self.use_enhanced_pair_encoding,
                enhanced_layers=self.enhanced_layers,
                disable_pair_encoding=disable_pair_encoding,
                predict_residual=False
            )
        elif state_decoder_type == 'autoregressive':
            self.abstract_dynamics = AutoregressiveStateDecoder(state_dim,z_dim,h_dim)

        if policy_decoder_type == 'mlp':
            self.ll_policy = LowLevelPolicy(state_dim, a_dim, z_dim, h_dim, a_dist, action_num, max_grid_size, fixed_sig=fixed_sig,
                                            state_emb_layer=self.state_emb_layer, pair_emb_layer=self.pair_emb_layer,
                                            use_enhanced_pair_encoding=self.use_enhanced_pair_encoding, enhanced_layers=self.enhanced_layers,
                                            disable_pair_encoding=disable_pair_encoding)
        elif policy_decoder_type == 'autoregressive':
            self.ll_policy = AutoregressiveLowLevelPolicy(state_dim,a_dim,z_dim,h_dim,a_dist=a_dist,fixed_sig=None)

        # self.emb_layer  = nn.Linear(state_dim+z_dim,h_dim)
        # self.fc = nn.Sequential(nn.Linear(state_dim+z_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())

        self.state_decoder_type = state_decoder_type
        self.policy_decoder_type = policy_decoder_type
        self.a_dist = a_dist

    def forward(self, states, clip, in_grid, actions, selection, z, pair_in, pair_out, state_decoder):

        '''
        INPUTS:
            states: batch_size x T x state_dim state sequence tensor
            z:      batch_size x 1 x z_dim sampled z/skill variable
        OUTPUTS:
            sT_mean: batch_size x 1 x state_dim tensor of terminal (time=T) state means
            sT_sig:  batch_size x 1 x state_dim tensor of terminal (time=T) state standard devs
            a_mean: batch_size x T x a_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  batch_size x T x a_dim tensor of action standard devs for each t in {0.,,,.T}
        '''
        # state decoder 쓰는 경우
        s_0 = states[:,0:1,:]
        clip_0 = clip[:,0:1,:]

        # MLP로 구현
        a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig = self.ll_policy(states, clip, in_grid, z, pair_in, pair_out)

        if state_decoder:
            s_T = states[:,-1:,:]
            if self.state_decoder_type == 'autoregressive':
                sT_mean, sT_sig = self.abstract_dynamics(s_0, s_T, z.detach())
                sT_logits = None  # autoregressive doesn't have color classifier
            elif self.state_decoder_type == 'mlp':
                # Old simple MLP decoder (no ARC context)
                sT_mean, sT_sig = self.abstract_dynamics(s_0, z.detach())
                sT_logits = None  # old MLP doesn't have color classifier
            elif self.state_decoder_type in ['arc_residual', 'arc_direct']:
                # New ARC-specific decoder (with full context and color classifier)
                sT_mean, sT_sig, sT_logits = self.abstract_dynamics(s_0, clip_0, in_grid, z.detach(), pair_in, pair_out)
            return sT_mean, sT_sig, sT_logits, a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig

        else:
            return a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig


class Prior(nn.Module):
    '''
    Decoder module.
    Decoder takes states, actions, and a sampled z and outputs parameters of P(s_T|s_0,z) and P(a_t|s_t,z) for all t in {0,...,T}
    P(s_T|s_0,z) is our "abstract dynamics model", because it predicts the resulting state transition over T timesteps given a skill 
    (so similar to regular dynamics model, but in skill space and also temporally extended)
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    We can try the following architecture:
    -embed z
    -Pass into fully connected network to get "state T features"
    '''

    def __init__(self,state_dim,z_dim,h_dim,goal_conditioned=False,goal_dim=2,max_grid_size=30,state_emb_layer=None,pair_emb_layer=None,use_enhanced_pair_encoding=False,enhanced_layers=None,disable_pair_encoding=False,use_concept_in_prior=False,concept_scale=1.0):

        super(Prior,self).__init__()

        self.state_dim = state_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.goal_conditioned = goal_conditioned
        self.disable_pair_encoding = disable_pair_encoding
        self.use_concept_in_prior = use_concept_in_prior
        self.concept_scale = concept_scale
        if(self.goal_conditioned):
            self.goal_dim = goal_dim
        else:
            self.goal_dim = 0

        # Adjust input dimension based on whether pair encoding is disabled and concept guidance
        input_dim = state_dim+state_dim+state_dim+self.goal_dim  # state, clip, in_grid
        if not disable_pair_encoding:
            input_dim += h_dim  # add pair dimension
        if use_concept_in_prior:
            input_dim += z_dim  # add concept embedding dimension (same as z_dim)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        #self.mean_layer = nn.Linear(h_dim,z_dim)
        self.mean_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim))
        #self.sig_layer  = nn.Sequential(nn.Linear(h_dim,z_dim),nn.Softplus())
        self.sig_layer  = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim),nn.Softplus())

        self.state_emb_layer = state_emb_layer
        self.pair_emb_layer = pair_emb_layer
        self.max_grid_size = max_grid_size
        self.use_enhanced_pair_encoding = use_enhanced_pair_encoding
        self.enhanced_layers = enhanced_layers

        # Concept embedding projection layer (to match z_dim)
        if use_concept_in_prior:
            self.concept_proj = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.ReLU(),
                nn.Linear(z_dim, z_dim)
            )

    def forward(self, s0, clip0, in_grid, pair_in, pair_out, goal=None, concept_emb=None):

        '''
        INPUTS:
            states: batch_size x T x state_dim state sequence tensor
            concept_emb: batch_size x z_dim concept embedding (optional)

        OUTPUTS:
            z_mean: batch_size x 1 x state_dim tensor of z means
            z_sig:  batch_size x 1 x state_dim tensor of z standard devs

        '''
        # state 임베딩
        s0_reshaped = s0.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous()
        s_emb = self.state_emb_layer(s0_reshaped)    # (batch * block_size, h_dim)
        s_emb = s_emb.reshape(s0.shape[0], s0.shape[1], -1) # Use -1 to infer actual embedding dimension
        if(self.goal_conditioned):
            s_emb = torch.cat([s_emb, goal],dim=-1)
            
        # clip 임베딩
        clip0_reshaped = clip0.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous()
        clip_emb = self.state_emb_layer(clip0_reshaped)
        clip_emb = clip_emb.reshape(clip0.shape[0], clip0.shape[1], -1)
        
        # in_grid 임베딩
        in_grid_reshaped = in_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous()
        in_grid_emb = self.state_emb_layer(in_grid_reshaped)
        in_grid_emb = in_grid_emb.reshape(in_grid.shape[0], in_grid.shape[1], -1)

        # input-output pair 임베딩 - Only if pair encoding is not disabled
        if not self.disable_pair_encoding:
            if self.use_enhanced_pair_encoding and self.enhanced_layers:
                # Enhanced: Process each input-output pair separately and learn transformations
                pair_transforms = []
                for i in range(3):  # 3 example pairs
                    input_grid = pair_in[:, i:i+1, :, :]  # (batch, 1, max_grid_size, max_grid_size)
                    output_grid = pair_out[:, i:i+1, :, :]

                    # Individual embeddings
                    input_emb = self.enhanced_layers['pair_input_emb_layer'](input_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())
                    output_emb = self.enhanced_layers['pair_output_emb_layer'](output_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous())

                    # Learn input->output transformation
                    pair_transform = self.enhanced_layers['pair_transform_layer'](torch.cat([input_emb, output_emb], dim=-1))
                    pair_transforms.append(pair_transform)

                # Combine all pair transformations
                combined_pairs = torch.stack(pair_transforms, dim=1)  # (batch, 3, h_dim)
                combined_pairs = combined_pairs.reshape(combined_pairs.shape[0], -1)  # (batch, 3*h_dim)
                pair_emb = self.enhanced_layers['enhanced_pair_combiner'](combined_pairs)  # (batch, h_dim)
                pair_emb = pair_emb.reshape(pair_emb.shape[0], 1, self.h_dim)
                pair_tiled = pair_emb.tile([1, s_emb.shape[1], 1])
            else:
                # Original method: concatenate all pairs
                pair = torch.cat([pair_in, pair_out], dim=1)
                pair_shape = pair.shape
                # print("Prior - pair shape : {0}".format(pair_shape))
                pair_reshaped = pair.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous()
                pair_emb = self.state_emb_layer(pair_reshaped)
                # print("Prior - Before pair_emb : {0}".format(pair_emb.shape))
                pair_emb = pair_emb.reshape(pair_shape[0], 1, -1)  # Use -1 to infer dimension
                # print("Prior - After pair_emb : {0}".format(pair_emb.shape))

                # Handle dimension mismatch by projecting to expected size
                expected_dim = 6 * self.h_dim  # 6 * 512 = 3072
                actual_dim = pair_emb.shape[-1]
                if actual_dim != expected_dim:
                    # Create a simple projection layer if not exists
                    if not hasattr(self, 'pair_projection'):
                        self.pair_projection = nn.Linear(actual_dim, expected_dim).to(pair_emb.device)
                    pair_emb = self.pair_projection(pair_emb)

                pair_emb = self.pair_emb_layer(pair_emb)
                pair_tiled = pair_emb.tile([1, s_emb.shape[1], 1])       # 차원 맞춰주려고 N번 반복

        # Ensure all embeddings have the expected dimension (h_dim)
        if s_emb.shape[-1] != self.h_dim:
            if not hasattr(self, 's_projection'):
                self.s_projection = nn.Linear(s_emb.shape[-1], self.h_dim).to(s_emb.device)
            s_emb = self.s_projection(s_emb)

        if clip_emb.shape[-1] != self.h_dim:
            if not hasattr(self, 'clip_projection'):
                self.clip_projection = nn.Linear(clip_emb.shape[-1], self.h_dim).to(clip_emb.device)
            clip_emb = self.clip_projection(clip_emb)

        if in_grid_emb.shape[-1] != self.h_dim:
            if not hasattr(self, 'in_grid_projection'):
                self.in_grid_projection = nn.Linear(in_grid_emb.shape[-1], self.h_dim).to(in_grid_emb.device)
            in_grid_emb = self.in_grid_projection(in_grid_emb)

        # Process concept embedding if provided
        if self.use_concept_in_prior and concept_emb is not None:
            # Project concept embedding
            concept_processed = self.concept_proj(concept_emb)
            # Scale concept embedding
            concept_processed = concept_processed * self.concept_scale
            # Expand to match batch and time dimensions
            if len(concept_processed.shape) == 2:  # (batch, z_dim)
                concept_processed = concept_processed.unsqueeze(1)  # (batch, 1, z_dim)
            # Tile to match time dimension
            concept_tiled = concept_processed.tile([1, s_emb.shape[1], 1])

        # Conditionally include pair information and concept
        if not self.disable_pair_encoding:
            if pair_tiled.shape[-1] != self.h_dim:
                if not hasattr(self, 'pair_tiled_projection'):
                    self.pair_tiled_projection = nn.Linear(pair_tiled.shape[-1], self.h_dim).to(pair_tiled.device)
                pair_tiled = self.pair_tiled_projection(pair_tiled)
            if self.use_concept_in_prior and concept_emb is not None:
                s_emb = torch.cat([s_emb, clip_emb, in_grid_emb, pair_tiled, concept_tiled], dim=-1)
            else:
                s_emb = torch.cat([s_emb, clip_emb, in_grid_emb, pair_tiled], dim=-1)
        else:
            if self.use_concept_in_prior and concept_emb is not None:
                s_emb = torch.cat([s_emb, clip_emb, in_grid_emb, concept_tiled], dim=-1)
            else:
                s_emb = torch.cat([s_emb, clip_emb, in_grid_emb], dim=-1)

        feats = self.layers(s_emb)
        # get mean and stand dev of action distribution
        z_mean = self.mean_layer(feats)
        z_sig  = self.sig_layer(feats)

        return z_mean, z_sig

    def get_loss(self, states, clip, in_grid, actions, pair_in, pair_out, goal=None):
        '''
        To be used only for low level action Prior training
        '''
        a_mean, a_sig = self.forward(states, clip, in_grid, pair_in, pair_out, goal)

        a_dist = Normal.Normal(a_mean, a_sig)
        return - torch.mean(a_dist.log_prob(actions))


class ARCDirectOutputPredictor(nn.Module):
    '''
    Direct output predictor: P(output|input, pair_in, pair_out)
    Predicts the output grid directly from input grid and example pairs, without skill latent z.
    This tests if the model can learn task transformation from examples alone.
    '''

    def __init__(self, h_dim, max_grid_size=10,
                 state_emb_layer=None, pair_emb_layer=None,
                 use_enhanced_pair_encoding=False, enhanced_layers=None):

        super(ARCDirectOutputPredictor, self).__init__()

        self.max_grid_size = max_grid_size
        self.h_dim = h_dim
        self.state_dim = max_grid_size * max_grid_size

        # Reuse existing embedding layers
        self.state_emb_layer = state_emb_layer
        self.pair_emb_layer = pair_emb_layer
        self.use_enhanced_pair_encoding = use_enhanced_pair_encoding
        self.enhanced_layers = enhanced_layers

        # Input dimension: input_emb + pair_emb
        input_dim = h_dim + h_dim

        # Feature extraction layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )

        # Output predictor
        self.output_predictor = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, self.state_dim * 11)  # 11 color classes per cell
        )

    def forward(self, input_grid, pair_in, pair_out):
        '''
        INPUTS:
            input_grid: batch_size x 1 x state_dim - test input grid (flattened)
            pair_in: batch_size x 3 x max_grid_size x max_grid_size - input examples
            pair_out: batch_size x 3 x max_grid_size x max_grid_size - output examples
        OUTPUTS:
            output_logits: batch_size x 1 x (state_dim * 11) - predicted output logits
        '''
        batch_size = input_grid.shape[0]

        # Embed input grid
        input_emb = self.state_emb_layer(
            input_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous()
        )
        input_emb = input_emb.reshape(batch_size, 1, self.h_dim)

        # Encode input-output pairs
        if self.use_enhanced_pair_encoding and self.enhanced_layers:
            # Enhanced pair encoding
            pair_transforms = []
            for i in range(3):
                pair_input = pair_in[:, i:i+1, :, :]
                pair_output = pair_out[:, i:i+1, :, :]

                input_enc = self.enhanced_layers['pair_input_emb_layer'](
                    pair_input.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous()
                )
                output_enc = self.enhanced_layers['pair_output_emb_layer'](
                    pair_output.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous()
                )

                pair_transform = self.enhanced_layers['pair_transform_layer'](
                    torch.cat([input_enc, output_enc], dim=-1)
                )
                pair_transforms.append(pair_transform)

            combined_pairs = torch.stack(pair_transforms, dim=1)
            combined_pairs = combined_pairs.reshape(batch_size, -1)
            pair_emb = self.enhanced_layers['enhanced_pair_combiner'](combined_pairs)
            pair_emb = pair_emb.reshape(batch_size, 1, self.h_dim)
        else:
            # Original pair encoding
            pair = torch.cat([pair_in, pair_out], dim=1)
            pair_shape = pair.shape

            pair_emb = self.state_emb_layer(
                pair.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous()
            )
            pair_emb = pair_emb.reshape(pair_shape[0], 1, pair_shape[1] * self.h_dim)
            pair_emb = self.pair_emb_layer(pair_emb)

        # Concatenate input and pair embeddings
        combined = torch.cat([input_emb, pair_emb], dim=-1)

        # Extract features
        feats = self.layers(combined)

        # Predict output logits
        output_logits = self.output_predictor(feats)

        return output_logits


class GenerativeModel(nn.Module):

    def __init__(self, decoder, prior):
        super().__init__()
        self.decoder = decoder
        self.prior = prior

    def forward(self):
        pass


class SkillModel(nn.Module):
    def __init__(self,state_dim,a_dim,z_dim,h_dim,horizon,a_dist='normal',beta=1.0,fixed_sig=None,encoder_type='gru',state_decoder_type='mlp',policy_decoder_type='mlp',
                 per_element_sigma=True,conditional_prior=True,train_diffusion_prior=False,normalize_latent=False,
                 color_num=11,action_num=36,max_grid_size=30,diffusion_steps=100, use_in_out=False ,diffusion_scale=1.0, use_enhanced_pair_encoding=False, disable_pair_encoding=False, use_shared_grid_embedding=False, use_split_pair_trajectory_encoding=False, use_direct_output_predictor=False, use_direct_output_for_diffusion=False, use_positional_encoding=False, use_concept_guidance=False, use_cfg_for_concept=True, use_concept_in_encoder=False, num_concepts=0, use_concept_in_prior=False, concept_scale=1.0):
        super(SkillModel, self).__init__()

        self.state_dim = state_dim # state dimension
        self.a_dim = a_dim # action dimension
        self.z_dim = z_dim
        self.encoder_type = encoder_type
        self.state_decoder_type = state_decoder_type
        self.policy_decoder_type = policy_decoder_type
        self.conditional_prior = conditional_prior
        self.train_diffusion_prior = train_diffusion_prior
        self.diffusion_prior = None
        self.a_dist = a_dist
        self.normalize_latent = normalize_latent
        self.max_grid_size = max_grid_size
        self.diffusion_scale = diffusion_scale
        self.use_in_out = use_in_out
        self.use_enhanced_pair_encoding = use_enhanced_pair_encoding
        self.disable_pair_encoding = disable_pair_encoding
        self.use_shared_grid_embedding = use_shared_grid_embedding
        self.use_split_pair_trajectory_encoding = use_split_pair_trajectory_encoding
        self.use_positional_encoding = use_positional_encoding
        self.use_concept_guidance = use_concept_guidance
        self.use_cfg_for_concept = use_cfg_for_concept
        self.use_concept_in_encoder = use_concept_in_encoder
        self.num_concepts = num_concepts
        self.use_concept_in_prior = use_concept_in_prior
        self.concept_scale = concept_scale

        # Create base CNN layers
        base_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            # nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        # Create flatten and linear layers
        flatten_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*self.max_grid_size*self.max_grid_size, h_dim),
            # nn.Flatten(), nn.Linear(32*28*28, h_dim),
            nn.Tanh()
        )

        # Wrap with positional encoding support
        self.state_emb_layer = StateEmbeddingWithPositionalEncoding(
            base_cnn_layers=base_cnn,
            flatten_and_linear=flatten_linear,
            use_positional_encoding=use_positional_encoding,
            pos_encoding_channels=32,
            max_grid_size=max_grid_size
        )
        self.pair_emb_layer = nn.Sequential(
            nn.Linear(6*h_dim, h_dim),  # input-output pair 총 3개씩
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )

        # Enhanced pair encoding: process each input-output pair individually
        if self.use_enhanced_pair_encoding:
            if getattr(self, 'use_shared_grid_embedding', False):
                # Use the same grid embedding layer for consistency
                self.pair_input_emb_layer = self.state_emb_layer
                self.pair_output_emb_layer = self.state_emb_layer
            else:
                # Individual grid embedding for pair processing (original)
                # Create pair input embedding with positional encoding support
                pair_input_cnn = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                )
                pair_input_flatten_linear = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(32*self.max_grid_size*self.max_grid_size, h_dim),
                    nn.Tanh()
                )
                self.pair_input_emb_layer = StateEmbeddingWithPositionalEncoding(
                    base_cnn_layers=pair_input_cnn,
                    flatten_and_linear=pair_input_flatten_linear,
                    use_positional_encoding=use_positional_encoding,
                    pos_encoding_channels=32,
                    max_grid_size=max_grid_size
                )

                # Create pair output embedding with positional encoding support
                pair_output_cnn = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                )
                pair_output_flatten_linear = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(32*self.max_grid_size*self.max_grid_size, h_dim),
                    nn.Tanh()
                )
                self.pair_output_emb_layer = StateEmbeddingWithPositionalEncoding(
                    base_cnn_layers=pair_output_cnn,
                    flatten_and_linear=pair_output_flatten_linear,
                    use_positional_encoding=use_positional_encoding,
                    pos_encoding_channels=32,
                    max_grid_size=max_grid_size
                )
            # Transformation encoding: learn input->output transformation
            self.pair_transform_layer = nn.Sequential(
                nn.Linear(2*h_dim, h_dim),  # input + output embeddings
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.Tanh()
            )
            # Combine multiple pairs
            self.enhanced_pair_combiner = nn.Sequential(
                nn.Linear(3*h_dim, h_dim),  # 3 pairs each with h_dim
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU()
            )

        # Prepare enhanced_layers for encoder (shared between GRU and Transformer)
        enhanced_layers = None
        if self.use_enhanced_pair_encoding:
            enhanced_layers = {
                'pair_input_emb_layer': self.pair_input_emb_layer,
                'pair_output_emb_layer': self.pair_output_emb_layer,
                'pair_transform_layer': self.pair_transform_layer,
                'enhanced_pair_combiner': self.enhanced_pair_combiner
            }

        if encoder_type == 'gru':
            self.encoder = GRUEncoder(state_dim,a_dim,z_dim,h_dim,normalize_latent=normalize_latent,
                                      color_num=color_num,action_num=action_num,max_grid_size=max_grid_size,
                                      state_emb_layer=self.state_emb_layer,
                                      pair_emb_layer=self.pair_emb_layer,
                                      use_enhanced_pair_encoding=self.use_enhanced_pair_encoding,
                                      enhanced_layers=enhanced_layers,
                                      disable_pair_encoding=self.disable_pair_encoding,
                                      use_split_pair_trajectory_encoding=self.use_split_pair_trajectory_encoding,
                                      use_concept_in_encoder=self.use_concept_in_encoder
                                      )
        elif encoder_type == 'transformer':
            self.encoder = TransformEncoder(state_dim, a_dim, z_dim, h_dim,
                                      horizon=horizon,
                                      n_transformer_layers=4, n_heads=8, dropout=0.1,
                                      normalize_latent=normalize_latent,
                                      color_num=11, action_num=action_num, max_grid_size=max_grid_size,
                                      state_emb_layer=self.state_emb_layer, pair_emb_layer=self.pair_emb_layer,
                                      use_enhanced_pair_encoding=self.use_enhanced_pair_encoding,
                                      enhanced_layers=enhanced_layers,
                                      disable_pair_encoding=self.disable_pair_encoding,
                                      use_split_pair_trajectory_encoding=self.use_split_pair_trajectory_encoding,
                                      use_concept_in_encoder=self.use_concept_in_encoder
                                      )

        # Concept classifier for auxiliary task (helps ensure z encodes concept)
        if self.use_concept_in_encoder and self.num_concepts > 0:
            self.concept_classifier = nn.Sequential(
                nn.Linear(z_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(h_dim, h_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(h_dim // 2, self.num_concepts)
            )
            print(f"Initialized concept classifier with {self.num_concepts} classes")
        else:
            self.concept_classifier = None

        enhanced_layers_for_decoder = None
        if self.use_enhanced_pair_encoding:
            enhanced_layers_for_decoder = {
                'pair_input_emb_layer': self.pair_input_emb_layer,
                'pair_output_emb_layer': self.pair_output_emb_layer,
                'pair_transform_layer': self.pair_transform_layer,
                'enhanced_pair_combiner': self.enhanced_pair_combiner
            }
        self.decoder = Decoder(state_dim,a_dim,z_dim,h_dim, a_dist, fixed_sig=fixed_sig,state_decoder_type=state_decoder_type,
                               policy_decoder_type=policy_decoder_type,per_element_sigma=per_element_sigma,
                               action_num=action_num, max_grid_size=max_grid_size, state_emb_layer=self.state_emb_layer, pair_emb_layer=self.pair_emb_layer,
                               use_enhanced_pair_encoding=self.use_enhanced_pair_encoding, enhanced_layers=enhanced_layers_for_decoder,
                               disable_pair_encoding=self.disable_pair_encoding)
        
        if conditional_prior:
            enhanced_layers = None
            if self.use_enhanced_pair_encoding:
                enhanced_layers = {
                    'pair_input_emb_layer': self.pair_input_emb_layer,
                    'pair_output_emb_layer': self.pair_output_emb_layer,
                    'pair_transform_layer': self.pair_transform_layer,
                    'enhanced_pair_combiner': self.enhanced_pair_combiner
                }
            self.prior = Prior(state_dim, z_dim, h_dim, state_emb_layer=self.state_emb_layer, pair_emb_layer=self.pair_emb_layer,max_grid_size=max_grid_size,
                              use_enhanced_pair_encoding=self.use_enhanced_pair_encoding, enhanced_layers=enhanced_layers,
                              disable_pair_encoding=self.disable_pair_encoding,
                              use_concept_in_prior=self.use_concept_in_prior,
                              concept_scale=self.concept_scale)
            self.gen_model = GenerativeModel(self.decoder, self.prior)
            
        if self.train_diffusion_prior:
            nn_model = Model_mlp(
                x_shape = state_dim,
                n_hidden = 512,
                y_dim = z_dim,
                embed_dim = 128,
                net_type ='unet',
                max_grid_size=self.max_grid_size,
                use_in_out=self.use_in_out,
                use_enhanced_pair_encoding=getattr(self, 'use_enhanced_pair_encoding', False),
                use_concept_guidance=getattr(self, 'use_concept_guidance', False),
                use_cfg_for_concept=getattr(self, 'use_cfg_for_concept', True),
                # net_type ='transformer'
            ).to('cuda')
            self.diffusion_prior = Model_Cond_Diffusion(
                nn_model,
                betas=(1e-4, 0.02),
                n_T=diffusion_steps,    # 원래 100
                device='cuda',
                x_dim=state_dim,
                y_dim=z_dim,
                drop_prob=0.0,
                guide_w=0.0,
                use_in_out=self.use_in_out,
            )

        # Direct output predictor (optional)
        self.use_direct_output_predictor = use_direct_output_predictor
        self.use_direct_output_for_diffusion = use_direct_output_for_diffusion
        self.direct_output_predictor = None
        if self.use_direct_output_predictor:
            enhanced_layers_for_predictor = None
            if self.use_enhanced_pair_encoding:
                enhanced_layers_for_predictor = {
                    'pair_input_emb_layer': self.pair_input_emb_layer,
                    'pair_output_emb_layer': self.pair_output_emb_layer,
                    'pair_transform_layer': self.pair_transform_layer,
                    'enhanced_pair_combiner': self.enhanced_pair_combiner
                }
            self.direct_output_predictor = ARCDirectOutputPredictor(
                h_dim=h_dim,
                max_grid_size=max_grid_size,
                state_emb_layer=self.state_emb_layer,
                pair_emb_layer=self.pair_emb_layer,
                use_enhanced_pair_encoding=self.use_enhanced_pair_encoding,
                enhanced_layers=enhanced_layers_for_predictor
            )

        self.beta = beta

    def forward(self, states, clip, in_grid, actions, selection, pair_in, pair_out, state_decoder, concept_emb=None, concept_scale=1.0):

        '''
        Takes states and actions, returns the distributions necessary for computing the objective function
        INPUTS:
            states: batch_size x T x state_dim state sequence tensor
            actions: batch_size x T x a_dim action sequence tensor
            concept_emb: (optional) batch_size x concept_dim tensor for concept conditioning
            concept_scale: (optional) scale factor for concept embeddings (default 1.0)
        OUTPUTS:
            s_T_mean:     batch_size x 1 x state_dim tensor of means of "decoder" distribution over terminal states
            S_T_sig:      batch_size x 1 x state_dim tensor of standard devs of "decoder" distribution over terminal states
            a_means:      batch_size x T x a_dim tensor of means of "decoder" distribution over actions
            a_sigs:       batch_size x T x a_dim tensor of stand devs
            z_post_means: batch_size x 1 x z_dim tensor of means of z posterior distribution
            z_post_sigs:  batch_size x 1 x z_dim tensor of stand devs of z posterior distribution
        '''

        # Scale concept embedding if provided
        scaled_concept_emb = None
        if concept_emb is not None and concept_scale != 1.0:
            scaled_concept_emb = concept_emb * concept_scale
        else:
            scaled_concept_emb = concept_emb

        # STEP 1. Encode states and actions to get posterior over z
        z_post_means, z_post_sigs = self.encoder(states, clip, in_grid, actions, selection, pair_in, pair_out, concept_emb=scaled_concept_emb)        
        
        # STEP 2. sample z from posterior
        if not self.normalize_latent: 
            z_sampled = self.reparameterize(z_post_means, z_post_sigs)
        else:
            z_sampled = z_post_means

        # STEP 3. Pass z_sampled and states through decoder
        if state_decoder:
            s_T_mean, s_T_sig, s_T_logits, a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig = self.decoder(states, clip, in_grid, actions, selection, z_sampled, pair_in, pair_out, state_decoder)
            return s_T_mean, s_T_sig, s_T_logits, a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig, z_post_means, z_post_sigs, z_sampled
        else:
            a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig = self.decoder(states, clip, in_grid, actions, selection, z_sampled, pair_in, pair_out, state_decoder)
            return a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig, z_post_means, z_post_sigs, z_sampled

    def get_losses(self, states, s_T, clip, in_grid, actions, selection, pair_in, pair_out, state_decoder, concept_emb=None, concept_labels=None, concept_scale=1.0, concept_loss_weight=0.0, concept_contrastive_weight=0.0, contrastive_temperature=0.1, use_concept_for_diffusion=False):
        '''
        Computes various components of the loss:
        L = E_q [log P(s_T|s_0,z)]
          + E_q [sum_t=0^T P(a_t|s_t,z)]
          - D_kl(q(z|s_0,...,s_T,a_0,...,a_T)||P(z_0|s_0))

        Additional concept-related inputs:
          concept_emb: (optional) batch_size x concept_dim tensor for concept conditioning
          concept_labels: (optional) batch_size tensor of concept IDs for classification loss
          concept_scale: (optional) scale factor for concept embeddings (default 1.0)
          concept_loss_weight: (optional) weight for concept classification loss (default 0.0)
        Distributions we need:
        '''
        T = states.shape[1]
        # loss terms corresponding to -logP(s_T|s_0,z) and -logP(a_t|s_t,z)

        if state_decoder:
            # s_T = states[:,-1:,:]
            s_T_mean, s_T_sig, s_T_logits, a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig, z_post_means, z_post_sigs, z_sampled = self.forward(states, clip, in_grid, actions, selection, pair_in, pair_out, state_decoder, concept_emb=concept_emb, concept_scale=concept_scale)

            # Compute state decoder loss using cross entropy for discrete color prediction
            if s_T_logits is not None and self.state_decoder_type in ['arc_residual', 'arc_direct']:
                # Reshape logits: (batch, 1, state_dim * 11) -> (batch, state_dim, 11)
                batch_size = s_T_logits.shape[0]
                s_T_logits_reshaped = s_T_logits.reshape(batch_size, self.max_grid_size * self.max_grid_size, 11)

                # Reshape targets: (batch, 1, state_dim) -> (batch, state_dim)
                s_T_targets = s_T.reshape(batch_size, self.max_grid_size * self.max_grid_size).long()

                # Cross entropy loss for color classification
                s_T_loss = F.cross_entropy(
                    s_T_logits_reshaped.reshape(-1, 11),  # (batch * state_dim, 11)
                    s_T_targets.reshape(-1),  # (batch * state_dim,)
                    reduction='mean'
                )
            else:
                # Fallback to Gaussian likelihood for older models
                s_T_dist = Normal.Normal(s_T_mean, s_T_sig)
                s_T_loss = -torch.mean(torch.sum(s_T_dist.log_prob(s_T), dim=-1)) / T
        else:
            a_mean, a_sig, x_mean, x_sig, y_mean, y_sig, h_mean, h_sig, w_mean, w_sig, z_post_means, z_post_sigs, z_sampled = self.forward(states, clip, in_grid, actions, selection, pair_in, pair_out, state_decoder, concept_emb=concept_emb, concept_scale=concept_scale)
        
        # 원래 있던 Loss 쓰는 경우 -> 카테고리화
        a_dist = torch.distributions.categorical.Categorical(a_mean)
        x_dist = torch.distributions.categorical.Categorical(x_mean)
        y_dist = torch.distributions.categorical.Categorical(y_mean)
        h_dist = torch.distributions.categorical.Categorical(h_mean)
        w_dist = torch.distributions.categorical.Categorical(w_mean)
        
        z_post_dist = Normal.Normal(z_post_means, z_post_sigs)
        
        if not self.normalize_latent:
            if self.conditional_prior:
                z_prior_means, z_prior_sigs = self.prior(states[:,0:1,:], clip[:,0:1,:], in_grid, pair_in, pair_out) 
                z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs) 
            else:
                z_prior_means = torch.zeros_like(z_post_means)
                z_prior_sigs = torch.ones_like(z_post_sigs)
                z_prior_dist = Normal.Normal(z_prior_means, z_prior_sigs)
        
        # 원래 있던 Loss 쓰는 경우
        # print(actions.shape, actions.squeeze(-1).shape, selection.shape, selection[:, :, 0].squeeze(-1).shape)
        a_loss = -torch.mean(torch.sum(a_dist.log_prob(actions.squeeze(-1)), dim=-1))
        x_loss = -torch.mean(torch.sum(x_dist.log_prob(selection[:, :, 0].squeeze(-1)), dim=-1))
        y_loss = -torch.mean(torch.sum(y_dist.log_prob(selection[:, :, 1].squeeze(-1)), dim=-1))
        h_loss = -torch.mean(torch.sum(h_dist.log_prob(selection[:, :, 2].squeeze(-1)), dim=-1))
        w_loss = -torch.mean(torch.sum(w_dist.log_prob(selection[:, :, 3].squeeze(-1)), dim=-1))
        
        # KL loss 계산
        if not self.normalize_latent:
            kl_loss = torch.mean(torch.sum(KL.kl_divergence(z_post_dist, z_prior_dist), dim=-1))/T 
        else:
            kl_loss = torch.tensor(0.0).cuda()

        # Diffusion prior loss 계산
        if self.train_diffusion_prior:
            # Calculate direct output embedding if needed (renamed to text_guide_emb)
            text_guide_emb_for_diffusion = None
            # Option 1: Use concept embedding directly for diffusion conditioning
            if use_concept_for_diffusion and concept_emb is not None:
                text_guide_emb_for_diffusion = concept_emb
            # Option 2: Use direct output for diffusion if concept features are enabled
            # (text_guide_proj expects concept embedding dimension)
            elif self.use_direct_output_for_diffusion and self.use_direct_output_predictor and self.use_concept_in_encoder:
                # Get direct output prediction
                with torch.no_grad():
                    direct_output_logits = self.direct_output_predictor(in_grid, pair_in, pair_out)
                    batch_size = direct_output_logits.shape[0]
                    grid_size = self.max_grid_size * self.max_grid_size
                    direct_logits_reshaped = direct_output_logits.reshape(batch_size, grid_size, 11)
                    # Get predicted output (argmax)
                    predicted_output = torch.argmax(direct_logits_reshaped, dim=-1)  # (batch, grid_size)
                    # Embed predicted output using state_emb_layer
                    predicted_output_grid = predicted_output.reshape(batch_size, 1, self.max_grid_size, self.max_grid_size).float()
                    text_guide_emb_for_diffusion = self.decoder.state_emb_layer(predicted_output_grid)

            # diffusion_loss = self.diffusion_prior.loss_on_batch(states[:, 0:1, :, :], clip[:, 0:1, :, :], in_grid, z_sampled[:, 0, :].detach(), predict_noise=0)
            diffusion_loss = self.diffusion_prior.loss_on_batch(
                states[:, 0:1, :, :], clip[:, 0:1, :, :], in_grid, pair_in, pair_out,
                z_sampled[:, 0, :].detach(), predict_noise=0,
                text_guide_emb=text_guide_emb_for_diffusion
            )
        else:
            diffusion_loss = 0.0

        # Direct output prediction loss (if enabled)
        direct_output_loss = 0.0
        if self.use_direct_output_predictor:
            # Use in_grid as test input, predict out_grid
            direct_output_logits = self.direct_output_predictor(in_grid, pair_in, pair_out)

            # Reshape logits and targets
            batch_size = direct_output_logits.shape[0]
            grid_size = self.max_grid_size * self.max_grid_size
            direct_logits_reshaped = direct_output_logits.reshape(batch_size, grid_size, 11)
            target_output = states[:, -1, :].reshape(batch_size, grid_size).long()

            # Cross entropy loss
            direct_output_loss = F.cross_entropy(
                direct_logits_reshaped.reshape(-1, 11),
                target_output.reshape(-1),
                reduction='mean'
            )

        # Concept classification loss (if enabled)
        concept_class_loss = 0.0
        if self.concept_classifier is not None and concept_labels is not None and concept_loss_weight > 0.0:
            # Use z_sampled to predict concept
            concept_logits = self.concept_classifier(z_sampled.squeeze(1))  # (batch, num_concepts)
            concept_class_loss = F.cross_entropy(concept_logits, concept_labels.long(), reduction='mean')
            concept_class_loss = concept_loss_weight * concept_class_loss

        # Concept contrastive loss (SupCon-style): pushes same-concept z's together, different-concept z's apart
        concept_contrastive_loss = 0.0
        if concept_labels is not None and concept_contrastive_weight > 0.0:
            # z_sampled shape: (batch, 1, z_dim) -> (batch, z_dim)
            z_for_contrastive = z_sampled.squeeze(1)  # (batch, z_dim)

            # L2 normalize for cosine similarity
            z_normalized = F.normalize(z_for_contrastive, p=2, dim=1)  # (batch, z_dim)

            # Compute similarity matrix
            similarity_matrix = torch.matmul(z_normalized, z_normalized.T) / contrastive_temperature  # (batch, batch)

            # Create mask for positive pairs (same concept)
            labels = concept_labels.long()
            batch_size = labels.shape[0]

            # mask[i,j] = 1 if labels[i] == labels[j], else 0
            positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # (batch, batch)

            # Remove diagonal (self-similarity)
            identity_mask = torch.eye(batch_size, device=z_normalized.device)
            positive_mask = positive_mask * (1 - identity_mask)

            # Count positive pairs per sample
            num_positives = positive_mask.sum(dim=1)  # (batch,)

            # For numerical stability, subtract max
            logits_max, _ = similarity_matrix.max(dim=1, keepdim=True)
            logits = similarity_matrix - logits_max.detach()

            # Compute log_softmax over all samples except self
            exp_logits = torch.exp(logits) * (1 - identity_mask)
            log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

            # Compute mean of log-likelihood over positive pairs
            # Only for samples that have at least one positive pair
            mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)

            # Only include samples with at least one positive pair
            valid_samples = (num_positives > 0).float()
            if valid_samples.sum() > 0:
                concept_contrastive_loss = -(mean_log_prob_pos * valid_samples).sum() / (valid_samples.sum() + 1e-8)
                concept_contrastive_loss = concept_contrastive_weight * concept_contrastive_loss
            else:
                concept_contrastive_loss = 0.0

        loss_tot = (a_loss + x_loss + y_loss + h_loss + w_loss) + self.beta * kl_loss + diffusion_loss + direct_output_loss + concept_class_loss + concept_contrastive_loss

        if state_decoder:
            loss_tot += s_T_loss
            return  loss_tot, s_T_loss, a_loss, x_loss, y_loss,  h_loss, w_loss, kl_loss, diffusion_loss, direct_output_loss, concept_class_loss, concept_contrastive_loss
        else:
            return  loss_tot, a_loss, x_loss, y_loss,  h_loss,  w_loss, kl_loss, diffusion_loss, direct_output_loss, concept_class_loss, concept_contrastive_loss

    def reparameterize(self, mean, std):
        eps = torch.normal(torch.zeros(mean.size()).cuda(), torch.ones(mean.size()).cuda())
        return mean + std*eps


# """미사용 코드"""


class SkillPolicy(nn.Module):
    def __init__(self, state_dim, z_dim, h_dim):
        super(SkillPolicy,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim)
        )

    def forward(self,state):

        return self.layers(state)