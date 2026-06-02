import torch
import torch.nn as nn
import numpy as np
import math


class TimeSiren(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(TimeSiren, self).__init__()
        # just a fully connected NN with sin activations
        self.lin1 = nn.Linear(input_dim, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class FCBlock(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        # one layer of non-linearities (just a useful building block to use below)
        self.model = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.BatchNorm1d(num_features=out_feats),
            nn.GELU(),
        )

    def forward(self, x):
        return self.model(x)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1).squeeze(dim=1)
        return embeddings


class Grid2DPositionalEncoding(nn.Module):
    """
    2D sinusoidal positional encoding for grid-based data.
    Adds positional information to spatial features (height x width grids).

    Args:
        channels: Number of channels to add positional encoding
        max_h: Maximum height of the grid
        max_w: Maximum width of the grid
    """

    def __init__(self, channels, max_h=30, max_w=30):
        super().__init__()
        self.channels = channels
        self.max_h = max_h
        self.max_w = max_w

        # Create positional encoding cache
        # channels should be divisible by 2 (half for height, half for width)
        assert channels % 2 == 0, "channels must be divisible by 2"

        # Generate positional encodings
        pe = torch.zeros(max_h, max_w, channels)

        # Height encoding (first half of channels)
        d_model_h = channels // 2
        div_term_h = torch.exp(torch.arange(0, d_model_h, 2).float() * (-math.log(10000.0) / d_model_h))
        pos_h = torch.arange(0, max_h).unsqueeze(1)
        pe[:, :, 0:d_model_h:2] = torch.sin(pos_h * div_term_h).unsqueeze(1).repeat(1, max_w, 1)
        pe[:, :, 1:d_model_h:2] = torch.cos(pos_h * div_term_h).unsqueeze(1).repeat(1, max_w, 1)

        # Width encoding (second half of channels)
        d_model_w = channels // 2
        div_term_w = torch.exp(torch.arange(0, d_model_w, 2).float() * (-math.log(10000.0) / d_model_w))
        pos_w = torch.arange(0, max_w).unsqueeze(1)
        pe[:, :, d_model_h::2] = torch.sin(pos_w * div_term_w).unsqueeze(0).repeat(max_h, 1, 1)
        pe[:, :, d_model_h+1::2] = torch.cos(pos_w * div_term_w).unsqueeze(0).repeat(max_h, 1, 1)

        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, channels, height, width)
        Returns:
            Tensor with positional encoding added: (batch_size, channels, height, width)
        """
        batch_size, channels, h, w = x.shape

        # Get positional encoding for the specific height and width
        # pe is (max_h, max_w, channels), we need (h, w, channels)
        pos_enc = self.pe[:h, :w, :].permute(2, 0, 1)  # (channels, h, w)
        pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (batch_size, channels, h, w)

        # Add positional encoding to input
        return x + pos_enc.to(x.device)


class ResBlock(nn.Module):
    def __init__(self, input_dim, t_embed_dim, state_embed_dim):
        super(ResBlock, self).__init__()

        self.time_layer = nn.Linear(t_embed_dim, input_dim*2)
        self.state_layer = nn.Linear(state_embed_dim, input_dim)
        self.layer1 = nn.Linear(2*input_dim, input_dim)
        self.layer2 = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.act = nn.GELU()

    def forward(self, x, t, s):
        t_embed = self.time_layer(t)
        s_embed = self.layer_norm(self.state_layer(s))
        h = torch.cat([x, s_embed], dim=1)
        h = self.layer1(h)
        h = self.layer_norm(h)
        scale_shift = t_embed.chunk(2, dim=1)
        scale, shift = scale_shift
        h = h * (scale + 1) + shift
        h = self.act(h)
        h = self.layer2(h)
        h = self.layer_norm(h)
        h = self.act(h)
        h = h + x
        return h


class UNet(nn.Module):
    def __init__(self, input_dim=64, state_dim=29, h_dims=[128,32,16,8], nheads=32, time_embed_dim=256, state_embed_dim=256):
        super(UNet, self).__init__()

        self.input_dim = input_dim
        self.h_dims = h_dims
        self.nheads = nheads
        self.state_dim = state_dim
        self.time_embed_dim = time_embed_dim
        self.state_embed_dim = state_embed_dim
        self.nheads = nheads

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(128),
            nn.Linear(128, time_embed_dim),
            nn.LayerNorm(time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.LayerNorm(time_embed_dim),
            nn.GELU()
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, state_embed_dim),
            nn.LayerNorm(state_embed_dim),
            nn.GELU(),
            nn.Linear(state_embed_dim, state_embed_dim),
            nn.LayerNorm(state_embed_dim),
            nn.GELU()
        )
        self.init_layer = nn.Linear(input_dim, h_dims[0])
        self.final_mlp = nn.Sequential(
            nn.Linear(h_dims[0], input_dim),
            # nn.LayerNorm(state_embed_dim),
            # nn.GELU(),
            # nn.Linear(z_dim, z_dim)
        )
        self.res_block1 = ResBlock(h_dims[0], time_embed_dim, state_embed_dim)
        self.res_block2 = ResBlock(h_dims[1], time_embed_dim, state_embed_dim)
        self.res_block3 = ResBlock(h_dims[2], time_embed_dim, state_embed_dim)
        self.res_block4 = ResBlock(h_dims[3], time_embed_dim, state_embed_dim)
        self.down1 = nn.Linear(h_dims[0], h_dims[1])
        self.down2 = nn.Linear(h_dims[1], h_dims[2])
        self.down3 = nn.Linear(h_dims[2], h_dims[3])
        self.up1 = nn.Linear(h_dims[3], h_dims[2])
        self.up2 = nn.Linear(h_dims[2], h_dims[1])
        self.up3 = nn.Linear(h_dims[1], h_dims[0])
        self.res_block5 = ResBlock(h_dims[2], time_embed_dim, state_embed_dim)
        self.res_block6 = ResBlock(h_dims[1], time_embed_dim, state_embed_dim)
        self.res_block7 = ResBlock(h_dims[0], time_embed_dim, state_embed_dim)
        self.final_res_block = ResBlock(h_dims[0], time_embed_dim, state_embed_dim)
        self.layer_norm1 = nn.LayerNorm(h_dims[0])
        self.layer_norm2 = nn.LayerNorm(h_dims[1])
        self.layer_norm3 = nn.LayerNorm(h_dims[2])
        self.layer_norm4 = nn.LayerNorm(h_dims[3])

    def forward(self, x, t, s):
        t_embed = self.time_mlp(t)
        s_embed = self.state_mlp(s)
        x = self.init_layer(x)
        #Down
        h1 = self.res_block1(x, t_embed, s_embed)
        h2 = nn.GELU()(self.layer_norm2(self.down1(h1)))
        h2 = self.res_block2(h2, t_embed, s_embed)
        h3 = nn.GELU()(self.layer_norm3(self.down2(h2)))
        h3 = self.res_block3(h3, t_embed, s_embed)
        h4 = nn.GELU()(self.layer_norm4(self.down3(h3)))
        h4 = self.res_block4(h4, t_embed, s_embed)
        #Up
        h = nn.GELU()(self.layer_norm3(self.up1(h4)))
        h = h+h3
        h = self.res_block5(h, t_embed, s_embed)
        h = nn.GELU()(self.layer_norm2(self.up2(h)))
        h = h+h2
        h = self.res_block6(h, t_embed, s_embed)
        h = nn.GELU()(self.layer_norm1(self.up3(h)))
        h = h+h1
        h = self.res_block7(h, t_embed, s_embed)
        h = h+x
        h = self.final_res_block(h, t_embed, s_embed)
        h = self.final_mlp(h)

        return h


class TransformerEncoderBlock(nn.Module):
    def __init__(self, trans_emb_dim, transformer_dim, nheads):
        super(TransformerEncoderBlock, self).__init__()
        # mainly going off of https://jalammar.github.io/illustrated-transformer/

        self.trans_emb_dim = trans_emb_dim
        self.transformer_dim = transformer_dim
        self.nheads = nheads

        self.input_to_qkv1 = nn.Linear(self.trans_emb_dim, self.transformer_dim * 3)
        self.multihead_attn1 = nn.MultiheadAttention(self.transformer_dim, num_heads=self.nheads)
        self.attn1_to_fcn = nn.Linear(self.transformer_dim, self.trans_emb_dim)
        self.attn1_fcn = nn.Sequential(
            nn.Linear(self.trans_emb_dim, self.trans_emb_dim * 4),
            nn.GELU(),
            nn.Linear(self.trans_emb_dim * 4, self.trans_emb_dim),
        )
        self.norm1a = nn.BatchNorm1d(self.trans_emb_dim)
        self.norm1b = nn.BatchNorm1d(self.trans_emb_dim)

    def split_qkv(self, qkv):
        assert qkv.shape[-1] == self.transformer_dim * 3
        q = qkv[:, :, :self.transformer_dim]
        k = qkv[:, :, self.transformer_dim: 2 * self.transformer_dim]
        v = qkv[:, :, 2 * self.transformer_dim:]
        return (q, k, v)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        qkvs1 = self.input_to_qkv1(inputs)
        # shape out = [3, batchsize, transformer_dim*3]

        qs1, ks1, vs1 = self.split_qkv(qkvs1)
        # shape out = [3, batchsize, transformer_dim]

        attn1_a = self.multihead_attn1(qs1, ks1, vs1, need_weights=False)
        attn1_a = attn1_a[0]
        # shape out = [3, batchsize, transformer_dim = trans_emb_dim x nheads]

        attn1_b = self.attn1_to_fcn(attn1_a)
        attn1_b = attn1_b / 1.414 + inputs / 1.414  # add residual
        # shape out = [3, batchsize, trans_emb_dim]

        # normalise
        attn1_b = self.norm1a(attn1_b.transpose(0, 2).transpose(0, 1))
        attn1_b = attn1_b.transpose(0, 1).transpose(0, 2)
        # batchnorm likes shape = [batchsize, trans_emb_dim, 3]
        # so have to shape like this, then return

        # fully connected layer
        attn1_c = self.attn1_fcn(attn1_b) / 1.414 + attn1_b / 1.414
        # shape out = [3, batchsize, trans_emb_dim]

        # normalise
        # attn1_c = self.norm1b(attn1_c)
        attn1_c = self.norm1b(attn1_c.transpose(0, 2).transpose(0, 1))
        attn1_c = attn1_c.transpose(0, 1).transpose(0, 2)
        return attn1_c


class Model_mlp_diff_embed(nn.Module):
    # this model embeds x, y, t, before input into a fc NN (w residuals)

    def __init__(
        self,
        x_dim,
        n_hidden,
        y_dim,
        embed_dim,
        output_dim=None,
        is_dropout=False,
        is_batch=False,
        activation="relu",
        net_type="fc",
        use_prev=False,
        h_dims=[128,32,16,8],
        max_grid_size=30,
        use_in_out=False,
        use_enhanced_pair_encoding=False,
        disable_pair_encoding=False,
        use_concept_guidance=False,
        use_cfg_for_concept=True,
        concept_scale=1.0,
    ):
        super(Model_mlp_diff_embed, self).__init__()
        self.embed_dim = embed_dim  # input embedding dimension
        self.n_hidden = n_hidden
        self.net_type = net_type
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.use_prev = use_prev # whether x contains previous timestep
        self.use_in_out = use_in_out  # whether to use in_grid and pair_in/out as input to model
        self.use_enhanced_pair_encoding = use_enhanced_pair_encoding
        self.concept_scale = concept_scale  # Scale factor for concept embedding (default 1.0 = no scaling)
        self.disable_pair_encoding = disable_pair_encoding
        self.use_concept_guidance = use_concept_guidance
        self.use_cfg_for_concept = use_cfg_for_concept  # If False, use direct conditioning instead of CFG
        
        if output_dim is None:
            self.output_dim = y_dim  # by default, just output size of action space
        else:
            self.output_dim = output_dim  # sometimes overwrite, eg for discretised, mean/variance, mixture density models
        self.max_grid_size = max_grid_size
        
        # ARC 그리드 인코딩
        self.state_emb_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0), 
            # nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(),
            nn.Flatten(), nn.Linear(32*self.max_grid_size*self.max_grid_size, x_dim), 
            # nn.Flatten(), nn.Linear(32*28*28, x_dim), 
            nn.Tanh()
        )
        
        self.pair_emb_layer = nn.Sequential(
            nn.Linear(6 * x_dim, x_dim),  # 3개 input + 3개 output = 6개 그리드
            nn.ReLU(),
            nn.Linear(x_dim, x_dim),
            nn.ReLU()
        )

        # Text guide embedding projection layer (to match pair_emb dimension)
        # NOTE: Removed final Tanh to allow larger magnitudes (avoid concept signal being too weak)
        self.text_guide_proj = nn.Sequential(
            nn.Linear(y_dim, x_dim),  # y_dim (concept emb dim) → x_dim (match other embeddings)
            nn.ReLU(),
            nn.Linear(x_dim, x_dim),
            # nn.Tanh()  # Removed: was constraining output to [-1,1] making concept embedding too small
        )

        # Enhanced pair encoding layers
        if self.use_enhanced_pair_encoding:
            # Individual grid embedding for pair processing
            self.pair_input_emb_layer = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32*self.max_grid_size*self.max_grid_size, x_dim),
                nn.Tanh()
            )
            self.pair_output_emb_layer = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32*self.max_grid_size*self.max_grid_size, x_dim),
                nn.Tanh()
            )
            # Transformation encoding: learn input->output transformation
            self.pair_transform_layer = nn.Sequential(
                nn.Linear(2*x_dim, x_dim),  # input + output embeddings
                nn.ReLU(),
                nn.Linear(x_dim, x_dim),
                nn.ReLU(),
                nn.Linear(x_dim, x_dim),
                nn.Tanh()
            )
            # Combine multiple pairs
            self.enhanced_pair_combiner = nn.Sequential(
                nn.Linear(3*x_dim, x_dim),  # 3 pairs each with x_dim
                nn.ReLU(),
                nn.Linear(x_dim, x_dim),
                nn.ReLU()
            )
        
        if self.use_in_out and not self.disable_pair_encoding:
            if self.use_concept_guidance:
                # With concept guidance: state, clip, in_grid, pair, concept
                self.s_p_emb_layer = nn.Sequential(
                    nn.Linear(x_dim*5, x_dim),
                    nn.ReLU(),
                    nn.Linear(x_dim, x_dim),
                    nn.ReLU()
                )
            else:
                # Without concept guidance: state, clip, in_grid, pair
                self.s_p_emb_layer = nn.Sequential(
                    nn.Linear(x_dim*4, x_dim),
                    nn.ReLU(),
                    nn.Linear(x_dim, x_dim),
                    nn.ReLU()
                )
        else:
            # No pair encoding: state, clip, in_grid (concept not supported without pairs)
            self.s_p_emb_layer = nn.Sequential(
                nn.Linear(x_dim*3, x_dim),
                nn.ReLU(),
                nn.Linear(x_dim, x_dim),
                nn.ReLU()
            )
            
        if self.net_type == 'unet':
            self.unet = UNet(input_dim=y_dim, state_dim=x_dim, h_dims=h_dims, nheads=32, time_embed_dim=self.x_dim, state_embed_dim=self.x_dim)
            # self.unet = UNet(input_dim=y_dim, state_dim=x_dim, h_dims=h_dims, nheads=32, time_embed_dim=256, state_embed_dim=256)
            return

        # embedding NNs
        if self.use_prev:
            self.x_embed_nn = nn.Sequential(
                nn.Linear(int(x_dim / 2), self.embed_dim),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )
        else:
            self.x_embed_nn = nn.Sequential(
                nn.Linear(x_dim, self.embed_dim),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )  # no prev hist
        self.y_embed_nn = nn.Sequential(
            nn.Linear(y_dim, self.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.t_embed_nn = TimeSiren(1, self.embed_dim)

        # fc nn layers
        if self.net_type == "fc":
            if self.use_prev:
                self.fc1 = nn.Sequential(FCBlock(self.embed_dim * 4, n_hidden))  # concat x, x_prev,
            else:
                self.fc1 = nn.Sequential(FCBlock(self.embed_dim * 3, n_hidden))  # no prev hist
            self.fc2 = nn.Sequential(FCBlock(n_hidden + y_dim + 1, n_hidden))  # will concat y and t at each layer
            self.fc3 = nn.Sequential(FCBlock(n_hidden + y_dim + 1, n_hidden))
            self.fc4 = nn.Sequential(nn.Linear(n_hidden + y_dim + 1, self.output_dim))

        # transformer layers
        elif self.net_type == "transformer":
            self.nheads = 16  # 16
            self.trans_emb_dim = 64
            self.transformer_dim = self.trans_emb_dim * self.nheads  # embedding dim for each of q,k and v (though only k and v have to be same I think)

            self.t_to_input = nn.Linear(self.embed_dim, self.trans_emb_dim)
            self.y_to_input = nn.Linear(self.embed_dim, self.trans_emb_dim)
            self.x_to_input = nn.Linear(self.embed_dim, self.trans_emb_dim)

            self.pos_embed = TimeSiren(1, self.trans_emb_dim)

            self.transformer_block1 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)
            self.transformer_block2 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)
            self.transformer_block3 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)
            self.transformer_block4 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)

            if self.use_prev:
                self.final = nn.Linear(self.trans_emb_dim * 4, self.output_dim)  # final layer params
            else:
                self.final = nn.Linear(self.trans_emb_dim * 3, self.output_dim)
        else:
            raise NotImplementedError

    def forward(self, y, x, clip, in_grid, t, context_mask, pair_in=None, pair_out=None, text_guide_emb=None):
        # embed y, x, t
        if self.net_type == 'unet':
            # print("input x shape: {0}".format(x.shape))
            s_emb = self.state_emb_layer(x)
            clip_emb = self.state_emb_layer(clip)
            in_grid_emb = self.state_emb_layer(in_grid)

            # print(pair_in.shape)
            # print(pair_out.shape)


            # state, clip, in_grid + pair (or text guidance) - only if pair encoding is not disabled
            if self.use_in_out and not getattr(self, 'disable_pair_encoding', False):
                # First, compute pair encoding
                pair_emb = None

                # Option 1: Enhanced pair encoding
                if getattr(self, 'use_enhanced_pair_encoding', False):
                    # Enhanced pair encoding: process individual input-output transformations
                    pair_transforms = []

                    # Process each input-output pair individually
                    for i in range(3):  # 3 pairs
                        input_grid = pair_in[:, i:i+1]  # (batch, 1, grid_h, grid_w)
                        output_grid = pair_out[:, i:i+1]  # (batch, 1, grid_h, grid_w)

                        # Individual embeddings
                        input_emb = self.pair_input_emb_layer(
                            input_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous().cuda()
                        )
                        output_emb = self.pair_output_emb_layer(
                            output_grid.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous().cuda()
                        )

                        # Learn input->output transformation
                        transform_input = torch.cat([input_emb, output_emb], dim=-1)
                        transform_emb = self.pair_transform_layer(transform_input)
                        pair_transforms.append(transform_emb)

                    # Combine all transformations
                    combined_transforms = torch.cat(pair_transforms, dim=-1)
                    pair_emb = self.enhanced_pair_combiner(combined_transforms)
                else:
                    # Original pair encoding (concatenation-based)
                    pair = torch.cat([pair_in, pair_out], dim=1)
                    pair_shape = pair.shape

                    # VAE와 동일: state_emb_layer로 처리
                    pair_emb = self.state_emb_layer(pair.reshape(-1, 1, self.max_grid_size, self.max_grid_size).type(torch.float32).contiguous().cuda())
                    pair_emb = pair_emb.reshape(pair_shape[0], 1, pair_shape[1] * self.x_dim)  # VAE와 동일한 reshape
                    pair_emb = self.pair_emb_layer(pair_emb)

                    # Diffusion에서는 time dimension이 없으므로 squeeze
                    pair_emb = pair_emb.squeeze(1)  # (batch, n_hidden)

                # Add text guidance (concept) to feature vector
                if text_guide_emb is not None:
                    # Project text_guide_emb to match x_dim
                    text_guide_projected = self.text_guide_proj(text_guide_emb)

                    # Scale concept embedding to control its influence (default 1.0 = no scaling)
                    concept_scale = getattr(self, 'concept_scale', 1.0)
                    if concept_scale != 1.0:
                        text_guide_projected = text_guide_projected * concept_scale

                    # Two modes for concept guidance:
                    # 1. CFG (Classifier-Free Guidance): use context_mask to enable/disable concept
                    # 2. Direct Conditioning: always use concept, no masking
                    if self.use_cfg_for_concept and context_mask is not None:
                        # CFG mode: Apply context_mask to disable concept guidance for unconditional samples
                        # context_mask: 0 for conditional (use concept), 1 for unconditional (mask concept)
                        mask_expanded = context_mask.unsqueeze(1)  # (batch, 1)
                        # Zero out concept embedding for masked (unconditional) samples
                        text_guide_projected = text_guide_projected * (1.0 - mask_expanded)
                    # else: Direct Conditioning mode - always use concept embedding as-is
                else:
                    # No concept guidance: use zero tensor to maintain dimension compatibility
                    text_guide_projected = torch.zeros_like(pair_emb)

                # Concatenate based on whether concept guidance is enabled
                if self.use_concept_guidance:
                    # With concept: state, clip, in_grid, pair_emb, text_guide
                    feat = torch.cat([s_emb, clip_emb, in_grid_emb, pair_emb, text_guide_projected], dim=-1)
                else:
                    # Without concept: state, clip, in_grid, pair_emb (no text_guide)
                    feat = torch.cat([s_emb, clip_emb, in_grid_emb, pair_emb], dim=-1)
            else:
                feat = torch.cat([s_emb, clip_emb, in_grid_emb], dim=-1)
            s_p_emb = self.s_p_emb_layer(feat)
            # print("s_emb shape: {0}".format(s_emb.shape))
            net_output = self.unet(y, t, s_p_emb)
            
        # 이건 안씀, 어차피 UNET 사용 
        else:
            if self.use_prev:
                x_e = self.x_embed_nn(x[:, :int(self.x_dim / 2)])
                x_e_prev = self.x_embed_nn(x[:, int(self.x_dim / 2):])
            else:
                x_e = self.x_embed_nn(x)  # no prev hist
                x_e_prev = None
            y_e = self.y_embed_nn(y)
            t_e = self.t_embed_nn(t)

            # mask out context embedding, x_e, if context_mask == 1
            context_mask = context_mask.repeat(x_e.shape[1], 1).T
            x_e = x_e * (-1 * (1 - context_mask))
            if self.use_prev:
                x_e_prev = x_e_prev * (-1 * (1 - context_mask))

            # pass through fc nn
            if self.net_type == "fc":
                net_output = self.forward_fcnn(x_e, x_e_prev, y_e, t_e, x, y, t)

            # or pass through transformer encoder
            elif self.net_type == "transformer":
                net_output = self.forward_transformer(x_e, x_e_prev, y_e, t_e, x, y, t)

        return net_output

    def forward_fcnn(self, x_e, x_e_prev, y_e, t_e, x, y, t):
        if self.use_prev:
            net_input = torch.cat((x_e, x_e_prev, y_e, t_e), 1)
        else:
            net_input = torch.cat((x_e, y_e, t_e), 1)
        nn1 = self.fc1(net_input)
        nn2 = self.fc2(torch.cat((nn1 / 1.414, y, t), 1)) + nn1 / 1.414  # residual and concat inputs again
        nn3 = self.fc3(torch.cat((nn2 / 1.414, y, t), 1)) + nn2 / 1.414
        net_output = self.fc4(torch.cat((nn3, y, t), 1))
        return net_output

    def forward_transformer(self, x_e, x_e_prev, y_e, t_e, x, y, t):
        # roughly following this: https://jalammar.github.io/illustrated-transformer/

        t_input = self.t_to_input(t_e)
        y_input = self.y_to_input(y_e)
        x_input = self.x_to_input(x_e)
        if self.use_prev:
            x_input_prev = self.x_to_input(x_e_prev)
        # shape out = [batchsize, trans_emb_dim]

        # add 'positional' encoding
        # note, here position refers to order tokens are fed into transformer
        t_input += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 1.0)
        y_input += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 2.0)
        x_input += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 3.0)
        if self.use_prev:
            x_input_prev += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 4.0)

        if self.use_prev:
            inputs1 = torch.cat(
                (
                    t_input[None, :, :],
                    y_input[None, :, :],
                    x_input[None, :, :],
                    x_input_prev[None, :, :],
                ),
                0,
            )
        else:
            inputs1 = torch.cat((t_input[None, :, :], y_input[None, :, :], x_input[None, :, :]), 0)
        # shape out = [3, batchsize, trans_emb_dim]

        block1 = self.transformer_block1(inputs1)
        block2 = self.transformer_block2(block1)
        block3 = self.transformer_block3(block2)
        block4 = self.transformer_block4(block3)

        # flatten and add final linear layer
        # transformer_out = block2
        transformer_out = block4
        transformer_out = transformer_out.transpose(0, 1)  # roll batch to first dim
        # shape out = [batchsize, 3, trans_emb_dim]

        flat = torch.flatten(transformer_out, start_dim=1, end_dim=2)
        # shape out = [batchsize, 3 x trans_emb_dim]

        out = self.final(flat)
        # shape out = [batchsize, n_dim]
        return out


def ddpm_schedules(beta1, beta2, T, schedule):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    # beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    # beta_t = (beta2 - beta1) * torch.arange(-1, T + 1, dtype=torch.float32) / T + beta1
    if schedule == 'linear':
        beta_t = (beta2 - beta1) * torch.arange(-1, T, dtype=torch.float32) / (T - 1) + beta1
    elif schedule == 'quadratic':
        beta_t = (beta2 - beta1) * torch.square(torch.arange(-1, T, dtype=torch.float32)) / torch.max(torch.square(torch.arange(-1, T, dtype=torch.float32))) + beta1
    elif schedule == 'cosine':
        s=0.008
        steps = T + 1
        x = np.linspace(-1, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
        beta_t = torch.cat([torch.tensor([1.0]), torch.tensor(betas_clipped, dtype=torch.float32)])
    else:
        raise NotImplementedError

    beta_t[0] = beta1  # modifying this so that beta_t[1] = beta1, and beta_t[n_T]=beta2, while beta[0] is never used
    # this is as described in Denoising Diffusion Probabilistic Models paper, section 4
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


def ddim_schedules(beta1, beta2, T, schedule, ddim_steps=None, ddim_discr='uniform', ddim_eta=0.0):
    """
    Returns pre-computed schedules for DDIM sampling, allowing for accelerated sampling.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    assert ddim_eta >= 0.0 and ddim_eta <= 1.0, "ddim_eta must be in [0, 1]"

    # Print only on first call
    if not hasattr(ddim_schedules, "_printed"):
        verbose = True
        ddim_schedules._printed = True
    else:
        verbose = False

    # Determine number of sampling steps for DDIM
    if ddim_steps is None:
        ddim_steps = T  # No acceleration if not specified

    if verbose:
        print(f"DDIM Schedules - beta1: {beta1}, beta2: {beta2}, T: {T}, schedule: {schedule}")
        print(f"DDIM config: steps={ddim_steps}, discretization={ddim_discr}, eta={ddim_eta}")

    # Calculate DDPM parameters first (same as in ddmp_schedules)
    if schedule == 'linear':
        beta_t = (beta2 - beta1) * torch.arange(-1, T, dtype=torch.float32) / (T - 1) + beta1
    elif schedule == 'quadratic':
        beta_t = (beta2 - beta1) * torch.square(torch.arange(-1, T, dtype=torch.float32)) / torch.max(torch.square(torch.arange(-1, T, dtype=torch.float32))) + beta1
    elif schedule == 'cosine':
        s = 0.008
        steps = T + 1
        x = np.linspace(-1, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
        beta_t = torch.cat([torch.tensor([1.0]), torch.tensor(betas_clipped, dtype=torch.float32)])
    else:
        raise NotImplementedError

    beta_t[0] = beta1

    # Calculate DDPM parameters
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    # Standard DDPM parameters
    sqrt_alphabar_t = torch.sqrt(alphabar_t)
    oneover_sqrt_alpha_t = 1 / torch.sqrt(alpha_t)
    sqrt_one_minus_alphabar_t = torch.sqrt(1 - alphabar_t)

    # Select subset of timesteps for DDIM
    if ddim_discr == 'uniform':
        c = T // ddim_steps
        ddim_timesteps = np.asarray(list(range(0, T, c)))
    elif ddim_discr == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(T * 0.8), ddim_steps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f"Unknown discretization method '{ddim_discr}'")

    # Add back T-1 as the last timestep if it's not already included
    if ddim_timesteps[-1] != T - 1:
        ddim_timesteps = np.append(ddim_timesteps, T - 1)

    # DDIM-specific parameters
    ddim_timesteps = torch.from_numpy(ddim_timesteps).long()

    # For each DDIM timestep, calculate alphas, sigmas and other parameters
    ddim_alphas = alphabar_t[ddim_timesteps]
    ddim_alphas_prev = torch.cat([torch.tensor([1.0]), alphabar_t[ddim_timesteps[:-1]]])

    # Calculate sigma_t for DDIM (controls the stochasticity)
    ddim_sigmas = ddim_eta * torch.sqrt(
        (1 - ddim_alphas_prev) / (1 - ddim_alphas) * (1 - ddim_alphas / ddim_alphas_prev)
    )

    # Calculate DDIM sampling coefficients (used in the sampling equations)
    ddim_sqrt_recip = torch.sqrt(1. / ddim_alphas_prev)
    ddim_sqrt_recip_minus_one = torch.sqrt(1. / ddim_alphas_prev - 1)

    # For noise prediction models: coef for current x, coef for predicted noise
    ddim_coef1 = ddim_sqrt_recip
    ddim_coef2 = ddim_sqrt_recip_minus_one

    # For direct x0 prediction models
    ddim_x0_coef = (ddim_sqrt_recip * (1 - ddim_alphas_prev)) / (1 - ddim_alphas)

    return {
        "alpha_t": alpha_t,
        "oneover_sqrt_alpha_t": oneover_sqrt_alpha_t,
        "sqrt_beta_t": torch.sqrt(beta_t),
        "alphabar_t": alphabar_t,
        "sqrt_alphabar_t": sqrt_alphabar_t,
        "sqrt_one_minus_alphabar_t": sqrt_one_minus_alphabar_t,
        "ddim_timesteps": ddim_timesteps,
        "ddim_num_steps": len(ddim_timesteps),
        "ddim_alphas": ddim_alphas,
        "ddim_alphas_prev": ddim_alphas_prev,
        "ddim_sigmas": ddim_sigmas,
        "ddim_sqrt_recip": ddim_sqrt_recip,
        "ddim_coef1": ddim_coef1,
        "ddim_coef2": ddim_coef2,
        "ddim_x0_coef": ddim_x0_coef,
        "ddim_eta": ddim_eta,
    }


class Model_Cond_Diffusion(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, x_dim, y_dim, drop_prob=0.1, guide_w=0.0, normalize_latent=False, schedule='linear',use_in_out=False):
        super(Model_Cond_Diffusion, self).__init__()
        for k, v in ddpm_schedules(betas[0], betas[1], n_T, schedule).items():
            self.register_buffer(k, v)

        self.nn_model = nn_model
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.guide_w = guide_w
        self.normalize_latent = normalize_latent
        self.use_in_out = use_in_out
        self.betas = betas
        self.schedule = schedule

        # Z-score normalization statistics (for denormalizing output)
        # These are set via set_latent_stats() after training with normalize_latent_zscore=1
        self.register_buffer('latent_mean', None)
        self.register_buffer('latent_std', None)
        self.use_zscore_denorm = False  # Flag to enable denormalization

    def set_latent_stats(self, mean, std):
        """Set latent statistics for z-score denormalization during sampling."""
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean).float()
        if isinstance(std, np.ndarray):
            std = torch.from_numpy(std).float()
        self.latent_mean = mean.to(self.device)
        self.latent_std = std.to(self.device)
        self.use_zscore_denorm = True
        print(f"[Model_Cond_Diffusion] Z-score denormalization enabled. Mean shape: {mean.shape}, Std shape: {std.shape}")

    def _denormalize(self, y):
        """Denormalize latent if z-score stats are set."""
        if self.use_zscore_denorm and self.latent_mean is not None and self.latent_std is not None:
            return y * self.latent_std + self.latent_mean
        return y

    def loss_on_batch(self, x_batch, clip, in_grid, pair_in_batch, pair_out_batch, y_batch, predict_noise=True, text_guide_emb=None, return_pred_x0=False):
        _ts = torch.randint(1, self.n_T + 1, (y_batch.shape[0], 1)).to(self.device)

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros(x_batch.shape[0]) + self.drop_prob).to(self.device)

        # randomly sample some noise, noise ~ N(0, 1)
        noise = torch.randn_like(y_batch).to(self.device)

        # add noise to clean target actions
        y_t = self.sqrtab[_ts] * y_batch + self.sqrtmab[_ts] * noise
        if self.normalize_latent:
            y_t = y_t/torch.norm(y_t, dim=-1).unsqueeze(-1)

        # use nn model to predict noise
        if self.nn_model.net_type == 'unet':
            noise_pred_batch = self.nn_model(y_t, x_batch, clip, in_grid, pair_in_batch, pair_out_batch, _ts, context_mask, text_guide_emb=text_guide_emb)
        else:
           noise_pred_batch = self.nn_model(y_t, x_batch, clip, in_grid, pair_in_batch, pair_out_batch, _ts / self.n_T, context_mask, text_guide_emb=text_guide_emb)

        if self.normalize_latent:
            noise_pred_batch = noise_pred_batch/torch.norm(noise_pred_batch, dim=-1).unsqueeze(-1)

        # Calculate predicted clean latent (x0) from predicted noise
        # Using the formula: x0 = (y_t - sqrt(1-alphabar_t) * noise_pred) / sqrt(alphabar_t)
        if return_pred_x0:
            pred_x0 = (y_t - self.sqrtmab[_ts] * noise_pred_batch) / self.sqrtab[_ts]

        # return mse between predicted and true noise
        if predict_noise:
            loss = self.loss_mse(noise, noise_pred_batch)
        else:
            loss = (torch.minimum((self.sqrtab[_ts] / self.sqrtmab[_ts]) ** 2, torch.tensor(5)) * ((y_batch - noise_pred_batch) ** 2)).mean()

        if return_pred_x0:
            return loss, pred_x0
        else:
            return loss

    def sample(self, x_batch, clip, in_grid, pair_in=None, pair_out=None, return_y_trace=False, extract_embedding=False, predict_noise=True, text_guide_emb=None):
        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]

        y_shape = (n_sample, self.y_dim)

        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)

        if not is_zero:
            if len(x_batch.shape) > 2:  # 이미지 같은 경우 (batch, channel, 가로, 세로)
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1, 1)
                clip = clip.repeat(2, 1, 1, 1)
                in_grid = in_grid.repeat(2, 1, 1, 1)
            else:                       # 시계열 데이터 같은 것 (batch, 사이즈)
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1)
                clip = clip.repeat(2, 1)
                in_grid = in_grid.repeat(2, 1)

            # Setup text_guide_emb for classifier-free guidance
            # First half: conditional (with guidance), Second half: unconditional (without guidance)
            if text_guide_emb is not None:
                # Create unconditional embeddings (zeros)
                unconditional_emb = torch.zeros_like(text_guide_emb)
                # Concatenate: [conditional, unconditional]
                text_guide_emb = torch.cat([text_guide_emb, unconditional_emb], dim=0)

            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        # 이거 안씀
        if extract_embedding:
            x_embed = self.nn_model.embed_context(x_batch)

        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            z = torch.randn(y_shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            if extract_embedding:   # 안씀
                eps = self.nn_model(y_i, x_batch, clip, in_grid, pair_in, pair_out, t_is, context_mask, x_embed, text_guide_emb=text_guide_emb)
            else:
                if self.nn_model.net_type == 'unet':
                    eps = self.nn_model(y_i, x_batch, clip, in_grid, pair_in, pair_out, t_is*self.n_T, context_mask, text_guide_emb=text_guide_emb)
                else:
                    eps = self.nn_model(y_i, x_batch, clip, in_grid, pair_in, pair_out, t_is, context_mask, text_guide_emb=text_guide_emb)
            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]

            if not predict_noise:
                x0 = eps
                eps = 1 / self.sqrtmab[i] * (y_i - self.sqrtab[i] * x0)

            y_i = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            if return_y_trace and (i % 20 == 0 or i == self.n_T or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())

        # Apply z-score denormalization if enabled
        y_i = self._denormalize(y_i)

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i

    def sample_extra(self, x_batch, clip, in_grid, pair_in, pair_out, extra_steps=4,return_y_trace=False, predict_noise=True, text_guide_emb=None):
        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]

        y_shape = (n_sample, self.y_dim)

        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)

        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1, 1)
                clip = clip.repeat(2, 1, 1, 1)
                in_grid = in_grid.repeat(2, 1, 1, 1)
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1)
                clip = clip.repeat(2, 1)
                in_grid = in_grid.repeat(2, 1)
            # Repeat text_guide_emb if provided
            if text_guide_emb is not None:
                text_guide_emb = text_guide_emb.repeat(2, 1)
            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            # context_mask = torch.zeros_like(x_batch[:,0]).to(self.device)
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        # for i_dummy in range(self.n_T, 0, -1):
        for i_dummy in range(self.n_T, -extra_steps, -1):
            i = max(i_dummy, 1)
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if self.normalize_latent:
                y_i = y_i/torch.norm(y_i, dim=-1).unsqueeze(-1)

            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            z = torch.randn(y_shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            if self.nn_model.net_type == 'unet':
                eps = self.nn_model(y_i, x_batch, clip, in_grid, pair_in=pair_in, pair_out=pair_out, t=t_is*self.n_T, context_mask=context_mask, text_guide_emb=text_guide_emb)
            else:
                eps = self.nn_model(y_i, x_batch, clip, in_grid, pair_in=pair_in, pair_out=pair_out, t=t_is, context_mask=context_mask, text_guide_emb=text_guide_emb)

            if self.normalize_latent:
                eps = eps/torch.norm(eps,dim=-1).unsqueeze(-1)

            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]

            if not predict_noise:
                x0 = eps
                eps = 1 / self.sqrtmab[i] * (y_i - self.sqrtab[i] * x0)

            # x0 = torch.clip((1 / self.sqrtab[i]) * (y_i - self.sqrtmab[i] * eps), -1.0, 1.0)
            # y_i = ((1 / self.oneover_sqrta[i]) * (1 - self.alphabar_t[i - 1])) / (1 - self.alphabar_t[i]) * y_i + ((self.sqrtab[i - 1] * (1 - self.alpha_t[i])) / (1 - self.alphabar_t[i])) * x0

            y_i = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            if return_y_trace and (i % 20 == 0 or i == self.n_T or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())

        if self.normalize_latent:
            y_i = y_i/torch.norm(y_i, dim=-1).unsqueeze(-1)

        # Apply z-score denormalization if enabled
        y_i = self._denormalize(y_i)

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i

    def sample_ddim(self, x_batch, clip, in_grid, pair_in=None, pair_out=None,
                ddim_steps=None, ddim_eta=0.0, ddim_discr='uniform',
                    return_y_trace=False, extract_embedding=False, predict_noise=True, noise_temperature=1.0, text_guide_emb=None):
        """
        DDIM sampling for accelerated inference with fewer sampling steps.

        Args:
            x_batch: Conditioning input
            clip, in_grid, pair_in, pair_out: Additional conditioning inputs
            ddim_steps: Number of DDIM sampling steps (fewer = faster sampling)
            ddim_eta: Controls the stochasticity (0=deterministic, 1=DDPM-like)
            ddim_discr: How to discretize timesteps ('uniform' or 'quad')
            return_y_trace: Whether to return intermediate denoising steps
            extract_embedding: Whether to pre-compute context embeddings
            predict_noise: Whether model predicts noise (True) or direct x0 (False)
            noise_temperature: Temperature scaling for sampling noise (higher = more diversity)

        Returns:
            Sampled output, and optionally the trajectory of intermediate steps
        """
        # Get DDIM parameters
        ddim_params = ddim_schedules(
            self.betas[0],
            self.betas[1],
            self.n_T,
            self.schedule,
            ddim_steps=ddim_steps,
            ddim_discr=ddim_discr,
            ddim_eta=ddim_eta
        )

        # Check if guide_w is effectively zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # Determine batch size and shape
        n_sample = x_batch.shape[0]
        y_shape = (n_sample, self.y_dim)

        # Sample initial noise with temperature scaling
        y_i = torch.randn(y_shape).to(self.device) * noise_temperature
        
        # Handle classifier-free guidance setup
        if not is_zero:
            if len(x_batch.shape) > 2:
                x_batch = x_batch.repeat(2, 1, 1, 1)
                clip = clip.repeat(2, 1, 1, 1)
                in_grid = in_grid.repeat(2, 1, 1, 1)
                if pair_in is not None:
                    pair_in = pair_in.repeat(2, 1, 1, 1)
                if pair_out is not None:
                    pair_out = pair_out.repeat(2, 1, 1, 1)
            else:
                x_batch = x_batch.repeat(2, 1)
                clip = clip.repeat(2, 1)
                in_grid = in_grid.repeat(2, 1)
                if pair_in is not None:
                    pair_in = pair_in.repeat(2, 1)
                if pair_out is not None:
                    pair_out = pair_out.repeat(2, 1)

            # Setup text_guide_emb for classifier-free guidance
            # First half: conditional (with guidance), Second half: unconditional (without guidance)
            if text_guide_emb is not None:
                # Create unconditional embeddings (zeros)
                unconditional_emb = torch.zeros_like(text_guide_emb)
                # Concatenate: [conditional, unconditional]
                text_guide_emb = torch.cat([text_guide_emb, unconditional_emb], dim=0)

            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0
        else:
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        if extract_embedding:
            x_embed = self.nn_model.embed_context(x_batch)

        # Get timesteps and run DDIM sampling
        timesteps = ddim_params["ddim_timesteps"]
        y_i_store = []

        for i, timestep in enumerate(reversed(timesteps)):
            t_is = torch.tensor([timestep / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            # Model prediction
            if extract_embedding:
                model_output = self.nn_model(y_i, x_batch, clip, in_grid, pair_in, pair_out, t_is, context_mask, x_embed, text_guide_emb=text_guide_emb)
            else:
                if self.nn_model.net_type == 'unet':
                    model_output = self.nn_model(y_i, x_batch, clip, in_grid, pair_in, pair_out, t_is*self.n_T, context_mask, text_guide_emb=text_guide_emb)
                else:
                    model_output = self.nn_model(y_i, x_batch, clip, in_grid, pair_in, pair_out, t_is, context_mask, text_guide_emb=text_guide_emb)
            
            # Apply classifier-free guidance
            if not is_zero:
                output1 = model_output[:n_sample]
                output2 = model_output[n_sample:]
                model_output = (1 + self.guide_w) * output1 - self.guide_w * output2
                y_i = y_i[:n_sample]

            # DDIM update step
            idx = len(timesteps) - 1 - i
            alpha_t = ddim_params["ddim_alphas"][idx]
            alpha_prev = ddim_params["ddim_alphas_prev"][idx]
            sigma = ddim_params["ddim_sigmas"][idx] if ddim_eta > 0 else 0

            if not predict_noise:
                x0_pred = model_output
                # Calculate noise from predicted x0
                eps_pred = (y_i - alpha_t.sqrt() * x0_pred) / (1 - alpha_t).sqrt()
                dir_xt = (1. - alpha_prev - sigma**2).sqrt() * eps_pred
            else:
                eps_pred = model_output
                x0_pred = (y_i - (1 - alpha_t).sqrt() * eps_pred) / alpha_t.sqrt()
                dir_xt = (1. - alpha_prev - sigma**2).sqrt() * eps_pred

            if ddim_eta > 0 and i < len(timesteps) - 1:
                noise = torch.randn(y_shape, device=self.device) * noise_temperature
            else:
                noise = torch.zeros(y_shape, device=self.device)

            y_i = alpha_prev.sqrt() * x0_pred + dir_xt + sigma * noise

            if return_y_trace:
                y_i_store.append(y_i.detach().cpu().numpy())

        # Apply z-score denormalization if enabled
        y_i = self._denormalize(y_i)

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i

    def ddim_sample_extra(self, x_batch, clip, in_grid, pair_in=None, pair_out=None,
                   ddim_steps=None, ddim_eta=0.0, ddim_discr='uniform', extra_steps=4,
                   return_y_trace=False, extract_embedding=False, predict_noise=False, noise_temperature=1.0, text_guide_emb=None):
        """
        DDIM sampling with extra refinement steps for higher quality results.
        """
        # Get DDIM parameters
        ddim_params = ddim_schedules(
            self.betas[0], 
            self.betas[1], 
            self.n_T,
            self.schedule,
            ddim_steps=ddim_steps,
            ddim_discr=ddim_discr,
            ddim_eta=ddim_eta
        )
        
        # Check if guide_w is effectively zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # Determine batch size and shape
        n_sample = x_batch.shape[0]
        y_shape = (n_sample, self.y_dim)

        # Sample initial noise with temperature scaling
        y_i = torch.randn(y_shape).to(self.device) * noise_temperature

        # Handle classifier-free guidance setup
        if not is_zero:
            if len(x_batch.shape) > 2:
                x_batch = x_batch.repeat(2, 1, 1, 1)
                clip = clip.repeat(2, 1, 1, 1)
                in_grid = in_grid.repeat(2, 1, 1, 1)
                if pair_in is not None:
                    pair_in = pair_in.repeat(2, 1, 1, 1)
                if pair_out is not None:
                    pair_out = pair_out.repeat(2, 1, 1, 1)
            else:
                x_batch = x_batch.repeat(2, 1)
                clip = clip.repeat(2, 1)
                in_grid = in_grid.repeat(2, 1)
                if pair_in is not None:
                    pair_in = pair_in.repeat(2, 1)
                if pair_out is not None:
                    pair_out = pair_out.repeat(2, 1)
            # Repeat text_guide_emb if provided
            if text_guide_emb is not None:
                text_guide_emb = text_guide_emb.repeat(2, 1)

            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0
        else:
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        if extract_embedding:
            x_embed = self.nn_model.embed_context(x_batch)

        # Get timesteps and run DDIM sampling
        timesteps = ddim_params["ddim_timesteps"]
        y_i_store = []
        
        # Regular DDIM steps
        for i, timestep in enumerate(reversed(timesteps)):
            t_is = torch.tensor([timestep / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            # Model prediction
            if extract_embedding:
                model_output = self.nn_model(y_i, x_batch, clip, in_grid, pair_in=pair_in, pair_out=pair_out, t=t_is, context_mask=context_mask, x_embed=x_embed, text_guide_emb=text_guide_emb)
            else:
                if self.nn_model.net_type == 'unet':
                    model_output = self.nn_model(y_i, x_batch, clip, in_grid, pair_in=pair_in, pair_out=pair_out, t=t_is*self.n_T, context_mask=context_mask, text_guide_emb=text_guide_emb)
                else:
                    model_output = self.nn_model(y_i, x_batch, clip, in_grid, pair_in=pair_in, pair_out=pair_out, t=t_is, context_mask=context_mask, text_guide_emb=text_guide_emb)
            
            # Apply classifier-free guidance
            if not is_zero:
                output1 = model_output[:n_sample]
                output2 = model_output[n_sample:]
                model_output = (1 + self.guide_w) * output1 - self.guide_w * output2
                y_i = y_i[:n_sample]

            # DDIM update step
            idx = len(timesteps) - 1 - i
            alpha_t = ddim_params["ddim_alphas"][idx]
            alpha_prev = ddim_params["ddim_alphas_prev"][idx]
            sigma = ddim_params["ddim_sigmas"][idx] if ddim_eta > 0 else 0

            if not predict_noise:
                x0_pred = model_output
                # Calculate noise from predicted x0
                eps_pred = (y_i - alpha_t.sqrt() * x0_pred) / (1 - alpha_t).sqrt()
                dir_xt = (1. - alpha_prev - sigma**2).sqrt() * eps_pred
            else:
                eps_pred = model_output
                x0_pred = (y_i - (1 - alpha_t).sqrt() * eps_pred) / alpha_t.sqrt()
                dir_xt = (1. - alpha_prev - sigma**2).sqrt() * eps_pred

            if ddim_eta > 0 and i < len(timesteps) - 1:
                noise = torch.randn(y_shape, device=self.device) * noise_temperature
            else:
                noise = torch.zeros(y_shape, device=self.device)

            y_i = alpha_prev.sqrt() * x0_pred + dir_xt + sigma * noise

            if return_y_trace:
                y_i_store.append(y_i.detach().cpu().numpy())

        # Extra refinement steps at t=1
        if extra_steps > 0:
            t_is = torch.tensor([1 / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)
            
            for extra_step in range(1, extra_steps + 1):
                if not is_zero:
                    y_i = y_i.repeat(2, 1)
                    t_is_doubled = t_is.repeat(2, 1)
                else:
                    t_is_doubled = t_is
                
                # Model prediction for extra step
                if extract_embedding:
                    model_output = self.nn_model(y_i, x_batch, clip, in_grid, pair_in=pair_in, pair_out=pair_out, t=t_is_doubled, context_mask=context_mask, x_embed=x_embed, text_guide_emb=text_guide_emb)
                else:
                    if self.nn_model.net_type == 'unet':
                        model_output = self.nn_model(y_i, x_batch, clip, in_grid, pair_in=pair_in, pair_out=pair_out, t=t_is_doubled*self.n_T, context_mask=context_mask, text_guide_emb=text_guide_emb)
                    else:
                        model_output = self.nn_model(y_i, x_batch, clip, in_grid, pair_in=pair_in, pair_out=pair_out, t=t_is_doubled, context_mask=context_mask, text_guide_emb=text_guide_emb)
                
                # Apply classifier-free guidance
                if not is_zero:
                    output1 = model_output[:n_sample]
                    output2 = model_output[n_sample:]
                    model_output = (1 + self.guide_w) * output1 - self.guide_w * output2
                    y_i = y_i[:n_sample]
                
                # Extra steps use exponential interpolation
                if not predict_noise:
                    x0_pred = model_output
                    interp_weight = 0.9 ** extra_step
                    y_i = interp_weight * y_i + (1 - interp_weight) * x0_pred
                else:
                    # Use last DDIM parameters for noise prediction case
                    alpha_t = ddim_params["ddim_alphas"][-1]
                    alpha_prev = ddim_params["ddim_alphas_prev"][-1]
                    x0_pred = (y_i - (1 - alpha_t).sqrt() * model_output) / alpha_t.sqrt()
                    dir_xt = (1. - alpha_prev).sqrt() * model_output
                    y_i = alpha_prev.sqrt() * x0_pred + dir_xt
                
                if return_y_trace:
                    y_i_store.append(y_i.detach().cpu().numpy())

        # Apply z-score denormalization if enabled
        y_i = self._denormalize(y_i)

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                x = x + x2
            else:
                x = x1 + x2
            return x / 1.414
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            return x


class Model_cnn_mlp(nn.Module):
    def __init__(self, x_shape, n_hidden, y_dim, embed_dim, net_type, max_grid_size=30, output_dim=None):
        super(Model_cnn_mlp, self).__init__()

        self.x_shape = x_shape
        self.n_hidden = n_hidden
        self.y_dim = y_dim
        self.embed_dim = embed_dim
        self.n_feat = 64
        self.net_type = net_type
        self.max_grid_size = max_grid_size
        if output_dim is None:
            self.output_dim = y_dim  # by default, just output size of action space
        else:
            self.output_dim = output_dim  # sometimes overwrite, eg for discretised, mean/variance, mixture density models

        # set up CNN for image
        self.conv_down1 = nn.Sequential(
            ResidualConvBlock(self.x_shape[-1], self.n_feat, is_res=True),
            nn.MaxPool2d(2),
        )
        self.conv_down3 = nn.Sequential(
            ResidualConvBlock(self.n_feat, self.n_feat * 2, is_res=True),
            nn.MaxPool2d(2),
        )
        self.imageembed = nn.Sequential(nn.AvgPool2d(8))

        cnn_out_dim = self.n_feat * 2  # how many features after flattening -- WARNING, will have to adjust this for diff size input resolution
        # it is the flattened size after CNN layers, and average pooling

        # then once have flattened vector out of CNN, just feed into previous Model_mlp_diff_embed
        self.nn_downstream = Model_mlp_diff_embed(
            cnn_out_dim,
            self.n_hidden,
            self.y_dim,
            self.embed_dim,
            self.output_dim,
            is_dropout=False,
            is_batch=False,
            activation="relu",
            net_type=self.net_type,
            use_prev=False,
            max_grid_size=self.max_grid_size
        )

    def forward(self, y, x, t, context_mask, x_embed=None):
        # torch expects batch_size, channels, height, width
        # but we feed in batch_size, height, width, channels

        if x_embed is None:
            x_embed = self.embed_context(x)
        else:
            # otherwise, we already extracted x_embed
            # e.g. outside of sampling loop
            pass

        return self.nn_downstream(y, x_embed, t, context_mask)

    def embed_context(self, x):
        x = x.permute(0, 3, 2, 1)
        x1 = self.conv_down1(x)
        x3 = self.conv_down3(x1)  # [batch_size, 128, 35, 18]
        # c3 is [batch size, 128, 4, 4]
        x_embed = self.imageembed(x3)
        # c_embed is [batch size, 128, 1, 1]
        x_embed = x_embed.view(x.shape[0], -1)
        # c_embed is now [batch size, 128]
        return x_embed


class Model_mlp(nn.Module):
    def __init__(self, x_shape, n_hidden, y_dim, embed_dim, net_type, max_grid_size=30, output_dim=None, use_in_out=False, use_enhanced_pair_encoding=False, disable_pair_encoding=False, use_concept_guidance=False, use_cfg_for_concept=True, concept_scale=1.0):
        super(Model_mlp, self).__init__()
        self.x_shape = x_shape
        self.n_hidden = n_hidden
        self.y_dim = y_dim
        self.embed_dim = embed_dim
        self.net_type = net_type
        self.max_grid_size = max_grid_size
        self.use_in_out = use_in_out
        self.use_enhanced_pair_encoding = use_enhanced_pair_encoding  # whether to use enhanced pair encoding
        self.disable_pair_encoding = disable_pair_encoding  # whether to disable pair encoding
        self.use_concept_guidance = use_concept_guidance  # whether to use concept guidance
        self.use_cfg_for_concept = use_cfg_for_concept  # whether to use CFG for concept (vs direct conditioning)
        self.concept_scale = concept_scale  # Scale factor for concept embedding (default 1.0 = no scaling)

        if output_dim is None:
            self.output_dim = y_dim  # by default, just output size of action space
        else:
            self.output_dim = output_dim  # sometimes overwrite, eg for discretised, mean/variance, mixture density models

        self.nn = Model_mlp_diff_embed(
            self.x_shape,
            self.n_hidden,
            self.y_dim,
            self.embed_dim,
            self.output_dim,
            is_dropout=False,
            is_batch=False,
            activation="relu",
            net_type=self.net_type,
            use_prev=False,
            #h_dims=[int(y_dim*4), int(y_dim*2), int(y_dim), int(y_dim/2)],
            h_dims=[128,32,16,8],
            max_grid_size=self.max_grid_size,
            use_in_out=self.use_in_out,
            use_enhanced_pair_encoding=self.use_enhanced_pair_encoding,
            use_concept_guidance=self.use_concept_guidance,
            use_cfg_for_concept=self.use_cfg_for_concept,
            concept_scale=self.concept_scale
        )

    def forward(self, y, x, clip, in_grid, pair_in=None, pair_out=None, t=None, context_mask=None, x_embed=None, text_guide_emb=None):
        # print(pair_in.shape)
        # print(pair_out.shape)

        return self.nn(y, x, clip, in_grid, t, context_mask, pair_in=pair_in, pair_out=pair_out, text_guide_emb=text_guide_emb)
