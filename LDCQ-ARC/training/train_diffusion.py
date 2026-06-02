import os
import sys
import datetime

curr_folder=os.path.abspath(__file__)
parent_folder=os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder) 

from argparse import ArgumentParser
import os
# from comet_ml import Experiment
try:
    import wandb
    if not hasattr(wandb, 'init'):
        wandb = None
except Exception:
    wandb = None

# import d4rl
# import gym
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from models.diffusion_models import (
    Model_mlp,
    Model_cnn_mlp,
    Model_Cond_Diffusion,
)
# from models.concept_encoder import ConceptEncoder  # imported conditionally below
# from models.discrete_concept_encoder import DiscreteConceptEncoder, create_concept_mapping_from_data  # imported conditionally below


class PriorDataset(Dataset):
    def __init__(
        self, data_dir, filename, train_or_test, test_prop, sample_z=False, use_direct_output_for_diffusion=False
    ):
        # just load it all into RAM
        self.state_all = np.load(os.path.join(data_dir, filename + "_states.npy"), allow_pickle=True)
        self.clip_all = np.load(os.path.join(data_dir, filename + "_clip.npy"), allow_pickle=True)
        self.in_grid_all = np.load(os.path.join(data_dir, filename + "_in_grid.npy"), allow_pickle=True)
        self.latent_all = np.load(os.path.join(data_dir, filename + "_latents.npy"), allow_pickle=True)
        self.pair_in_all = np.load(os.path.join(data_dir, filename + "_pair_in.npy"), allow_pickle=True)
        self.pair_out_all = np.load(os.path.join(data_dir, filename + "_pair_out.npy"), allow_pickle=True)

        # Load concepts if available
        concepts_path = os.path.join(data_dir, filename + "_concepts.npy")
        if os.path.exists(concepts_path):
            self.concepts_all = np.load(concepts_path, allow_pickle=True)
        else:
            # If concepts don't exist, use empty strings
            self.concepts_all = np.array([''] * len(self.state_all), dtype=object)

        # Load direct output embeddings if available (for diffusion guidance)
        self.use_direct_output_for_diffusion = use_direct_output_for_diffusion
        if use_direct_output_for_diffusion:
            direct_output_emb_path = os.path.join(data_dir, filename + "_direct_output_emb.npy")
            if os.path.exists(direct_output_emb_path):
                self.direct_output_emb_all = np.load(direct_output_emb_path, allow_pickle=True)
                print(f"Loaded direct output embeddings: shape {self.direct_output_emb_all.shape}")
            else:
                raise FileNotFoundError(f"Direct output embeddings not found at {direct_output_emb_path}. "
                                      f"Please run collect_diffusion_data.py with --use_direct_output_for_diffusion 1 first.")
        else:
            self.direct_output_emb_all = None

        if sample_z:
            self.latent_all_std = np.load(os.path.join(data_dir, filename + "_latents_std.npy"), allow_pickle=True)

        # self.state_mean = self.state_all.mean(axis=0)
        # self.state_std = self.state_all.std(axis=0)
        #self.state_all = (self.state_all - self.state_mean) / self.state_std

        # self.latent_mean = self.latent_all.mean(axis=0)
        # self.latent_std = self.latent_all.std(axis=0)
        #self.latent_all = (self.latent_all - self.latent_mean) / self.latent_std
        
        self.sample_z = sample_z
        n_train = int(self.state_all.shape[0] * (1 - test_prop))
        
        if train_or_test == "train":
            self.state_all = self.state_all[:n_train]
            self.clip_all = self.clip_all[:n_train]
            self.in_grid_all = self.in_grid_all[:n_train]
            self.latent_all = self.latent_all[:n_train]
            self.pair_in_all = self.pair_in_all[:n_train]
            self.pair_out_all = self.pair_out_all[:n_train]
            self.concepts_all = self.concepts_all[:n_train]
            if self.direct_output_emb_all is not None:
                self.direct_output_emb_all = self.direct_output_emb_all[:n_train]
            if sample_z:
                self.latent_all_std = self.latent_all_std[:n_train]
        elif train_or_test == "test":
            self.state_all = self.state_all[n_train:]
            self.clip_all = self.clip_all[n_train:]
            self.in_grid_all = self.in_grid_all[n_train:]
            self.latent_all = self.latent_all[n_train:]
            self.pair_in_all = self.pair_in_all[n_train:]
            self.pair_out_all = self.pair_out_all[n_train:]
            self.concepts_all = self.concepts_all[n_train:]
            if self.direct_output_emb_all is not None:
                self.direct_output_emb_all = self.direct_output_emb_all[n_train:]
            if sample_z:
                self.latent_all_std = self.latent_all_std[n_train:]
        else:
            raise NotImplementedError

    def __len__(self):
        return self.state_all.shape[0]

    def __getitem__(self, index):
        state = self.state_all[index]
        clip = self.clip_all[index]
        in_grid = self.in_grid_all[index]
        latent = self.latent_all[index]
        pair_in = self.pair_in_all[index]
        pair_out = self.pair_out_all[index]
        concept = self.concepts_all[index]
        direct_output_emb = self.direct_output_emb_all[index] if self.direct_output_emb_all is not None else None
        if self.sample_z:
            latent_std = self.latent_all_std[index]
            latent = np.random.normal(latent,latent_std)
            # latent = (latent - self.latent_mean) / self.latent_std
        #else:
        #    latent = (latent - self.latent_mean) / self.latent_std
        return (state, clip, in_grid, pair_in, pair_out, latent, concept, direct_output_emb)


def custom_collate_fn(batch):
    """Custom collate function to handle None values in direct_output_emb"""
    # Unpack the batch
    states, clips, in_grids, pair_ins, pair_outs, latents, concepts, direct_output_embs = zip(*batch)

    # Convert to tensors (except concepts and direct_output_embs which may contain None)
    states = torch.stack([torch.tensor(s) for s in states])
    clips = torch.stack([torch.tensor(c) for c in clips])
    in_grids = torch.stack([torch.tensor(ig) for ig in in_grids])
    pair_ins = torch.stack([torch.tensor(pi) for pi in pair_ins])
    pair_outs = torch.stack([torch.tensor(po) for po in pair_outs])
    latents = torch.stack([torch.tensor(l) for l in latents])

    # Concepts: keep as list
    concepts = list(concepts)

    # Direct output embeddings: keep as list (may contain None)
    direct_output_embs = list(direct_output_embs)

    return states, clips, in_grids, pair_ins, pair_outs, latents, concepts, direct_output_embs

def train(args):
    # get datasets set up
    torch_data_train = PriorDataset(
        args.data_dir, args.skill_model_filename[:-4], train_or_test="train", test_prop=args.test_split, sample_z=args.sample_z,
        use_direct_output_for_diffusion=getattr(args, 'use_direct_output_for_diffusion', False)
    )
    dataload_train = DataLoader(
        torch_data_train, batch_size=args.batch_size, shuffle=True, num_workers=8,
        collate_fn=custom_collate_fn, pin_memory=True, persistent_workers=True
    )

    if args.test_split > 0.0:
        torch_data_test = PriorDataset(
            args.data_dir, args.skill_model_filename[:-4], train_or_test="test", test_prop=args.test_split, sample_z=args.sample_z,
            use_direct_output_for_diffusion=getattr(args, 'use_direct_output_for_diffusion', False)
        )
        dataload_test = DataLoader(
            torch_data_test, batch_size=args.batch_size, shuffle=True, num_workers=8,
            collate_fn=custom_collate_fn, pin_memory=True, persistent_workers=True
        )

    # x_shape = torch_data_train.state_all.shape[1]
    x_shape = args.s_dim
    y_dim = torch_data_train.latent_all.shape[1]

    # Check if using concept guidance
    use_concept_guidance = getattr(args, 'use_concept_guidance', False)
    use_discrete_concepts = getattr(args, 'use_discrete_concepts', False)  # Default to text-based (0)

    # create model
    nn_model = Model_mlp(
        x_shape = x_shape,
        n_hidden = args.n_hidden,
        y_dim = y_dim,
        embed_dim = 128,    # h_dim*8 = 16*8 = 128
        net_type = args.net_type,
        max_grid_size=args.max_grid_size,
        use_in_out=args.use_in_out,  # 0: False, 1: True
        use_enhanced_pair_encoding=args.use_enhanced_pair_encoding,  # 0: False, 1: True
        disable_pair_encoding=args.disable_pair_encoding,  # 0: use pairs, 1: disable pairs
        use_concept_guidance=use_concept_guidance,  # Whether to use concept guidance (adds concept to input)
        use_cfg_for_concept=bool(getattr(args, 'use_cfg_for_concept', True)),  # Whether to use CFG (vs direct conditioning)
        concept_scale=getattr(args, 'concept_scale', 1.0),  # Scale factor for concept embedding (default 1.0 = no scaling)
    ).to(args.device)
    
    model = Model_Cond_Diffusion(
        nn_model,
        betas=(1e-4, 0.02),
        n_T=args.diffusion_steps,
        device=args.device,
        x_dim=x_shape,
        y_dim=y_dim,
        drop_prob=args.drop_prob,
        guide_w=args.cfg_weight,
        normalize_latent=args.normalize_latent,
        schedule=args.schedule,
        use_in_out=args.use_in_out,  # 0: False, 1: True
    ).to(args.device)

    # Apply torch.compile to diffusion model if enabled
    use_compile = bool(args.use_compile)
    if use_compile:
        print(f"Compiling diffusion model with mode='{args.compile_mode}'...")
        print("⏳ First iteration will be slower due to compilation...")
        model = torch.compile(model, mode=args.compile_mode)
        print("Diffusion model compiled!")

    # Initialize ConceptEncoder if using concept guidance
    if use_concept_guidance:
        from models.concept_encoder import ConceptEncoder
        from models.discrete_concept_encoder import DiscreteConceptEncoder, create_concept_mapping_from_data
        if use_discrete_concepts:
            print("Initializing DiscreteConceptEncoder for concept-guided diffusion training...")

            # Create concept mapping from data
            concept_to_id = create_concept_mapping_from_data(args.data_dir, "_concepts.npy")
            num_concepts = len(concept_to_id)

            if num_concepts == 0:
                print("WARNING: No concepts found in data! Disabling concept guidance.")
                concept_encoder = None
            else:
                concept_encoder = DiscreteConceptEncoder(
                    num_concepts=num_concepts,
                    embedding_dim=y_dim,  # Match latent dimension
                    concept_to_id=concept_to_id,
                    device=args.device
                )
                # Save concept mappings AND weights for later use
                base_filename = args.skill_model_filename[:-4]
                mapping_path = os.path.join(args.checkpoint_dir, base_filename + '_concept_mapping.json')
                weights_path = os.path.join(args.checkpoint_dir, base_filename + '_concept_weights.pth')
                concept_encoder.save_mappings(mapping_path)
                concept_encoder.save_weights(weights_path)
        else:
            print("Initializing text-based ConceptEncoder for diffusion training...")
            concept_encoder = ConceptEncoder(
                model_name='all-MiniLM-L6-v2',
                projection_dim=y_dim,  # Match latent dimension
                device=args.device
            )
            concept_encoder.eval()  # Keep encoder frozen during diffusion training

            # Save text-based ConceptEncoder projection weights for later use
            base_filename = args.skill_model_filename[:-4]
            text_encoder_weights_path = os.path.join(args.checkpoint_dir, base_filename + '_text_concept_encoder.pth')
            concept_encoder.save_projection(text_encoder_weights_path)
            print(f"Saved text-based ConceptEncoder projection to: {text_encoder_weights_path}")
    else:
        concept_encoder = None

    # Load skill model's concept classifier if using concept classifier loss
    skill_model_concept_classifier = None
    if getattr(args, 'use_concept_classifier_loss', 0):
        print("Loading skill model's concept classifier for concept classifier loss...")
        from models.skill_model import SkillModel

        # Load skill model to get concept classifier
        skill_checkpoint_path = os.path.join(args.checkpoint_dir, args.skill_model_filename)
        skill_checkpoint = torch.load(skill_checkpoint_path, map_location=args.device)

        # Get model config from checkpoint
        z_dim = y_dim
        h_dim = args.n_hidden
        s_dim = args.s_dim
        a_dim = 36  # Fixed for ARCLE

        # Create skill model with same config as training
        temp_skill_model = SkillModel(
            state_dim=s_dim,
            a_dim=a_dim,
            z_dim=z_dim,
            h_dim=h_dim,
            horizon=5,
            policy_decoder_type='mlp',
            state_decoder_type='mlp',
            train_diffusion_prior=True,
            conditional_prior=True,
            normalize_latent=args.normalize_latent,
            max_grid_size=args.max_grid_size,
            use_in_out=bool(args.use_in_out),
            use_enhanced_pair_encoding=bool(getattr(args, 'use_enhanced_pair_encoding', 0)),
            use_shared_grid_embedding=bool(getattr(args, 'use_shared_grid_embedding', 0)),
            disable_pair_encoding=bool(getattr(args, 'disable_pair_encoding', 0)),
            use_split_pair_trajectory_encoding=bool(getattr(args, 'use_split_pair_trajectory_encoding', 0)),
            use_direct_output_predictor=False,
            use_direct_output_for_diffusion=False,
            use_positional_encoding=bool(args.use_positional_encoding),
            use_concept_guidance=bool(getattr(args, 'use_concept_guidance', 0)),
            use_cfg_for_concept=False,
            use_concept_in_encoder=bool(getattr(args, 'use_concept_in_encoder', 0)),
            num_concepts=num_concepts if use_concept_guidance and use_discrete_concepts else 0,
            encoder_type='gru'
        ).to(args.device)

        # Load weights
        temp_skill_model.load_state_dict(skill_checkpoint['model_state_dict'], strict=False)

        # Extract concept classifier and freeze it
        skill_model_concept_classifier = temp_skill_model.concept_classifier
        skill_model_concept_classifier.eval()
        for param in skill_model_concept_classifier.parameters():
            param.requires_grad = False

        # Create ID to concept mapping for loss calculation
        if use_concept_guidance and use_discrete_concepts and concept_to_id:
            id_to_concept = {v: k for k, v in concept_to_id.items()}
            print(f"Concept classifier loss enabled with weight {args.concept_classifier_weight}")
            print(f"Using {num_concepts} concepts for classification")
        else:
            print("WARNING: Concept classifier loss enabled but no concept mapping found!")
            skill_model_concept_classifier = None

        del temp_skill_model  # Free memory

    # Select Optimizer
    if(args.optimizer == "Adam"):
        optim = torch.optim.Adam(model.parameters(), lr=args.lrate)
    elif(args.optimizer == "AdamW"):
        optim = torch.optim.AdamW(model.parameters(), lr=args.lrate)
    else:
        ValueError("Invalid optimizer")

    # Initialize GradScaler for AMP
    use_amp = bool(getattr(args, 'use_amp', 0))
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print("AMP (Automatic Mixed Precision) enabled")

    best_test_loss = 10000000

    for ep in tqdm(range(args.n_epoch), desc="Epoch", mininterval=300.0):
        model.train()

        # lrate decay
        #optim.param_groups[0]["lr"] = args.lrate * ((np.cos((ep / args.n_epoch) * np.pi) + 1) / 2)
        optim.param_groups[0]["lr"] = args.lrate * ((np.cos((ep / 75) * np.pi) + 1))

        # train loop
        model.train()
        pbar = tqdm(dataload_train, mininterval=300.0)
        loss_ep, n_batch = 0, 0

        for x_batch, clip_batch, in_grid_batch, pair_in_batch, pair_out_batch, y_batch, concept_batch, direct_output_emb_batch in pbar:
            x_batch = x_batch.type(torch.FloatTensor).to(args.device)   # (Batch, 1, 30, 30)
            clip_batch = clip_batch.type(torch.FloatTensor).to(args.device)   # (Batch, 1, 30, 30)
            in_grid_batch = in_grid_batch.type(torch.FloatTensor).to(args.device)   # (Batch, 1, 30, 30)
            pair_in_batch = pair_in_batch.type(torch.FloatTensor).to(args.device)
            pair_out_batch = pair_out_batch.type(torch.FloatTensor).to(args.device)

            y_batch = y_batch.type(torch.FloatTensor).to(args.device)   # (Batch, z_dim)

            # Determine text_guide_emb: use direct output embedding OR concept encoding
            text_guide_emb = None

            # Priority 1: Direct output guidance (if enabled and available)
            if getattr(args, 'use_direct_output_for_diffusion', False) and direct_output_emb_batch is not None:
                # Filter out None values and convert to tensor
                valid_embeddings = [emb for emb in direct_output_emb_batch if emb is not None]
                if valid_embeddings:
                    text_guide_emb = torch.tensor(np.array(valid_embeddings), dtype=torch.float32).to(args.device)
                    # Expand to match batch size if needed
                    if text_guide_emb.shape[0] < x_batch.shape[0]:
                        text_guide_emb = text_guide_emb.repeat(x_batch.shape[0] // text_guide_emb.shape[0] + 1, 1)[:x_batch.shape[0]]

            # Priority 2: Concept guidance (if direct output not used)
            elif use_concept_guidance and concept_encoder is not None:
                # Filter out empty concepts
                valid_concepts = [c for c in concept_batch if c]
                if valid_concepts:
                    with torch.no_grad():
                        text_guide_emb = concept_encoder(valid_concepts, batch_size=1)
                        # Expand to match batch size if needed
                        if text_guide_emb.shape[0] < x_batch.shape[0]:
                            # Repeat for items without concepts
                            text_guide_emb = text_guide_emb.repeat(x_batch.shape[0] // text_guide_emb.shape[0] + 1, 1)[:x_batch.shape[0]]

            # Calculate diffusion loss (and optionally get predicted x0 for concept classifier loss)
            with autocast(enabled=use_amp):
                if skill_model_concept_classifier is not None and getattr(args, 'use_concept_classifier_loss', 0):
                    # Get both loss and predicted clean latent
                    diffusion_loss, pred_x0 = model.loss_on_batch(
                        x_batch, clip_batch, in_grid_batch, pair_in_batch, pair_out_batch, y_batch,
                        args.predict_noise, text_guide_emb=text_guide_emb, return_pred_x0=True
                    )

                    # Calculate concept classifier loss on predicted clean latent
                    # Get concept IDs from concept names
                    concept_ids = []
                    for concept_name in concept_batch:
                        if concept_name and concept_name in concept_to_id:
                            concept_ids.append(concept_to_id[concept_name])
                        else:
                            concept_ids.append(-1)  # Invalid concept

                    # Filter out samples with invalid concepts
                    valid_indices = [i for i, cid in enumerate(concept_ids) if cid != -1]

                    if len(valid_indices) > 0:
                        # Get predictions for valid concepts only
                        valid_pred_x0 = pred_x0[valid_indices]
                        valid_concept_ids = torch.tensor([concept_ids[i] for i in valid_indices], dtype=torch.long).to(args.device)

                        # Get concept classifier predictions
                        concept_logits = skill_model_concept_classifier(valid_pred_x0)

                        # Calculate cross entropy loss
                        import torch.nn.functional as F
                        concept_classifier_loss = F.cross_entropy(concept_logits, valid_concept_ids)

                        # Combined loss
                        loss = diffusion_loss + args.concept_classifier_weight * concept_classifier_loss

                        # Log both losses
                        if wandb is not None:
                            try:
                                wandb.log({
                                    "train_diffusion/diffusion_loss": diffusion_loss.item(),
                                    "train_diffusion/concept_classifier_loss": concept_classifier_loss.item(),
                                    "train_diffusion/total_loss": loss.item()
                                })
                            except Exception: pass
                    else:
                        # No valid concepts in this batch, use only diffusion loss
                        loss = diffusion_loss
                        if wandb is not None:
                            try: wandb.log({"train_diffusion/loss": loss.item()})
                            except Exception: pass
                else:
                    # Standard training without concept classifier loss
                    loss = model.loss_on_batch(x_batch, clip_batch, in_grid_batch, pair_in_batch, pair_out_batch, y_batch, args.predict_noise, text_guide_emb=text_guide_emb)
                    if wandb is not None:
                        try: wandb.log({"train_diffusion/loss": loss.item()})
                        except Exception: pass

            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            loss_ep += loss.detach().item()
            n_batch += 1
            pbar.set_description(f"train loss: {loss_ep/n_batch:.4f}")
        # experiment.log_metric("train_loss", loss_ep/n_batch, step=ep)
        if wandb is not None:
            try: wandb.log({"train_diffusion/mean_loss": loss_ep/n_batch})
            except Exception: pass
        torch.save(nn_model, os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4] + '_diffusion_prior.pt'))

        # test loop
        if args.test_split > 0.0:
            model.eval()
            pbar = tqdm(dataload_test, mininterval=300.0)
            loss_ep, n_batch = 0, 0

            with torch.no_grad():
                for x_batch, clip_batch, in_grid_batch, pair_in_batch, pair_out_batch, y_batch, concept_batch, direct_output_emb_batch in pbar:
                    x_batch = x_batch.type(torch.FloatTensor).to(args.device)
                    clip_batch = clip_batch.type(torch.FloatTensor).to(args.device)
                    in_grid_batch = in_grid_batch.type(torch.FloatTensor).to(args.device)
                    pair_in_batch = pair_in_batch.type(torch.FloatTensor).to(args.device)
                    pair_out_batch = pair_out_batch.type(torch.FloatTensor).to(args.device)
                    y_batch = y_batch.type(torch.FloatTensor).to(args.device)

                    # Determine text_guide_emb: use direct output embedding OR concept encoding
                    text_guide_emb = None

                    # Priority 1: Direct output guidance (if enabled and available)
                    if getattr(args, 'use_direct_output_for_diffusion', False) and direct_output_emb_batch is not None:
                        valid_embeddings = [emb for emb in direct_output_emb_batch if emb is not None]
                        if valid_embeddings:
                            text_guide_emb = torch.tensor(np.array(valid_embeddings), dtype=torch.float32).to(args.device)
                            if text_guide_emb.shape[0] < x_batch.shape[0]:
                                text_guide_emb = text_guide_emb.repeat(x_batch.shape[0] // text_guide_emb.shape[0] + 1, 1)[:x_batch.shape[0]]

                    # Priority 2: Concept guidance (if direct output not used)
                    elif use_concept_guidance and concept_encoder is not None:
                        valid_concepts = [c for c in concept_batch if c]
                        if valid_concepts:
                            text_guide_emb = concept_encoder(valid_concepts, batch_size=1)
                            if text_guide_emb.shape[0] < x_batch.shape[0]:
                                text_guide_emb = text_guide_emb.repeat(x_batch.shape[0] // text_guide_emb.shape[0] + 1, 1)[:x_batch.shape[0]]

                    # Calculate test loss (same logic as training)
                    if skill_model_concept_classifier is not None and getattr(args, 'use_concept_classifier_loss', 0):
                        # Get both loss and predicted clean latent
                        diffusion_loss, pred_x0 = model.loss_on_batch(
                            x_batch, clip_batch, in_grid_batch, pair_in_batch, pair_out_batch, y_batch,
                            args.predict_noise, text_guide_emb=text_guide_emb, return_pred_x0=True
                        )

                        # Calculate concept classifier loss on predicted clean latent
                        concept_ids = []
                        for concept_name in concept_batch:
                            if concept_name and concept_name in concept_to_id:
                                concept_ids.append(concept_to_id[concept_name])
                            else:
                                concept_ids.append(-1)

                        valid_indices = [i for i, cid in enumerate(concept_ids) if cid != -1]

                        if len(valid_indices) > 0:
                            valid_pred_x0 = pred_x0[valid_indices]
                            valid_concept_ids = torch.tensor([concept_ids[i] for i in valid_indices], dtype=torch.long).to(args.device)
                            concept_logits = skill_model_concept_classifier(valid_pred_x0)

                            import torch.nn.functional as F
                            concept_classifier_loss = F.cross_entropy(concept_logits, valid_concept_ids)
                            loss = diffusion_loss + args.concept_classifier_weight * concept_classifier_loss
                        else:
                            loss = diffusion_loss
                    else:
                        loss = model.loss_on_batch(x_batch, clip_batch, in_grid_batch, pair_in_batch, pair_out_batch, y_batch, args.predict_noise, text_guide_emb=text_guide_emb)

                    loss_ep += loss.detach().item()
                    n_batch += 1
                    pbar.set_description(f"test loss: {loss_ep/n_batch:.4f}")
            # experiment.log_metric("test_loss", loss_ep/n_batch, step=ep)
            if wandb is not None:
                try: wandb.log({"train_diffusion/test_loss": loss_ep/n_batch})
                except Exception: pass

            if loss_ep < best_test_loss:
                best_test_loss = loss_ep
                torch.save(nn_model, os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4] + '_diffusion_prior_best.pt'))

        # elif ep%75==0:
        #     torch.save(nn_model, os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4] + '_diffusion_prior_best.pt'))

        if(ep%args.save_cycle == 0):
            torch.save(nn_model, os.path.join(args.checkpoint_dir, args.skill_model_filename[:-4]+'_'+str(ep)+'_epoch'+'.pt'))

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--lrate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--net_type', type=str, default='unet')
    parser.add_argument('--n_hidden', type=int, default=512)
    parser.add_argument('--test_split', type=float, default=0.05)
    parser.add_argument('--sample_z', type=int, default=0)

    parser.add_argument('--solar_dir', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=parent_folder+'/checkpoints/')
    parser.add_argument('--data_dir', type=str, default=parent_folder+'/data/')
    parser.add_argument('--skill_model_filename', type=str)
    parser.add_argument('--append_goals', type=int, default=0)

    parser.add_argument('--drop_prob', type=float, default=0.1)
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--cfg_weight', type=float, default=0.0)
    parser.add_argument('--predict_noise', type=int, default=0)
    parser.add_argument('--normalize_latent', type=int, default=0)
    parser.add_argument('--schedule', type=str, default='linear')

    # parser.add_argument('--a_dim', type=int, default=36)
    # parser.add_argument('--z_dim', type=int, default=128)
    # parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--s_dim', type=int, default=256)
    parser.add_argument('--date', type=str, default='00.00')

    parser.add_argument('--save_cycle', type=int, default=100)
    parser.add_argument('--gpu_name', type=str, required=True)
    parser.add_argument('--optimizer', type=str, default="AdamW")
    parser.add_argument('--max_grid_size', type=int, default=30)
    parser.add_argument('--use_in_out', type=int, default=0)  # 0: False, 1: True
    parser.add_argument('--use_enhanced_pair_encoding', type=int, default=0)  # 0: False, 1: True
    parser.add_argument('--use_shared_grid_embedding', type=int, default=0, help='Use shared grid embedding for pair encoding (0=disable, 1=enable)')
    parser.add_argument('--disable_pair_encoding', type=int, default=0)  # 0: use pairs, 1: disable pairs
    parser.add_argument('--use_split_pair_trajectory_encoding', type=int, default=0, help='Split pair and trajectory encoding (0=disable, 1=enable)')
    parser.add_argument('--use_concept_guidance', type=int, default=0, help='Use concept guidance for diffusion (0=disable, 1=enable)')
    parser.add_argument('--use_discrete_concepts', type=int, default=0, help='Use discrete concept IDs instead of text encoding (0=text, 1=discrete)')
    parser.add_argument('--use_cfg_for_concept', type=int, default=1, help='Use CFG for concept guidance (0=direct conditioning, 1=CFG)')
    parser.add_argument('--concept_scale', type=float, default=1.0, help='Scale factor for concept embedding strength (default 1.0 = no scaling, >1.0 = stronger concept)')
    parser.add_argument('--use_concept_in_encoder', type=int, default=0, help='Use concept in skill encoder (0=disable, 1=enable)')
    parser.add_argument('--num_concepts', type=int, default=0, help='Number of concepts (0=auto-detect from data)')
    parser.add_argument('--use_direct_output_for_diffusion', type=int, default=0, help='Use direct output predictor embedding for diffusion guidance (0=disable, 1=enable)')
    parser.add_argument('--use_concept_classifier_loss', type=int, default=0, help='Add concept classifier loss to diffusion training (0=disable, 1=enable)')
    parser.add_argument('--concept_classifier_weight', type=float, default=1.0, help='Weight for concept classifier loss (default 1.0)')
    parser.add_argument('--use_positional_encoding', type=int, default=1, help='Use 2D positional encoding for state embeddings (0=disable, 1=enable)')
    parser.add_argument('--use_amp', type=int, default=0, help='Use Automatic Mixed Precision for faster training (0=disable, 1=enable)')
    parser.add_argument('--use_compile', type=int, default=0, help='Use torch.compile for faster training (0=disable, 1=enable, requires PyTorch 2.0+)')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead', choices=['default', 'reduce-overhead', 'max-autotune'], help='Compilation mode for torch.compile')
    args = parser.parse_args()

    d = datetime.datetime.now()
    file_info = args.env+'_'+args.date
    filename = args.gpu_name+'_'+'diffusion_'+file_info
    task_name = args.solar_dir.split("/")[-1]
    # Handle different task name formats (e.g., "train.task_name.s10.H5" vs "five_task")
    if "." in task_name:
        task = task_name.split(".")[1]
    else:
        task = task_name
    base_config = vars(args) if 'args' in locals() else {}
    additional_config = {
            'task':task_name,
            'batch_size':args.batch_size,
            'sample_z':args.sample_z,
            'filename':filename,
            'net_type':args.net_type,
            'diffusion_steps':args.diffusion_steps,
            'skill_model_filename':args.skill_model_filename,
            'normalize_latent':args.normalize_latent,
            'schedule': args.schedule,
            'test_split': args.test_split,
            'append_goals': args.append_goals
        }

    config = {**base_config, **additional_config}

    # Hardcoded wandb config for now
    wandb_config = {
        'entity': 'dbsgh797210',
        'project': 'LDCQ_single',
        'api_key': '391af36b1546e19e6e1eb483f69c989abf5d202a'
    }
    
    os.environ["WANDB_API_KEY"] = wandb_config['api_key']
    
    run = None
    if wandb is not None:
        try:
            run = wandb.init(
                entity=wandb_config['entity'],
                project=wandb_config['project'],
                name='LDCQ_'+args.gpu_name+'_'+'diffusion'+'_'+task+'_'+str(d.month)+'.'+str(d.day)+'_'+str(d.hour)+'.'+str(d.minute),
                config=config,
                mode='offline',
            )
        except Exception as e:
            print(f"[WARN] wandb.init failed: {e}")
            run = None
    print("wandb run name: ", run.name if run is not None else 'local')
    train(args)
