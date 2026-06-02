import os
import sys
import datetime
from zoneinfo import ZoneInfo

curr_folder=os.path.abspath(__file__)
parent_folder=os.path.dirname(os.path.dirname(curr_folder))
sys.path.append(parent_folder)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.distributions.normal as Normal
from models.skill_model import SkillModel
# Conditionally import ConceptEncoder only when needed (to avoid sentence_transformers dependency)
# from models.concept_encoder import ConceptEncoder
# import h5py
from utils.utils import get_dataset, ARC_Segment_Dataset
import pickle
from tqdm import tqdm
import argparse
try:
    import wandb
    if not hasattr(wandb, 'init'):
        wandb = None
except Exception:
    wandb = None

def train(model, optimizer, train_loader, train_state_decoder, return_concept=False, concept_encoder=None, concept_to_id=None, concept_scale=1.0, concept_loss_weight=0.0, concept_contrastive_weight=0.0, contrastive_temperature=0.1, use_concept_for_diffusion=False, use_amp=False, scaler=None):

	losses = []

	pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train skill model : ", mininterval=600.0)

	for idx, batch in pbar:
		if return_concept:
			state, s_T, clip, clip_T, selection, operation, reward, terminated, _, in_grid, out_grid, ex_in, ex_out, concept = batch
		else:
			state, s_T, clip, clip_T, selection, operation, reward, terminated, _, in_grid, out_grid, ex_in, ex_out = batch

		states = state.cuda()
		clip = clip.cuda()
		s_T = s_T.cuda()
		in_grid = in_grid.cuda()
		actions = operation.cuda()
		selection = selection.cuda()
		pair_in = ex_in.cuda()
		pair_out = ex_out.cuda()

		# Prepare concept embeddings and labels if using concept guidance
		concept_emb = None
		concept_labels = None
		if return_concept and concept_encoder is not None:
			# concept is a tuple of strings, convert to embeddings (with projection)
			concept_emb = concept_encoder(list(concept)).cuda()
			# Also create concept labels for classification loss
			if concept_to_id is not None:
				concept_labels = torch.tensor([concept_to_id.get(c, len(concept_to_id)) for c in concept]).cuda()

		model.zero_grad()

		if use_amp:
			with autocast():
				if train_state_decoder:
					loss_tot, s_T_loss, a_loss, x_loss, y_loss,  h_loss, w_loss, kl_loss, diffusion_loss, direct_output_loss, concept_class_loss, concept_contrastive_loss = model.get_losses(states, s_T, clip, in_grid, actions, selection, pair_in, pair_out, train_state_decoder, concept_emb=concept_emb, concept_labels=concept_labels, concept_scale=concept_scale, concept_loss_weight=concept_loss_weight, concept_contrastive_weight=concept_contrastive_weight, contrastive_temperature=contrastive_temperature, use_concept_for_diffusion=use_concept_for_diffusion)
				else:
					loss_tot, a_loss, x_loss, y_loss,  h_loss, w_loss, kl_loss, diffusion_loss, direct_output_loss, concept_class_loss, concept_contrastive_loss = model.get_losses(states, s_T, clip, in_grid, actions, selection, pair_in, pair_out, train_state_decoder, concept_emb=concept_emb, concept_labels=concept_labels, concept_scale=concept_scale, concept_loss_weight=concept_loss_weight, concept_contrastive_weight=concept_contrastive_weight, contrastive_temperature=contrastive_temperature, use_concept_for_diffusion=use_concept_for_diffusion)

			scaler.scale(loss_tot).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			if train_state_decoder:
				loss_tot, s_T_loss, a_loss, x_loss, y_loss,  h_loss, w_loss, kl_loss, diffusion_loss, direct_output_loss, concept_class_loss, concept_contrastive_loss = model.get_losses(states, s_T, clip, in_grid, actions, selection, pair_in, pair_out, train_state_decoder, concept_emb=concept_emb, concept_labels=concept_labels, concept_scale=concept_scale, concept_loss_weight=concept_loss_weight, concept_contrastive_weight=concept_contrastive_weight, contrastive_temperature=contrastive_temperature, use_concept_for_diffusion=use_concept_for_diffusion)
			else:
				loss_tot, a_loss, x_loss, y_loss,  h_loss, w_loss, kl_loss, diffusion_loss, direct_output_loss, concept_class_loss, concept_contrastive_loss = model.get_losses(states, s_T, clip, in_grid, actions, selection, pair_in, pair_out, train_state_decoder, concept_emb=concept_emb, concept_labels=concept_labels, concept_scale=concept_scale, concept_loss_weight=concept_loss_weight, concept_contrastive_weight=concept_contrastive_weight, contrastive_temperature=contrastive_temperature, use_concept_for_diffusion=use_concept_for_diffusion)

			loss_tot.backward()
			optimizer.step()
  
		# log losses
		log_dict = {
			"train_skill/loss": loss_tot.item(),
			"train_skill/a_loss": a_loss.item(),
			"train_skill/x_loss": x_loss.item(),
			"train_skill/y_loss": y_loss.item(),
			"train_skill/h_loss": h_loss.item(),
			"train_skill/w_loss": w_loss.item(),
			"train_skill/kl_loss": kl_loss.item(),
			"train_skill/diffusion_loss": diffusion_loss.item() if train_diffusion_prior else diffusion_loss,
		}
		if train_state_decoder:
			log_dict["train_skill/s_T_loss"] = s_T_loss.item()
		if getattr(model, "use_direct_output_predictor", False):
			log_dict["train_skill/direct_output_loss"] = direct_output_loss.item()
		if isinstance(concept_class_loss, torch.Tensor):
			log_dict["train_skill/concept_class_loss"] = concept_class_loss.item()
		if isinstance(concept_contrastive_loss, torch.Tensor):
			log_dict["train_skill/concept_contrastive_loss"] = concept_contrastive_loss.item()
		if wandb is not None:
				try: wandb.log(log_dict)
				except Exception: pass
		losses.append(loss_tot.item())

		# pbar.set_description('Loss: %.3f' % loss_tot.item())
	
	mean_losses = np.mean(losses)

	if wandb is not None:
		try: wandb.log({"train_skill/mean_loss": mean_losses})
		except Exception: pass
 
	return mean_losses

def test(model, test_loader, test_state_decoder, test_num, return_concept=False, use_amp=False):

	losses = []
	s_T_losses = []
	a_losses = []
	kl_losses = []
	s_T_ents = []
	diffusion_losses = []

	with torch.no_grad():
		pbar = tqdm(enumerate(test_loader), total=test_num, desc="Test skill model : ", mininterval=300.0)

		for i, batch in pbar:
			if(i >= test_num):
				break

			if return_concept:
				state, s_T, clip, clip_T, selection, operation, reward, terminated, _, in_grid, out_grid, ex_in, ex_out, concept = batch
			else:
				state, s_T, clip, clip_T, selection, operation, reward, terminated, _, in_grid, out_grid, ex_in, ex_out = batch

			states = state.cuda()
			clip = clip.cuda()
			s_T = s_T.cuda()
			in_grid = in_grid.cuda()
			actions = operation.cuda()
			selection = selection.cuda()
			pair_in = ex_in.cuda()
			pair_out = ex_out.cuda()

			if use_amp:
				with autocast():
					if test_state_decoder:
						loss_tot, s_T_loss, a_loss, x_loss, y_loss,  h_loss, w_loss, kl_loss, diffusion_loss, direct_output_loss, concept_class_loss, concept_contrastive_loss = model.get_losses(states, s_T, clip, in_grid, actions, selection, pair_in, pair_out, test_state_decoder)
						s_T_losses.append(s_T_loss.item())
					else:
						loss_tot, a_loss, x_loss, y_loss,  h_loss, w_loss, kl_loss, diffusion_loss, direct_output_loss, concept_class_loss, concept_contrastive_loss = model.get_losses(states, s_T, clip, in_grid, actions, selection, pair_in, pair_out, test_state_decoder)
			else:
				if test_state_decoder:
					loss_tot, s_T_loss, a_loss, x_loss, y_loss,  h_loss, w_loss, kl_loss, diffusion_loss, direct_output_loss, concept_class_loss, concept_contrastive_loss = model.get_losses(states, s_T, clip, in_grid, actions, selection, pair_in, pair_out, test_state_decoder)
					s_T_losses.append(s_T_loss.item())
				else:
					loss_tot, a_loss, x_loss, y_loss,  h_loss, w_loss, kl_loss, diffusion_loss, direct_output_loss, concept_class_loss, concept_contrastive_loss = model.get_losses(states, s_T, clip, in_grid, actions, selection, pair_in, pair_out, test_state_decoder)

			# log losses
			losses.append(loss_tot.item())
			a_losses.append(a_loss.item())
			kl_losses.append(kl_loss.item())
			diffusion_losses.append(diffusion_loss.item() if train_diffusion_prior else diffusion_loss)

	mean_losses = np.mean(losses)

	if wandb is not None:
		try: wandb.log({"train_skill/test_loss": mean_losses})
		except Exception: pass
 
	if train_diffusion_prior:
		return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses), np.mean(diffusion_losses)
	return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(kl_losses), None

def test_acc(model, test_loader, test_num, test_state_decoder=False, return_concept=False, use_amp=False):

	s_T_ents = []

	total_num = 0
	correct = 0
	correct_0 = 0
	total_0 = 0
	total_op, total_x, total_y, total_h, total_w = 0, 0, 0, 0, 0
	correct_op, correct_x, correct_y, correct_h, correct_w = 0, 0, 0, 0, 0

	# State decoder: count perfect grid matches
	total_state_grids = 0
	correct_state_grids = 0

	# Direct output predictor: count perfect grid matches
	total_direct_output_grids = 0
	correct_direct_output_grids = 0

	with torch.no_grad():
		pbar = tqdm(enumerate(test_loader), total=test_num, desc="Test Accuracy skill model : ", mininterval=300.0)

		for i, batch in pbar:
			if(i >= test_num):
				break

			if return_concept:
				state, s_T, clip, clip_T, selection, operation, reward, terminated, _, in_grid, out_grid, ex_in, ex_out, concept = batch
			else:
				state, s_T, clip, clip_T, selection, operation, reward, terminated, _, in_grid, out_grid, ex_in, ex_out = batch

			states = state.cuda()
			s_T = s_T.cuda()
			clip = clip.cuda()
			in_grid = in_grid.cuda()
			operation = operation.cuda()
			selection = selection.cuda()
			pair_in = ex_in.cuda()
			pair_out = ex_out.cuda()

			# Use AMP if enabled
			if use_amp:
				with autocast():
					z_post_means, z_post_sigs = model.encoder(states, clip, in_grid, operation, selection, pair_in, pair_out)
			else:
				z_post_means, z_post_sigs = model.encoder(states, clip, in_grid, operation, selection, pair_in, pair_out)

			if not model.normalize_latent:
				z_sampled = model.reparameterize(z_post_means, z_post_sigs)
			else:
				z_sampled = z_post_means


			# Test direct output predictor accuracy if enabled
			if getattr(model, 'use_direct_output_predictor', False):
				if use_amp:
					with autocast():
						# Predict output from in_grid and example pairs
						direct_output_logits = model.direct_output_predictor(in_grid, pair_in, pair_out)
				else:
					direct_output_logits = model.direct_output_predictor(in_grid, pair_in, pair_out)

				# Get predicted state from logits (argmax per cell)
				batch_size = direct_output_logits.shape[0]
				grid_size = args.max_grid_size * args.max_grid_size
				direct_logits_reshaped = direct_output_logits.reshape(batch_size, grid_size, 11)
				predicted_output = torch.argmax(direct_logits_reshaped, dim=-1)  # (batch, grid_size)

				# Ground truth output (final state)
				target_output = states[:, -1, :].reshape(batch_size, grid_size).long()

				# Check if entire grid matches perfectly
				grid_match = torch.all(predicted_output == target_output, dim=-1)  # (batch,)
				correct_direct_output_grids += grid_match.sum().item()
				total_direct_output_grids += batch_size
			total_0 += 1
			for h in range(H):
				if use_amp:
					with autocast():
						pred_operation, pred_x, pred_y, pred_h, pred_w = model.decoder.ll_policy.tensor_policy(
							states[:, h, :, :], clip[:, h, :, :], in_grid, z_sampled, pair_in, pair_out)
				else:
					pred_operation, pred_x, pred_y, pred_h, pred_w = model.decoder.ll_policy.tensor_policy(
						states[:, h, :, :], clip[:, h, :, :], in_grid, z_sampled, pair_in, pair_out)

				p_operation = torch.argmax(pred_operation, dim=-1)
				p_x = torch.argmax(pred_x, dim=-1)
				p_y = torch.argmax(pred_y, dim=-1)
				p_h = torch.argmax(pred_h, dim=-1)
				p_w = torch.argmax(pred_w, dim=-1)

				if operation[0, h, 0] == p_operation:
					correct_op += 1
				if selection[0, h, 0] == p_x:
					correct_x += 1
				if selection[0, h, 1] == p_y:
					correct_y += 1
				if selection[0, h, 2] == p_h:
					correct_h += 1
				if selection[0, h, 3] == p_w:
					correct_w += 1

				total_op += 1
				total_x += 1
				total_y += 1
				total_h += 1
				total_w += 1

				if (operation[0, h, 0] == p_operation and
					selection[0, h, 0] == p_x and
					selection[0, h, 1] == p_y and
					selection[0, h, 2] == p_h and
					selection[0, h, 3] == p_w):
					correct += 1
					if h == 0:
						correct_0 += 1

					total_num += 1

			# Test state decoder accuracy if enabled
			if test_state_decoder and model.state_decoder_type in ['arc_residual', 'arc_direct']:
				s_0 = states[:, 0:1, :]
				clip_0 = clip[:, 0:1, :]

				# Get state decoder prediction
				if use_amp:
					with autocast():
						sT_mean, sT_sig, sT_logits = model.decoder.abstract_dynamics(
							s_0, clip_0, in_grid, z_sampled.detach(), pair_in, pair_out
						)
				else:
					sT_mean, sT_sig, sT_logits = model.decoder.abstract_dynamics(
						s_0, clip_0, in_grid, z_sampled.detach(), pair_in, pair_out
					)

				# Get predicted state from logits (argmax per cell)
				batch_size = sT_logits.shape[0]
				sT_logits_reshaped = sT_logits.reshape(batch_size, args.max_grid_size*args.max_grid_size, 11)
				predicted_state = torch.argmax(sT_logits_reshaped, dim=-1)  # (batch, state_dim)

				# Ground truth state
				target_state = s_T[:, -1, :].reshape(batch_size, args.max_grid_size*args.max_grid_size).long()

				# Check if entire grid matches perfectly
				grid_match = torch.all(predicted_state == target_state, dim=-1)  # (batch,)
				correct_state_grids += grid_match.sum().item()
				total_state_grids += batch_size
   
	# Avoid division by zero
	if total_num == 0:
		print("Warning: No test samples processed (total_num=0). Skipping accuracy calculation.")
		return 0.0

	log_dict = {
		"train_skill/test_acc_whole": 100.0 * correct / total_num if total_num > 0 else 0.0,
		"train_skill/test_acc_s0": 100.0 * correct_0 / total_0 if total_0 > 0 else 0.0,
		"train_skill/test_acc_operation": 100.0 * correct_op / total_op if total_op > 0 else 0.0,
		"train_skill/test_acc_x": 100.0 * correct_x / total_x if total_x > 0 else 0.0,
		"train_skill/test_acc_y": 100.0 * correct_y / total_y if total_y > 0 else 0.0,
		"train_skill/test_acc_h": 100.0 * correct_h / total_h if total_h > 0 else 0.0,
		"train_skill/test_acc_w": 100.0 * correct_w / total_w if total_w > 0 else 0.0,
	}

	# Add state decoder accuracy if tested
	if test_state_decoder and total_state_grids > 0:
		state_acc = 100.0 * correct_state_grids / total_state_grids
		log_dict["train_skill/test_acc_state_decoder"] = state_acc
		print(f"State Decoder Accuracy: {state_acc:.2f}% ({correct_state_grids}/{total_state_grids} perfect matches)")

	# Add direct output predictor accuracy if enabled
	if getattr(model, 'use_direct_output_predictor', False) and total_direct_output_grids > 0:
		direct_output_acc = 100.0 * correct_direct_output_grids / total_direct_output_grids
		log_dict["train_skill/test_acc_direct_output"] = direct_output_acc
		print(f"Direct Output Predictor Accuracy: {direct_output_acc:.2f}% ({correct_direct_output_grids}/{total_direct_output_grids} perfect matches)")

	if wandb is not None:
		try: wandb.log(log_dict)
		except Exception: pass
	return correct/total_num

def test_acc_prior(model, test_loader, test_num, return_concept=False, concept_to_id=None, use_amp=False):

	total_num = 0
	correct = 0

	# Concept classification tracking
	if return_concept and concept_to_id is not None and model.concept_classifier is not None:
		num_concepts = len(concept_to_id)
		id_to_concept = {v: k for k, v in concept_to_id.items()}
		concept_correct = {i: 0 for i in range(num_concepts)}
		concept_total = {i: 0 for i in range(num_concepts)}
		concept_pred_dist = {i: {j: 0 for j in range(num_concepts)} for i in range(num_concepts)}
		track_concept = True
	else:
		track_concept = False

	with torch.no_grad():
		pbar = tqdm(enumerate(test_loader), total=test_num, desc="Test Accuracy skill model : ", mininterval=300.0)

		for i, batch in pbar:
			if(i >= test_num):
				break

			if return_concept:
				state, s_T, clip, clip_T, selection, operation, reward, terminated, _, in_grid, out_grid, ex_in, ex_out, concept = batch
			else:
				state, s_T, clip, clip_T, selection, operation, reward, terminated, _, in_grid, out_grid, ex_in, ex_out = batch

			states = state.cuda()
			clip = clip.cuda()
			in_grid = in_grid.cuda()
			actions = operation.cuda()
			selection = selection.cuda()
			pair_in = ex_in.cuda()
			pair_out = ex_out.cuda()

			# print("state_0 : {0}".format(states.shape))
			# print("clip_0 : {0}".format(clip.shape))
			# print("pair_in : {0}".format(pair_in.shape))
			# print("pair_out : {0}".format(pair_out.shape))

			if use_amp:
				with autocast():
					latent, latent_prior_std = model.prior(states[:, 0:1, :, :], clip[:, 0:1, :, :], in_grid, pair_in, pair_out)
					pred_operation, pred_x, pred_y, pred_h, pred_w = model.decoder.ll_policy.tensor_policy(states[:, 0, :, :], clip[:, 0, :, :], in_grid, latent, pair_in, pair_out)
			else:
				latent, latent_prior_std = model.prior(states[:, 0:1, :, :], clip[:, 0:1, :, :], in_grid, pair_in, pair_out)
				pred_operation, pred_x, pred_y, pred_h, pred_w = model.decoder.ll_policy.tensor_policy(states[:, 0, :, :], clip[:, 0, :, :], in_grid, latent, pair_in, pair_out)

			# if not model.normalize_latent:
			# 	z_sampled = model.reparameterize(z_post_means, z_post_sigs)
			# else:
			# 	z_sampled = z_post_means
			p_operation = torch.argmax(pred_operation)
			p_x = torch.argmax(pred_x)
			p_y = torch.argmax(pred_y)
			p_h = torch.argmax(pred_h)
			p_w = torch.argmax(pred_w)

			# print("operation : {0}, pred_operation : {1}".format(operation[0, 0, 0].shape, p_operation.shape))
			# print("x : {0}, p_x : {1}, p_x = {2}".format(selection[0, 0, 0].shape, p_x.shape, p_x))
			# print("y : {0}, p_y : {1}, p_y = {2}".format(selection[0, 0, 1].shape, p_y.shape, p_y))
			# print("h : {0}, p_h : {1}, p_h = {2}".format(selection[0, 0, 2].shape, p_h.shape, p_h))
			# print("w : {0}, p_w : {1}, p_w = {2}".format(selection[0, 0, 3].shape, p_w.shape, p_w))

			if (operation[0, 0, 0] == p_operation and
				selection[0, 0, 0] == p_x and
				selection[0, 0, 1] == p_y and
				selection[0, 0, 2] == p_h and
				selection[0, 0, 3] == p_w):
				correct = correct + 1

			total_num = total_num + 1

			# Concept classification evaluation
			if track_concept:
				# Get ground truth concept ID
				gt_concept_id = concept_to_id[concept[0]]
				concept_total[gt_concept_id] += 1

				# Classify the sampled latent
				if use_amp:
					with autocast():
						concept_logits = model.concept_classifier(latent)
				else:
					concept_logits = model.concept_classifier(latent)
				pred_concept_id = torch.argmax(concept_logits, dim=-1).item()

				# Track prediction distribution
				concept_pred_dist[gt_concept_id][pred_concept_id] += 1

				# Check if correct
				if pred_concept_id == gt_concept_id:
					concept_correct[gt_concept_id] += 1

	if wandb is not None:
		try: wandb.log({"train_skill/test_prior_acc": 100.0*correct/total_num})
		except Exception: pass

	# Log concept classification results
	if track_concept:
		# Calculate per-concept accuracy
		for concept_id in range(num_concepts):
			if concept_total[concept_id] > 0:
				acc = 100.0 * concept_correct[concept_id] / concept_total[concept_id]
				if wandb is not None:
					try: wandb.log({f"train_skill/concept_{concept_id}_sampling_acc": acc})
					except Exception: pass

		# Calculate overall concept accuracy
		total_concept_correct = sum(concept_correct.values())
		total_concept_samples = sum(concept_total.values())
		if total_concept_samples > 0:
			overall_concept_acc = 100.0 * total_concept_correct / total_concept_samples
			if wandb is not None:
				try: wandb.log({"train_skill/concept_sampling_overall_acc": overall_concept_acc})
				except Exception: pass

			# Print results
			print("\n" + "="*60)
			print("VAE PRIOR CONCEPT SAMPLING RESULTS")
			print("="*60)
			print(f"Overall Accuracy: {overall_concept_acc:.2f}% ({total_concept_correct}/{total_concept_samples})")
			print()

			for concept_id in range(num_concepts):
				if concept_total[concept_id] > 0:
					acc = 100.0 * concept_correct[concept_id] / concept_total[concept_id]
					concept_name = id_to_concept[concept_id]
					print(f"  [{concept_id}] {concept_name}")
					print(f"      Accuracy: {acc:.2f}% ({concept_correct[concept_id]}/{concept_total[concept_id]})")
					print(f"      Prediction Distribution: {dict(concept_pred_dist[concept_id])}")

			print("="*60 + "\n")

	return correct/total_num


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='ARCLE')
parser.add_argument('--beta', type=float, default=0.1)

parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--policy_decoder_type', type=str, default='mlp')
parser.add_argument('--state_decoder_type', type=str, default='mlp')
parser.add_argument('--a_dist', type=str, default='normal')
parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--separate_test_trajectories', type=int, default=0)
parser.add_argument('--test_on', type=bool, default=False)
parser.add_argument('--test_cycle', type=int, default=20)
parser.add_argument('--save_cycle', type=int, default=50)
parser.add_argument('--test_num', type=int, default=500)
parser.add_argument('--get_rewards', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=50000)
parser.add_argument('--start_training_state_decoder_after', type=int, default=10000)
parser.add_argument('--normalize_latent', type=int, default=0)

parser.add_argument('--append_goals', type=int, default=0)

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--solar_dir', type=str, default=None)
parser.add_argument('--test_solar_dir', type=str, default=None)
parser.add_argument('--checkpoint_dir', type=str, default=parent_folder+'/checkpoints/')

parser.add_argument('--date', type=str, default='00.00')
parser.add_argument('--conditional_prior', type=int, default=1)
parser.add_argument('--train_diffusion_prior', type=int, default=1)
parser.add_argument('--diffusion_steps', type=int, default=500)

parser.add_argument('--gpu_name', type=str, required=True)
parser.add_argument('--optimizer', type=str, default="AdamW")

parser.add_argument('--a_dim', type=int, default=36)
parser.add_argument('--z_dim', type=int, default=128)
parser.add_argument('--h_dim', type=int, default=256)
parser.add_argument('--s_dim', type=int, default=256)
parser.add_argument('--max_grid_size', type=int, default=256)
parser.add_argument('--diffusion_scale', type=float, default=1.0)
parser.add_argument('--use_in_out', type=int, default=0)
parser.add_argument('--use_enhanced_pair_encoding', type=int, default=0, help='Use enhanced pair encoding (0=disable, 1=enable)')
parser.add_argument('--disable_pair_encoding', type=int, default=0, help='Disable pair encoding completely (0=use pairs, 1=disable pairs)')
parser.add_argument('--use_shared_grid_embedding', type=int, default=0, help='Use shared grid embedding for pair encoding (0=disable, 1=enable)')
parser.add_argument('--use_split_pair_trajectory_encoding', type=int, default=0, help='Split pair and trajectory encoding (0=disable, 1=enable)')
parser.add_argument('--use_direct_output_predictor', type=int, default=0, help='Use direct output predictor (0=disable, 1=enable)')
parser.add_argument('--use_direct_output_for_diffusion', type=int, default=0, help='Use direct output embedding for diffusion conditioning (0=disable, 1=enable)')
parser.add_argument('--use_positional_encoding', type=int, default=0, help='Use 2D positional encoding for state embeddings (0=disable, 1=enable)')
parser.add_argument('--use_concept_guidance', type=int, default=0, help='Use concept guidance for diffusion prior (0=disable, 1=enable)')
parser.add_argument('--use_cfg_for_concept', type=int, default=1, help='Use CFG for concept guidance (0=direct conditioning, 1=CFG)')
parser.add_argument('--use_concept_in_encoder', type=int, default=0, help='Use concept in skill encoder (0=disable, 1=enable)')
parser.add_argument('--num_concepts', type=int, default=0, help='Number of concepts (0=auto-detect from data, only used when use_concept_in_encoder=1)')
parser.add_argument('--concept_scale', type=float, default=1.0, help='Scale factor for concept embeddings during training (default=1.0)')
parser.add_argument('--concept_loss_weight', type=float, default=0.0, help='Weight for concept classification loss (default=0.0, set >0 to enable)')
parser.add_argument('--concept_contrastive_weight', type=float, default=0.0, help='Weight for concept contrastive loss (SupCon-style, default=0.0, set >0 to enable)')
parser.add_argument('--contrastive_temperature', type=float, default=0.1, help='Temperature for contrastive loss (default=0.1)')
parser.add_argument('--use_concept_for_diffusion', type=int, default=0, help='Use concept embedding for diffusion prior conditioning (0=disable, 1=enable)')
parser.add_argument('--encoder_type', type=str, default='gru', choices=['gru', 'transformer'], help='Encoder type (gru or transformer)')
parser.add_argument('--use_amp', type=int, default=0, help='Use Automatic Mixed Precision for faster training (0=disable, 1=enable)')
parser.add_argument('--use_compile', type=int, default=0, help='Use torch.compile for faster training (0=disable, 1=enable, requires PyTorch 2.0+)')
parser.add_argument('--compile_mode', type=str, default='reduce-overhead', choices=['default', 'reduce-overhead', 'max-autotune'], help='Compilation mode for torch.compile')

args = parser.parse_args()

batch_size = args.batch_size #default 128

h_dim = args.h_dim
z_dim = args.z_dim
lr = args.lr #5e-5
wd = 0.0
H = args.horizon
stride = 1
n_epochs = args.num_epochs
# test_split = args.test_split
a_dist = args.a_dist	#'normal' # 'tanh_normal' or 'normal'
encoder_type = args.encoder_type  # 'gru' or 'transformer'
state_decoder_type = args.state_decoder_type
policy_decoder_type = args.policy_decoder_type
load_from_checkpoint = False
per_element_sigma = True
start_training_state_decoder_after = args.start_training_state_decoder_after
train_diffusion_prior = args.train_diffusion_prior	# False
test_on = args.test_on
normalize_latent = args.normalize_latent

beta = args.beta # 1.0 # 0.1, 0.01, 0.001
conditional_prior = args.conditional_prior # True

checkpoint_dir = args.checkpoint_dir
env_name = args.env

action_num = args.a_dim
date = args.date
now = datetime.datetime.now(ZoneInfo('Asia/Seoul'))
nowtime = now.strftime("%m.%d")


state_dim = args.s_dim
a_dim = args.a_dim

if 'ARCLE' in args.env:
	# Enable return_concept if using concept in encoder
	return_concept = bool(args.use_concept_in_encoder)

	dataset = ARC_Segment_Dataset(
		data_path=args.solar_dir,
		return_concept=return_concept
	)
	test_dataset = ARC_Segment_Dataset(
		data_path=args.test_solar_dir,
		return_concept=return_concept
	)
else:
    raise ValueError(f"Unsupported env: {args.env}")

file_info = env_name  + '_' + date
filename = args.gpu_name+'_' + 'skill_model_' + file_info

# org_checkpoint_dir = checkpoint_dir+'/'+args.gpu_name+'_'+ date

# suffix = 0
# checkpoint_dir = org_checkpoint_dir

# while os.path.exists(checkpoint_dir):
#     checkpoint_dir = org_checkpoint_dir + f'_{suffix}'
#     suffix += 1

# os.makedirs(checkpoint_dir)

checkpoint_dir = checkpoint_dir+'/'+args.gpu_name+'_'+ date
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    print(f"Created checkpoint dir: {checkpoint_dir}")
else:
    if not os.listdir(checkpoint_dir):
        print(f"Using existing empty dir: {checkpoint_dir}")
    else:
        raise FileExistsError(f"Checkpoint dir already exists and is non-empty: {checkpoint_dir}")

# Auto-detect number of concepts if needed
num_concepts = args.num_concepts
if args.use_concept_in_encoder:
    import json

    concept_mapping_path = os.path.join(checkpoint_dir, filename + '_concept_mapping.json')

    if os.path.exists(concept_mapping_path):
        # Load existing concept mapping
        with open(concept_mapping_path, 'r') as f:
            concept_to_id = json.load(f)
        num_concepts = len(concept_to_id)
        print(f"Loaded concept mapping with {num_concepts} concepts from {concept_mapping_path}")
    elif num_concepts == 0:
        # Auto-detect from training data
        print("Auto-detecting number of concepts from training data...")
        all_concepts = set()

        # Sample from dataset to find unique concepts
        from torch.utils.data import DataLoader
        temp_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

        for batch in tqdm(temp_loader, desc="Detecting concepts"):
            if return_concept:
                *_, concepts = batch
                for concept in concepts:
                    all_concepts.add(concept)

        # Create concept mapping
        concept_to_id = {concept: idx for idx, concept in enumerate(sorted(all_concepts))}
        num_concepts = len(concept_to_id)

        # Save concept mapping
        with open(concept_mapping_path, 'w') as f:
            json.dump(concept_to_id, f, indent=2)

        print(f"Auto-detected {num_concepts} concepts: {list(concept_to_id.keys())}")
        print(f"Saved concept mapping to {concept_mapping_path}")
    else:
        # Use manually specified num_concepts, create empty mapping
        concept_to_id = {}
        print(f"Using manually specified num_concepts={num_concepts}")
else:
    num_concepts = 0
    concept_to_id = {}
    print("ℹ️  Concept-in-encoder is disabled (use_concept_in_encoder=0)")

# Check model option
print("Normalize_latent : {0}".format(normalize_latent))
print("Diffusion prior : {0}".format(train_diffusion_prior))
print("Conditional_prior : {0}".format(conditional_prior))
print("Use concept in encoder : {0}".format(args.use_concept_in_encoder))
print("Number of concepts : {0}".format(num_concepts))

model = SkillModel(
    state_dim,
    a_dim,
    z_dim,
    h_dim,
    horizon=H,
    a_dist=a_dist,
    beta=beta,
    fixed_sig=None,
    encoder_type=encoder_type,
    state_decoder_type=state_decoder_type,
    policy_decoder_type=policy_decoder_type,
    per_element_sigma=per_element_sigma,
    conditional_prior=conditional_prior,
    train_diffusion_prior=train_diffusion_prior,
    diffusion_steps=args.diffusion_steps,
    normalize_latent=normalize_latent,
    color_num=11,
    action_num=action_num,
    max_grid_size=args.max_grid_size,
    diffusion_scale=args.diffusion_scale,
    use_in_out=args.use_in_out,
    use_enhanced_pair_encoding=bool(args.use_enhanced_pair_encoding),
    disable_pair_encoding=bool(args.disable_pair_encoding),
    use_shared_grid_embedding=bool(args.use_shared_grid_embedding),
    use_split_pair_trajectory_encoding=bool(args.use_split_pair_trajectory_encoding),
    use_direct_output_predictor=bool(args.use_direct_output_predictor),
    use_direct_output_for_diffusion=bool(args.use_direct_output_for_diffusion),
    use_positional_encoding=bool(args.use_positional_encoding),
    use_concept_guidance=bool(args.use_concept_guidance),
    use_cfg_for_concept=bool(args.use_cfg_for_concept),
    use_concept_in_encoder=bool(args.use_concept_in_encoder),
    num_concepts=num_concepts
).cuda()

# Apply torch.compile if enabled (PyTorch 2.0+)
use_compile = bool(args.use_compile)
if use_compile:
    print(f"Compiling model with mode='{args.compile_mode}'...")
    print("⏳ First iteration will be slower due to compilation...")
    model = torch.compile(model, mode=args.compile_mode)
    print("Model compiled successfully!")
else:
    print("ℹ️  torch.compile disabled")

if(args.optimizer == "Adam"):
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
elif(args.optimizer == "AdamW"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
else:
    ValueError("올바른 Optimizer를 입력해주세요")

# Initialize GradScaler for AMP if enabled
use_amp = bool(args.use_amp)
scaler = GradScaler() if use_amp else None
if use_amp:
    print("Automatic Mixed Precision (AMP) enabled")
else:
    print("ℹ️  Using full precision (FP32)")

d=datetime.datetime.now()
# Wandb 기록
task_name = args.solar_dir.split("/")[-1]
# Handle different task name formats (e.g., "train.task_name.s10.H5" vs "five_task")
if "." in task_name:
    task = task_name.split(".")[1]
else:
    task = task_name

# Hardcoded wandb config for now
wandb_config = {
    'entity': 'dbsgh797210',
    'project': 'LDCQ_single',
    'api_key': '391af36b1546e19e6e1eb483f69c989abf5d202a'
}

os.environ["WANDB_API_KEY"] = wandb_config['api_key']

# WandB will use the API key from environment variable
print("WandB API key configured from environment")

base_config = vars(args) if 'args' in locals() else {}
additional_config = {
        'task':task_name,
		'lr':lr,
		'h_dim':h_dim,
		'z_dim':z_dim,
		'a_dim':a_dim,
		'l2_reg':wd,
		'beta':beta,
		'a_dist':a_dist,
		'filename':filename,
		'encoder_type':encoder_type,
		'state_decoder_type':state_decoder_type,
		'policy_decoder_type':policy_decoder_type,
		'per_element_sigma':per_element_sigma,
		'conditional_prior': conditional_prior,
		'train_diffusion_prior': train_diffusion_prior,
		# 'test_split': test_split,
		'separate_test_trajectories': args.separate_test_trajectories,
		'get_rewards': args.get_rewards,
		'normalize_latent': args.normalize_latent,
		'append_goals': args.append_goals,
		'use_amp': use_amp,
		'use_compile': use_compile,
		'compile_mode': args.compile_mode if use_compile else None
    }

config = {**base_config, **additional_config}

run = None
if wandb is not None:
    try:
        run = wandb.init(
            entity=wandb_config['entity'],
            project=wandb_config['project'],
            name='LDCQ_'+args.gpu_name+'_'+'skill'+'_'+task+'_'+date,
            config=config,
            mode='offline',
        )
    except Exception as e:
        print(f"[WARN] wandb.init failed: {e}")
        run = None

run_name = run.name if run is not None else 'local'
print(f"WandB Run Name: {run_name}")

train_loader = DataLoader(
	dataset=dataset,
	batch_size=batch_size,
	num_workers=4,
	shuffle=True)

test_loader = DataLoader(
	dataset=test_dataset,
	batch_size=1,
	num_workers=4,
	shuffle=True
 )

# Initialize concept encoder and concept-to-id mapping if using concepts
concept_encoder = None
concept_to_id = None
if return_concept and args.use_concept_in_encoder:
	print("Initializing concept encoder...")
	# Import only when actually needed (requires sentence_transformers)
	from models.concept_encoder import ConceptEncoder
	concept_encoder = ConceptEncoder(model_name='sentence-transformers/all-MiniLM-L6-v2')

	# Build concept vocabulary from dataset
	unique_concepts = set()
	for data in dataset:
		if len(data) > 13:  # Has concept field
			unique_concepts.add(data[13])  # concept is 14th element (index 13)

	concept_to_id = {concept: idx for idx, concept in enumerate(sorted(unique_concepts))}
	print(f"Found {len(concept_to_id)} unique concepts")

	# Verify num_concepts matches if specified
	if args.num_concepts > 0 and len(concept_to_id) != args.num_concepts:
		print(f"Warning: --num_concepts={args.num_concepts} but found {len(concept_to_id)} concepts in data")

min_test_loss = 10**10
min_test_s_T_loss = 10**10
min_test_a_loss = 10**10
for i in range(n_epochs):
	# if(test_on and i % 50 == 0):
	if(test_on and i % args.test_cycle == 0):
		# Test Loss
		test_loss, test_s_T_loss, test_a_loss, test_kl_loss, test_diffusion_loss = test(model, train_loader, test_state_decoder = i > start_training_state_decoder_after, test_num=args.test_num, return_concept=return_concept, use_amp=use_amp)
		# test_loss, test_s_T_loss, test_a_loss, test_kl_loss, test_diffusion_loss = 0.0, 0.0, 0.0, 0.0, 0.0

		# Test Accuracy
		accuracy = test_acc(model, test_loader, test_num=args.test_num, test_state_decoder=i > start_training_state_decoder_after, return_concept=return_concept, use_amp=use_amp)
		# accuracy = test_acc(model, test_loader, test_num=args.test_num)
		prior_accuracy = test_acc_prior(model, test_loader, test_num=args.test_num, return_concept=return_concept, concept_to_id=concept_to_id, use_amp=use_amp)

		print("--------TEST---------")
		
		print('test_loss: ', test_loss)
		print('test_s_T_loss: ', test_s_T_loss)
		print('test_a_loss: ', test_a_loss)
		print('test_kl_loss: ', test_kl_loss)
		print('test_Acc: ', accuracy*100.0,'%')
		if test_diffusion_loss is not None:
			print('test_diffusion_loss ', test_diffusion_loss)
		print(i)
		if test_diffusion_loss is not None:
			pass
		
		if test_loss < min_test_loss:
			min_test_loss = test_loss	
			checkpoint_path = os.path.join(checkpoint_dir, filename+'_best.pth')
			torch.save({'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
		if test_s_T_loss < min_test_s_T_loss:
			min_test_s_T_loss = test_s_T_loss

			checkpoint_path = os.path.join(checkpoint_dir, filename+'_best_sT.pth')
			torch.save({'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
		if test_a_loss < min_test_a_loss:
			min_test_a_loss = test_a_loss

			checkpoint_path = os.path.join(checkpoint_dir, filename+'_best_a.pth')
			torch.save({'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)

	loss = train(model, optimizer, train_loader, train_state_decoder = i > start_training_state_decoder_after, return_concept=return_concept, concept_encoder=concept_encoder, concept_to_id=concept_to_id, concept_scale=args.concept_scale, concept_loss_weight=args.concept_loss_weight, concept_contrastive_weight=args.concept_contrastive_weight, contrastive_temperature=args.contrastive_temperature, use_concept_for_diffusion=bool(args.use_concept_for_diffusion), use_amp=use_amp, scaler=scaler)
	
	print("--------TRAIN---------")
	
	print('Loss: ', loss)
	print("Epoch: {0}/{1}".format(i, n_epochs))
	# experiment.log_metric("Train loss", loss, step=i)

	# if i % 50 == 0:
	if i % args.save_cycle == 0:
		checkpoint_path = os.path.join(checkpoint_dir, filename+'_'+str(i)+'_'+'.pth')
		torch.save({'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)