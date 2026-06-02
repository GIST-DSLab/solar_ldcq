import numpy as np
import gymnasium as gym
import utils
import time
import os
from tqdm import tqdm
import argparse
import shutil
from datetime import datetime
import json
import hashlib
def compute_grid_hash(in_grid, out_grid):
    grid_str = str(in_grid) + str(out_grid)
    return hashlib.sha256(grid_str.encode()).hexdigest()
def get_ratio(task_name):
    if '-2-1' in task_name:
        return (2, 1)
    elif '-1-2' in task_name:
        return (1, 2)
    else:
        return (None, None)
def is_duplicate(in_grid, out_grid, existing_hashes):
    if not existing_hashes:
        return False
    grid_hash = compute_grid_hash(in_grid, out_grid)
    if grid_hash in existing_hashes:
        return True
    return False
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='ARCLE/O2ARCv2Env-v0')
parser.add_argument('--tasks', nargs='+', type=str, required=True)
parser.add_argument('--data_folder_path', type=str, default="SOLAR_data")
parser.add_argument('--num_samples', type=int, default=10000)
parser.add_argument('--num_examples', type=int, default=3)
parser.add_argument('--max_grid_dim', nargs=2, type=int, default=(30, 30))
parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--save_whole_trace', type=str, default="True")
parser.add_argument('--save_seg_trace', type=str, default="True")
parser.add_argument('--render_mode', type=str, default="None")
parser.add_argument('--delete_existing_data', type=str, default="False")
parser.add_argument('--rand_seed', type=int, default=0)
parser.add_argument('--segment_train_dir', type=str, default=None)
parser.add_argument('--segment_test_dir', type=str, default=None)
parser.add_argument('--whole_train_dir', type=str, default=None)
parser.add_argument('--whole_test_dir', type=str, default=None)
parser.add_argument('--max_duplicate_attempts', type=int, default=10000)
parser.add_argument('--skip_on_error', type=str, default="False")
parser.add_argument('--subfolder', type=str, default=None)
parser.add_argument('--validate_all', type=str, default="False")
parser.add_argument('--validate_expert_only', type=str, default="True")
args = parser.parse_args()
env_name = args.env
data_folder_path = args.data_folder_path
num_samples = args.num_samples
max_grid_dim = args.max_grid_dim
H = args.horizon
save_whole_trace = args.save_whole_trace.lower()
save_seg_trace = args.save_seg_trace.lower()
num_examples = args.num_examples
delete_existing_data = args.delete_existing_data.lower()
rand_seed = args.rand_seed
max_duplicate_attempts = args.max_duplicate_attempts
skip_on_error = args.skip_on_error.lower() == "true"
validate_all = args.validate_all.lower() == "true"
validate_expert_only = args.validate_expert_only.lower() == "true"
custom_dirs = {
    'segment_train': args.segment_train_dir,
    'segment_test': args.segment_test_dir,
    'whole_train': args.whole_train_dir,
    'whole_test': args.whole_test_dir
}
if args.render_mode.lower() == "none":
    render_mode = None
else:
    render_mode = args.render_mode.lower()
tasks = args.tasks
for file in tasks:
    if file == "all":
        task_list = []
        if args.subfolder:
            subfolder_path = f'maker/{args.subfolder}'
            if os.path.exists(subfolder_path):
                for task in os.listdir(subfolder_path):
                    task_path = f'{subfolder_path}/{task}'
                    if os.path.isdir(task_path) and task != "__pycache__":
                        if os.path.exists(f'{task_path}/grid_maker.py'):
                            task_list.append(f'{args.subfolder}/{task}')
                        else:
                            for subtask in os.listdir(task_path):
                                subtask_path = f'{task_path}/{subtask}'
                                if os.path.isdir(subtask_path) and subtask != "__pycache__":
                                    if os.path.exists(f'{subtask_path}/grid_maker.py'):
                                        task_list.append(f'{args.subfolder}/{task}/{subtask}')
        else:
            task_list = [task for task in os.listdir('maker') if os.path.isdir(f'maker/{task}') and task != "__pycache__"]
        tasks = task_list
    elif ".txt" in file:
        tasks.remove(file)
        tasks_list = []
        with open(f'maker/{file}', 'r') as f:
            for line in f:
                tasks_list.append(line.strip())
        tasks.extend(tasks_list)
if args.subfolder and "all" not in args.tasks:
    tasks = [f"{args.subfolder}/{task}" if not task.startswith(f"{args.subfolder}/") else task for task in tasks]
print("target: ", tasks)
now = datetime.now()
formatted_time = now.strftime("%y.%m.%d")
all_test_hashes = set()
current_task_test_hashes = {}
for i in tqdm(range(len(tasks)), desc="total", position=0):
    task_name = tasks[i].split('/')[-1] if '/' in tasks[i] else tasks[i]
    if delete_existing_data == "true":
        for data_type in ['train', 'test']:
            if custom_dirs[f'whole_{data_type}']:
                whole_folder_path = custom_dirs[f'whole_{data_type}'] + f"/{data_type}.{task_name}.s{max_grid_dim[0]}.{formatted_time}"
            else:
                whole_folder_path = data_folder_path+'/whole'+f"/{data_type}.{task_name}.s{max_grid_dim[0]}.{formatted_time}"
            if os.path.exists(whole_folder_path):
                shutil.rmtree(whole_folder_path)
            if custom_dirs[f'segment_{data_type}']:
                seg_folder_path = custom_dirs[f'segment_{data_type}'] + f"/{data_type}.{task_name}.s{max_grid_dim[0]}.H{H}.{formatted_time}"
            else:
                seg_folder_path = data_folder_path+'/segment'+f"/{data_type}.{task_name}.s{max_grid_dim[0]}.H{H}.{formatted_time}"
            if os.path.exists(seg_folder_path):
                shutil.rmtree(seg_folder_path)
            wrong_folder_path = data_folder_path+'/wrong'+f"/{data_type}.{task_name}.s{max_grid_dim[0]}.{formatted_time}"
            if os.path.exists(wrong_folder_path):
                shutil.rmtree(wrong_folder_path)
    for t in ['test','train']:
        if save_seg_trace == "true" or save_whole_trace == "true":
            if delete_existing_data != "true":
                if custom_dirs[f'whole_{t}']:
                    whole_folder_path = custom_dirs[f'whole_{t}'] + f"/{t}.{task_name}.s{max_grid_dim[0]}.{formatted_time}"
                else:
                    whole_folder_path = data_folder_path+'/whole'+f"/{t}.{task_name}.s{max_grid_dim[0]}.{formatted_time}"
                if os.path.exists(whole_folder_path):
                    print(f"dataset exists already for {tasks[i]}")
                    print("change '--delete_existing_data option' to delete old dataset then save new dataset")
                    continue
                if custom_dirs[f'segment_{t}']:
                    seg_folder_path = custom_dirs[f'segment_{t}'] + f"/{t}.{task_name}.s{max_grid_dim[0]}.H{H}.{formatted_time}"
                else:
                    seg_folder_path = data_folder_path+'/segment'+f"/{t}.{task_name}.s{max_grid_dim[0]}.{formatted_time}"
                if os.path.exists(seg_folder_path):
                    print(f"dataset exists already for {tasks[i]}")
                    print("change '--delete_existing_data option' to delete old dataset then save new dataset")
                    continue
        try:
            if t == 'test':
                target_samples = 100
            else:
                if '-half' in tasks[i]:
                    target_samples = num_samples
                else:
                    target_samples = num_samples
            if t=='test':
                base_test_seed = 1000 + rand_seed
                if '-1-2' in tasks[i] or '-2-1' in tasks[i] or '-half' in tasks[i]:
                    actual_generation_count = max(max_duplicate_attempts, 3000)
                else:
                    actual_generation_count = max(max_duplicate_attempts, 1000)
                print(f"Test data generation - generating up to {actual_generation_count} samples to find {target_samples} independent expert samples")
                grid_maker = utils.import_library_for_task(tasks[i], actual_generation_count, max_grid_dim, num_examples=num_examples, rand_seed=base_test_seed)
            else:
                test_buffer = len(all_test_hashes)
                large_buffer = max(test_buffer * 3, 50000)
                if '-1-2' in tasks[i] or '-2-1' in tasks[i]:
                    grid_maker_samples = (target_samples + large_buffer) // 3
                    if (target_samples + large_buffer) % 3 != 0:
                        grid_maker_samples += 1
                elif '-half' in tasks[i]:
                    grid_maker_samples = (target_samples + large_buffer) // 2 if (target_samples + large_buffer) % 2 == 0 else (target_samples + large_buffer + 1) // 2
                else:
                    grid_maker_samples = target_samples + large_buffer
                grid_maker = utils.import_library_for_task(tasks[i], grid_maker_samples, max_grid_dim, num_examples=num_examples, rand_seed=rand_seed)
        except Exception as e:
            print(f"Task {tasks[i]} failed: {e}")
            if skip_on_error:
                print(f"   Skipping and continuing with next task...")
                break
            else:
                raise
        env = gym.make(env_name, render_mode=render_mode, data_loader=grid_maker, max_grid_size=max_grid_dim, colors=10, max_episode_steps=None, max_trial=3)
        successful_samples = 0
        duplicate_count = 0
        error_count = 0
        current_hashes = set()
        unique_count = 0
        expert_accepted_count = 0
        expert_data_pending = None
        expert_samples_count = 0
        pending_samples = []
        check_hashes = set()
        if t == 'train':
            check_hashes.update(all_test_hashes)
            print(f"Train generation: checking against {len(all_test_hashes)} total test hashes (all tasks)")
        def get_base_iteration(sample_id):
            normalized = sample_id.replace('-', '_')
            parts = normalized.split('_')
            for i, part in enumerate(parts):
                if part in ['expert', 'random', 'standard']:
                    if i + 1 < len(parts):
                        try:
                            return int(parts[i + 1])
                        except ValueError:
                            pass
            return None
        is_ratio_version = '-1-2' in tasks[i] or '-2-1' in tasks[i]
        sample_groups = {}
        group_skip_status = {}
        if is_ratio_version and t == 'train':
            for idx in range(len(grid_maker.data)):
                ex_in_tmp, ex_out_tmp, pr_in_tmp, pr_out_tmp, desc_tmp = grid_maker.pick(data_index=idx)
                base_iter = get_base_iteration(desc_tmp['id'])
                if base_iter is not None:
                    if base_iter not in sample_groups:
                        sample_groups[base_iter] = []
                    sample_groups[base_iter].append(idx)
            print(f"Pre-checking {len(sample_groups)} ratio groups for test overlap...")
            for base_iter, group_indices in sample_groups.items():
                should_skip_group = False
                for idx in group_indices:
                    ex_in_tmp, ex_out_tmp, pr_in_tmp, pr_out_tmp, desc_tmp = grid_maker.pick(data_index=idx)
                    is_expert_tmp = 'expert' in desc_tmp['id'] or 'gold_standard' in desc_tmp['id']
                    if is_expert_tmp:
                        in_grid_tmp = pr_in_tmp[0]
                        out_grid_tmp = pr_out_tmp[0]
                        grid_hash_tmp = compute_grid_hash(in_grid_tmp, out_grid_tmp)
                        if check_hashes and grid_hash_tmp in check_hashes:
                            should_skip_group = True
                            break
                group_skip_status[base_iter] = should_skip_group
            skipped_groups = sum(1 for skip in group_skip_status.values() if skip)
            print(f"Will skip {skipped_groups}/{len(sample_groups)} groups due to test overlap")
        progress_bar = tqdm(range(len(grid_maker.data)), desc=f"task-{tasks[i]}-{t}", position=1)
        skip_indices = set()
        for j in progress_bar:
            if j in skip_indices:
                continue
            if t == 'test':
                if expert_samples_count >= 100:
                    print(f"Test stopping: expert_samples_count={expert_samples_count}")
                    break
            elif successful_samples >= target_samples:
                break
            try:
                ex_in, ex_out, pr_in, pr_out, desc = grid_maker.pick(data_index=j)
                is_expert = 'expert' in desc['id'] or 'gold_standard' in desc['id']
                is_random = 'random' in desc['id']
                if t == 'test':
                    if is_random:
                        continue
                    if is_expert:
                        in_grid = pr_in[0]
                        out_grid = pr_out[0]
                        grid_hash = compute_grid_hash(in_grid, out_grid)
                        if grid_hash in current_hashes or grid_hash in all_test_hashes:
                            duplicate_count += 1
                            continue
                        current_hashes.add(grid_hash)
                group_should_skip = False
                if is_ratio_version and t == 'train':
                    base_iter = get_base_iteration(desc['id'])
                    if base_iter in group_skip_status and group_skip_status[base_iter]:
                        if base_iter in sample_groups:
                            for group_idx in sample_groups[base_iter]:
                                if group_idx > j:
                                    skip_indices.add(group_idx)
                        group_should_skip = True
                        duplicate_count += 1
                        continue
                if is_ratio_version and t == 'train' and is_random:
                    expert_ratio, random_ratio = get_ratio(tasks[i])
                    if expert_ratio == 2 and random_ratio == 1:
                        if expert_accepted_count % 2 != 0:
                            continue
                    elif expert_ratio == 1 and random_ratio == 2:
                        pass
                data = {
                    "desc": {"id": desc['id'], "concept": desc.get('concept', '')},
                    "step": [],
                    "selection": [],
                    "operation": [],
                    "operation_name": [],
                    "reward": [],
                    "terminated": [],
                    "grid_dim": [],
                    "in_grid": [],
                    "out_grid": [],
                    "grid": [],
                    "clip_dim": [],
                    "clip": [],
                    "selection_mask": [],
                    "ex_in": [],
                    "ex_out": [],
                    "ex_in_grid_dim": [],
                    "ex_out_grid_dim": []
                }
                obs, info = env.reset(options={'prob_index': j, 'subprob_index': 0, 'adaptation' : False})
                whole_operations = desc['operations']
                whole_selections = desc['selections']
                grid = np.full(max_grid_dim, 10, dtype=np.int8)
                sel = [0, 0, 0, 0]
                sel_mask = np.zeros(max_grid_dim, dtype=np.int8)
                reward = 0
                term = False
                g_h, g_w = obs['grid_dim']
                grid_pad = grid.copy()
                grid_pad[:g_h, :g_w] = obs['grid'][:g_h, :g_w]
                c_h, c_w = obs['clip_dim']
                clip_pad = grid.copy()
                clip_pad[:c_h, :c_w] = obs['clip'][:c_h, :c_w].astype(np.uint8)
                for ei, eo in zip(ex_in, ex_out):
                    utils.append_example(data, ei, eo, grid)
                time.sleep(0) if render_mode == None else time.sleep(1)
                for s in range(len(whole_operations)):
                    time.sleep(0) if render_mode == None else time.sleep(1)
                    try:
                        operation = whole_operations[s]
                        selection = whole_selections[s]
                        selection_mask = utils.sel_bbox_to_mask(
                            selection, max_grid_dim)
                        action = {'selection': selection_mask.astype(bool), 'operation': operation}
                        obs, reward, term, trunc, info = env.step(action)
                        utils.append_data(data, grid_pad, selection, g_h, g_w, clip_pad, c_h, c_w, selection_mask, operation, reward, term, s)
                        g_h, g_w = obs['grid_dim']
                        grid_pad = grid.copy()
                        grid_pad[:g_h, :g_w] = obs['grid'][:g_h, :g_w]
                        c_h, c_w = obs['clip_dim']
                        clip_pad = grid.copy()
                        clip_pad[:c_h, :c_w] = obs['clip'][:c_h, :c_w]
                    except Exception as e:
                        print(f" {tasks[i]} : something wrong in trace! skip this problem")
                        print(e)
                        utils.save_wrong(data, tasks[i], t, max_grid_dim[0], data_folder_path)
                        error_count += 1
                        if is_ratio_version and t == 'train' and is_expert:
                            base_iter = get_base_iteration(desc['id'])
                            if base_iter in sample_groups:
                                for group_idx in sample_groups[base_iter]:
                                    if group_idx > j:
                                        skip_indices.add(group_idx)
                        if t == 'train':
                            progress_bar.set_postfix({"generated": successful_samples, "unique": unique_count, "errors": error_count})
                        else:
                            progress_bar.set_postfix({"unique": successful_samples, "duplicates": duplicate_count, "errors": error_count})
                        break
                should_validate = False
                if validate_all:
                    should_validate = True
                elif validate_expert_only:
                    should_validate = ('gold_standard' in data['desc']['id'] or 'expert' in data['desc']['id'])
                if should_validate:
                    if len(data['grid']) == 0 or not np.array_equal(np.array(data['grid'][-1])[:g_h, :g_w], pr_out[0]):
                        print(f" {tasks[i]} not correct answer")
                        utils.save_wrong(data, tasks[i], t, max_grid_dim[0], data_folder_path)
                        error_count += 1
                        if is_ratio_version and t == 'train' and is_expert:
                            base_iter = get_base_iteration(desc['id'])
                            if base_iter in sample_groups:
                                for group_idx in sample_groups[base_iter]:
                                    if group_idx > j:
                                        skip_indices.add(group_idx)
                        if t == 'train':
                            progress_bar.set_postfix({"generated": successful_samples, "unique": unique_count, "errors": error_count})
                        else:
                            progress_bar.set_postfix({"unique": successful_samples, "duplicates": duplicate_count, "errors": error_count})
                        continue
                data['in_grid'] = data['grid'][0]
                data['out_grid'] = data['grid'][-1]
                in_grid = pr_in[0]
                out_grid = pr_out[0]
                grid_hash = compute_grid_hash(in_grid, out_grid)
                should_skip = False
                if t == 'train':
                    if not (is_ratio_version and is_random):
                        if check_hashes and grid_hash in check_hashes:
                            duplicate_count += 1
                            should_skip = True
                        else:
                            if grid_hash not in current_hashes:
                                unique_count += 1
                            current_hashes.add(grid_hash)
                    else:
                        if grid_hash not in current_hashes:
                            unique_count += 1
                        current_hashes.add(grid_hash)
                else:
                    if grid_hash not in current_hashes:
                        unique_count += 1
                if should_skip:
                    if is_ratio_version and t == 'train' and is_expert:
                        base_iter = get_base_iteration(desc['id'])
                        if base_iter in sample_groups:
                            for group_idx in sample_groups[base_iter]:
                                if group_idx > j:
                                    skip_indices.add(group_idx)
                    progress_bar.set_postfix({"unique": successful_samples, "duplicates": duplicate_count, "errors": error_count})
                    continue
                data = utils.convert_npint_to_int(data)
                if save_whole_trace == "true":
                    custom_whole_dir = custom_dirs[f'whole_{t}']
                    all_empty = all(dir == "" for dir in custom_dirs.values() if dir is not None)
                    if all_empty or custom_whole_dir is None:
                        utils.save_whole(data, tasks[i], t, max_grid_dim[0], data_folder_path, None)
                    elif custom_whole_dir != "":
                        utils.save_whole(data, tasks[i], t, max_grid_dim[0], data_folder_path, custom_whole_dir)
                if save_seg_trace == "true":
                    sel = [0, 0, 0, 0]
                    sel_mask = np.zeros(max_grid_dim, dtype=np.int8)
                    for _ in range(H-1):
                        utils.append_data(data, grid.copy(), sel, 0, 0, grid.copy(), 0, 0, sel_mask, 35, 0, True, len(data['step']))
                    custom_seg_dir = custom_dirs[f'segment_{t}']
                    all_empty = all(dir == "" for dir in custom_dirs.values() if dir is not None)
                    if all_empty or custom_seg_dir is None:
                        utils.save_seg(data, tasks[i], H, t, max_grid_dim[0], data_folder_path, None)
                    elif custom_seg_dir != "":
                        utils.save_seg(data, tasks[i], H, t, max_grid_dim[0], data_folder_path, custom_seg_dir)
                successful_samples += 1
                if t == 'test' and ('expert' in desc['id'] or 'gold_standard' in desc['id']):
                    expert_samples_count += 1
                if t == 'train' and is_ratio_version and ('expert' in desc['id'] or 'gold_standard' in desc['id']):
                    expert_accepted_count += 1
                if t == 'train':
                    progress_bar.set_postfix({"generated": successful_samples, "unique": unique_count, "errors": error_count})
                else:
                    progress_bar.set_postfix({"expert": expert_samples_count, "total": successful_samples, "duplicates": duplicate_count, "errors": error_count})
            except Exception as e:
                error_count += 1
                if skip_on_error:
                    print(f"\nSample {j} failed: {e}")
                    print(f"   Skipping and continuing with next sample...")
                    if t == 'train':
                        progress_bar.set_postfix({"generated": successful_samples, "unique": unique_count, "errors": error_count})
                    else:
                        progress_bar.set_postfix({"unique": successful_samples, "duplicates": duplicate_count, "errors": error_count})
                    continue
                else:
                    raise
        if t == 'test':
            current_task_test_hashes[tasks[i]] = current_hashes.copy()
            all_test_hashes.update(current_hashes)
            print(f"  Global test hash count: {len(all_test_hashes)}")
        print(f"\n{tasks[i]}-{t} generation complete:")
        if t == 'train':
            print(f"  Samples generated: {successful_samples}/{target_samples}")
            if successful_samples > 0:
                print(f"  Unique samples: {unique_count}/{successful_samples} ({unique_count/successful_samples*100:.1f}% unique)")
            else:
                print(f"  Unique samples: {unique_count}/{successful_samples} (0.0% unique)")
            print(f"  Errors encountered: {error_count}")
        else:
            print(f"  Unique samples: {successful_samples}/{target_samples}")
            print(f"  Duplicates filtered: {duplicate_count}")
            print(f"  Errors encountered: {error_count}")
            if check_hashes:
                success_rate = successful_samples / len(grid_maker.data) * 100 if len(grid_maker.data) > 0 else 0
                print(f"  Success rate: {success_rate:.1f}%")
        env.close()
