import random
import numpy as np
import hashlib
import os
from pathlib import Path
import json
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--train_file_path', type=str, default='data/train')
parser.add_argument('--test_file_path', type=str, default='data/test')
args = parser.parse_args()
train_file_path = args.train_file_path
test_file_path = args.test_file_path
train_hash_set = set()
train_hash_to_id = {}
train_duplicate_groups = {}
num_train = 0
num_train_duplicate = 0
num_test = 0
num_duplicate = 0
for path, _, files in os.walk(Path(train_file_path)):
    for name in files:
        if 'gold_standard' in name or 'expert' in name:
            num_train += 1
            p = os.path.join(path, name)
            with open(p, 'r') as f:
                trajectory = json.load(f)
            grid_train_str = str(trajectory['in_grid']) + str(trajectory['out_grid'])
            train_hash = hashlib.sha256(grid_train_str.encode()).hexdigest()
            train_id = trajectory['desc']['id']
            if train_hash in train_hash_to_id:
                if train_hash not in train_duplicate_groups:
                    train_duplicate_groups[train_hash] = [train_hash_to_id[train_hash]]
                train_duplicate_groups[train_hash].append(train_id)
            else:
                train_hash_to_id[train_hash] = train_id
            train_hash_set.add(train_hash)
num_train_duplicate = sum(len(group) for group in train_duplicate_groups.values())
duplicate_id_list = []
test_only_id_list = []
for path, _, files in os.walk(Path(test_file_path)):
    for name in files:
        if 'gold_standard' in name or 'expert' in name:
            num_test += 1
            p = os.path.join(path, name)
            with open(p, 'r') as f:
                trajectory = json.load(f)
            grid_eval_str = str(trajectory['in_grid']) + str(trajectory['out_grid'])
            eval_hash = hashlib.sha256(grid_eval_str.encode()).hexdigest()
            if eval_hash in train_hash_set:
                num_duplicate += 1
                duplicate_id_list.append(trajectory['desc']['id'])
            else:
                test_only_id_list.append(trajectory['desc']['id'])
duplicate_id_list.sort()
test_only_id_list.sort()
print(f"=== Train Dataset Internal Duplicates ===")
print(f"Number of Train files: {num_train}")
print(f"Number of duplicated Train files: {num_train_duplicate}")
print(f"Number of unique duplicate groups in Train: {len(train_duplicate_groups)}")
if train_duplicate_groups:
    print("Train duplicate groups:")
    for i, (hash_val, id_group) in enumerate(train_duplicate_groups.items(), 1):
        print(f"  Group {i}: {id_group}")
print(f"\n=== Train-Test Comparison ===")
print(f"Number of Test files: {num_test}")
print(f"Number of Test files duplicated with Train: {num_duplicate}")
print(f"Number of Test-only files: {len(test_only_id_list)}")
print(f"Test files duplicated with Train: {duplicate_id_list}")
