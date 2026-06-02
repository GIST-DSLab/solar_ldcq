#!/bin/bash

#change the arguments below
python generate_trajectory.py \
    --env ARCLE/O2ARCv2Env-v0 \
    --data_folder_path /home/jovyan/beomi/yunho/ldcq_arc_working/combo \
    --segment_train_dir "" \
    --segment_test_dir "" \
    --whole_train_dir ""  \
    --whole_test_dir ""  \
    --subfolder simple-combo \
    --tasks simple-combo-4258a5f9-colorfix-4258a5f9-middle \
    --num_samples 10 \
    --num_examples 3 \
    --max_grid_dim 10 10 \
    --horizon 5 \
    --save_whole_trace True \
    --save_seg_trace True \
    --render_mode None \
    --delete_existing_data True \
    --rand_seed 0 \
    --max_duplicate_attempts 100000 \
    --skip_on_error False \
    --validate_all False \
    --validate_expert_only True



:<<"OPTIONS"
explanation of arguments
-env: RL environment. If you change this, the data type and functions are all changed.
-tasks:
    1) task_id_1 task_id_2 .... :list of task_ids.
    2) tasks.txt : A file that contains one task ID per line.
    3) all : all tasks in the  'maker'SOLAR-Generator/maker/ folder
-num_samples: number of samples to generate for each task
-num_examples: number of example pairs for each trace data
-max_grid_dim: maximum grid dim h, w
-horizon: step length of segment trace
-save_whole_trace: whether save the whole trace or not
-save_seg_trace: whether save segment trace or not
-render_mode: 'none' for generating trace quickly, 'ansi' for watching the step of generating trace.
-delete_existing_data: whether delete existing trace or not
-data_folder_path: path to save trace data files.
-rand_seed: random seed that all grid_makers share.
-segment_train_dir: custom directory for segment training data (optional, leave empty for default)
-segment_test_dir: custom directory for segment test data (optional, leave empty for default)
-whole_train_dir: custom directory for whole training data (optional, leave empty for default)
-whole_test_dir: custom directory for whole test data (optional, leave empty for default)
-subfolder: subfolder in maker/ directory (arc-handcraft, simple-combo, test-train, unseen, or None for root)
-validate_all: True to validate all samples, False to use validate_expert_only
-validate_expert_only: True to validate only expert/gold_standard samples, False to skip validation (only meaningful when validate_all=False)

# NOTE: generates both train and test splits simultaneously.
# train and test use different random seeds (train=rand_seed, test=rand_seed+5) to avoid overlap.

# Validation behavior:
# - validate_all=True: validate all samples (validate_expert_only is ignored)
# - validate_all=False, validate_expert_only=True: validate only expert/gold_standard (default)
# - validate_all=False, validate_expert_only=False: no validation

OPTIONS