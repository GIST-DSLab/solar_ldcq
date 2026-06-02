from maker.base_grid_maker import BaseGridMaker
from typing import Dict, List, Tuple
from numpy.typing import NDArray
import numpy as np
import random
class GridMaker(BaseGridMaker):
    def parse(self, **kwargs) -> List[Tuple[List[NDArray], List[NDArray], List[NDArray], List[NDArray], Dict]]:
        dat = []
        num = 0
        num_samples = kwargs['num_samples']
        max_h, max_w = kwargs['max_grid_dim']
        num_examples = kwargs['num_examples']
        while num < num_samples:
            pr_in: List[NDArray] = []
            pr_out: List[NDArray] = []
            ex_in: List[NDArray] = []
            ex_out: List[NDArray] = []
            operations = []
            selections = []
            background_color = 0
            available_colors = [c for c in range(1, 10)]
            num_object_colors = random.randint(3, min(6, len(available_colors)))
            object_colors = random.sample(available_colors, num_object_colors)
            j = 0
            while (j < num_examples + 1):
                grid_h = random.randint(4, max_h)
                grid_w = random.randint(4, max_w)
                input_grid = np.full((grid_h, grid_w), background_color, dtype=np.int8)
                for j_col in range(grid_w):
                    column_color = random.choice(object_colors)
                    for i in range(grid_h):
                        prob = 0.3 * (1 - i / grid_h) + 0.1
                        if random.random() < prob:
                            input_grid[i, j_col] = column_color
                output_grid = np.full((grid_h, grid_w), background_color, dtype=np.int8)
                for col in range(grid_w):
                    color_counts = {}
                    for row in range(grid_h):
                        color = input_grid[row, col]
                        if color != background_color:
                            color_counts[color] = color_counts.get(color, 0) + 1
                    current_row = grid_h - 1
                    for color in sorted(color_counts.keys()):
                        count = color_counts[color]
                        for _ in range(count):
                            output_grid[current_row, col] = color
                            current_row -= 1
                if j < num_examples:
                    ex_in.append(input_grid)
                    ex_out.append(output_grid)
                else:
                    pr_in.append(input_grid)
                    pr_out.append(output_grid)
                j += 1
            working_grid = pr_in[0].copy()
            for col in range(grid_w):
                something_moved = True
                while something_moved:
                    something_moved = False
                    for row in range(grid_h - 2, -1, -1):
                        if working_grid[row, col] != background_color:
                            object_color = working_grid[row, col]
                            block_bottom = row
                            block_top = row
                            while (block_top - 1 >= 0 and
                                   working_grid[block_top - 1, col] == object_color):
                                block_top -= 1
                            block_height = block_bottom - block_top + 1
                            max_fall_distance = 0
                            for check_row in range(block_bottom + 1, grid_h):
                                if working_grid[check_row, col] == background_color:
                                    max_fall_distance += 1
                                else:
                                    break
                            if max_fall_distance > 0:
                                for step in range(max_fall_distance):
                                    operations.append(21)
                                    selections.append([block_top + step, col, block_height - 1, 0])
                                for clear_idx in range(block_height):
                                    working_grid[block_top + clear_idx, col] = background_color
                                for place_idx in range(block_height):
                                    working_grid[block_top + max_fall_distance + place_idx, col] = object_color
                                something_moved = True
            operations.append(34)
            selections.append([0, 0, grid_h-1, grid_w-1])
            num += 1
            desc = {'id': f'1e0a9b12-expert_{num}',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
            wrong_directions = [20, 22, 23]
            for rand_idx in range(1):
                if len(operations) > 2:
                    branch_idx = random.randint(1, len(operations) - 2)
                else:
                    branch_idx = 0
                rand_operations = operations[:branch_idx]
                rand_selections = selections[:branch_idx]
                rand_working_grid = pr_in[0].copy()
                for op_idx in range(branch_idx):
                    op = operations[op_idx]
                    sel = selections[op_idx]
                    if op == 21:
                        sel_row, sel_col, sel_h, sel_w = sel
                        block_height = sel_h + 1
                        for h_idx in range(block_height - 1, -1, -1):
                            obj_color = rand_working_grid[sel_row + h_idx, sel_col]
                            rand_working_grid[sel_row + h_idx, sel_col] = background_color
                            rand_working_grid[sel_row + h_idx + 1, sel_col] = obj_color
                moves_added = 0
                attempts = 0
                current_row, current_col = None, None
                while attempts < 30 and current_row is None:
                    attempts += 1
                    rand_row = random.randint(0, grid_h - 1)
                    rand_col = random.randint(0, grid_w - 1)
                    if rand_working_grid[rand_row, rand_col] != background_color:
                        current_row, current_col = rand_row, rand_col
                if current_row is None:
                    continue
                while moves_added < 6 and attempts < 60:
                    attempts += 1
                    direction = random.choice(wrong_directions)
                    can_move = False
                    new_row, new_col = current_row, current_col
                    if direction == 20:
                        if current_row > 0 and rand_working_grid[current_row - 1, current_col] == background_color:
                            can_move = True
                            new_row = current_row - 1
                    elif direction == 22:
                        if current_col > 0 and rand_working_grid[current_row, current_col - 1] == background_color:
                            can_move = True
                            new_col = current_col - 1
                    elif direction == 23:
                        if current_col < grid_w - 1 and rand_working_grid[current_row, current_col + 1] == background_color:
                            can_move = True
                            new_col = current_col + 1
                    if can_move:
                        rand_operations.append(direction)
                        rand_selections.append([current_row, current_col, 0, 0])
                        moves_added += 1
                        obj_color = rand_working_grid[current_row, current_col]
                        rand_working_grid[current_row, current_col] = background_color
                        rand_working_grid[new_row, new_col] = obj_color
                        current_row, current_col = new_row, new_col
                rand_operations.append(34)
                rand_selections.append([0, 0, grid_h - 1, grid_w - 1])
                desc = {'id': f'1e0a9b12-random_{num}_{rand_idx}',
                        'selections': rand_selections,
                        'operations': rand_operations}
                dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
