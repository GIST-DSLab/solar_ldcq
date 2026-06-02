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
        available_colors = list(range(1, 10))
        while num < num_samples:
            num += 1
            pr_in: List[NDArray] = []
            pr_out: List[NDArray] = []
            ex_in: List[NDArray] = []
            ex_out: List[NDArray] = []
            operations = []
            selections = []
            observed_color_mapping = {}
            for j in range(num_examples):
                grid_size = np.random.randint(2, max_h + 1)
                h = grid_size
                w = grid_size
                input_grid = np.zeros((h, w), dtype=np.uint8)
                input_colors_in_example = []
                for col in range(w):
                    color = random.choice(available_colors)
                    input_grid[:, col] = color
                    if color not in input_colors_in_example:
                        input_colors_in_example.append(color)
                available_target_colors = available_colors.copy()
                used_target_colors = list(observed_color_mapping.values())
                for used_color in used_target_colors:
                    if used_color in available_target_colors:
                        available_target_colors.remove(used_color)
                for input_color in input_colors_in_example:
                    if input_color not in observed_color_mapping:
                        if available_target_colors:
                            target_color = random.choice(available_target_colors)
                            available_target_colors.remove(target_color)
                        else:
                            target_color = random.choice(available_colors)
                        observed_color_mapping[input_color] = target_color
                output_grid = np.zeros((h, w), dtype=np.uint8)
                for col in range(w):
                    original_color = input_grid[0, col]
                    new_color = observed_color_mapping[original_color]
                    output_grid[:, col] = new_color
                ex_in.append(input_grid)
                ex_out.append(output_grid)
            if observed_color_mapping:
                grid_size = np.random.randint(2, max_h + 1)
                h = grid_size
                w = grid_size
                input_grid = np.zeros((h, w), dtype=np.uint8)
                observed_input_colors = list(observed_color_mapping.keys())
                for col in range(w):
                    color = random.choice(observed_input_colors)
                    input_grid[:, col] = color
                output_grid = np.zeros((h, w), dtype=np.uint8)
                for col in range(w):
                    original_color = input_grid[0, col]
                    new_color = observed_color_mapping[original_color]
                    output_grid[:, col] = new_color
                    selections.append([0, col, h-1, 0])
                    operations.append(new_color)
                operations.append(34)
                selections.append([0, 0, h-1, w-1])
                pr_in.append(input_grid)
                pr_out.append(output_grid)
            desc = {'id': f'0d3d703e-expert_{num}',
                    'selections': selections,
                    'operations': operations,
                    'observed_color_mapping': observed_color_mapping}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
            for i in range(1):
                selections_random, operations_random = self.random_trajectory(
                    h, w, input_grid, observed_color_mapping, available_colors)
                desc = {'id': f'0d3d703e-random_{num}_{i+1}',
                        'selections': selections_random.copy(),
                        'operations': operations_random.copy(),
                        'observed_color_mapping': observed_color_mapping}
                dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
    def random_trajectory(self, h, w, input_grid, observed_color_mapping, available_colors):
        selections = []
        operations = []
        branch_point = random.randint(0, w - 1)
        for col in range(w):
            original_color = input_grid[0, col]
            correct_color = observed_color_mapping[original_color]
            if col < branch_point:
                selections.append([0, col, h-1, 0])
                operations.append(correct_color)
            else:
                random_color = random.choice(available_colors)
                selections.append([0, col, h-1, 0])
                operations.append(random_color)
        operations.append(34)
        selections.append([0, 0, h-1, w-1])
        return selections, operations
