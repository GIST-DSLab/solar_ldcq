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
            num += 1
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
                input_grid = np.full((grid_h, grid_w), background_color, dtype=np.uint8)
                for j_col in range(grid_w):
                    column_color = random.choice(object_colors)
                    for i in range(grid_h):
                        prob = 0.3 * (1 - i / grid_h) + 0.1
                        if random.random() < prob:
                            input_grid[i, j_col] = column_color
                output_grid = np.full((grid_h, grid_w), background_color, dtype=np.uint8)
                for col in range(grid_w):
                    objects = []
                    for row in range(grid_h):
                        if input_grid[row, col] != background_color:
                            objects.append(input_grid[row, col])
                    for i, obj in enumerate(objects):
                        bottom_row = grid_h - 1 - i
                        output_grid[bottom_row, col] = obj
                if j < num_examples:
                    ex_in.append(input_grid)
                    ex_out.append(output_grid)
                else:
                    pr_in.append(input_grid)
                    pr_out.append(output_grid)
                j += 1
            working_grid = input_grid.copy()
            for col in range(grid_w):
                for row in range(grid_h-2, -1, -1):
                    if working_grid[row, col] != background_color:
                        fall_distance = 0
                        for check_row in range(row + 1, grid_h):
                            if working_grid[check_row, col] == background_color:
                                fall_distance += 1
                            else:
                                break
                        if fall_distance > 0:
                            current_row = row
                            for step in range(fall_distance):
                                operations.append(21)
                                selections.append([current_row, col, 0, 0])
                                current_row += 1
                            object_color = working_grid[row, col]
                            working_grid[row, col] = background_color
                            working_grid[row + fall_distance, col] = object_color
            operations.append(34)
            selections.append([0, 0, grid_h-1, grid_w-1])
            desc = {'id': f'1e0a9b12-expert_{num}',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
