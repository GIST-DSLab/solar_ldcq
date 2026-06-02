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
            h = np.random.randint(3, min(max_h // 2, 5))
            w = h
            answer_h = 2 * h
            answer_w = w
            border_color = np.random.randint(1, 10)
            numbers = [i for i in range(1, 10) if i != border_color]
            selected_numbers = random.sample(numbers, 2)
            j = 0
            while (j < num_examples + 1):
                rand_grid = np.random.choice(selected_numbers, size=[h, w]).astype(np.uint8)
                flipped_grid = np.flipud(rand_grid)
                step1_grid = np.concatenate((flipped_grid, rand_grid), axis=0)
                answer_grid = step1_grid.copy()
                answer_grid[0, :] = border_color
                answer_grid[answer_h - 1, :] = border_color
                answer_grid[:, 0] = border_color
                answer_grid[:, answer_w - 1] = border_color
                if (j == num_examples):
                    selections.append([0, 0, answer_h - 1, answer_w - 1])
                    operations.append(33)
                    selections.append([0, 0, h - 1, w - 1])
                    operations.append(29)
                    selections.append([h, 0, h - 1, w - 1])
                    operations.append(30)
                    selections.append([0, 0, h - 1, w - 1])
                    operations.append(27)
                    selections.append([0, 0, 0, answer_w - 1])
                    operations.append(border_color)
                    selections.append([answer_h - 1, 0, 0, answer_w - 1])
                    operations.append(border_color)
                    selections.append([0, 0, answer_h - 1, 0])
                    operations.append(border_color)
                    selections.append([0, answer_w - 1, answer_h - 1, 0])
                    operations.append(border_color)
                    selections.append([0, 0, answer_h - 1, answer_w - 1])
                    operations.append(34)
                    pr_in.append(rand_grid)
                    pr_out.append(answer_grid)
                    j += 1
                else:
                    ex_in.append(rand_grid)
                    ex_out.append(answer_grid)
                    j += 1
            desc = {'id': f'simple-combo-4c4377d9-6f8cd79b-expert_{num}',
                    'selections': selections,
                    'operations': operations,
                    'concept': "flip vertically and concatenate below, drawing border lines with given color"}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
