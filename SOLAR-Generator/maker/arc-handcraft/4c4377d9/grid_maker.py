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
            j = 0
            while (j < num_examples+1):
                h = np.random.randint(3, max_h//2)
                w = np.random.randint(3, max_w//2)
                rand_grid = np.zeros((h, w), dtype=np.uint8)
                numbers = list(range(10))
                selected_numbers = random.sample(numbers, 2)
                for y in range(len(rand_grid)):
                    for x in range(len(rand_grid[0])):
                        rand_grid[y][x] = random.sample(selected_numbers, 2)[0]
                down_grid = np.flipud(rand_grid)
                answer_grid = np.concatenate((rand_grid, down_grid))
                if (j == num_examples):
                    answer_h = 2 * h
                    answer_w = w
                    selections.append([0, 0, answer_h-1, answer_w-1])
                    operations.append(33)
                    selections.append([0, 0, answer_h-1, answer_w-1])
                    operations.append(32)
                    selections.append([0, 0, h-1, w-1])
                    operations.append(28)
                    selections.append([0, 0, h-1, w-1])
                    operations.append(30)
                    selections.append([h, 0, h-1, w-1])
                    operations.append(30)
                    selections.append([h, 0, h-1, w-1])
                    operations.append(27)
                    operations.append(34)
                    selections.append([0, 0, answer_h-1, answer_w-1])
                    pr_in.append(rand_grid)
                    pr_out.append(answer_grid)
                    j = j + 1
                else:
                    ex_in.append(rand_grid)
                    ex_out.append(answer_grid)
                    j = j + 1
            desc = {'id': '4c4377d9',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
