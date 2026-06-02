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
            colors_list = [3, 2, 1]
            j = 0
            while (j < num_examples+1):
                h = np.random.randint(6, max_h-1)
                w = np.random.randint(h+1, max_w)
                rand_grid = np.zeros((h, w), dtype=np.uint8)
                p_x = np.random.randint(1, h-1)
                p_y = np.random.randint(1, w-1)
                while p_x + p_y >= w or p_x + p_y < h:
                    p_x = np.random.randint(1, h-1)
                    p_y = np.random.randint(1, w-1)
                rand_grid[p_x][:p_y+1] = colors_list[1]
                answer_grid = rand_grid.copy()
                choice = np.random.randint(2)
                if choice == 0:
                    y_up = p_y
                    for x_up in range(p_x-1, -1, -1):
                        if y_up < w-1:
                            y_up += 1
                            answer_grid[x_up][:y_up+1] = colors_list[0]
                            if (j == num_examples):
                                selections.append([x_up, 0, 0, y_up])
                                operations.append(colors_list[0])
                    y_down = p_y
                    for x_down in range(p_x+1, h):
                        if y_down > 0:
                            y_down -= 1
                            answer_grid[x_down][:y_down+1] = colors_list[2]
                            if (j == num_examples):
                                selections.append([x_down, 0, 0, y_down])
                                operations.append(colors_list[2])
                if choice == 1:
                    y_down = p_y
                    for x_down in range(p_x+1, h):
                        if y_down > 0:
                            y_down -= 1
                            answer_grid[x_down][:y_down+1] = colors_list[2]
                            if (j == num_examples):
                                selections.append([x_down, 0, 0, y_down])
                                operations.append(colors_list[2])
                    y_up = p_y
                    for x_up in range(p_x-1, -1, -1):
                        if y_up < w-1:
                            y_up += 1
                            answer_grid[x_up][:y_up+1] = colors_list[0]
                            if (j == num_examples):
                                selections.append([x_up, 0, 0, y_up])
                                operations.append(colors_list[0])
                if (j == num_examples):
                    operations.append(34)
                    selections.append([0, 0, h-1, w-1])
                    pr_in.append(rand_grid)
                    pr_out.append(answer_grid)
                    j = j + 1
                else:
                    ex_in.append(rand_grid)
                    ex_out.append(answer_grid)
                    j = j + 1
            desc = {'id': f'a65b410d-expert_{num}',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
