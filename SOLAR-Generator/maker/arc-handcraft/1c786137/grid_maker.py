from maker.base_grid_maker import BaseGridMaker
from typing import Dict, List, Tuple
from numpy.typing import NDArray
import numpy as np
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
                h = np.random.randint(6, max_h)
                w = np.random.randint(6, max_w)
                rand_grid = np.zeros((h, w), dtype=np.uint8)
                numbers = list(range(10))
                selected_numbers = np.random.choice(numbers, 4, replace=False)
                for x in range(h):
                    for y in range(w):
                        rand_grid[x, y] = np.random.choice(selected_numbers)
                l_y_1 = np.random.randint(1, w-1)
                l_y_2 = np.random.randint(1, w-1)
                l_x_1 = np.random.randint(1, h-1)
                l_x_2 = np.random.randint(1, h-1)
                while abs(l_y_2 - l_y_1) < 2:
                    l_y_2 = np.random.randint(1, w-1)
                while abs(l_x_2 - l_x_1) < 2:
                    l_x_2 = np.random.randint(1, h-1)
                if l_x_1 > l_x_2:
                    l_x_2, l_x_1 = l_x_1, l_x_2
                if l_y_1 > l_y_2:
                    l_y_2, l_y_1 = l_y_1, l_y_2
                unselected_numbers = [x for x in numbers if x not in selected_numbers]
                l_color = np.random.choice(unselected_numbers)
                for x in range(len(rand_grid)):
                    for y in range(len(rand_grid[0])):
                        if x == l_x_1 and y >= l_y_1 and y <= l_y_2:
                            rand_grid[x, y] = l_color
                        if x == l_x_2 and y >= l_y_1 and y <= l_y_2:
                            rand_grid[x, y] = l_color
                        if y == l_y_1 and x >= l_x_1 and x <= l_x_2:
                            rand_grid[x, y] = l_color
                        if y == l_y_2 and x >= l_x_1 and x <= l_x_2:
                            rand_grid[x, y] = l_color
                answer_grid = rand_grid.copy()
                answer_grid = rand_grid[l_x_1+1: l_x_2, l_y_1+1: l_y_2]
                if (j == num_examples):
                    answer_h = l_x_2 - l_x_1 - 1
                    answer_w = l_y_2 - l_y_1 - 1
                    selections.append([0, 0, l_x_2 - l_x_1 - 2, l_y_2 - l_y_1 - 2])
                    operations.append(33)
                    selections.append([0, 0, l_x_2 - l_x_1 - 2, l_y_2 - l_y_1 - 2])
                    operations.append(32)
                    selections.append([l_x_1+1, l_y_1+1, l_x_2 - l_x_1 - 2, l_y_2 - l_y_1 - 2])
                    operations.append(28)
                    selections.append([0, 0, l_x_2 - l_x_1 - 2, l_y_2 - l_y_1 - 2])
                    operations.append(30)
                    operations.append(34)
                    selections.append([0, 0, answer_h-1, answer_w-1])
                    pr_in.append(rand_grid)
                    pr_out.append(answer_grid)
                    j = j + 1
                else:
                    ex_in.append(rand_grid)
                    ex_out.append(answer_grid)
                    j = j + 1
            desc = {'id': '1c786137',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
