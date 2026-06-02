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
                h = np.random.randint(5, max_h)
                w = np.random.randint(5, max_w)
                rand_grid = np.zeros((h, w), dtype=np.uint8)
                p_h1 = np.random.randint(h)
                p_h2 = p_h1
                while p_h1 == p_h2:
                    p_h2 = np.random.randint(h)
                if p_h1 > p_h2:
                    p_h1, p_h2 = p_h2, p_h1
                p_w1 = np.random.randint(w)
                p_w2 = p_w1
                while p_w1 == p_w2:
                    p_w2 = np.random.randint(w)
                if p_w1 > p_w2:
                    p_w1, p_w2 = p_w2, p_w1
                p_color = np.random.randint(1, 10)
                while all(all(element == 0 for element in sub_array) for sub_array in rand_grid):
                    for x in range(p_h1, p_h2+1):
                        for y in range(p_w1, p_w2+1):
                            rand_grid[x][y] = np.random.randint(0, 2) * p_color
                min_x, min_y = w-1, h-1
                max_x, max_y = 0, 0
                for y in range(len(rand_grid)):
                    for x in range(len(rand_grid[0])):
                        if rand_grid[y][x] != 0 and x < min_x:
                            min_x = x
                        if rand_grid[y][x] != 0 and y < min_y:
                            min_y = y
                        if rand_grid[y][x] != 0 and y > max_y:
                            max_y = y
                        if rand_grid[y][x] != 0 and x > max_x:
                            max_x = x
                answer_grid = rand_grid.copy()
                answer_grid = answer_grid[min_y: max_y+1, min_x: max_x+1]
                if (j == num_examples):
                    answer_h = max_y - min_y + 1
                    answer_w = max_x - min_x + 1
                    selections.append([0, 0, answer_h - 1, answer_w - 1])
                    operations.append(33)
                    selections.append([0, 0, answer_h - 1, answer_w - 1])
                    operations.append(32)
                    selections.append([min_y, min_x, answer_h - 1, answer_w - 1])
                    operations.append(28)
                    selections.append([0, 0, answer_h - 1, answer_w - 1])
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
            desc = {'id': '1cf80156',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
