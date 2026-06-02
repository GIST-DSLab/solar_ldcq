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
            color_list = random.sample(range(1, 10), 2)
            j = 0
            while (j < num_examples+1):
                h = np.random.randint(4, max_h)
                w = np.random.randint(4, max_w)
                rand_grid = np.zeros((h, w), dtype=np.uint8)
                p_x1 = np.random.randint(h)
                p_x2 = p_x1
                while abs(p_x1 - p_x2) < 3:
                    p_x1 = np.random.randint(h)
                    p_x2 = np.random.randint(h)
                if p_x1 > p_x2:
                    p_x1, p_x2 = p_x2, p_x1
                p_y1 = np.random.randint(w)
                p_y2 = p_y1
                while abs(p_y1 - p_y2) < 3:
                    p_y1 = np.random.randint(w)
                    p_y2 = np.random.randint(w)
                if p_y1 > p_y2:
                    p_y1, p_y2 = p_y2, p_y1
                p_color = color_list[0]
                rand_grid[p_x1, p_y1:p_y2+1] = p_color
                rand_grid[p_x2, p_y1:p_y2+1] = p_color
                rand_grid[p_x1:p_x2+1, p_y1] = p_color
                rand_grid[p_x1:p_x2+1, p_y2] = p_color
                points = []
                for x in range(p_x1, p_x2+1):
                    for y in range(p_y1, p_y2+1):
                        if x == p_x1 or x == p_x2 or y == p_y1 or y == p_y2:
                            if (x == p_x1 and y == p_y1) or (x == p_x2 and y == p_y1) or (x == p_x1 and y == p_y2) or (x == p_x2 and y == p_y2):
                                continue
                            else:
                                points.append((x, y))
                hole = random.choice(points)
                h_x, h_y = hole
                rand_grid[h_x, h_y] = 0
                answer_grid = rand_grid.copy()
                for x in range(p_x1+1, p_x2):
                    for y in range(p_y1+1, p_y2):
                        answer_grid[x][y] = color_list[1]
                if (j == num_examples):
                    selections.append([p_x1+1, p_y1+1, p_x2 - p_x1 - 2, p_y2 - p_y1 - 2])
                    operations.append(color_list[1])
                if h_y == p_y1:
                    answer_grid[h_x, :h_y+1] = color_list[1]
                    if (j == num_examples):
                        selections.append([h_x, 0, 0, p_y1])
                        operations.append(color_list[1])
                elif h_y == p_y2:
                    answer_grid[h_x, h_y:] = color_list[1]
                    if (j == num_examples):
                        selections.append([h_x, p_y2, 0, w - p_y2])
                        operations.append(color_list[1])
                elif h_x == p_x1:
                    answer_grid[:h_x+1, h_y] = color_list[1]
                    if (j == num_examples):
                        selections.append([0, h_y, p_x1, 0])
                        operations.append(color_list[1])
                elif h_x == p_x2:
                    answer_grid[h_x:, h_y] = color_list[1]
                    if (j == num_examples):
                        selections.append([h_x, h_y, h - h_x, 0])
                        operations.append(color_list[1])
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
            desc = {'id': 'd4f3cd78',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
