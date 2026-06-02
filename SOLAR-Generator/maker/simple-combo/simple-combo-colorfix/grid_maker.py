from maker.base_grid_maker import BaseGridMaker
from typing import Dict, List, Tuple
from numpy.typing import NDArray
import numpy as np
import random
class GridMaker(BaseGridMaker):
    def is_diagonal_intersect(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return abs(x1 - x2) == abs(y1 - y2) or x1 + y1 == x2 + y2
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
            rect1_color = 1
            rect2_color = 2
            border_color = np.random.randint(1, 10)
            j = 0
            while (j < num_examples+1):
                h = random.randint(7, min(max_h, 9))
                w = h
                rand_grid = np.zeros((h, w), dtype=np.uint8)
                x1 = random.randint(1, h-3)
                y1 = random.randint(1, w-3)
                x2 = random.randint(1, h-3)
                y2 = random.randint(1, w-3)
                attempts = 0
                max_attempts = 50
                while (self.is_diagonal_intersect((x1, y1), (x2, y2)) or \
                       self.is_diagonal_intersect((x1+1, y1), (x2, y2)) or \
                       self.is_diagonal_intersect((x1, y1+1), (x2, y2)) or \
                       abs(x1-x2) < 2 or abs(y1-y2) < 2) and attempts < max_attempts:
                    x2 = random.randint(1, h-3)
                    y2 = random.randint(1, w-3)
                    attempts += 1
                if attempts >= max_attempts:
                    x2 = min(h-3, x1 + 3)
                    y2 = min(w-3, y1 + 3)
                for dx, dy in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                    rand_grid[x1+dx][y1+dy] = rect1_color
                    rand_grid[x2+dx][y2+dy] = rect2_color
                step1_grid = rand_grid.copy()
                f_len = min(x1, y1)
                s1 = (x1, y1)
                for i in range(f_len):
                    s1 = (s1[0]-1, s1[1]-1)
                    x, y = s1
                    step1_grid[x][y] = rect1_color
                s_len = min(h-x2-1, w-y2-1)
                s2 = (x2+1, y2+1)
                for i in range(s_len-1):
                    s2 = (s2[0]+1, s2[1]+1)
                    x, y = s2
                    step1_grid[x][y] = rect2_color
                answer_grid = step1_grid.copy()
                answer_grid[0, :] = border_color
                answer_grid[h-1, :] = border_color
                answer_grid[:, 0] = border_color
                answer_grid[:, w-1] = border_color
                if (j == num_examples):
                    s1 = (x1, y1)
                    for i in range(f_len):
                        s1 = (s1[0]-1, s1[1]-1)
                        x, y = s1
                        selections.append([x, y, 0, 0])
                        operations.append(rect1_color)
                    s2 = (x2+1, y2+1)
                    for i in range(s_len-1):
                        s2 = (s2[0]+1, s2[1]+1)
                        x, y = s2
                        selections.append([x, y, 0, 0])
                        operations.append(rect2_color)
                    selections.append([0, 0, 0, w-1])
                    operations.append(border_color)
                    selections.append([h-1, 0, 0, w-1])
                    operations.append(border_color)
                    selections.append([0, 0, h-1, 0])
                    operations.append(border_color)
                    selections.append([0, w-1, h-1, 0])
                    operations.append(border_color)
                    operations.append(34)
                    selections.append([0, 0, h-1, w-1])
                    pr_in.append(rand_grid)
                    pr_out.append(answer_grid)
                    j += 1
                else:
                    ex_in.append(rand_grid)
                    ex_out.append(answer_grid)
                    j += 1
            desc = {'id': f'simple-combo-colorfix-expert_{num}',
                    'selections': selections,
                    'operations': operations,
                    'concept':"diagonal line based on two colored squares, drawing border lines with given color"}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
