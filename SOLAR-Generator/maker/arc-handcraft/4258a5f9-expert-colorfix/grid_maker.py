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
            selections: List[NDArray] = []
            operations: List[NDArray] = []
            p_color = 5
            r_color = 1
            j = 0
            while (j < num_examples+1):
                h = np.random.randint(5, max_h)
                w = h
                rand_grid = np.zeros((h, w), dtype=np.uint8)
                max_points = h * h // 9
                weights = list(range(1, max_points))
                num_p = random.choices(range(1, max_points), weights=weights, k=1)[0]
                points = []
                for _ in range(num_p):
                    x = random.randint(1,h-2)
                    y = random.randint(1,w-2)
                    new_point = (x, y)
                    if all(abs(new_point[0] - point[0]) >= 2 and abs(new_point[1] - point[1]) >= 2 for point in points):
                        points.append(new_point)
                    else:
                        continue
                for x, y in points:
                    rand_grid[x][y] = p_color
                answer_grid = rand_grid.copy()
                for x, y in points:
                    for dx, dy in [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]:
                        answer_grid[x+dx][y+dy] = r_color
                if (j == num_examples):
                    choice = 0
                    if choice == 0:
                        all_cells = []
                        for x, y in points:
                            for dx, dy in [(1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1)]:
                                all_cells.append((x+dx, y+dy))
                        for x, y in all_cells:
                            selections.append([x, y, 0, 0])
                            operations.append(r_color)
                    operations.append(34)
                    selections.append([0, 0, h-1, w-1])
                    pr_in.append(rand_grid)
                    pr_out.append(answer_grid)
                    j = j + 1
                else:
                    ex_in.append(rand_grid)
                    ex_out.append(answer_grid)
                    j = j + 1
            desc = {'id': f'4258a5f9-gold_standard_{num}',
                    'selections': selections,
                    'operations': operations,
                    "concept":f'filling 8-directional neighbors around points'}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
