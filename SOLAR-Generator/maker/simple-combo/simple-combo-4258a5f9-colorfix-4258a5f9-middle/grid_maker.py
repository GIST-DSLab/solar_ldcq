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
            p_color = 5
            r_color = 1
            border_color = np.random.randint(1, 10)
            while border_color == p_color or border_color == r_color:
                border_color = np.random.randint(1, 10)
            j = 0
            while (j < num_examples+1):
                h = np.random.randint(7, min(max_h, 10))
                w = h
                rand_grid = np.zeros((h, w), dtype=np.uint8)
                max_points = (h - 2) * (h - 2) // 9
                weights = list(range(1, max(2, max_points)))
                num_p = random.choices(range(1, max(2, max_points)), weights=weights, k=1)[0]
                points = []
                for _ in range(num_p):
                    x = random.randint(2, h-3)
                    y = random.randint(2, w-3)
                    new_point = (x, y)
                    if all(abs(new_point[0] - point[0]) >= 3 and abs(new_point[1] - point[1]) >= 3 for point in points):
                        points.append(new_point)
                if len(points) == 0:
                    points.append((h//2, w//2))
                for x, y in points:
                    rand_grid[x][y] = p_color
                step1_grid = rand_grid.copy()
                for x, y in points:
                    for dx, dy in [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]:
                        step1_grid[x+dx][y+dy] = r_color
                answer_grid = step1_grid.copy()
                answer_grid[0, :] = border_color
                answer_grid[h-1, :] = border_color
                answer_grid[:, 0] = border_color
                answer_grid[:, w-1] = border_color
                if (j == num_examples):
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
                    pr_out.append(step1_grid)
                    j += 1
                else:
                    ex_in.append(rand_grid)
                    ex_out.append(answer_grid)
                    j += 1
            desc = {'id': f'simple-combo-4258a5f9-colorfix-4258a5f9-middle-expert_{num}',
                    'selections': selections,
                    'operations': operations,
                    'concept': "filling 8-directional neighbors around points, drawing border lines with given color"}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
