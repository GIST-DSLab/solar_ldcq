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
            j = 0
            while (j < num_examples + 1):
                h = np.random.randint(4, min(max_h // 2, 6))
                w = h
                rand_grid = np.zeros((h, w), dtype=np.uint8)
                max_points = (h - 2) * (h - 2) // 9
                max_points = max(2, max_points)
                weights = list(range(1, max_points))
                if len(weights) == 0:
                    weights = [1]
                num_p = random.choices(range(1, max_points), weights=weights, k=1)[0]
                points = []
                for _ in range(num_p):
                    x = random.randint(1, h - 2)
                    y = random.randint(1, w - 2)
                    new_point = (x, y)
                    if all(abs(new_point[0] - point[0]) >= 3 and abs(new_point[1] - point[1]) >= 3 for point in points):
                        points.append(new_point)
                if len(points) == 0:
                    points.append((h // 2, w // 2))
                for x, y in points:
                    rand_grid[x][y] = p_color
                step1_grid = rand_grid.copy()
                for x, y in points:
                    for dx, dy in [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]:
                        step1_grid[x + dx][y + dy] = r_color
                flipped_grid = np.flipud(step1_grid)
                answer_grid = np.concatenate((flipped_grid, step1_grid), axis=0)
                answer_h = 2 * h
                answer_w = w
                if (j == num_examples):
                    all_cells = []
                    for x, y in points:
                        for dx, dy in [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]:
                            all_cells.append((x + dx, y + dy))
                    for x, y in all_cells:
                        selections.append([x, y, 0, 0])
                        operations.append(r_color)
                    selections.append([0, 0, answer_h - 1, answer_w - 1])
                    operations.append(33)
                    selections.append([0, 0, h - 1, w - 1])
                    operations.append(29)
                    selections.append([h, 0, h - 1, w - 1])
                    operations.append(30)
                    selections.append([0, 0, h - 1, w - 1])
                    operations.append(27)
                    selections.append([0, 0, answer_h - 1, answer_w - 1])
                    operations.append(34)
                    pr_in.append(rand_grid)
                    pr_out.append(answer_grid)
                    j += 1
                else:
                    ex_in.append(rand_grid)
                    ex_out.append(answer_grid)
                    j += 1
            desc = {'id': f'simple-combo-4258a5f9-4c4377d9-expert_{num}',
                    'selections': selections,
                    'operations': operations,
                    'concept': "filling 8-directional neighbors around points, flip vertically and concatenate below"}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
