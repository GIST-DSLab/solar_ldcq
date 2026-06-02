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
                num_p = random.choice(range(1, h * h // 9))
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
                    for dx, dy in [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]:                        answer_grid[x+dx][y+dy] = r_color
                if (j == num_examples):
                    choice = 0
                    if choice == 0:
                        all_cells = []
                        for x, y in points:
                            for dx, dy in [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]:
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
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
            for i in range(1):
                new_operations = []
                new_selections = []
                forbidden_positions = set()
                for x, y in points:
                    forbidden_positions.add((x, y))
                    for dx, dy in [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]:
                        if 0 <= x+dx < h and 0 <= y+dy < w:
                            forbidden_positions.add((x+dx, y+dy))
                available_positions = []
                for x in range(h):
                    for y in range(w):
                        if (x, y) not in forbidden_positions:
                            available_positions.append((x, y))
                if available_positions:
                    strategy = 0
                    if strategy == 0:
                        num_colors = min(np.random.randint(5, 15), len(available_positions))
                        selected_positions = random.sample(available_positions, num_colors)
                        for x, y in selected_positions:
                            rand_color=np.random.choice([0,r_color])
                            new_selections.append([x, y, 0, 0])
                            new_operations.append(rand_color)
                    elif strategy == 1:
                        corner_edge_positions = []
                        for x, y in available_positions:
                            if (x == 0 or x == h-1 or y == 0 or y == w-1):
                                corner_edge_positions.append((x, y))
                        if corner_edge_positions:
                            num_colors = min(np.random.randint(3, 8), len(corner_edge_positions))
                            selected_positions = random.sample(corner_edge_positions, num_colors)
                        else:
                            num_colors = min(np.random.randint(3, 8), len(available_positions))
                            selected_positions = random.sample(available_positions, num_colors)
                        for x, y in selected_positions:
                            new_selections.append([x, y, 0, 0])
                            new_operations.append(r_color)
                    else:
                        distant_positions = []
                        for x, y in available_positions:
                            min_distance = float('inf')
                            for px, py in points:
                                distance = abs(x - px) + abs(y - py)
                                min_distance = min(min_distance, distance)
                            if min_distance >= 3:
                                distant_positions.append((x, y))
                        if distant_positions:
                            num_colors = min(np.random.randint(3, 10), len(distant_positions))
                            selected_positions = random.sample(distant_positions, num_colors)
                        else:
                            num_colors = min(np.random.randint(3, 8), len(available_positions))
                            selected_positions = random.sample(available_positions, num_colors)
                        for x, y in selected_positions:
                            new_selections.append([x, y, 0, 0])
                            new_operations.append(r_color)
                new_operations.append(34)
                new_selections.append([0, 0, h-1, w-1])
                desc = {
                    'id': f'4258a5f9-random_{num}_{i}',
                    'selections': new_selections,
                    'operations': new_operations,
                }
                dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
