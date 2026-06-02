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
            p_colors = random.sample(range(1, 10), 2)
            j = 0
            while (j < num_examples+1):
                h = 10
                w = h
                rand_grid = np.zeros((h, w), dtype=np.uint8)
                points = []
                x1 = np.random.randint(1, h-2)
                y1 = np.random.randint(1, w-2)
                x2, y2 = x1, y1
                while self.is_diagonal_intersect((x1, y1), (x2, y2)) or self.is_diagonal_intersect((x1+1, y1), (x2, y2)) or self.is_diagonal_intersect((x1, y1+1), (x2, y2)):
                    x2 = np.random.randint(1, h-2)
                    y2 = np.random.randint(1, w-2)
                for dx, dy in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                    rand_grid[x1+dx][y1+dy] = p_colors[0]
                    rand_grid[x2+dx][y2+dy] = p_colors[1]
                answer_grid = rand_grid.copy()
                f_len = min(x1, y1)
                s1 = (x1, y1)
                for i in range(f_len):
                    s1 = (s1[0]-1, s1[1]-1)
                    x, y = s1
                    answer_grid[x][y] = p_colors[0]
                    if (j == num_examples):
                        selections.append([x, y, 0, 0])
                        operations.append(p_colors[0])
                s_len = min(h-x2-1, w-y2-1)
                s2 = (x2+1, y2+1)
                for i in range(s_len-1):
                    s2 = (s2[0]+1, s2[1]+1)
                    x, y = s2
                    answer_grid[x][y] = p_colors[1]
                    if (j == num_examples):
                        selections.append([x, y, 0, 0])
                        operations.append(p_colors[1])
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
            desc = {'id': f'5c0a986e-colordiff-gold_standard_{num}',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
            for i in range(1):
                protected_areas = self._get_protected_areas(h, w, num)
                branch_idx = random.randint(1, len(operations) - 2)
                rand_operations = operations[:branch_idx]
                rand_selections = selections[:branch_idx]
                for _ in range(8):
                    rand_x1, rand_y1 = self._get_safe_random_position(h, w, protected_areas)
                    rand_operations.append(random.choice([0, p_colors[0], p_colors[1]]))
                    rand_selections.append([rand_x1, rand_y1, 0, 0])
                rand_operations.append(34)
                rand_selections.append([0, 0, h-1, w-1])
                desc = {'id': f'5c0a986e-random_{num}_{i}',
                        'selections': rand_selections,
                        'operations': rand_operations}
                dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
    def _get_protected_areas(self, h, w, num):
        temp_state = np.random.get_state()
        np.random.seed(num)
        x1 = np.random.randint(1, h-2)
        y1 = np.random.randint(1, w-2)
        x2, y2 = x1, y1
        while self.is_diagonal_intersect((x1, y1), (x2, y2)) or self.is_diagonal_intersect((x1+1, y1), (x2, y2)) or self.is_diagonal_intersect((x1, y1+1), (x2, y2)):
            x2 = np.random.randint(1, h-2)
            y2 = np.random.randint(1, w-2)
        protected_areas = set()
        for dx, dy in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            protected_areas.add((x1+dx, y1+dy))
            protected_areas.add((x2+dx, y2+dy))
        np.random.set_state(temp_state)
        return protected_areas
    def _get_safe_random_position(self, h, w, protected_areas, max_attempts=50):
        for _ in range(max_attempts):
            rand_x = random.randint(0, h-1)
            rand_y = random.randint(0, w-1)
            if (rand_x, rand_y) not in protected_areas:
                return rand_x, rand_y
        return rand_x, rand_y
