from typing import Dict, List, Tuple
from numpy.typing import NDArray
import numpy as np
from maker.base_grid_maker import BaseGridMaker
class GridMaker(BaseGridMaker):
    def parse(self, **kwargs) -> List[Tuple[List[NDArray], List[NDArray], List[NDArray], List[NDArray], Dict, List[NDArray], List[NDArray]]]:
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
            for _ in range(num_examples):
                stack_direction = np.random.randint(4)
                if stack_direction == 0:
                    h = np.random.randint(1, max_h+1)
                    w = np.random.randint(1, max_w//2+1)
                elif stack_direction == 1:
                    h = np.random.randint(1, max_h//2+1)
                    w = np.random.randint(1, max_w+1)
                elif stack_direction == 2:
                    h = np.random.randint(1, max_h//2+1)
                    w = np.random.randint(1, max_w//2+1)
                else:
                    h = np.random.randint(1, max_h+1)
                    w = np.random.randint(1, max_w+1)
                input_matrix = self.no_repeat_matrix(h, w)
                ans_grid, _, _ = self.make_grid(
                    input_matrix, max_h, max_w, stack_direction)
                ex_in.append(ans_grid)
                ex_out.append(input_matrix)
            stack_direction = np.random.randint(3)
            if stack_direction == 0:
                h = np.random.randint(1, max_h+1)
                w = np.random.randint(1, max_w//2+1)
            elif stack_direction == 1:
                h = np.random.randint(1, max_h//2+1)
                w = np.random.randint(1, max_w+1)
            elif stack_direction == 2:
                h = np.random.randint(1, max_h//2+1)
                w = np.random.randint(1, max_w//2+1)
            else:
                h = np.random.randint(1, max_h+1)
                w = np.random.randint(1, max_w+1)
            input_matrix = self.no_repeat_matrix(h, w)
            ans_grid, n_hor, n_ver = self.make_grid(
                input_matrix, max_h, max_w, stack_direction)
            pr_in.append(ans_grid)
            pr_out.append(input_matrix)
            sel_h = np.random.randint(n_ver)
            sel_w = np.random.randint(n_hor)
            selections.append([h*sel_h, w*sel_w, h-1, w-1])
            operations.append(33)
            selections.append([0, 0, h-1, w-1])
            operations.append(34)
            desc = {'id': '7b7f7511',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
    def no_repeat_matrix(self, h, w):
        repeated_matrix = True
        while repeated_matrix:
            rand_grid = np.random.randint(
                0, 10, size=[h, w], dtype=np.uint8)
            repeated_matrix = self.check_repeat(rand_grid)
        return rand_grid
    def check_repeat(self, random_grid):
        n, m = random_grid.shape
        for h in range(1, n//2+1):
            if n % h != 0:
                continue
            for w in range(1, m//2+1):
                if m % w != 0:
                    continue
                if np.array_equal(random_grid[:h, :w], random_grid[h:2*h, :w]) or np.array_equal(random_grid[:h, :w], random_grid[:h, w:2*w]):
                    return True
        return False
    def make_grid(self, grid, max_h, max_w, stack_direction):
        h, w = grid.shape
        if stack_direction == 0:
            n_hor = np.random.randint(2, max_w//w+1)
            n_ver = 1
        elif stack_direction == 1:
            n_hor = 1
            n_ver = np.random.randint(2, max_h//h+1)
        elif stack_direction == 2:
            n_hor = np.random.randint(2, max_w//w+1)
            n_ver = np.random.randint(2, max_h//h+1)
        else:
            n_hor = 1
            n_ver = 1
        new_grid = np.zeros((h*n_ver, w*n_hor), dtype=np.uint8)
        for i in range(n_hor):
            for j in range(n_ver):
                new_grid[h*j:h*(j+1), w*i:w*(i+1)] = grid.copy()
        return new_grid, n_hor, n_ver
