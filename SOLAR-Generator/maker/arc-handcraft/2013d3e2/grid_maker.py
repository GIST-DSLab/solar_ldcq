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
            possible_list = [[25, 25, 25], [26, 27, 26]]
            action_list = possible_list[np.random.randint(2)]
            for _ in range(num_examples):
                h = np.random.randint(2, max_h//2+1)
                w = h
                rand_grid = np.random.randint(0, 10, size=[h, w])
                new_grid, _, _ = self.make_grid(rand_grid, action_list, max_h)
                ex_in.append(new_grid.copy())
                ex_out.append(rand_grid.copy())
            h = np.random.randint(2, max_h//2+1)
            w = h
            rand_grid = np.random.randint(0, 4, size=[h, w])
            new_grid, p_h, p_w = self.make_grid(rand_grid, action_list, max_h)
            pr_in.append(new_grid.copy())
            pr_out.append(rand_grid.copy())
            selections.append([0, 0, h-1, w-1])
            operations.append(33)
            selections.append([0, 0, h-1, w-1])
            operations.append(32)
            selections.append([p_h, p_w, h-1, w-1])
            operations.append(28)
            selections.append([0, 0, h-1, w-1])
            operations.append(30)
            selections.append([0, 0, h-1, w-1])
            operations.append(34)
            desc = {'id': '2013d3e2',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
    def make_grid(self, grid, action_list, max_h):
        h, w = grid.shape
        repeated_grid = np.zeros((2*h, 2*w))
        repeated_grid[0:h, 0:w] = grid.copy()
        for i, opr in enumerate(action_list):
            if i == 0:
                target_grid = grid.copy()
            else:
                target_grid = repeated_grid[h*((i)//2):h*((i)//2)+h, w*((i+1)//2):w*((i+1)//2)+w]
            if opr == 25:
                repeated_grid[h*((i+1)//2):h*((i+1)//2)+h, w*((3-i)//2):w*((3-i)//2)+w] = np.rot90(target_grid, 3).copy()
            elif opr == 26:
                repeated_grid[h*((i+1)//2):h*((i+1)//2)+h, w*((3-i)//2):w*((3-i)//2)+w] = np.flip(target_grid, 1).copy()
            elif opr == 27:
                repeated_grid[h*((i+1)//2):h*((i+1)//2)+h, w*((3-i)//2):w*((3-i)//2)+w] = np.flip(target_grid, 0).copy()
            else:
                raise NotImplementedError
        new_h = np.random.randint(2*h, max_h+1)
        new_w = new_h
        new_grid = np.zeros((new_h, new_w))
        p_h = np.random.randint(0, new_h-2*h+1)
        p_w = np.random.randint(0, new_w-2*w+1)
        new_grid[p_h:p_h+2*h, p_w:p_w+2*w] = repeated_grid.copy()
        return new_grid, p_h, p_w
