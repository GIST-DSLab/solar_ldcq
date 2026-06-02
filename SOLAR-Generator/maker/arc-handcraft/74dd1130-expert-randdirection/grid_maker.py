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
            flip_direction = np.random.randint(2)
            for _ in range(num_examples):
                h = np.random.randint(2, max_h+1)
                w = h
                rand_grid = np.random.randint(0, 10, size=[h, w])
                ex_in.append(rand_grid.copy())
                ex_out.append(np.flip(np.rot90(rand_grid), flip_direction).copy())
            h = np.random.randint(2, max_h+1)
            w = h
            rand_grid = np.random.randint(0, 10, size=[h, w])
            operations, selections = self.make_trace(rand_grid, flip_direction)
            pr_in.append(rand_grid.copy())
            pr_out.append(np.flip(np.rot90(rand_grid), flip_direction).copy())
            desc = {'id': f'74dd1130-gold_standard_{num}',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
    def make_trace(self, grid, flip_direction):
        h, w = grid.shape
        opr = []
        sel = []
        method = 0
        temp_sel = [0, 0, h-1, w-1]
        sel.append(temp_sel)
        sel.append(temp_sel)
        if flip_direction == 0:
            if method == 0:
                opr = [25, 26]
            elif method == 1:
                opr = [26, 24]
            elif method == 2:
                opr = [24, 27]
            elif method == 3:
                opr = [27, 25]
            else:
                raise NotImplementedError
        else:
            if method == 0:
                opr = [26, 25]
            elif method == 1:
                opr = [24, 26]
            elif method == 2:
                opr = [27, 24]
            elif method == 3:
                opr = [25, 27]
            else:
                raise NotImplementedError
        sel.append(temp_sel)
        opr.append(34)
        return opr, sel
