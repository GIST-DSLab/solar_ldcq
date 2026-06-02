from maker.base_grid_maker import BaseGridMaker
from typing import Dict, List, Tuple
from numpy.typing import NDArray
import numpy as np
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
            l_color = np.random.randint(1, 10)
            l_thick = 1
            j = 0
            while (j < num_examples+1):
                h = np.random.randint(2 * l_thick + 1, max_h)
                w = np.random.randint(2 * l_thick + 1, max_w)
                rand_grid = np.zeros((h, w), dtype=np.uint8)
                answer_grid = rand_grid.copy()
                answer_grid[0:l_thick, :] = l_color
                answer_grid[h-l_thick:, :] = l_color
                answer_grid[:, :l_thick] = l_color
                answer_grid[:, w-l_thick:] = l_color
                if (j == num_examples):
                    choice = np.random.randint(3)
                    if choice == 0:
                        selections.append([0, 0, h - 1, l_thick - 1])
                        operations.append(l_color)
                        selections.append([0, w - l_thick, h - 1, l_thick - 1])
                        operations.append(l_color)
                        selections.append([0, 0, l_thick - 1, w - 1])
                        operations.append(l_color)
                        selections.append([h - l_thick, 0, l_thick - 1, w - 1])
                        operations.append(l_color)
                    elif choice == 1:
                        selections.append([0, 0, h - 1, l_thick - 1])
                        operations.append(l_color)
                        selections.append([0, w - l_thick, h - 1, l_thick - 1])
                        operations.append(l_color)
                        selections.append([0, l_thick, l_thick - 1, w - 1 - 2 * l_thick])
                        operations.append(l_color)
                        selections.append([h - l_thick, l_thick, l_thick - 1, w - 1 - 2 * l_thick])
                        operations.append(l_color)
                    elif choice == 2:
                        selections.append([l_thick, 0, h - l_thick - 1, l_thick - 1])
                        operations.append(l_color)
                        selections.append([l_thick, w - l_thick, h - l_thick - 1, l_thick - 1])
                        operations.append(l_color)
                        selections.append([0, 0, l_thick - 1, w - 1])
                        operations.append(l_color)
                        selections.append([h - l_thick, 0, l_thick - 1, w - 1])
                        operations.append(l_color)
                    operations.append(34)
                    selections.append([0, 0, h-1, w-1])
                    pr_in.append(rand_grid)
                    pr_out.append(answer_grid)
                    j = j + 1
                else:
                    ex_in.append(rand_grid)
                    ex_out.append(answer_grid)
                    j = j + 1
            desc = {'id': f'6f8cd79b_expert_{num}',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
            for i in range(9):
                branch_idx = np.random.randint(1, len(operations) - 2)
                new_operations = operations[:branch_idx]
                new_selections = selections[:branch_idx]
                random_color = np.random.randint(1, 10)
                num_remaining = len(operations) - 1 - branch_idx
                new_operations += [random_color] * num_remaining
                new_operations.append(34)
                new_selections = selections[:len(new_operations)]
                desc = {
                    'id': f'6f8cd79b-random_{num}_{i}',
                    'selections': new_selections,
                    'operations': new_operations,
                }
                dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
