from typing import Dict, List, Tuple
from numpy.typing import NDArray
import numpy as np
import random
from maker.base_grid_maker import BaseGridMaker
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
            selection: List[NDArray] = []
            operation: List[NDArray] = []
            for _ in range(num_examples):
                h = np.random.randint(1, max_h//2+1)
                w = h
                rand_grid = np.random.randint(0, 10, size=[h, w], dtype=np.uint8)
                ex_in.append(rand_grid)
                ex_out.append(self.make_answer(rand_grid))
            h=3
            w = h
            rand_grid=np.array([[1, 4, 1], [4, 9, 4], [9, 1, 9]])
            pr_in.append(rand_grid)
            pr_out.append(self.make_answer(rand_grid))
            method = self.get_random_method()
            operation, selection = method(h, w)
            desc = {'id': f'46442a0e-golden-standard_{num}',
                    'selections': selection,
                    'operations': operation}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
            for i in range(2):
                rand_operations, rand_selections = self.random_trajectory(
                    expert_operations = operation,
                    expert_selections = selection,
                    h=h,
                    w=w,
                    answer_h=2*h,
                    answer_w=2*w,
                    num=num,
                    i=i,
                    num_random_ops = 4
                )
                desc = {'id': f'46442a0e-random_{num}_{i}',
                        'selections': rand_selections,
                        'operations': rand_operations}
                dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
    def make_answer(self, grid):
        h, w = grid.shape
        ans = np.zeros([2*h, 2*w], dtype=np.int8)
        ans[:h, :w] = grid.copy()
        ans[h:2*h, :w] = np.rot90(grid).copy()
        ans[h:2*h, w:2*w] = np.rot90(grid, 2).copy()
        ans[:h, w:2*w] = np.rot90(grid, 3).copy()
        return ans
    def get_random_method(self):
        methods = [method for method in dir(
            self) if method.startswith("method")]
        random_method = random.choice(methods)
        return getattr(self, random_method)
    def method1(self, h, w):
        operations = [33, 29, 30, 24, 29, 30, 26, 27, 34]
        selections = [[0, 0, 2*h-1, 2*w-1],
                      [0, 0, h-1, w-1],
                      [h, 0, h-1, w-1],
                      [h, 0, h-1, w-1],
                      [0, 0, 2*h-1, w-1],
                      [0, w, 2*h-1, w-1],
                      [0, w, 2*h-1, w-1],
                      [0, w, 2*h-1, w-1],
                      [0, 0, 2*h-1, 2*w-1]
                      ]
        return operations, selections
    def method2(self, h, w):
        operations = [33, 29, 30, 25, 29, 30, 26, 27, 34]
        selections = [[0, 0, 2*h-1, 2*w-1],
                      [0, 0, h-1, w-1],
                      [0, w, h-1, w-1],
                      [0, w, h-1, w-1],
                      [0, 0, h-1, 2*w-1],
                      [h, 0, h-1, 2*w-1],
                      [h, 0, h-1, 2*w-1],
                      [h, 0, h-1, 2*w-1],
                      [0, 0, 2*h-1, 2*w-1]
                      ]
        return operations, selections
    def method3(self, h, w):
        operations = [33, 29, 30, 24, 29, 30, 24, 29, 30, 24, 34]
        selections = [[0, 0, 2*h-1, 2*w-1],
                      [0, 0, h-1, w-1],
                      [h, 0, h-1, w-1],
                      [h, 0, h-1, w-1],
                      [h, 0, h-1, w-1],
                      [h, w, h-1, w-1],
                      [h, w, h-1, w-1],
                      [h, w, h-1, w-1],
                      [0, w, h-1, w-1],
                      [0, w, h-1, w-1],
                      [0, 0, 2*h-1, 2*w-1]
                      ]
        return operations, selections
    def method4(self, h, w):
        operations = [33, 29, 30, 25, 29, 30, 25, 29, 30, 25, 34]
        selections = [[0, 0, 2*h-1, 2*w-1],
                      [0, 0, h-1, w-1],
                      [0, w, h-1, w-1],
                      [0, w, h-1, w-1],
                      [0, w, h-1, w-1],
                      [h, w, h-1, w-1],
                      [h, w, h-1, w-1],
                      [h, w, h-1, w-1],
                      [h, 0, h-1, w-1],
                      [h, 0, h-1, w-1],
                      [0, 0, 2*h-1, 2*w-1]
                      ]
        return operations, selections
    def random_trajectory(self,
                        expert_operations: List[int],
                        expert_selections: List[List[int]],
                        h: int,
                        w: int,
                        answer_h: int,
                        answer_w: int,
                        num : int,
                        i: int,
                        num_random_ops: int = 3) :
        state = np.random.get_state()
        np.random.seed(num + i)
        insertion_point = np.random.randint(1, len(expert_operations) - 2)
        selections = expert_selections[:insertion_point]
        operations = expert_operations[:insertion_point]
        possible_selections = [
                            [0, 0, h-1, w-1],
                            [h, 0, h-1, w-1],
                            [0, w, h-1, w-1],
                            [h, w, h-1, w-1]
                        ]
        for _ in range(num_random_ops):
            random_op = random.choice([24, 25, 26, 27, 29])
            random_sel = random.choice(possible_selections)
            operations.append(random_op)
            selections.append(random_sel)
            if random_op == 29 :
                operations.append(30)
                radom_sel_paste = random.choice(possible_selections)
                selections.append(radom_sel_paste)
        np.random.set_state(state)
        selections.append([0, 0, answer_h-1, answer_w-1])
        operations.append(34)
        return operations, selections
