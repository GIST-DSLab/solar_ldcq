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
            stack_direction = 0
            start_left = 1
            if stack_direction == 1:
                available_space = max_h
                unit_width = min(max_w, 5)
            else:
                available_space = max_w
                unit_width = min(max_h, 5)
            d = 1
            max_unit_size = min(unit_width, (available_space - d) // 3)
            if max_unit_size < 2:
                max_unit_size = 2
            if max_unit_size >= 3:
                unit_choices = [3, 3, 3] + list(range(2, max_unit_size + 1))
                unit_size = np.random.choice(unit_choices)
            else:
                unit_size = max_unit_size
            max_possible_grids = (available_space + d) // (unit_size + d)
            if max_possible_grids < 2:
                d = 1
                unit_size = 2
                max_possible_grids = (available_space + d) // (unit_size + d)
            if max_possible_grids >= 3:
                num_small_grid = np.random.randint(2, max_possible_grids + 1)
            else:
                num_small_grid = max_possible_grids
            possible_list = [24, 25]
            action_list = np.random.choice(possible_list, size=num_small_grid-1)
            wall_color = np.random.randint(1, 10)
            j = 0
            while j < num_examples + 1:
                h = unit_size
                w = h
                if h == 1:
                    other_color_list = [
                        i for i in range(1, 10) if i != wall_color]
                else:
                    other_color_list = [
                        i for i in range(10) if i != wall_color]
                rand_grid = np.random.choice(other_color_list, size=[h, w])
                in_grid = self.make_grid(
                    rand_grid, h, w, start_left, num_small_grid, d, wall_color, stack_direction)
                if j == num_examples:
                    pr_in.append(in_grid.copy())
                    pr_out.append(self.make_answer_grid(in_grid, h, w, start_left, d, operations, selections, action_list, stack_direction, True))
                else:
                    ex_in.append(in_grid.copy())
                    ex_out.append(self.make_answer_grid(in_grid, h, w, start_left, d, operations, selections, action_list, stack_direction, False))
                j += 1
            desc = {'id': f'8e5a5113-expert_{num}',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
    def make_grid(self, rand_grid, h, w, start_left, num_small_grid, d, wall_color, stack_direction):
        if stack_direction == 1:
            in_h = (h+d)*num_small_grid-d
            in_w = w
            wall_grid = np.full((d, w), wall_color)
        else:
            in_h = h
            in_w = (w+d)*num_small_grid-d
            wall_grid = np.full((h, d), wall_color)
        in_grid = np.zeros((in_h, in_w))
        if start_left:
            in_grid[:h, :w] = rand_grid.copy()
        else:
            if stack_direction == 1:
                in_grid[in_h-h:in_h, :in_w] = rand_grid.copy()
            else:
                in_grid[:in_h, in_w - w:in_w] = rand_grid.copy()
        for i in range(1, num_small_grid):
            if stack_direction == 1:
                in_grid[h*i+d*i-d:h*i+d*i, :in_w] = wall_grid.copy()
            else:
                in_grid[:in_h, w*i+d*i-d:w*i+d*i] = wall_grid.copy()
        return in_grid
    def make_answer_grid(self, in_grid, h, w, start_left, d, operations, selections, action_list, stack_direction, for_test):
        out_grid = in_grid.copy()
        if stack_direction == 1:
            if start_left:
                for i, opr in enumerate(action_list):
                    low_h = h*i + d*i
                    high_h = h*(i+1) + d*i
                    if h == 1:
                        out_grid[low_h+h+d:high_h+h+d, :w] = out_grid[low_h:high_h, :w].copy()
                    elif opr == 24:
                        out_grid[low_h+h+d:high_h+h+d, :w] = np.rot90(out_grid[low_h:high_h, :w]).copy()
                    elif opr == 25:
                        out_grid[low_h+h+d:high_h+h+d, :w] = np.rot90(out_grid[low_h:high_h, :w], 3).copy()
                    elif opr == 26:
                        out_grid[low_h+h+d:high_h+h+d, :w] = np.flip(out_grid[low_h:high_h, :w], 1).copy()
                    elif opr == 27:
                        out_grid[low_h+h+d:high_h+h+d, :w] = np.flip(out_grid[low_h:high_h, :w], 0).copy()
                    else:
                        raise NotImplementedError
                    if for_test:
                        sel = [h*i+d*i, 0, h-1, w-1]
                        selections.append(sel.copy())
                        operations.append(29)
                        sel = [h*(i+1)+d*(i+1), 0, h-1, w-1]
                        selections.append(sel.copy())
                        operations.append(30)
                        if h != 1:
                            selections.append(sel.copy())
                            operations.append(opr)
            else:
                for i, opr in enumerate(action_list):
                    l = len(action_list)
                    low_h = h*(l-i) + d*(l-i)
                    high_h = h*(l-i+1) + d*(l-i)
                    if h == 1:
                        out_grid[low_h-h-d:high_h-h-d, :w] = out_grid[low_h:high_h, :w].copy()
                    elif opr == 24:
                        out_grid[low_h-h-d:high_h-h-d, :w] = np.rot90(out_grid[low_h:high_h, :w]).copy()
                    elif opr == 25:
                        out_grid[low_h-h-d:high_h-h-d, :w] = np.rot90(out_grid[low_h:high_h, :w], 3).copy()
                    elif opr == 26:
                        out_grid[low_h-h-d:high_h-h-d, :w] = np.flip(out_grid[low_h:high_h, :w], 1).copy()
                    elif opr == 27:
                        out_grid[low_h-h-d:high_h-h-d, :w] = np.flip(out_grid[low_h:high_h, :w], 0).copy()
                    else:
                        raise NotImplementedError
                    if for_test:
                        sel = [h*(l-i)+d*(l-i), 0, h-1, w-1]
                        selections.append(sel.copy())
                        operations.append(29)
                        sel = [h*(l-i-1)+d*(l-i-1), 0, h-1, w-1]
                        selections.append(sel.copy())
                        operations.append(30)
                        if h != 1:
                            selections.append(sel.copy())
                            operations.append(opr)
        else:
            if start_left:
                for i, opr in enumerate(action_list):
                    low_w = w*i + d*i
                    high_w = w*(i+1) + d*i
                    if h == 1:
                        out_grid[:h, low_w+w+d:high_w+w+d] = out_grid[:h, low_w:high_w].copy()
                    elif opr == 24:
                        out_grid[:h, low_w+w+d:high_w+w+d] = np.rot90(out_grid[:h, low_w:high_w]).copy()
                    elif opr == 25:
                        out_grid[:h, low_w+w+d:high_w+w+d] = np.rot90(out_grid[:h, low_w:high_w], 3).copy()
                    elif opr == 26:
                        out_grid[:h, low_w+w+d:high_w+w+d] = np.flip(out_grid[:h, low_w:high_w], 1).copy()
                    elif opr == 27:
                        out_grid[:h, low_w+w+d:high_w+w+d] = np.flip(out_grid[:h, low_w:high_w], 0).copy()
                    else:
                        raise NotImplementedError
                    if for_test:
                        sel = [0, w*i+d*i, h-1, w-1]
                        selections.append(sel.copy())
                        operations.append(29)
                        sel = [0, w*(i+1)+d*(i+1), h-1, w-1]
                        selections.append(sel.copy())
                        operations.append(30)
                        if h != 1:
                            selections.append(sel.copy())
                            operations.append(opr)
            else:
                for i, opr in enumerate(action_list):
                    l = len(action_list)
                    low_w = w*(l-i) + d*(l-i)
                    high_w = w*(l-i+1) + d*(l-i)
                    if h == 1:
                        out_grid[:h, low_w-w-d:high_w-w-d] = out_grid[:h, low_w:high_w].copy()
                    elif opr == 24:
                        out_grid[:h, low_w-w-d:high_w-w-d] = np.rot90(out_grid[:h, low_w:high_w]).copy()
                    elif opr == 25:
                        out_grid[:h, low_w-w-d:high_w-w-d] = np.rot90(out_grid[:h, low_w:high_w], 3).copy()
                    elif opr == 26:
                        out_grid[:h, low_w-w-d:high_w-w-d] = np.flip(out_grid[:h, low_w:high_w], 1).copy()
                    elif opr == 27:
                        out_grid[:h, low_w-w-d:high_w-w-d] = np.flip(out_grid[:h, low_w:high_w], 0).copy()
                    else:
                        raise NotImplementedError
                    if for_test:
                        sel = [0, w*(l-i)+d*(l-i), h-1, w-1]
                        selections.append(sel.copy())
                        operations.append(29)
                        sel = [0, w*(l-i-1)+d*(l-i-1), h-1, w-1]
                        selections.append(sel.copy())
                        operations.append(30)
                        if h != 1:
                            selections.append(sel.copy())
                            operations.append(opr)
        if for_test:
            out_h, out_w = in_grid.shape
            selections.append([0, 0, out_h-1, out_w-1])
            operations.append(34)
        return out_grid
