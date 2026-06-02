import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import colors
import shutil
import ffmpeg
import os
import utils
import numpy as np
import math
cmap = colors.ListedColormap(
    [
        '#000000',
        '#0074D9',
        '#FF4136',
        '#2ECC40',
        '#FFDC00',
        '#AAAAAA',
        '#F012BE',
        '#FF851B',
        '#7FDBFF',
        '#870C25',
        '#FFFFFF',
    ])
norm = colors.Normalize(vmin=0, vmax=10)
def plot_one(mode, is_ex, data, ax, i, seg_id):
    if is_ex:
        input_matrix = data
        if i == 0:
            ax.set_title(f'demonstration input {is_ex}', fontsize=20)
        else:
            ax.set_title(f'demonstration output {is_ex}', fontsize=20)
    else:
        if mode == "inout":
            if i == 0:
                input_matrix = data['in_grid']
                ax.set_title('input')
            else:
                input_matrix = data['out_grid']
                ax.set_title('output')
        else:
            input_matrix = data['grid'][i]
            operation_num = data['operation'][i]
            operation_name = utils.mapping_operation(operation_num)
            if i == 0 and seg_id == 0:
                ax.set_title(f"test input", fontsize=20)
            else:
                ax.set_title(f"step {seg_id+i}",fontsize=20)
            ax.text(0.5, -0.05, f'{operation_num}  {operation_name}', ha='center', transform=ax.transAxes, fontsize=20)
    local_norm = colors.Normalize(vmin=0, vmax=10)
    ax.pcolormesh(np.flip(input_matrix, 0), cmap=cmap, norm=local_norm, edgecolors='lightgrey', linewidth=0.05)
    ax.set_aspect('equal')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
def save_for_gif(is_ex, data, i, task_id, trace_id, save_folder_path):
    input_matrix = data['grid'][i]
    operation_num = data['operation'][i]
    operation_name = utils.mapping_operation(operation_num)
    plt.figure(figsize=(5, 5))
    if i == 0:
        plt.suptitle(f"{task_id}_{trace_id}\ninput", fontsize=30)
    else:
        plt.suptitle(f"{task_id}_{trace_id}\nstep{i}", fontsize=30)
    plt.text(0.5, -0.1, f'{operation_num}  {operation_name}', ha='center', transform=plt.gca().transAxes,  fontsize=20)
    plt.pcolormesh(np.flip(input_matrix, 0), cmap=cmap, norm=norm, edgecolors='lightgrey', linewidth=0.01)
    plt.gca().set_aspect('equal')
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    plt.tight_layout()
    if is_ex:
        if i == 0:
            file_name = f"ex_{is_ex}_in"
        else:
            file_name = f"ex_{is_ex}_out"
    else:
        file_name = f"trace_{i}"
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    if not os.path.exists(f"{save_folder_path}/{task_id}"):
        os.makedirs(f"{save_folder_path}/{task_id}")
    if not os.path.exists(f"{save_folder_path}/{task_id}/gif"):
        os.makedirs(f"{save_folder_path}/{task_id}/gif")
    if not os.path.exists(f"{save_folder_path}/{task_id}/gif/pngs_{task_id}_{trace_id}"):
        os.makedirs(f"{save_folder_path}/{task_id}/gif/pngs_{task_id}_{trace_id}")
    plt.savefig(f"{save_folder_path}/{task_id}/gif/pngs_{task_id}_{trace_id}/{file_name}.png", bbox_inches='tight', dpi=600)
def plot_task(mode, data, task_id, trace_id, save_folder_path, make_task_folder=False):
    num_step = len(data['step'])
    num_examples = len(data['ex_in'])
    exi = []
    exo = []
    axs = []
    seg_id = 0
    if mode == "gif":
        if os.path.exists(f"{save_folder_path}/{task_id}/gif/pngs_{task_id}_{trace_id}"):
            shutil.rmtree(f"{save_folder_path}/{task_id}/gif/pngs_{task_id}_{trace_id}")
        for i in range(num_step):
            save_for_gif(0, data, i, task_id, trace_id, save_folder_path)
    elif mode == "inout":
        fig = plt.figure(figsize=(10, 5*(num_examples+1)))
        gs = GridSpec(nrows=num_examples+1, ncols=2)
        is_ex = 0
        for h in range(num_examples):
            is_ex = h+1
            exi.append(fig.add_subplot(gs[h, 0]))
            plot_one(mode, is_ex, data['ex_in'][h], exi[h], 0, seg_id)
            exo.append(fig.add_subplot(gs[h, 1]))
            plot_one(mode, is_ex, data['ex_out'][h], exo[h], 1, seg_id)
        is_ex = 0
        ax_in = fig.add_subplot(gs[num_examples, 0])
        ax_out = fig.add_subplot(gs[num_examples, 1])
        plot_one(mode, is_ex, data, ax_in, 0, seg_id)
        plot_one(mode, is_ex, data, ax_out, 1, seg_id)
    else:
        step_rows = math.ceil(num_step / 5)
        step_cols = min(num_step, 5)
        fig = plt.figure(figsize=(5*5 , 5*(num_examples+step_rows)))
        gs = GridSpec(nrows=num_examples+step_rows, ncols=5)
        is_ex = 0
        if mode == "segment":
            seg_id = int(data['desc']['id'].split('.')[0].split('_')[-1])
        for h in range(num_examples):
            is_ex = h+1
            exi.append(fig.add_subplot(gs[h, 0]))
            plot_one(mode, is_ex, data['ex_in'][h], exi[h], 0, seg_id)
            exo.append(fig.add_subplot(gs[h, 1]))
            plot_one(mode, is_ex, data['ex_out'][h], exo[h], 1, seg_id)
        is_ex = 0
        for i in range(num_step):
            row = num_examples + (i // 5)
            col = i % 5
            axs.append(fig.add_subplot(gs[row, col]))
            plot_one(mode, is_ex, data, axs[i], i, seg_id)
    if mode != "gif":
        if 'expert' in data['desc']['id']:
            title = data['desc']['id'].replace('expert', 'gold-standard')
        else:
            title = data['desc']['id']
        fig.suptitle(f"{title}, {mode}\n",fontsize=20)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        if make_task_folder == "true":
            if not os.path.exists(f"{save_folder_path}/{task_id}"):
                os.makedirs(f"{save_folder_path}/{task_id}")
            if not os.path.exists(f"{save_folder_path}/{task_id}/{mode}"):
                os.makedirs(f"{save_folder_path}/{task_id}/{mode}")
            if mode == 'segment':
                if not os.path.exists(f"{save_folder_path}/{task_id}/{mode}/{task_id}_{trace_id}"):
                    os.makedirs(f"{save_folder_path}/{task_id}/{mode}/{task_id}_{trace_id}")
                plt.tight_layout()
                plt.savefig(f"{save_folder_path}/{task_id}/{mode}/{task_id}_{trace_id}/{data['desc']['id']}.png", dpi=600)
            else:
                plt.tight_layout()
                plt.savefig(f"{save_folder_path}/{task_id}/{mode}/{data['desc']['id']}.png", dpi=600)
        else:
            plt.tight_layout()
            plt.savefig(f"{save_folder_path}/{data['desc']['id']}.png", dpi=600)
    plt.close()
def make_gif(png_folder_path, output_filename):
    (
        ffmpeg
        .input(f'{png_folder_path}/trace_%d.png', framerate=4)
        .output(output_filename, filter_complex='[0:v] setsar=1/1,fps=4,scale=w=1024:h=-1,split [a][b];[a] palettegen=stats_mode=full [p];[b][p] paletteuse=new=1')
        .run()
    )
def find_data(file_path, path_list=None, json_list=None):
    if path_list == None:
        path_list = []
    if json_list == None:
        json_list = []
    if isinstance(file_path, list):
        for path in file_path:
            pl, jl = find_data(path)
            path_list.extend(pl)
            json_list.extend(jl)
    elif os.path.isdir(file_path):
        for file in os.listdir(file_path):
            if file.split('.')[-1] == 'json':
                path_list.append(os.path.join(file_path, file))
                json_list.append(file)
            else:
                pls, jls = find_data(os.path.join(file_path, file))
                path_list.extend(pls)
                json_list.extend(jls)
    elif file_path.split('.')[-1] == 'json':
        path_list.append(file_path)
        json_list.append(file_path.split('/')[-1])
    return path_list, json_list
