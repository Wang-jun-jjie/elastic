from pathlib import Path
# pytorch
import torch
import torch.nn as nn
from torchvision.utils import save_image
# NVIDIA apex
from apex import amp
# math and showcase
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from lib.utils import input_normalize
from lib.loom import Loom

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_trans(grid_hist, real_hist, prefix, image_size=224, epsilon=1):
    image_dir = Path('./picture') / prefix
    image_dir.mkdir(parents=True, exist_ok=True)
    grid_trans = image_dir / 'grids.png'
    if len(grid_hist) < 1:
        print('Iteration smaller than 1, cannot show the grid transistion')
        return
    print('==> Show the grid transistion')
    value_range = (2.0/image_size)*epsilon
    fig, axs = plt.subplots(4, len(grid_hist), figsize=(6*len(grid_hist)+1, 24))
    for i, grid in enumerate(grid_hist):
        axs[0, i].imshow(grid[0,0,:,:], cmap='rainbow', vmin=-value_range, vmax=value_range)
        axs[1, i].imshow(grid[0,1,:,:], cmap='rainbow', vmin=-value_range, vmax=value_range)
    for i, grid in enumerate(real_hist):
        axs[2, i].imshow(grid[0,0,:,:], cmap='rainbow', vmin=-value_range, vmax=value_range)
        axs[3, i].imshow(grid[0,1,:,:], cmap='rainbow', vmin=-value_range, vmax=value_range)
    plt.savefig(str(grid_trans))
    plt.close()

def show_grid(samp_hist, prefix, image_size=224, epsilon=1):
    image_dir = Path('./picture') / prefix
    image_dir.mkdir(parents=True, exist_ok=True)
    grid_trans = image_dir / 'gridline.png'
    print('==> Show the grid in lines')
    fig, axs = plt.subplots(figsize=(64, 64))
    last = samp_hist.detach().cpu().numpy()
    grid_x, grid_y = last[0,0,:,:], last[0,1,:,:]
    seg1 = np.stack((grid_x, grid_y), axis=2)
    seg2 = seg1.transpose(1,0,2)
    axs.add_collection(LineCollection(seg1))
    axs.add_collection(LineCollection(seg2))
    axs.autoscale()
    plt.savefig(str(grid_trans))
    plt.close()

def show_example(data, adv, prefix):
    image_dir = Path('./picture') / prefix
    image_dir.mkdir(parents=True, exist_ok=True)
    normal_path = image_dir / 'example_normal.png'
    adv_path = image_dir / 'example_adversarial.png'
    # use the torchvision save_image method
    save_image(data[0:4], str(normal_path))
    save_image(adv[0:4], str(adv_path))

# adversarial attack
def baseline_1_1(model, loom, optimizer, data, label, epsilon, step_size, iter, adv_label=None, record=False):
    '''
    Spatial adversarial attack baseline 1.1
        Inside loom:
        primitive grid -> sampling grid -> image binding
        Outside:
        primitive grid clipping (for attack budget)
    '''
    grid_hist = [] # the original grid (sampling grid)
    real_hist = [] # the final grid that bind with image

    data = data.clone().detach().to(device)
    grid = loom.init_prim_grid().detach().to(device)
    label = label.clone().detach().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(iter):
        grid.requires_grad = True
        # loom
        prim_grid = grid
        samp_grid = loom.prim_grid_2_samp_grid(prim_grid)
        adv = loom.image_binding(data, samp_grid)
        # ==========
        output = model(input_normalize(adv))
        if adv_label != None:
            loss = -1*criterion(output, adv_label)
        else:
            loss = criterion(output, label)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        sign_data_grad = grid.grad.sign()
        grad = torch.sum(torch.abs(grid.grad)[0].type(torch.float64))
        grid = grid + step_size*sign_data_grad
        grid = grid.detach()
        # clippping
        prim_grid = grid 
        prim_grid = loom.prim_clip(prim_grid, rho=epsilon)
        grid = prim_grid
        # =========
        if record:
            grid_hist.append(grid.clone().detach().cpu().numpy())
            # pass grid through the dataflow to get the real effect of grid
            # only differ while clipping / other operations inside gradient computation
            real_hist.append(grid.clone().detach().cpu().numpy())
    # bind it back to image
    samp_grid = loom.prim_grid_2_samp_grid(grid)
    adv = loom.image_binding(data, samp_grid)
    if record:
        # show_trans(grid_hist, real_hist, 'baseline_1_1')
        show_example(data, adv, 'baseline_1_1')
    return adv

def baseline_2_1(model, loom, optimizer, data, label, epsilon, kernel_size, step_size, iter, adv_label=None, record=False):
    '''
    Spatial adversarial attack baseline 2.1
        add gaussian kernel for more smooth image outcome
        Inside loom:
        primitive grid -> gaussian blur -> sampling grid -> image binding
        Outside:
        primitive grid clipping (for attack budget)
    '''
    grid_hist = [] # the original grid
    real_hist = [] # the final grid that bind with image

    data = data.clone().detach().to(device)
    grid = loom.init_prim_grid().detach().to(device)
    label = label.clone().detach().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(iter):
        grid.requires_grad = True
        # loom
        prim_grid = grid
        prim_grid = loom.gaussian_blur(prim_grid, kernel_size)
        samp_grid = loom.prim_grid_2_samp_grid(prim_grid)
        adv = loom.image_binding(data, samp_grid)
        # ==========
        output = model(input_normalize(adv))
        if adv_label != None:
            loss = -1*criterion(output, adv_label)
        else:
            loss = criterion(output, label)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        sign_data_grad = grid.grad.sign()
        grad = torch.sum(torch.abs(grid.grad)[0].type(torch.float64))
        grid = grid + step_size*sign_data_grad
        grid = grid.detach()
        # clippping
        prim_grid = grid 
        prim_grid = loom.prim_clip(prim_grid, rho=epsilon)
        grid = prim_grid
        # =========
        if record:
            grid_hist.append(grid.clone().detach().cpu().numpy())
            # pass grid through the dataflow to get the real effect of grid
            # only differ while clipping / other operations inside gradient computation
            prim_grid = loom.gaussian_blur(grid, kernel_size)
            real_hist.append(prim_grid.clone().detach().cpu().numpy())
    # bind it back to image
    prim_grid = loom.gaussian_blur(grid, kernel_size)
    samp_grid = loom.prim_grid_2_samp_grid(prim_grid)
    adv = loom.image_binding(data, samp_grid)
    if record:
        # show_trans(grid_hist, real_hist, 'baseline_2_1')
        show_example(data, adv, 'baseline_2_1')
    return adv
