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

def show_grid(samp_hist, prefix, sparsity=None, image_size=224, epsilon=1):
    image_dir = Path('./picture') / prefix
    image_dir.mkdir(parents=True, exist_ok=True)
    grid_trans = image_dir / 'gridline.png'
    print('==> Show the grid in lines')
    fig, axs = plt.subplots(1, len(samp_hist), figsize=(6*len(samp_hist), 6))
    # fig, axs = plt.subplots(figsize=(16, 16))
    for i, grid in enumerate(samp_hist):
        grid_x, grid_y = grid[0,0,:,:], grid[0,1,:,:]
        # prune the grid for better showcasing
        if sparsity:
            delete = [idx for idx in range(0, grid_x.shape[0]) if idx%sparsity!=0 ]
            grid_x = np.delete(grid_x, delete, axis=0)
            grid_y = np.delete(grid_y, delete, axis=0)
            grid_x = np.delete(grid_x, delete, axis=1)
            grid_y = np.delete(grid_y, delete, axis=1)
        seg1 = np.stack((grid_x, grid_y), axis=2)
        seg2 = seg1.transpose(1,0,2)
        axs[i].add_collection(LineCollection(seg1))
        axs[i].add_collection(LineCollection(seg2))
        axs[i].autoscale()
    plt.tight_layout()
    plt.savefig(str(grid_trans))
    plt.close()

def show_final_grid(samp_hist, prefix, sparsity=None, image_size=224, epsilon=1):
    # only show the last one that bind with the image
    image_dir = Path('./picture') / prefix
    image_dir.mkdir(parents=True, exist_ok=True)
    grid_trans = image_dir / 'gridline_final.png'
    print('==> Show the final grid')
    fig, axs = plt.subplots(figsize=(16, 16))
    grid = samp_hist[-1] # last
    grid_x, grid_y = grid[0,0,:,:], grid[0,1,:,:]
    # prune the grid for better showcasing
    if sparsity:
        delete = [idx for idx in range(0, grid_x.shape[0]) if idx%sparsity!=0 ]
        grid_x = np.delete(grid_x, delete, axis=0)
        grid_y = np.delete(grid_y, delete, axis=0)
        grid_x = np.delete(grid_x, delete, axis=1)
        grid_y = np.delete(grid_y, delete, axis=1)
    seg1 = np.stack((grid_x, grid_y), axis=2)
    seg2 = seg1.transpose(1,0,2)
    axs.add_collection(LineCollection(seg1))
    axs.add_collection(LineCollection(seg2))
    axs.set_xlim(-1,1)
    axs.set_ylim(-1,1)
    # axs.autoscale()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(str(grid_trans))
    plt.close()

def show_example(data, adv, prefix, n=4):
    image_dir = Path('./picture') / prefix
    image_dir.mkdir(parents=True, exist_ok=True)
    normal_path = image_dir / 'examples_normal.png'
    adv_path = image_dir / 'examples_adversarial.png'
    # use the torchvision save_image method
    save_image(data[0:n], str(normal_path))
    save_image(adv[0:n], str(adv_path))

def show_one_example(data, adv, prefix):
    # show the same example as to function show_final_grid (first in a batch)
    image_dir = Path('./picture') / prefix
    image_dir.mkdir(parents=True, exist_ok=True)
    normal_path = image_dir / 'picture_normal.png'
    adv_path = image_dir / 'picture_adversarial.png'
    # use matplotlib
    data = data[0].permute(1,2,0).detach().cpu().numpy()
    adv = adv[0].permute(1,2,0).detach().cpu().numpy()
    fig, axs = plt.subplots(figsize=(16, 16))
    axs.imshow(data)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(str(normal_path))
    plt.close()
    fig, axs = plt.subplots(figsize=(16, 16))
    axs.imshow(adv)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(str(adv_path))
    plt.close()

def baseline_0_1(model, loom, data, epsilon, alpha, kernel_size, record=False):
    '''
    Spatial adversarial attack baseline 0.1
    Pure elastic transform
    '''
    grid_hist = [] # the original grid (sampling grid)
    real_hist = [] # the final grid that bind with image
    samp_hist = [] # the sampling grid that bind with image

    data = data.clone().detach().to(device)
    grid = loom.init_prim_grid().detach().to(device)
    # generate random field
    value_range = (2.0/data.shape[-1])*epsilon*alpha
    grid = grid + torch.empty_like(grid).uniform_(-value_range, value_range)
    prim_grid = loom.gaussian_blur(grid, kernel_size)
    # clippping
    prim_grid = loom.prim_clip(prim_grid, rho=epsilon)
    prim_grid = loom.gaussian_blur(prim_grid, kernel_size)
    # =========
    samp_grid = loom.prim_grid_2_samp_grid(prim_grid)
    adv = loom.image_binding(data, samp_grid)
    if record:
        real_hist.append(prim_grid.clone().detach().cpu().numpy())
        samp_hist.append(samp_grid.clone().detach().cpu().numpy())
        show_example(data, adv, 'baseline_0_1')
        show_final_grid(samp_hist, 'baseline_0_1', sparsity=2)
    return adv


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
    samp_hist = [] # the sampling grid that bind with image

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
            samp_grid = loom.prim_grid_2_samp_grid(grid)
            samp_hist.append(samp_grid.clone().detach().cpu().numpy())
            real_hist.append(grid.clone().detach().cpu().numpy())
    # bind it back to image
    samp_grid = loom.prim_grid_2_samp_grid(grid)
    adv = loom.image_binding(data, samp_grid)
    if record:
        # show_trans(grid_hist, real_hist, 'baseline_1_1')
        show_example(data, adv, 'baseline_1_1')
        show_grid(samp_hist, 'baseline_1_1', sparsity=2)
        show_final_grid(samp_hist, 'baseline_2_1', sparsity=2)
    return adv

def baseline_2_1(model, loom, optimizer, data, label, epsilon, kernel_size, step_size, iter, adv_label=None, record=False):
    '''
    Spatial adversarial attack baseline 2.1
        add gaussian kernel for more smooth image outcome
        Inside loom:
        primitive grid -> sampling grid -> image binding
        Outside:
        primitive grid clipping (for attack budget) -> gaussian blur
    '''
    grid_hist = [] # the original grid
    real_hist = [] # the final grid that bind with image
    samp_hist = [] # the sampling grid that bind with image

    data = data.clone().detach().to(device)
    grid = loom.init_prim_grid().detach().to(device)
    label = label.clone().detach().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(iter):
        grid.requires_grad = True
        # loom
        samp_grid = loom.prim_grid_2_samp_grid(grid)
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
        grid = loom.prim_clip(grid, rho=epsilon)
        grid = loom.gaussian_blur(grid, kernel_size)
        # =========
        if record:
            grid_hist.append(grid.clone().detach().cpu().numpy())
            # pass grid through the dataflow to get the real effect of grid
            # only differ while clipping / other operations inside gradient computation
            samp_grid = loom.prim_grid_2_samp_grid(grid)
            samp_hist.append(samp_grid.clone().detach().cpu().numpy())
            real_hist.append(grid.clone().detach().cpu().numpy())
    # bind it back to image
    samp_grid = loom.prim_grid_2_samp_grid(grid)
    adv = loom.image_binding(data, samp_grid)
    if record:
        # show_trans(grid_hist, real_hist, 'baseline_2_1')
        show_example(data, adv, 'baseline_2_1')
        show_grid(samp_hist, 'baseline_2_1', sparsity=2)
        show_final_grid(samp_hist, 'baseline_2_1', sparsity=2)
        show_one_example(data, adv, 'baseline_2_1')
    return adv

def baseline_2_2(model, loom, optimizer, data, label, epsilon, kernel_size, step_size, iter, adv_label=None, record=False):
    '''
    Spatial adversarial attack baseline 2.2
        add gaussian kernel for more smooth image outcome
        Inside loom:
        primitive grid -> gaussian filter -> sampling grid -> image binding
        Outside:
        primitive grid clipping (for attack budget) -> gaussian filter
        test the possibility for setting it as a constrain
    '''
    grid_hist = [] # the original grid
    real_hist = [] # the final grid that bind with image
    samp_hist = [] # the sampling grid that bind with image

    data = data.clone().detach().to(device)
    grid = loom.init_prim_grid().detach().to(device)
    label = label.clone().detach().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(iter):
        grid.requires_grad = True
        # loom
        prim_grid = loom.gaussian_blur(grid, kernel_size)
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
        
        grid = grid + step_size*sign_data_grad
        grid = grid.detach()
        # clippping
        grid = loom.prim_clip(grid, rho=epsilon)
        grid = loom.gaussian_blur(grid, kernel_size)
        # =========
        if record:
            grid_hist.append(grid.clone().detach().cpu().numpy())
            # pass grid through the dataflow to get the real effect of grid
            # only differ while clipping / other operations inside gradient computation
            samp_grid = loom.prim_grid_2_samp_grid(grid)
            samp_hist.append(samp_grid.clone().detach().cpu().numpy())
            real_hist.append(grid.clone().detach().cpu().numpy())
    # bind it back to image
    samp_grid = loom.prim_grid_2_samp_grid(grid)
    adv = loom.image_binding(data, samp_grid)
    if record:
        # show_trans(grid_hist, real_hist, 'baseline_2_1')
        show_example(data, adv, 'baseline_2_2')
        show_grid(samp_hist, 'baseline_2_2', sparsity=2)
        show_final_grid(samp_hist, 'baseline_2_2', sparsity=2)
        show_one_example(data, adv, 'baseline_2_2')
    return adv

def baseline_2_3(model, loom, optimizer, data, label, epsilon, kernel_size, step_size, iter, adv_label=None, record=False):
    '''
    Spatial adversarial attack baseline 2.3
        add gaussian kernel for more smooth image outcome
        Inside loom:
        primitive grid -> gaussian filter -> sampling grid -> image binding
        Outside:
        primitive grid clipping (for attack budget) -> gaussian filter
        see the graph to find out the different with 2.2
    '''
    grid_hist = [] # the original grid
    real_hist = [] # the final grid that bind with image
    samp_hist = [] # the sampling grid that bind with image

    data = data.clone().detach().to(device)
    grid = loom.init_prim_grid().detach().to(device)
    label = label.clone().detach().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(iter):
        grid.requires_grad = True
        # loom
        prim_grid = loom.gaussian_blur(grid, kernel_size)
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
        # here is the difference
        # grid = loom.gaussian_blur(grid, kernel_size)
        grid = prim_grid + step_size*sign_data_grad
        grid = grid.detach()
        # clippping
        grid = loom.prim_clip(grid, rho=epsilon)
        grid = loom.gaussian_blur(grid, kernel_size)
        # =========
        if record:
            grid_hist.append(grid.clone().detach().cpu().numpy())
            # pass grid through the dataflow to get the real effect of grid
            # only differ while clipping / other operations inside gradient computation
            samp_grid = loom.prim_grid_2_samp_grid(grid)
            samp_hist.append(samp_grid.clone().detach().cpu().numpy())
            real_hist.append(grid.clone().detach().cpu().numpy())
    # bind it back to image
    samp_grid = loom.prim_grid_2_samp_grid(grid)
    adv = loom.image_binding(data, samp_grid)
    if record:
        # show_trans(grid_hist, real_hist, 'baseline_2_1')
        show_example(data, adv, 'baseline_2_3')
        show_grid(samp_hist, 'baseline_2_3', sparsity=2)
        show_final_grid(samp_hist, 'baseline_2_3', sparsity=2)
        show_one_example(data, adv, 'baseline_2_3')
    return adv

def baseline_2_4(model, loom, optimizer, data, label, epsilon, kernel_size, step_size, iter, adv_label=None, record=False):
    '''
    Spatial adversarial attack baseline 2.4
        add gaussian kernel for more smooth image outcome
        Inside loom:
        primitive grid -> gaussian filter -> sampling grid -> image binding
        Outside:
        primitive grid clipping (for attack budget) -> gaussian filter (last iter)
    '''
    grid_hist = [] # the original grid
    real_hist = [] # the final grid that bind with image
    samp_hist = [] # the sampling grid that bind with image

    data = data.clone().detach().to(device)
    grid = loom.init_prim_grid().detach().to(device)
    label = label.clone().detach().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(iter):
        grid.requires_grad = True
        # loom
        prim_grid = loom.gaussian_blur(grid, kernel_size)
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

        grid = grid + step_size*sign_data_grad
        grid = grid.detach()
        # clippping
        grid = loom.prim_clip(grid, rho=epsilon)
        # =========
        if record:
            grid_hist.append(grid.clone().detach().cpu().numpy())
            # pass grid through the dataflow to get the real effect of grid
            # only differ while clipping / other operations inside gradient computation
            prim_grid = loom.gaussian_blur(grid, kernel_size)
            samp_grid = loom.prim_grid_2_samp_grid(prim_grid)
            samp_hist.append(samp_grid.clone().detach().cpu().numpy())
            real_hist.append(prim_grid.clone().detach().cpu().numpy())
    # bind it back to image
    grid = loom.gaussian_blur(grid, kernel_size)
    samp_grid = loom.prim_grid_2_samp_grid(grid)
    adv = loom.image_binding(data, samp_grid)
    if record:
        # show_trans(grid_hist, real_hist, 'baseline_2_1')
        show_example(data, adv, 'baseline_2_4')
        show_grid(samp_hist, 'baseline_2_4', sparsity=2)
        show_final_grid(samp_hist, 'baseline_2_4', sparsity=2)
        show_one_example(data, adv, 'baseline_2_4')
    return adv

def baseline_2_5(model, loom, optimizer, data, label, epsilon, kernel_size, step_size, iter, adv_label=None, record=False):
    '''
    Spatial adversarial attack baseline 2.5
        add gaussian kernel for more smooth image outcome
        Inside loom:
        primitive grid -> gaussian filter -> sampling grid -> image binding
        Outside:
        primitive grid clipping (last iter) -> gaussian filter (last iter)
    '''
    grid_hist = [] # the original grid
    real_hist = [] # the final grid that bind with image
    samp_hist = [] # the sampling grid that bind with image

    data = data.clone().detach().to(device)
    grid = loom.init_prim_grid().detach().to(device)
    label = label.clone().detach().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(iter):
        grid.requires_grad = True
        # loom
        prim_grid = loom.gaussian_blur(grid, kernel_size)
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

        grid = grid + step_size*sign_data_grad
        grid = grid.detach()
        # clippping
        # grid = loom.prim_clip(grid, rho=epsilon)
        # =========
        if record:
            grid_hist.append(grid.clone().detach().cpu().numpy())
            # pass grid through the dataflow to get the real effect of grid
            # only differ while clipping / other operations inside gradient computation
            prim_grid = loom.prim_clip(grid, rho=epsilon)
            prim_grid = loom.gaussian_blur(prim_grid, kernel_size)
            samp_grid = loom.prim_grid_2_samp_grid(prim_grid)
            samp_hist.append(samp_grid.clone().detach().cpu().numpy())
            real_hist.append(prim_grid.clone().detach().cpu().numpy())
    # bind it back to image
    grid = loom.prim_clip(grid, rho=epsilon)
    grid = loom.gaussian_blur(grid, kernel_size)
    samp_grid = loom.prim_grid_2_samp_grid(grid)
    adv = loom.image_binding(data, samp_grid)
    if record:
        # show_trans(grid_hist, real_hist, 'baseline_2_1')
        show_example(data, adv, 'baseline_2_5')
        show_grid(samp_hist, 'baseline_2_5', sparsity=2)
        show_final_grid(samp_hist, 'baseline_2_5', sparsity=2)
        show_one_example(data, adv, 'baseline_2_5')
    return adv

def baseline_2_6(model, loom, optimizer, data, label, epsilon, kernel_size, step_size, iter, adv_label=None, record=False):
    '''
    Spatial adversarial attack baseline 2.6
        add gaussian kernel for more smooth image outcome
        Inside loom:
        primitive grid -> primitive grid clipping -> gaussian filter -> sampling grid -> image binding
        Outside:
        primitive grid clipping (last iter) -> gaussian filter (last iter)
    '''
    grid_hist = [] # the original grid
    real_hist = [] # the final grid that bind with image
    samp_hist = [] # the sampling grid that bind with image

    data = data.clone().detach().to(device)
    grid = loom.init_prim_grid().detach().to(device)
    label = label.clone().detach().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(iter):
        grid.requires_grad = True
        # loom
        prim_grid = loom.prim_clip(grid, rho=epsilon)
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

        grid = grid + step_size*sign_data_grad
        grid = grid.detach()
        # clippping
        # grid = loom.prim_clip(grid, rho=epsilon)
        # =========
        if record:
            grid_hist.append(grid.clone().detach().cpu().numpy())
            # pass grid through the dataflow to get the real effect of grid
            # only differ while clipping / other operations inside gradient computation
            prim_grid = loom.prim_clip(grid, rho=epsilon)
            prim_grid = loom.gaussian_blur(grid, kernel_size)
            samp_grid = loom.prim_grid_2_samp_grid(prim_grid)
            samp_hist.append(samp_grid.clone().detach().cpu().numpy())
            real_hist.append(prim_grid.clone().detach().cpu().numpy())
    # bind it back to image
    grid = loom.prim_clip(grid, rho=epsilon)
    grid = loom.gaussian_blur(grid, kernel_size)
    samp_grid = loom.prim_grid_2_samp_grid(grid)
    adv = loom.image_binding(data, samp_grid)
    if record:
        # show_trans(grid_hist, real_hist, 'baseline_2_1')
        show_example(data, adv, 'baseline_2_6')
        show_grid(samp_hist, 'baseline_2_6', sparsity=2)
        show_final_grid(samp_hist, 'baseline_2_6', sparsity=2)
        show_one_example(data, adv, 'baseline_2_6')
    return adv

def baseline_2_7(model, loom, optimizer, data, label, epsilon, kernel_size, step_size, iter, adv_label=None, record=False):
    '''
    Spatial adversarial attack baseline 2.7
        old 2.2 with wrong smooth constrain
    '''
    grid_hist = [] # the original grid
    real_hist = [] # the final grid that bind with image
    samp_hist = [] # the sampling grid that bind with image

    data = data.clone().detach().to(device)
    grid = loom.init_prim_grid().detach().to(device)
    label = label.clone().detach().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(iter):
        grid.requires_grad = True
        # loom
        prim_grid = loom.gaussian_blur(grid, kernel_size)
        prim_grid = loom.prim_clip(prim_grid, rho=epsilon)
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
        
        grid = grid + step_size*sign_data_grad
        grid = grid.detach()
        # clippping
        # grid = loom.prim_clip(grid, rho=epsilon)
        # grid = loom.gaussian_blur(grid, kernel_size)
        # =========
        if record:
            grid_hist.append(grid.clone().detach().cpu().numpy())
            # pass grid through the dataflow to get the real effect of grid
            # only differ while clipping / other operations inside gradient computation
            samp_grid = loom.prim_grid_2_samp_grid(grid)
            samp_hist.append(samp_grid.clone().detach().cpu().numpy())
            real_hist.append(grid.clone().detach().cpu().numpy())
    # bind it back to image
    prim_grid = loom.gaussian_blur(grid, kernel_size)
    prim_grid = loom.prim_clip(prim_grid, rho=epsilon)
    samp_grid = loom.prim_grid_2_samp_grid(prim_grid)
    adv = loom.image_binding(data, samp_grid)
    if record:
        # show_trans(grid_hist, real_hist, 'baseline_2_1')
        show_example(data, adv, 'baseline_2_7')
        show_grid(samp_hist, 'baseline_2_7', sparsity=2)
        show_final_grid(samp_hist, 'baseline_2_7', sparsity=2)
        show_one_example(data, adv, 'baseline_2_7')
    return adv

def baseline_2_8(model, loom, optimizer, data, label, epsilon, kernel_size, step_size, iter, adv_label=None, record=False):
    '''
    Spatial adversarial attack baseline 2.8
        Inside loom:
        primitive grid -> primitive grid clipping -> gaussian filter -> sampling grid -> image binding
        Outside:
        primitive grid clipping -> gaussian filter
    '''
    grid_hist = [] # the original grid
    real_hist = [] # the final grid that bind with image
    samp_hist = [] # the sampling grid that bind with image

    data = data.clone().detach().to(device)
    grid = loom.init_prim_grid().detach().to(device)
    label = label.clone().detach().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(iter + 1):
        grid.requires_grad = True
        # loom
        prim_grid = loom.prim_clip(grid, rho=epsilon)
        prim_grid = loom.gaussian_blur(prim_grid, kernel_size)
        samp_grid = loom.prim_grid_2_samp_grid(prim_grid)
        if i == 0:
            data_rolled = torch.roll(data, shifts=-1, dims=0)
            adv = loom.image_binding(data_rolled, samp_grid)
        else:
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

        grid = grid + step_size*sign_data_grad
        grid = grid.detach()
        # clippping
        # grid = loom.prim_clip(grid, rho=epsilon)
        # grid = loom.gaussian_blur(grid, kernel_size)
        # =========
        if record:
            grid_hist.append(grid.clone().detach().cpu().numpy())
            # pass grid through the dataflow to get the real effect of grid
            # only differ while clipping / other operations inside gradient computation
            prim_grid = loom.prim_clip(grid, rho=epsilon)
            prim_grid = loom.gaussian_blur(grid, kernel_size)
            samp_grid = loom.prim_grid_2_samp_grid(prim_grid)
            samp_hist.append(samp_grid.clone().detach().cpu().numpy())
            real_hist.append(prim_grid.clone().detach().cpu().numpy())
    # bind it back to image
    grid = loom.prim_clip(grid, rho=epsilon)
    grid = loom.gaussian_blur(grid, kernel_size)
    samp_grid = loom.prim_grid_2_samp_grid(grid)
    adv = loom.image_binding(data, samp_grid)
    if record:
        # show_trans(grid_hist, real_hist, 'baseline_2_1')
        show_example(data, adv, 'baseline_2_8')
        show_grid(samp_hist, 'baseline_2_8', sparsity=2)
        show_final_grid(samp_hist, 'baseline_2_8', sparsity=2)
        show_one_example(data, adv, 'baseline_2_8')
    return adv

