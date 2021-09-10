import torch
import torch.nn as nn
import torch.nn.functional as F

# Loom
class Loom(nn.Module):
    def __init__(self, args):
        super(Loom, self).__init__()
        self.args = args
        # some constant
        self.identity_mean, self.identity_offset = 0.5, 4.0
        self.deforming_offset = (2.0 + (self.identity_offset/(args.image_size-1))) / 2.0
        # filter for converting accumulated grid and sampling grid
        # accu_filter_x = torch.ones((1, 1, 1, args.image_size), requires_grad=False)
        # accu_filter_y = torch.ones((1, 1, args.image_size, 1), requires_grad=False)
        # dvia_filter_x = torch.cat((torch.full((1, 1, 1, 1), 1.), torch.full((1, 1, 1, 1), -1.)), 3)
        # dvia_filter_y = torch.cat((torch.full((1, 1, 1, 1), 1.), torch.full((1, 1, 1, 1), -1.)), 2)
        # self.register_buffer('accu_filter_x', accu_filter_x)
        # self.register_buffer('accu_filter_y', accu_filter_y)
        # self.register_buffer('dvia_filter_x', dvia_filter_x)
        # self.register_buffer('dvia_filter_y', dvia_filter_y)
        # identity sampling gird for converting primitive gird and sampling grid
        samp_iden = self.init_samp_grid()
        self.register_buffer('samp_iden', samp_iden)
        # gaussian kernel
        gaussian_kernel2d = self.get_gaussian_kernel2d(args.kernel_size)
        self.register_buffer('gaussian_kernel2d', gaussian_kernel2d)

    # all kinds of grid initialization
    def init_prim_grid(self):
        # output shape (N, 2. H, W)
        prim_grid = torch.zeros(self.args.batch_size, 2, self.args.image_size, self.args.image_size)
        return prim_grid
    def init_samp_grid(self):
        # output shape (N, 2. H, W)
        sequence = torch.arange(-(self.args.image_size-1), (self.args.image_size), 2)/(self.args.image_size-1.0)
        samp_grid_x = sequence.repeat(self.args.image_size,1)
        samp_grid_y = samp_grid_x.t()
        samp_grid = torch.cat((samp_grid_x.unsqueeze(0), samp_grid_y.unsqueeze(0)), 0)
        samp_grid = samp_grid.unsqueeze(0).repeat(self.args.batch_size, 1, 1, 1)
        return samp_grid

    # gaussian blur related function
    def get_gaussian_kernel2d(self, kernel_size, sigma=None):
        if sigma == None:
            sigma = kernel_size * 0.15 + 0.35
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()
        kernel2d = torch.mm(kernel1d[:, None], kernel1d[None, :])
        return kernel2d
    def gaussian_blur(self, prim_grid, kernel_size, sigma=None):
        # kernel_size should have odd and positive integers
        kernel = self.gaussian_kernel2d
        if kernel_size != self.args.kernel_size:
            kernel = self.get_gaussian_kernel2d(kernel_size)
        kernel = kernel.expand(prim_grid.shape[-3], 1, kernel.shape[0], kernel.shape[1])
        # padding = (left, right, top, bottom)
        padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]
        prim_grid = F.pad(prim_grid, padding, mode="reflect")
        prim_grid = F.conv2d(prim_grid, kernel, groups=prim_grid.shape[-3])
        return prim_grid

    # grid conversion function
    def prim_grid_2_samp_grid(self, prim_grid):
        samp_grid = prim_grid + self.samp_iden
        return samp_grid
    def samp_grid_2_prim_grid(self, samp_grid):
        prim_grid = samp_grid - self.samp_iden
        return prim_grid

    # clipping function
    # primitive gird clipping to enforce attack budget or control the loss surface 
    def prim_clip(self, prim_grid, rho=1.0):
        pixel_width = 2.0/(self.args.image_size)
        prim_grid = torch.clamp(prim_grid, -pixel_width*rho, pixel_width*rho)
        return prim_grid
    # sampling grid clipping to enforce feasible image(spatial domain)
    def samp_clip(self, samp_grid, eta=1.0):
        return F.hardtanh(samp_grid, min_val=-eta, max_val=eta)
    
    # forwarding
    def forward_grid(self, prim_grid, gaussian=True, rho=1, eta=1.0):
        if gaussian:
            prim_grid = self.gaussian_blur(prim_grid, self.args.kernel_size)
        samp_grid = self.prim_grid_2_samp_grid(prim_grid)
        samp_grid = self.samp_clip(samp_grid, eta)
        return samp_grid
    def image_binding(self, image, samp_grid):
        # binding grid with image using sampling grid
        binding_grid = samp_grid.permute(0,2,3,1)
        distort_image = F.grid_sample(image, binding_grid, align_corners=True)
        return distort_image
    def forward(self, image, prim_grid, gaussian=True, rho=1, eta=1.0):
        samp_grid = self.forward_grid(prim_grid, gaussian, rho, eta)
        distort_image = self.image_binding(image, samp_grid)
        return distort_image