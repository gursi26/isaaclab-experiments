import torch

def create_grid(N, spacing=1.0):
    offset = (N - 1) * spacing / 2
    x = torch.linspace(-offset, offset, N)
    y = torch.linspace(-offset, offset, N)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack([xx.flatten(), yy.flatten(), torch.zeros(N**2)], dim=1)
    return grid

