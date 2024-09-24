import torch
import torch.nn as nn
import torch.nn.functional as F

class EnvironmentLight(nn.Module):
    def __init__(self, num_lobes=32, device='cpu'):
        super(EnvironmentLight, self).__init__()
        self.num_lobes = num_lobes
        self.mu = nn.Parameter(torch.randn(num_lobes, 3, device=device))
        self.c = nn.Parameter(torch.ones(num_lobes, 3, device=device))
        self.sigma = nn.Parameter(torch.ones(num_lobes, device=device) * 0.5)

    def forward(self, directions):
        mu_normalized = F.normalize(self.mu, dim=-1)
        dot_product = torch.matmul(directions, mu_normalized.transpose(0, 1))
        gaussian = torch.exp(-2 * (1 - dot_product) / (self.sigma ** 2).unsqueeze(0))
        return torch.matmul(gaussian, self.c)

    def get_light_directions(self):
        return F.normalize(self.mu, dim=-1)

    def get_aggregated_light(self):
        directions = self.get_light_directions()
        colors = self(directions)
        aggregated_direction = directions.mean(dim=0, keepdim=True)
        aggregated_color = colors.mean(dim=0, keepdim=True)
        return aggregated_direction, aggregated_color


class ToneMapping(nn.Module):
    def __init__(self, num_bins=5):
        super(ToneMapping, self).__init__()
        self.num_bins = num_bins
        self.widths = nn.Parameter(torch.ones(num_bins) / num_bins)
        self.heights = nn.Parameter(torch.linspace(0, 1, num_bins + 1))
        self.slopes = nn.Parameter(torch.ones(num_bins + 1))

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        bin_idx = torch.searchsorted(torch.cumsum(self.widths, 0), x)
        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)
        
        x_low = torch.gather(torch.cat([torch.zeros(1), torch.cumsum(self.widths, 0)[:-1]]), 0, bin_idx)
        x_high = x_low + torch.gather(self.widths, 0, bin_idx)
        y_low = torch.gather(self.heights[:-1], 0, bin_idx)
        y_high = torch.gather(self.heights[1:], 0, bin_idx)
        slope_low = torch.gather(self.slopes[:-1], 0, bin_idx)
        slope_high = torch.gather(self.slopes[1:], 0, bin_idx)
        
        t = (x - x_low) / (x_high - x_low)
        y = y_low + (y_high - y_low) * (slope_low * t ** 2 + 2 * t * (1 - t)) / ((slope_low + slope_high) * t ** 2 + 2 * (slope_low + slope_high - 2) * t + 2)
        return y