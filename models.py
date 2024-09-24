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
    def __init__(self, num_bins=5, device='cpu'):
        super(ToneMapping, self).__init__()
        self.num_bins = num_bins
        self.widths = nn.Parameter(torch.ones(num_bins, device=device) / num_bins)
        self.heights = nn.Parameter(torch.linspace(0, 1, num_bins + 1, device=device))
        self.slopes = nn.Parameter(torch.ones(num_bins + 1, device=device))

    def forward(self, x):
        original_shape = x.shape
        x = x.flatten()
        x = torch.clamp(x, 0, 1)
        device = x.device

        bin_edges = torch.cumsum(self.widths, 0)
        bin_idx = torch.searchsorted(bin_edges, x)
        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)
        
        widths_cumsum = torch.cat([torch.zeros(1, device=device), bin_edges[:-1]])
        x_low = widths_cumsum[bin_idx]
        x_high = (widths_cumsum + self.widths)[bin_idx]
        y_low = self.heights[:-1][bin_idx]
        y_high = self.heights[1:][bin_idx]
        slope_low = self.slopes[:-1][bin_idx]
        slope_high = self.slopes[1:][bin_idx]
        
        t = (x - x_low) / (x_high - x_low + 1e-8)
        numerator = slope_low * t ** 2 + 2 * t * (1 - t)
        denominator = (slope_low + slope_high) * t ** 2 + 2 * (slope_low + slope_high - 2) * t + 2
        y = y_low + (y_high - y_low) * numerator / denominator
        
        return y.reshape(original_shape)