import torch.nn.functional as F
from torch import nn
import torch
from torch.signal import windows

class DoyPositionEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.position_encodings = nn.Linear(365, embedding_dim)

    def forward(self, days_of_year):
        days = F.one_hot(days_of_year, num_classes=365).to(torch.float32)
        return self.position_encodings(days)


class TilesDoyPositionEncoder(nn.Module):
    def __init__(self, embedding_dim: int, time_length_in_each_tile: tuple):
        super().__init__()
        self.time_length_in_each_tile = time_length_in_each_tile
        self.position_encodings = nn.ModuleList([nn.Linear(365, embedding_dim) for _ in range(4)])

    def forward(self, days_of_year):
        """
        Args:
            days_of_year: b t
        Returns:
           position encoding: b t d
        """
        time_length = days_of_year.size(dim=1)
        tile_id = self.time_length_in_each_tile.index(time_length)
        days = F.one_hot(days_of_year, num_classes=365).to(torch.float32)
        return self.position_encodings[tile_id](days)


class GaussianTilesDoyPositionEncoder(nn.Module):
    def __init__(self, embedding_dim: int, time_length_in_each_tile: tuple):
        super().__init__()
        self.time_length_in_each_tile = time_length_in_each_tile
        self.position_encodings = nn.ModuleList([nn.Linear(365, embedding_dim) for _ in range(4)])
        self.gaussian_window = windows.gaussian(365)

    def forward(self, days_of_year):
        device = next(self.parameters()).device
        b, t = days_of_year.size()
        gaussian_encodings = torch.zeros((b, t, 365)).to(device)
        for i in range(t):
            doy = days_of_year[0, i].item()
            gaussian_encodings[:, i, :] = self.gaussian_window.roll(shifts=doy-182, dims=0)

        tile_id = self.time_length_in_each_tile.index(t)
        return self.position_encodings[tile_id](gaussian_encodings)
