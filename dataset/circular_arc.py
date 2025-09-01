import math
import torch
from torch.utils.data import Dataset, DataLoader

class CircularArcDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        dt: float = 0.1,
        radius_range: tuple = (1.0, 1.0),
        omega_range: tuple = (1.0, 1.0),
        center_range: tuple = (-2., 2.)
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.dt = dt
        self.radius_range = radius_range
        self.omega_range = omega_range
        self.center_range = center_range

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        r = torch.empty(1).uniform_(*self.radius_range).item()
        omega = torch.empty(1).uniform_(*self.omega_range).item()
        theta0 = torch.empty(1).uniform_(to=2 * math.pi).item()

        center_x = torch.empty(1).uniform_(*self.center_range).item()
        center_y = torch.empty(1).uniform_(*self.center_range).item()

        traj = torch.zeros(self.seq_len, 2)
        for t in range(self.seq_len):
            theta = theta0 + omega * self.dt * t
            traj[t, 0] = center_x + r * math.cos(theta)
            traj[t, 1] = center_y + r * math.sin(theta)

        return traj