import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat


class SimulatedSPADDataset(Dataset):
    def __init__(self, root_dir, exclude_folders=None):
        self.root_dir = Path(root_dir)
        all_mat_files = list(self.root_dir.rglob("*.mat"))
        
        if exclude_folders:
            self.mat_files = []
            for path in all_mat_files:
                if not any(f in path.parts for f in exclude_folders):
                    self.mat_files.append(path)
        else:
            self.mat_files = all_mat_files

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        mat_path = self.mat_files[idx]
        data = loadmat(str(mat_path))

        B = data['B'].astype(np.float32)
        edges_gt = data['edges_gt'].astype(np.float32)

        B_tensor = torch.tensor(B)
        edges_gt_tensor = torch.tensor(edges_gt)

        return B_tensor, edges_gt_tensor, str(mat_path)


def get_dataloader(root_dir, batch_size=1, shuffle=True, exclude_folders=None):
    dataset = SimulatedSPADDataset(root_dir, exclude_folders=exclude_folders)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)