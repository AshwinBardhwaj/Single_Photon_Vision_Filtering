import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from filtering.learned_filter_bank import LearnedLogGaborBank3D
from vision.feature_detection.phase_congruency_torch import phase_congruency_3D_torch
from utils.dataset import get_dataloader
from fileio.fileio import save_video_imageio
from visualize.visualize import vis_edge


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def validate_and_save(model, val_mat_path, output_dir, device, epoch):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    data = loadmat(val_mat_path)
    B_val = data['B'].astype(np.float32)
    B_tensor = torch.tensor(B_val, device=device)

    with torch.no_grad():
        model.input_size = B_val.shape
        responses, filter_energies, dirs = model(B_tensor)

        PCs = phase_congruency_3D_torch(
            responses, dirs,
            model.num_scales, model.num_orientations, model.num_velocities,
            flux_noise_std=torch.tensor(1.0, device=device),
            filter_energies=filter_energies
        )

        pc_x2, pc_y2, pc_t2, pc_xy, pc_yt, pc_xt = PCs
        edge_strength = torch.sqrt(torch.relu(pc_x2 + pc_y2 + pc_t2) + 1e-8)

    estr_np = edge_strength.cpu().numpy()

    estr_v = vis_edge(estr_np, False, [75, 97])

    out_name = Path(val_mat_path).stem
    save_path = os.path.join(output_dir, f"{out_name}_epoch_{epoch}.mp4")
    save_video_imageio(np.moveaxis(estr_v, -1, 0), save_path)
    print(f"Validation video saved to {save_path}")
    model.train()


def train_loop(subset_size=None):
    device = get_device()
    print(f"Using device: {device}")

    train_dir = root_dir / "scripts" / "data" / "sim_xvfi_1bit"
    val_mat = root_dir / "scripts" / "data" / "fig05_vision_0604-jump-1.mat"
    val_out_dir = root_dir / "scripts" / "output" / "validation_results"
    weights_out_dir = root_dir / "scripts" / "output" / "weights"
    os.makedirs(weights_out_dir, exist_ok=True)

    dataloader = get_dataloader(train_dir, batch_size=1, exclude_folders=['014', '016'])

    sample_B, _, _ = next(iter(dataloader))
    input_shape = sample_B[0].shape

    model = LearnedLogGaborBank3D(input_size=input_shape, num_velocities=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    num_epochs = 3
    best_avg_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (B_batch, gt_batch, paths) in enumerate(dataloader):
            B_train = B_batch[0].to(device)
            gt_train = gt_batch[0].to(device)

            model.input_size = B_train.shape

            optimizer.zero_grad()

            responses, filter_energies, dirs = model(B_train)

            PCs = phase_congruency_3D_torch(
                responses, dirs,
                model.num_scales, model.num_orientations, model.num_velocities,
                flux_noise_std=torch.tensor(1.0, device=device),
                filter_energies=filter_energies
            )

            pc_x2, pc_y2, pc_t2, pc_xy, pc_yt, pc_xt = PCs
            edge_strength = torch.sqrt(torch.relu(pc_x2 + pc_y2 + pc_t2) + 1e-8)

            loss = criterion(edge_strength, gt_train)
            loss.backward()

            # Final safeguard
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx} Loss: {loss.item():.4f}")

            if subset_size is not None and batch_idx >= subset_size - 1:
                break

        actual_batches = min(len(dataloader), subset_size) if subset_size else len(dataloader)
        avg_loss = total_loss / actual_batches
        print(f"--- Epoch {epoch} Average Loss: {avg_loss:.4f} ---")

        # validate_and_save(model, val_mat, val_out_dir, device, epoch)

        # Save model checkpoint
        checkpoint_path = os.path.join(weights_out_dir, f"model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_model_path = os.path.join(weights_out_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved BEST model: {best_model_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--mini':
        print("Running mini-training session...")
        train_loop(subset_size=10)
    else:
        train_loop()