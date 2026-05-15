# Single-Photon Vision Filtering

Filter bank comparison for Eulerian single-photon video: Log Gabor, Adaptive IIR. Extends Gupta et al. ICCV 2023.

## Data

Download the data zip from [Google Drive](https://drive.google.com/file/d/1po4Ju8f2sJV9zwVyAtnxi9Dwi97tLP8G/view?usp=sharing) and extract its contents into:

```
scripts/data/
```

The directory should contain `.mat` files for real data and a `sim_xvfi_1bit/` folder for simulated data.

## Main Evaluation

```bash
python scripts/eval_full_comparison.py
# real data only:
python scripts/eval_full_comparison.py --real-only
# simulated data only:
python scripts/eval_full_comparison.py --sim-only
```

Results saved to `scripts/output/full_comparison/`.

## Individual Reconstructions

### Real data (Log Gabor + Adaptive IIR, edge videos + flow)

```bash
python scripts/fig05_01_edge_SPAD_video.py   # warf scene
python scripts/fig08_11_edge_SPAD_video.py   # ball-1, λ_min=3, 3-orient
python scripts/fig08_12_edge_SPAD_video.py   # ball-1, λ_min=6.3, 3-orient
python scripts/fig08_13_edge_SPAD_video.py   # ball-1, λ_min=6.3, 6-orient
python scripts/fig08_21_edge_SPAD_video.py   # ball-3, λ_min=3
python scripts/fig08_22_edge_SPAD_video.py   # ball-3, learned filter
python scripts/fig08_23_edge_SPAD_video.py   # ball-3, steerable
python scripts/fig09_01_edge_SPAD_video.py   # ball-1
python scripts/fig09_02_edge_SPAD_video.py   # bicycle
python scripts/fig09_03_edge_SPAD_video.py   # jump
python scripts/fig09_04_edge_SPAD_video.py   # train-dark
```

### Simulated data (single clip)

Edit `PARENT_DIR`, `VID_NAME`, `PPP` at the top of the script, then:

```bash
python scripts/eval_sim.py
```

## Dependencies

```bash
pip install numpy scipy scikit-image imageio matplotlib torch
```
