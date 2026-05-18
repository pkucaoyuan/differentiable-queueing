# Shareable Artifacts

What's available, where to find it, and how it maps to the paper.

## 1. Training logs (W&B not used at runtime; equivalent local JSON/CSV)

`PPO/utils/softmax_policy.py` imports `wandb`, but this file is only used by the
Stable-Baselines3 PPO variant in `PPO/` — and `wandb` is not in `pyproject.toml`
or the conda env, so the import would fail. The STE training path
(`train/train_policy.py`, what I ran) does not import wandb. Equivalent
per-epoch metrics live in:

- **`loss/<env>_<model>.json`** — `[{epoch, test_loss, train_loss, test_loss_std}, …]` per epoch.
- **`training_logs/csv/<env>_<model>.csv`** — same data, CSV.
- **`training_logs/csv/all_training_curves.csv`** — all 11 runs concatenated (env, epoch, ...).
- **`training_logs/summary.csv`** — one row per env: initial / final / min test cost.

## 2. JSON/CSV outputs (ablation + reproduction numbers)

- **`results/reproduction/`** — every reproduction script's JSON output (M/M/1, §5.1 gradient cossim, §5.2 CμRule full + 5-class + 4 ablations, §8 Theorem 2 v1 + v2).
- **`results/admission_control_*.json`** — §5.3 admission control: per-env mean/std for PATHWISE_B1 and SPSA_{B=10,100,1000}.
- **`results/revision/`** — outputs of revision experiments E2/E5/E6/E7/E8.

## 3. Checkpoints

- 11 envs × 100 epochs = 1,100 `.pt` files, 284 MB raw / 255 MB gzipped.
- Bundled as `training_logs/checkpoints.tar.gz` — uploaded as a GitHub Release asset (download link in README) to keep the repo light.
- Layout inside the tarball: `models/<env>/ppg_softmax_<epoch>.pt`.
- Each checkpoint is the full state_dict of `PriorityNet` (~200 KB).

## 4. Per-epoch policy plots (optional)

- 1,100 PNG plots in `plot/<env>/ppg_softmax_<epoch>.png` (18 MB total). Visualization of the learned softmax priorities at each epoch.

## 5. Reproduction provenance

- **`REPRODUCTION_LEDGER.md`** — 25-entry table: experiment, status, numerical-match %, notes.
- **`logs/COMMANDS_LOG.md`** — every cluster submission with Job ID and timestamp.
- **`logs/BLOCKED_GPU.md`** — note on §4.3.1 GPU experiments being unavailable.

## Quick consistency check

```bash
# Stdout claim → loss JSON → results JSON should all match
grep "min cost" REPRODUCTION_LEDGER.md
column -t -s, training_logs/summary.csv
```
