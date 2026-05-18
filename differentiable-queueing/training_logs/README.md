# Training Logs & Artifacts

What's in this directory and how to read it.

## Note on W&B

The STE training path (`train/train_policy.py`) does not use W&B. One file
(`PPO/utils/softmax_policy.py`) does `import wandb`, but it's only used by
the Stable-Baselines3 PPO variant in `PPO/` and `wandb` is not installed in
the env. So no W&B runs / dashboards exist. Equivalent per-epoch metrics
are logged locally as JSON / CSV:

| W&B field | Local equivalent |
|---|---|
| `train_loss` (per step) | `epoch` + `train_loss` in `loss/*.json` |
| `eval_metric` | `test_loss` (mean) and `test_loss_std` |
| Run config | `configs/env/*.yaml`, `configs/model/*.yaml` |
| Checkpoint artifacts | `models/<env>/ppg_softmax_<epoch>.pt` |
| Plot artifacts | `plot/<env>/ppg_softmax_<epoch>.png` |

## Files

- `summary.csv` — one row per env: epochs, initial / final / min test cost.
- `csv/<env>_ppg_softmax.csv` — per-epoch training curve (epoch, test_loss, train_loss, test_loss_std).
- `csv/all_training_curves.csv` — all envs concatenated (env, epoch, ...).

## Mapping to paper sections

| Env | Paper section | Min cost |
|---|---|---:|
| `criss_cross_bh_ppg_softmax` | §6/§7 WC softmax | 15.20 |
| `criss_cross_bh_ppg_vanilla` | §6 Vanilla softmax | 17.21 |
| `reentrant_2_ppg_softmax` | §7 | 14.71 |
| `reentrant_3_ppg_softmax` | §7 | 21.99 |
| `reentrant_4_ppg_softmax` | §7 | 32.20 |
| `reentrant_5_ppg_softmax` | §7 | 30.10 |
| `reentrant_6_ppg_softmax` | §7 | 36.12 |
| `reentrant_7_ppg_softmax` | §7 | 37.26 |
| `reentrant_8_ppg_softmax` | §7 | 64.33 |
| `reentrant_9_ppg_softmax` | §7 | 72.92 |
| `reentrant_10_ppg_softmax` | §7 | 80.25 |

## Reproducing any single run

```bash
cd /user/yc4911/DJ_OR/differentiable-queueing
python train/train_policy.py -e=<env>.yaml -m=ppg_softmax.yaml --algo ste
```

Outputs land in:
- `loss/<env>_ppg_softmax.json` — training curve
- `models/<env>/ppg_softmax_<epoch>.pt` — checkpoint each epoch
- `plot/<env>/ppg_softmax_<epoch>.png` — policy plot each epoch

## Checkpoints

100 epochs × 11 envs ≈ 280 MB of `.pt` files; too large for the repo's
main history. Shared separately as a tarball (see PROFESSOR_PACKAGE.md).
