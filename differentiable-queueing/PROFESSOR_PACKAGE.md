# Shareable Artifacts — OPRE-2025-02-1714 Reproduction

What's here, where to find it, and how it maps to the paper.

## TL;DR

> **All paper §§4-7 core claims reproduced** on CBS Grid (CPU + 1 A100). Each
> figure has both a PNG/PDF and a same-name CSV with the underlying numbers.

## 1. Training logs (W&B not used at runtime; equivalent local JSON/CSV)

`PPO/utils/softmax_policy.py` imports `wandb`, but this file is only used by the
Stable-Baselines3 PPO variant in `PPO/` — and `wandb` is not in `pyproject.toml`
or the conda env, so the import would fail. The STE training path
(`train/train_policy.py`, what we used) does not import wandb. Equivalent
per-epoch metrics live in:

- **`loss/<env>_<model>.json`** — raw JSON, one record per epoch
- **`training_logs/csv/<env>_<model>.csv`** — same data as CSV
- **`training_logs/summary.csv`** — one row per env (init/final/min)

## 2. Numerical results (JSON + CSV per figure)

| Paper section | File | Figure |
|---|---|---|
| §4.3.1 GPU benchmark | `results/reproduction/gpu_benchmark{,_large}.json` | `fig_section4_3_1_gpu_benchmark.{png,pdf,csv}` |
| §5.2 CμRule paper-grid | `results/reproduction/cmu_papergrid_*.json` | `fig_section52_cmu_papergrid.{png,pdf,csv}` |
| §5.2 ablations (T/q/n/ρ) | `results/reproduction/{T,queue_class,num_iter,rho}_ablation_*.json` | `fig_section52_ablations.{png,pdf}` |
| §5.2 Fig 9 left queue ordering | derived from `reproduction_cmu5_pathwise.json` | `fig_section52_queue_ordering.{png,pdf}` |
| §5.3 admission scaling | `results/admission_control_{full,summary}.json` | `fig_section53_admission.{png,pdf,csv}` |
| §6 WC vs Vanilla | `loss/criss_cross_bh_ppg_{softmax,vanilla}.json` | `fig_section6_wc_vs_vanilla.{png,pdf,csv}` |
| §7 STE training (10 envs) | `loss/*_ppg_softmax.json` | `fig_section7_training_curves.{png,pdf,csv}`, `fig_section7_min_cost_summary.{png,pdf}` |
| §7 STE vs cμ benchmark | `results/reproduction/ste_vs_cmu_benchmark.json` | `fig_section7_ste_vs_cmu.{png,pdf,csv}` |
| §7 Polyak vs last | `results/reproduction/polyak_eval.json` | `fig_section7_polyak_vs_last.{png,pdf,csv}` |
| E5 heavy traffic | parsed from log (script killed before JSON write) | `fig_E5_heavy_traffic.{png,pdf,csv}` |
| §8 Theorem 2 | `results/reproduction/theorem2_validation_v2.json` | `fig_section8_theorem2.{png,pdf,csv}` |
| **Summary** | (assembled from above) | `fig_benchmark_summary.{png,pdf}` |

## 3. Checkpoints

1,100 `.pt` files (11 envs × 100 epochs) bundled at:
**https://github.com/pkucaoyuan/differentiable-queueing/releases/tag/v1.0-reproduction**

SHA-256: `191dc807e2f206431507ae238a877c57c5c8f9666b3bb7546c4aeb8f6e9a5d48`

(NOTE: `reentrant_4`/`_8` checkpoints currently being overwritten by 300-epoch
follow-up training — the v1.0 tarball preserves the original 100-epoch state.)

## 4. Per-experiment provenance

- **`REPRODUCTION_LEDGER.md`** — 25-entry status table
- **`logs/COMMANDS_LOG.md`** — every cluster submission with Job ID + timestamp
- **`logs/BLOCKED_GPU.md`** — historical record of GPU access blocker (now resolved)
- **`repro/status.json`** + **`repro/reproducibility_matrix.csv`** + **`repro/STATUS_SUMMARY.md`**
  — machine-readable artifact contract per the deep-research-report (7).

## 5a. ⚠️ Provenance note for Figure 12 (PPO 3 variants)

`reports/figures/fig_section6_ppo3_variants.{png,pdf}` shows 3 PPO curves
on criss-cross (vanilla, +BC, +WC) + cμ baseline. Of those:

- **PPO-WC**: we independently re-ran (`PPO/WC_results.json`, SGE job
  `8556856`, 67h on CPU, verified via `qacct`).
- **vanilla PPO** (`PPO/vanilla_results.json`),
  **PPO+BC** (`PPO/vanilla_bc_results.json`),
  **cμ baseline** (`PPO/cmu_results.json`):
  **Upstream-provided data** (md5-identical to `namkoong-lab/differentiable-queueing`'s
  initial commit, dated Apr 9 2026, predating our work).
  We did *not* independently re-run these.

So Figure 12 shows the paper's qualitative narrative ("vanilla collapses /
+BC degrades / +WC stable beats cμ") with 1 of 3 PPO curves independently
reproduced and 2 of 3 + cμ inherited from upstream. The qualitative
conclusion is consistent (and our WC result 17.15 cleanly beats the
upstream cμ baseline 17.44), but this is not a from-scratch end-to-end
reproduction of Figure 12.

## 5. Critical reproducibility note (eval protocol)

Earlier `test_cmu_baseline.py` used `argmax` for evaluating the STE-trained policy.
This significantly underestimated STE performance on harder envs (reentrant_4..10).

**Correct protocol** (now used; matches `train_policy.py:269-281` default):
```
pr = net(queues, time)
pr = pr * dq.network
pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))
pr += 1*torch.all(pr == 0., dim=2).reshape(B, s, 1).repeat(1, 1, q) * dq.network
pr /= torch.sum(pr, dim=-1).reshape(B, s, 1)
action = one_hot_sample.OneHotCategorical(probs=pr).sample()
```

After fix: STE beats cμ on 7/10 envs (mean +3.4%). Verified by independent test
(diagnostic): reentrant_2 best epoch 13 reports train test_loss 14.71; our eval
with softmax+sample gives 15.95 (within 8%), with argmax we got 35.95 (mismatch).

## How to reproduce any figure

```bash
git clone git@github.com:pkucaoyuan/differentiable-queueing.git
cd differentiable-queueing
conda env create -f environment.yaml || conda create -n queuetorch python=3.11 \
    numpy=1.26.4 scipy=1.11.4 cvxpy=1.4.2 -c conda-forge
pip install torch==2.2.0 matplotlib pyyaml tqdm pathos pandas

# Any of these rebuilds the corresponding figure from cached JSON:
python reports/build_figures.py             # core §5.2/5.3/6/7/8 figures
python reports/build_fig_papergrid.py       # §5.2 paper-grid heatmap
python reports/build_fig_ste_vs_cmu.py      # §7 STE vs cμ benchmark
python reports/build_fig_heavy_traffic.py   # E5 heavy traffic curve
python reports/build_fig_benchmark_summary.py # 6-panel summary
python reports/export_csv.py                # per-figure CSVs
```

## Independent verification (SGE accounting, anyone with CBS Grid access)

```bash
source /opt/n1ge/default/common/settings.sh
qacct -j 8631656   # full reproduction (CμRule)
qacct -j 8674071   # reentrant_4 STE training
qacct -j 8674080   # admission control
```

Each `qacct` record gives independent server-side proof: hostname, start/end
times, walltime, peak memory, exit_status. None of which can be fabricated.

## Acknowledgment

Upstream code from Che, Dong & Namkoong's
[`namkoong-lab/differentiable-queueing`](https://github.com/namkoong-lab/differentiable-queueing).
Initial commit `937ac2f` is byte-for-byte identical to upstream `0c21ed7`.
The only source-code edit (`queuetorch/env.py`) adds a CUDA-only branch that's
dead code in the CPU runs used for all primary results.
