# How to Reproduce — Step-by-Step Guide

This document records the exact commands, scripts, and logs for every reproduction experiment we ran. Anyone (including future-self) should be able to follow this to reproduce or extend the results.

---

## 1. Environment Setup

### One-time setup
```bash
# Install Miniconda (in shared NFS home, will be available on all nodes)
curl -sLo Miniconda3-latest-Linux-x86_64.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash

# Create env and install dependencies
conda create -n gpu python=3.11 -y
conda activate gpu
pip install torch==2.2.0 numpy==1.26.4 scipy==1.11.4 cvxpy==1.4.2 pathos pyyaml tqdm matplotlib

# Install queuetorch in editable mode
cd /user/yc4911/DJ_OR/differentiable-queueing
pip install -e .

# For PPO experiments only:
pip install 'stable-baselines3==2.3.0' 'gymnasium==0.29.1' 'shimmy==0.2.1'
```

### Per-session activation
```bash
eval "$(/user/yc4911/miniconda3/bin/conda shell.bash hook)"
conda activate gpu
source /opt/n1ge/default/common/settings.sh
```

---

## 2. Cluster Submission Rules

**Critical**: Never run compute on login nodes. Always submit via `grid_run`:

```bash
cd /user/yc4911/DJ_OR/differentiable-queueing
grid_run --grid_submit=batch --grid_mem=<MEM>G --grid_ncpus=<N> ./jobs/<script>.sh
```

Job management:
- `qstat` — view your jobs
- `qstat -j <jobID>` — detailed status
- `qdel <jobID>` — cancel
- Logs auto-saved as `<scriptname>.{o,e}<jobID>` in submission directory

Conventions used:
- All scripts auto-detect `NSLOTS` for multiprocessing pool size
- All scripts source conda + cd to correct directory
- All scripts log start/end timestamps

---

## 3. Reproduction Experiments — Step by Step

### 3.1 M/M/1 Sanity Check + Section 5.1 Gradient Comparison + Section 5.2 CMU Rule

**Single comprehensive script** that does all three:

```bash
# Submit
grid_run --grid_submit=batch --grid_mem=50G --grid_ncpus=16 \
    ./jobs/run_full_reproduction.sh
```

| Property | Value |
|----------|-------|
| Script | `experiments/full_reproduction.py` |
| Wrapper | `jobs/run_full_reproduction.sh` |
| Cores | 16 |
| Memory | 50G |
| Wall time | 5.8 hours |
| Job ID | 8631656 |

**What it does (in order):**
1. M/M/1 simulator validation (200K events, batch=50, ρ=0.9)
2. Section 5.1 gradient comparison: 50 PATHWISE samples (B=1) + 50 REINFORCE samples (B=1000) for sPR and sMW policies on criss-cross_bh
3. Section 5.2 CMU rule: 50 trials × 4 alphas (0.01, 0.1, 0.5, 1.0) × 4 gaps (1.0, 0.5, 0.05, 0.01) for both PATHWISE and REINFORCE on 10-class queue, ρ=0.95
4. Auto-verification: compares results with `cmu/pathwise_wc_cmu_multiclass10.json` and `cmu/wc_reinforce_baseline_cmu_B100_multiclass10.json`

**Outputs:**
- `results/reproduction_mm1.json` — M/M/1 sanity result
- `results/reproduction_gradient.json` — Section 5.1 cossim summary
- `results/reproduction_cmu_pathwise.json` — Section 5.2 PATHWISE (16 alpha-gap combos × 50 trials)
- `results/reproduction_cmu_reinforce.json` — Section 5.2 REINFORCE (16 alpha-gap combos × 50 trials)
- `logs/reproduction/run_full_reproduction.sh.o8631656` — stdout (1.8MB, includes verification table at end)
- `logs/reproduction/run_full_reproduction.sh.e8631656` — stderr (mostly tqdm progress bars)

**Verify it worked:**
```bash
# Check verification table at end of stdout
grep -v "^tensor(" logs/reproduction/run_full_reproduction.sh.o8631656 | tail -50

# Should see PATHWISE and REINFORCE comparison tables, all marked [OK]
```

---

### 3.2 STE Training (Section 7) on criss-cross

```bash
grid_run --grid_submit=batch --grid_mem=8G \
    ./jobs/run_ste_criss_cross.sh
```

| Property | Value |
|----------|-------|
| Script | `train/train_policy.py -e=criss_cross_bh.yaml -m=ppg_softmax.yaml --algo ste` |
| Wrapper | `jobs/run_ste_criss_cross.sh` |
| Cores | 1 |
| Memory | 8G |
| Wall time | 2.5 hours |
| Job ID | 8556850 |

**What it does:** Trains a PriorityNet (3-layer MLP, 128 hidden) using STE/PATHWISE on the criss-cross_bh network for 100 epochs. Each epoch: 20K-step training rollout + 200K-step evaluation rollout.

**Outputs:**
- `train/models/criss_cross_bh-ppg_softmax-ste.pt` — trained policy checkpoint
- `train/loss/criss_cross_bh-ppg_softmax-ste.json` — training loss curve
- `train/plot/criss_cross_bh-ppg_softmax-ste-*.png` — diagnostic plots
- `logs/reproduction/run_ste_criss_cross.sh.o8556850`

**Expected result:** test cost decreases from ~18 to ~16 over 100 epochs. Min cost ~15.20 around epoch 63.

---

### 3.3 PPO Training (Section 7) on criss-cross

```bash
grid_run --grid_submit=batch --grid_mem=16G --grid_ncpus=4 \
    ./jobs/run_ppo_criss_cross.sh
```

| Property | Value |
|----------|-------|
| Script | `PPO/train.py wc_softmax criss_cross_bh` |
| Wrapper | `jobs/run_ppo_criss_cross.sh` |
| Cores | 4 |
| Memory | 16G |
| Wall time | 67 hours (yes, really) |
| Job ID | 8556856 |

**What it does:** Trains a WC-Softmax policy via PPO (Stable Baselines 3) on criss-cross_bh for 101 iterations. Each iteration: 20 actors × 50K episode_steps rollout + 50K-step evaluation + 3 PPO epochs.

**Outputs:**
- `PPO/models/...` — checkpoints
- `logs/reproduction/run_ppo_criss_cross.sh.o8556856`

**Expected result:** test cost decreases slowly from ~17.7 to ~17.2 over 101 iters. Min cost ~16.62.

**Note:** PPO is much slower to converge than STE. STE achieves better cost (15.20 min) in 2.5h vs PPO (16.62 min) in 67h.

---

### 3.4 STE Training on reentrant_2 (Section 7 reentrant network)

```bash
grid_run --grid_submit=batch --grid_mem=8G \
    ./jobs/run_ste_reentrant_2.sh
```

| Property | Value |
|----------|-------|
| Script | `train/train_policy.py -e=reentrant_2.yaml -m=ppg_softmax.yaml --algo ste` |
| Wrapper | `jobs/run_ste_reentrant_2.sh` |
| Cores | 1 |
| Memory | 8G |
| Wall time | 35 minutes |
| Job ID | 8631657 |

**What it does:** First reproduction of a reentrant network (6 queues, 2 servers).

**Outputs:**
- `train/models/reentrant_2-ppg_softmax-ste.pt`
- `logs/reproduction/run_ste_reentrant_2.sh.o8631657`

**Expected result:** test cost has min around 14.71 (epoch 14). Cost is noisy in late training; paper uses Polyak averaging which is not in this script.

---

## 4. Revision Experiments

### E2: GLR Comparison (M/M/1)

```bash
grid_run --grid_submit=batch --grid_mem=8G --grid_ncpus=16 \
    ./jobs/run_e2_glr.sh
```

| Property | Value |
|----------|-------|
| Script | `experiments/glr_comparison.py` |
| Wall time | 6 seconds (yes) |
| Job ID | 8557913 |
| Output | `results/E2_glr_comparison.json` |

**Result:** PATHWISE < GLR < REINFORCE in MSE across all 16 (ρ, T) combos.

---

### E5: Heavy-Traffic Curve

```bash
grid_run --grid_submit=batch --grid_mem=30G --grid_ncpus=16 \
    ./jobs/run_e5_heavy_traffic.sh
```

| Property | Value |
|----------|-------|
| Script | `experiments/heavy_traffic_curve.py` |
| Wall time | ~10 hours |
| Job ID | 8557914 |
| Output | `results/E5_heavy_traffic_*.json` |

**What it does:** Sweeps ρ ∈ [0.80, 0.99] (11 values) × 3 gaps × 2 methods × 200 trials.

---

### E6: 3-Way Factorial Ablation

```bash
grid_run --grid_submit=batch --grid_mem=30G --grid_ncpus=16 \
    ./jobs/run_e6_ablation.sh
```

| Property | Value |
|----------|-------|
| Script | `experiments/ablation_3way.py` |
| Wall time | ~6 hours |
| Job ID | 8576491 (after fixing two bugs) |
| Output | `results/E6_ablation_3way.json` |

**Bugs fixed before final run:**
1. `PATHWISE_TEMP = 0.1` → `1e-6` (matched existing code)
2. Method 3 (Reparam) noise was added in action space → moved to logit space (consistent with Method 4)

---

### E7: Hyperparameter Sensitivity

```bash
grid_run --grid_submit=batch --grid_mem=30G --grid_ncpus=16 \
    ./jobs/run_e7_hyperparam.sh
```

| Property | Value |
|----------|-------|
| Script | `experiments/hyperparam_sensitivity.py` |
| Wall time | ~3.5 hours |
| Job ID | 8563087 (after fixing temp bug) |
| Output | `results/E7_hyperparam_*.json` |

---

### E8: Parameter Tables (zero compute)

```bash
# Run on login node — zero compute, just data extraction
cd experiments && python extract_parameters.py
```

| Property | Value |
|----------|-------|
| Wall time | < 5 seconds |
| Output | `results/E8_*.json` |

---

## 5. Standard Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│  1. Submit:  grid_run --grid_submit=batch ... ./job.sh      │
│  2. Monitor: qstat -j <jobID>                               │
│  3. Logs:    tail -f <script>.o<jobID>                      │
│  4. Verify:  python -c "import json; ..."                   │
│  5. Archive: mv <script>.* logs/{reproduction|revision}/    │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Job ID Index

| Job ID | Type | Date | Wall time | What |
|--------|------|------|-----------|------|
| 8556850 | reproduction | 2026-04-10 | 2.5h | STE criss-cross |
| 8556852 | reproduction | 2026-04-11 | ~30min | First CMU reproduction (20 trials) |
| 8556856 | reproduction | 2026-04-11 | 67h | PPO criss-cross |
| 8557913 | revision | 2026-04-12 | 6s | E2 GLR |
| 8557914 | revision | 2026-04-12 | 10h | E5 Heavy traffic |
| 8563087 | revision | 2026-04-12 | 3.5h | E7 Hyperparams (after temp fix) |
| 8576491 | revision | 2026-04-23 | 6h | E6 Ablation (after both bug fixes) |
| 8631656 | reproduction | 2026-05-10 | 5.8h | Full reproduction (50 trials × 16 combos × 2 methods) |
| 8631657 | reproduction | 2026-05-10 | 35min | STE reentrant_2 |

---

## 7. Files Created

### Experiment scripts
| File | Purpose |
|------|---------|
| `experiments/full_reproduction.py` | Comprehensive M/M/1 + 5.1 + 5.2 reproduction |
| `experiments/reproduce_main.py` | Initial 20-trial reproduction |
| `experiments/extract_parameters.py` | E8 parameter table extraction |
| `experiments/glr_comparison.py` | E2 GLR vs PATHWISE vs REINFORCE on M/M/1 |
| `experiments/heavy_traffic_curve.py` | E5 ρ sweep |
| `experiments/ablation_3way.py` | E6 6-method factorial |
| `experiments/hyperparam_sensitivity.py` | E7 4-axis sweep |
| `experiments/ste_bias_variance.py` | E1 (not yet run) |
| `experiments/criss_cross_nonwc.py` | E4 (not yet run) |
| `experiments/gpu_benchmarks.py` | E3 (blocked on GPU access) |

### Job submission wrappers
| File | Used for |
|------|----------|
| `jobs/run_full_reproduction.sh` | Comprehensive reproduction |
| `jobs/run_ste_criss_cross.sh` | STE on criss-cross |
| `jobs/run_ste_reentrant_2.sh` | STE on reentrant_2 |
| `jobs/run_ppo_criss_cross.sh` | PPO on criss-cross |
| `jobs/run_cmu_reproduce.sh` | First CMU reproduction |
| `jobs/run_e2_glr.sh` | E2 GLR |
| `jobs/run_e5_heavy_traffic.sh` | E5 Heavy traffic |
| `jobs/run_e6_ablation.sh` | E6 Ablation |
| `jobs/run_e7_hyperparam.sh` | E7 Hyperparams |

### Code modifications
- `queuetorch/env.py`: added GPU-native sampling path (`_draw_service_gpu`)
- `configs/env/criss_cross_IIB.yaml`: new config for E4

### Documentation
- `REPRODUCTION_GUIDE.md` — this file (how to run)
- `PROGRESS.md` — current status of all experiments
- `logs/REPRODUCTION_REPORT.md` — detailed numerical comparison vs reference data
- `docs/revision_experiments/REVISION_GUIDE.md` — full revision plan
- `docs/revision_experiments/E1-E8_*.md` — per-experiment specs
