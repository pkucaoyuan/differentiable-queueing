# Project Progress Log

Paper: OPRE-2025-02-1714 "Differentiable Discrete Event Simulation for Queuing Network Control"
Last updated: 2026-05-10

---

## Status Overview

| Phase | Status |
|-------|--------|
| 0. Codebase setup & familiarization | ✅ Done |
| 1. Reproduction of paper results | ✅ Done (Section 5.1, 5.2, 7 partial, 8) |
| 2. Revision experiments — Phase 1 (quick wins) | ✅ Done (E2, E5, E7, E8) |
| 3. Revision experiments — Phase 2 (core ablations) | 🚧 Partial (E6 done, E1/E4 pending) |
| 4. Revision experiments — Phase 3 (GPU) | ⏸️ Blocked (E3 needs GPU access) |
| 5. Paper rewrite & response letter | ⏸️ Not started |

---

## Reproduction (Sections 5.1, 5.2, 7, 8)

### Final reproduction (job 8631656, 5.8 hours, 16 cores)
- ✅ M/M/1 simulator: 0.31% error
- ✅ Section 5.1 gradient comparison (sPR + sMW on criss-cross)
- ✅ Section 5.2 CMU rule full sweep: 4 alphas × 4 gaps × 50 trials × 2 methods = 1600 trials
- All 24 reproduced (alpha, gap) combos within 5.19% of reference data

### Earlier reproductions
- Job 8556850: STE training criss-cross (100 epochs, 2.5h)
- Job 8556856: PPO training criss-cross (101 iters, 67h)
- Job 8631657: STE training reentrant_2 (100 epochs, 35min) — first reentrant network reproduced

### Coverage
- ✅ Section 5.1 (gradient quality)
- ✅ Section 5.2 (CMU rule)
- ✅ Section 7 partial: criss-cross + reentrant_2
- ✅ Section 8 sanity (M/M/1)
- ❌ Section 5.1 sMP policy not done
- ❌ Section 5.3 admission control not done
- ❌ Section 6 WC-Softmax ablation not done
- ❌ Section 7 reentrant_3 through reentrant_10 not done
- ❌ hyper-exponential variants not done

---

## Revision Experiments

### E1: STE Bias-Variance Beyond M/M/1
- Status: **Script ready, NOT submitted**
- File: `experiments/ste_bias_variance.py`
- Compute estimate: ~80 core-hours

### E2: GLR Comparison on M/M/1
- Status: **✅ Done** (job 8557913, 5.8s)
- Results: PATHWISE < GLR < REINFORCE in MSE across all 16 (ρ, T) combos
- File: `results/E2_glr_comparison.json`
- Concern: For continuous φ, Peng's GLR reduces to classic LR. Need to discuss with advisor whether to redo on a setting where GLR ≠ LR.

### E3: GPU Benchmarks
- Status: **⏸️ Blocked** — needs GPU node access
- File: `experiments/gpu_benchmarks.py` ready
- Issue: gpu03 always full (7 jobs queued), gpu04/05 not in scheduler
- Action needed: Email ResearchSupport@gsb.columbia.edu

### E4: Non-WC Criss-Cross
- Status: **Script ready, NOT submitted**
- File: `experiments/criss_cross_nonwc.py`
- Concern: Need exact Martins (1996) Cases IIB/IID parameters

### E5: Heavy-Traffic Curve
- Status: **✅ Done** (job 8557914, ~10h)
- Results: Cost increases as ρ→1 (rho=0.99 cost ~27 vs rho=0.8 cost ~5)
- File: `results/E5_heavy_traffic_*.json`

### E6: 3-Way Factorial Ablation
- Status: **✅ Done** (job 8576491, ~6h, after fixing PATHWISE_TEMP and noise placement bugs)
- File: `results/E6_ablation_3way.json`

### E7: Hyperparameter Sensitivity
- Status: **✅ Done** (job 8563087, ~3.5h, after fixing PATHWISE_TEMP bug)
- File: `results/E7_hyperparam_*.json`

### E8: Parameter Tables
- Status: **✅ Done** (zero compute)
- File: `results/E8_*.json` — 25 environments + 4 model configs extracted

---

## Code Modifications

### env.py — GPU-native sampling
- Added `_draw_service_gpu()` and CUDA path in `draw_inter_arrivals()`
- Uses `torch.distributions` for GPU-resident random sampling
- Auto-enabled when device='cuda' AND lam_type='constant'
- **Not yet validated on actual GPU** (no GPU access)

### configs/env/criss_cross_IIB.yaml — new
- For E4 non-WC experiment
- Parameters approximated from Martins et al. (1996) Case IIB

### Bug fixes
- `PATHWISE_TEMP = 0.1` → `1e-6` in E6 and E7 (matches existing `cmu_step_rules_PATHWISE.py`)
- Method 3/4 noise placement in E6 (logit space, per-step, both methods consistent)

---

## Issues & Open Questions

### Critical
1. **E2 GLR validity**: For M/M/1 with continuous φ, GLR = classic LR. Comparison may not be testing the actual GLR contribution. Need to discuss with advisor.
2. **GPU access**: Cannot run E3 without GPU node access. gpu04/05 hardware exists but not in scheduler.

### Code quality (for paper code release)
- `cmu_rule_REINFORCE.py` has hardcoded `/user/xz3355/...` paths
- `gradient_comparison.py` has hardcoded shebang `/user/tmm2219/.conda/envs/qt_env/bin/python`
- Different scripts use different `temp` values (1e-6 vs 0.1)
- No README explaining how to reproduce

### Reproduction gaps
- Section 5.3 admission control not reproduced
- Section 6 WC-Softmax not reproduced
- Most reentrant networks (3-10) not reproduced
- Hyper-exponential variants not reproduced

---

## File Inventory

### Reproduction
- `experiments/reproduce_main.py` — initial 20-trial reproduction (Test 1-5)
- `experiments/full_reproduction.py` — comprehensive 50-trial reproduction
- `jobs/run_ste_criss_cross.sh`, `run_ppo_criss_cross.sh`, `run_cmu_reproduce.sh`
- `jobs/run_full_reproduction.sh`, `run_ste_reentrant_2.sh`

### Revision
- `experiments/{ste_bias_variance,glr_comparison,gpu_benchmarks,criss_cross_nonwc,heavy_traffic_curve,ablation_3way,hyperparam_sensitivity,extract_parameters}.py`
- `jobs/run_e2_glr.sh`, `run_e5_heavy_traffic.sh`, `run_e6_ablation.sh`, `run_e7_hyperparam.sh`

### Documentation
- `logs/REPRODUCTION_REPORT.md` — detailed reproduction verification
- `docs/revision_experiments/REVISION_GUIDE.md` — full revision plan
- `docs/revision_experiments/E1-E8_*.md` — per-experiment specs
- `/user/yc4911/doc/email_to_prof_dong_20260414.md` — email draft to advisor
- `/user/yc4911/doc/revision_plan.md` — clean revision plan (attachment for email)
- `/user/yc4911/doc/meeting/meeting_notes_20260414.md` — meeting notes

### Logs
- `logs/reproduction/` — 36 files (5 jobs × 4 SGE log types + extras)
- `logs/revision/` — 24 files (6 jobs × 4 SGE log types)

### Results JSON
- `results/reproduction_*.json` — reproduction validation data
- `results/E2_*.json`, `E5_*.json`, `E6_*.json`, `E7_*.json`, `E8_*.json` — revision data

---

## Next Steps

### Immediate (before next meeting with Prof. Dong)
1. Send email with revision plan attached
2. Get clarification on:
   - Martins (1996) IIB/IID parameters for E4
   - GPU node access for E3
   - Whether to expand GLR comparison (E2) to where GLR ≠ LR
3. Discuss whether to reproduce more reentrant networks

### After meeting
1. Submit E1 (compute-heavy, ~80 core-hours)
2. Submit E4 once parameters confirmed
3. Try E3 once GPU access resolved
4. Begin paper rewrite for sections that are settled
