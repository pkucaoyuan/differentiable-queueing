# Comprehensive Reproduction Plan

Last updated: 2026-05-15

---

## Status of Reproduction So Far

### ✅ Confirmed reproduced

| # | Experiment | Note |
|---|-----------|------|
| 1 | M/M/1 sanity (E[Q]=9.0) | Deterministic, 0.31% error. Two runs identical |
| 3 | Section 5.2 PATHWISE (16 combos × 50 trials) | Within 5% of reference |
| 4 | Section 5.2 REINFORCE (16 combos × 50 trials) | Within 2% of reference |
| 5 | STE criss-cross training | Trend matches paper |
| 6 | PPO criss-cross training | Trend matches paper (STE 27x faster) |
| 7 | STE reentrant_2 training | Trend ok, caveat: last-iterate noisy |

### ⚠️ Issues found

| # | Issue | Severity |
|---|-------|----------|
| 2 | **Section 5.1 unstable** — 100K ground truth gives different sMW values across runs (0.15 vs -0.55). Original script defaults to **1M** | HIGH |
| 7 | reentrant_2 last-iterate ≠ paper's avg-iterate | MEDIUM |
| - | **GPU claim never validated** — paper experiments are all CPU | Paper-level |

### 📋 Not yet reproduced

| Section | Status | Reason |
|---------|--------|--------|
| 5.1 sMP policy | ❌ | Not in our previous runs |
| 5.1 different horizon T | ❌ | Only T=1000 done |
| 5.2 5-class queue | ❌ | Only 10-class done |
| 5.2 ablations (rho, T, queue_class, num_iter) | ❌ | Reference data exists in cmu/ |
| 5.3 admission control | ❌ | Different setup, separate script |
| 6 WC-Softmax ablation | ❌ | Separate experiment |
| 7 reentrant_3 to reentrant_10 | ❌ | Compute-heavy |
| 7 hyperexp service variants | ❌ | Separate configs |

### ❌ Cannot reproduce (record as blocker)

| Section | Blocker |
|---------|---------|
| GPU benchmarks (claimed but never tested) | GPU nodes 04/05 not in scheduler queue; gpu03 always full |
| GPU device path in env.py | torch installed is `2.2.0+cpu` build, no CUDA support |

---

## Comprehensive Plan — Submit in Parallel

### Phase A: Canonical Section 5.1 with proper 1M ground truth (CRITICAL)

Use the original `experiments/gradient_comparison.py` (paper's authoritative script) with paper's default settings:
- gt_batch = 1,000,000
- num_samples = 100
- estimators_per_sample = 100
- 3 policies: sPR, sMW, sMP

This is the **canonical Section 5.1 reproduction**, not our hand-rolled version.

### Phase B: Section 5.2 5-class queue (parallel to A)

The repo has reference data `cmu/pathwise_wc_cmu_multiclass5_all_eps_950_more_runs.json` (1000 runs!). We can run 5-class queue and verify against this.

### Phase C: Section 5.2 ablations (parallel to A, B)

Reference data exists for:
- rho ablation: 0.9, 0.95, 0.99
- T ablation: 500, 1000, 2000, 5000
- queue_class ablation: 5, 10, 15, 20

Reproduce one axis (e.g., rho) to verify scaling.

### Phase D: Section 7 reentrant_3 (medium scale)

Smallest reentrant network we haven't tested. Estimated wall time ~1h.

### Phase E (BLOCKED): Section 7 reentrant_5/10, GPU benchmarks

reentrant_10 takes 10+ hours; PPO takes 60+ hours. Skip unless time permits.

---

## Execution Plan

Submit phases A, B, C, D in parallel. Total compute budget estimate:
- A: ~3 hours (1M GT × 6 = 6h of GT + 200 samples × 1min = 3h, parallelized 16 cores)
- B: ~1.5 hours (5-class 50 trials × 16 combos × 2 methods)
- C: ~30 min (1 ablation axis, 4-5 values)
- D: ~1 hour (reentrant_3 STE, 100 epochs)

Total: ~6 hours wall time if run in parallel.
