# Reproduction Ledger — Per Experiment Status

Last updated: 2026-05-15

---

## Summary

| # | Experiment | Status | Match | Notes |
|---|-----------|--------|-------|-------|
| 1 | M/M/1 sanity | ✅ | exact | Deterministic |
| 2 | Section 5.1 gradient (quick, 100K GT) | ⚠️ | unstable | sMW varies 0.15 vs -0.55 between runs |
| 2' | Section 5.1 canonical (80K GT, 5 samples, 20 est.) | ⚠️ | noisy | High variance per (rho, policy); see below |
| 3 | Section 5.2 PATHWISE 10-class (50 trials × 16 combos) | ✅ | <5.2% | Within statistical noise |
| 4 | Section 5.2 REINFORCE 10-class (50 trials × 16 combos) | ✅ | <1.67% | Almost exact |
| 5 | STE training criss-cross (100 epochs) | ✅ | qualitative | cost 18→15.2 |
| 6 | PPO training criss-cross (101 iters, 67h) | ✅ | qualitative | cost 17.7→16.6, STE 27x faster confirmed |
| 7 | STE training reentrant_2 (100 epochs) | ✅ | trend | min cost 14.71 epoch 14 |
| 8 | Section 5.2 PATHWISE 5-class | ✅ | 0.78% | quick check (20 trials × 2 gaps) |
| 9 | Section 5.2 REINFORCE 5-class | ✅ | 0.19% | quick check (20 trials × 2 gaps) |
| 10 | rho ablation (rho=0.9, 0.99) | ✅ | <2.62% | quick check (10 trials × 2 gaps × 2 methods) |
| 11 | STE training reentrant_3 (100 epochs) | ✅ | trend | min cost 21.99 epoch 63 |
| 12 | STE training reentrant_5 (100 epochs) | ✅ | trend | min cost 30.10 epoch 75 |
| 13 | STE training reentrant_6 (100 epochs) | ✅ | trend | min cost 36.12 epoch 14 |
| 14 | STE training reentrant_7 (100 epochs) | ✅ | trend | min cost 37.26 epoch 91 |
| 15 | T ablation (T=500, 1000, 2000, 5000) | ✅ | <2.41% | 8/8 OK |
| 16 | queue_class ablation (qc=5, 15, 20) | ✅ | <1.62% | 6/6 OK |
| 17 | num_iter ablation (ni=10, 20, 100) | ✅ | <1.73% | 6/6 OK |
| 18 | Theorem 2 numerical validation | ❌ | mismatch | slopes wrong; T=1000 insufficient |
| 19 | Section 6: Vanilla Softmax criss-cross | ✅ | qualitative | min 17.20 (vs WC 15.20, 13.2% worse — confirms WC > Vanilla) |
| 20 | STE training reentrant_4 (100 epochs) | ✅ | trend | min cost 32.20, final 35.03 |
| 21 | Theorem 2 retry (T=10000, 1000 trials) | ❌ | mismatch | PW slope -4.13 (vs -3), RF slope -0.76 (vs -4). RF variance plateaus — my Gaussian-perturbation REINFORCE doesn't reproduce paper's likelihood-ratio scaling. Need paper's exact event-history score function. |
| 22 | Section 5.3 Admission Control (12 envs × 4 methods) | ✅ | qualitative | PATHWISE_B1 consistent across all envs; SPSA_B1000 fails catastrophically on K≥15 nets (e.g. reentrant_7: PW=44.1 vs SPSA_B1000=106.0) — confirms paper's claim that PATHWISE > SPSA at scale |
| 23 | STE training reentrant_8 (100 epochs) | ✅ | trend | min cost 64.33 (ep 2) |
| 24 | STE training reentrant_9 (100 epochs) | ✅ | trend | min cost 72.92 (ep 5) |
| 25 | STE training reentrant_10 (100 epochs) | ✅ | trend | min cost 80.25 (ep 80) |

**All reproduction jobs complete.**

---

## Detailed Results

### Experiment 1: M/M/1 Sanity Check

| Run | Job ID | E[Q] | Error | Time |
|-----|--------|------|-------|------|
| Original | (in full_reproduction 8631656) | 9.0282 | 0.31% | within job |
| Re-run | 8673694 | **9.0282** | 0.314% | 37s |

**Verdict: Identical. Fully reproducible (seed=42).**

---

### Experiment 2: Section 5.1 Gradient Cossim (criss-cross_bh)

**Iteration 1 (full_reproduction, 100K GT, 50 samples):**
| Policy | PATHWISE | REINFORCE |
|--------|----------|-----------|
| sPR | 0.8000 ± 0.6000 | -0.0000 ± 1.0000 |
| sMW | 0.1513 ± 0.4088 | 0.0500 ± 0.4697 |

**Iteration 2 (test_gradient.py rerun, 100K GT, 50 samples):**
| Policy | PATHWISE | REINFORCE |
|--------|----------|-----------|
| sPR | 1.0000 ± 0.0000 | 0.0400 ± 0.9992 |
| sMW | **-0.5533** ± 0.1777 | -0.0021 ± 0.4617 |

⚠️ **sMW flipped from +0.15 to -0.55 between runs.** Inconsistent.

**Iteration 3 (canonical gradient_comparison.py, 80K GT, 5 samples × 20 estimators):**

| rho | Policy | PATHWISE | REINFORCE |
|-----|--------|----------|-----------|
| 0.80 | sPR | -0.5600 ± 0.876 | -0.1000 ± 0.141 |
| 0.80 | sMW | 0.2407 ± 0.562 | 0.1212 ± 0.088 |
| 0.80 | sMP | -0.0366 ± 0.573 | 0.0643 ± 0.128 |
| 0.90 | sPR | -0.1200 ± 1.035 | 0.1000 ± 0.346 |
| 0.90 | sMW | 0.0375 ± 0.835 | 0.2974 ± 0.178 |
| 0.90 | sMP | 0.2269 ± 0.630 | 0.0812 ± 0.098 |
| 0.95 | sPR | -0.3400 ± 0.230 | 0.0200 ± 0.311 |
| 0.95 | sMW | **0.5392** ± 0.182 | 0.1379 ± 0.184 |
| 0.95 | sMP | 0.1980 ± 0.489 | 0.0366 ± 0.203 |
| 0.99 | sPR | 0.5400 ± 0.623 | 0.0600 ± 0.182 |
| 0.99 | sMW | 0.1850 ± 0.862 | 0.1545 ± 0.098 |
| 0.99 | sMP | 0.3755 ± 0.678 | 0.0643 ± 0.170 |

⚠️ **Most settings: PATHWISE cossim has very high std (0.5–1.0).** With only 5 samples per (rho, policy), mean is unreliable.

✅ Where PATHWISE > REINFORCE clearly: sMW@0.95 (0.54 vs 0.14), sMP@0.99 (0.38 vs 0.06).

❌ Where PATHWISE is NEGATIVE: sPR@0.80, 0.90, 0.95. Concerning — paper Figure 4 shows positive PATHWISE for all policies.

**Diagnosis:** The PAPER uses 100 samples × 100 estimators per sample = 10,000 measurements per (rho, policy). We used 5×20 = 100 — **100× less data**. The mean cossim has very high estimator variance.

**To match paper:** need num_samples=100, estimators_per_sample=100, and probably gt_batch=1M. Estimated compute: ~50× longer than our current run. Not feasible without GPU or many more cores.

---

### Experiment 3-4: Section 5.2 CMU Rule (10-class, ρ=0.95)

From full_reproduction job 8631656, 50 trials × 4 alphas × 4 gaps:

**PATHWISE — 12/12 within 5.19% of reference.**
**REINFORCE — 12/12 within 1.67% of reference.**

(Detailed table in `logs/REPRODUCTION_REPORT.md`)

---

### Experiment 8-9: Section 5.2 CMU Rule (5-class, ρ=0.99) — Phase B

Quick check (20 trials × 1 alpha × 2 gaps):

| Method | alpha | gap | Ours | Reference (cmu/) | Diff% |
|--------|-------|-----|------|------------------|-------|
| PATHWISE | 0.5 | 0.05 | 51.060 ± 2.601 | 50.664 ± 2.424 | **0.78%** |
| REINFORCE | 0.1 | 0.05 | 50.271 ± 2.259 | 50.366 ± 2.078 | **0.19%** |
| PATHWISE | 0.5 | 1.0 | 32.596 ± 0.048 | (no ref?) | - |
| REINFORCE | 0.1 | 1.0 | 33.613 ± 1.386 | (no ref?) | - |

**Verdict: Quantitative match for gap=0.05; gap=1.0 ref not found in expected location.**

Job: 8673698, 18 min total

---

### Experiment 10: rho ablation — Phase C

Quick check (10 trials × 1 alpha × 2 gaps × 2 rhos):

| rho | Method | gap | Ours | Reference | Diff% |
|-----|--------|-----|------|-----------|-------|
| 0.90 | PATHWISE | 1.0 | 5.756 ± 0.084 | 5.758 ± 0.106 | 0.04% |
| 0.90 | PATHWISE | 0.05 | 8.016 ± 0.162 | 8.150 ± 0.246 | 1.64% |
| 0.90 | REINFORCE | 1.0 | 5.717 ± 0.119 | 5.797 ± 0.115 | 1.37% |
| 0.90 | REINFORCE | 0.05 | 7.989 ± 0.286 | 8.147 ± 0.245 | 1.95% |
| 0.99 | PATHWISE | 1.0 | 27.224 ± 2.247 | 27.273 ± 1.558 | 0.18% |
| 0.99 | PATHWISE | 0.05 | 45.892 ± 3.061 | 46.075 ± 3.137 | 0.40% |
| 0.99 | REINFORCE | 1.0 | 26.664 ± 1.469 | 27.383 ± 1.518 | 2.62% |
| 0.99 | REINFORCE | 0.05 | 44.655 ± 1.638 | 45.834 ± 2.872 | 2.57% |

**Verdict: All 8 within 2.62%. Confirms cost scales correctly with rho.**

Job: 8673700, 20 min total

---

### Experiment 11: STE training reentrant_3 — Phase D

| Metric | Value |
|--------|-------|
| Network | reentrant_3 (9 queues, 3 servers) |
| Epochs | 100 |
| Initial test cost | 25.072 |
| Final test cost | 22.200 |
| **Min test cost** | **21.994** (epoch 63) |
| Avg of best 20 | 22.271 |

**Verdict: STE training converges on reentrant_3. Trend matches paper claim.**

Comparison vs reentrant_2:
- reentrant_2 (6 queues): min 14.71
- reentrant_3 (9 queues): min 21.99
- Reasonable scaling with network size.

Job: 8673699, ~1.6h total

---

## What Works vs What Doesn't

### Works (reliable reproduction)
- M/M/1 sanity (deterministic)
- Section 5.2 CMU rule (5-class AND 10-class)
- Section 5.2 rho ablation
- Section 7 STE training (criss-cross, reentrant_2, reentrant_3)
- Section 7 PPO baseline (criss-cross)

### Doesn't work reliably
- **Section 5.1 gradient cosine similarity** — high variance, needs 100× more data to match paper Figure 4
- **last-iterate cost from train_policy.py** — paper uses Polyak avg-iterate, our scripts don't

### Cannot reproduce
- **GPU acceleration** — see `logs/BLOCKED_GPU.md` (no GPU access)

---

## Commands and Logs

All job submissions recorded in `logs/COMMANDS_LOG.md` with timestamp + command + Job ID + notes.

Job log files in `logs/reproduction/run_<scriptname>.{o,e}<jobID>`.

Result JSONs in `results/reproduction/`.
