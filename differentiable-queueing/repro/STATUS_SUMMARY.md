# Reproduction Status Summary

Project: **differentiable-queueing**  ·  Paper: `OPRE-2025-02-1714`  ·  Updated: 2026-06-07

## Status distribution

| Status | Count | Meaning |
|---|---:|---|
| ✅ pass | 11 | Numerical match within acceptance window |
| ⚠️ noisy | 1 | High variance, compute budget too small |
| ✅⚠️ pass_with_caveat | 1 | Pass but with documented limitation |
| ✅ pass_last_iterate | 1 | Trend matches; uses last-iterate vs paper avg-iterate |
| ❌ methodology_mismatch | 1 | Our test setup doesn't exercise the paper's exact claim |
| **TOTAL** | **15** | |

## Per-artifact table

| Artifact | §Section | Status | Result / Note |
|---|---|---|---|
| `mm1_sanity` | simulator validation | ✅ pass | E[Q]=9.0282, analytical=9.0, error_pct=0.31 |
| `section_5_1_gradient_quick` | §5.1 Fig 4 | ⚠️ noisy | rerun with paper's 100x100 samples on GPU |
| `section_5_2_cmu_10class_quickgrid` | §5.2 Fig 9 right | ✅⚠️ pass_with_caveat | PW_max_diff_pct=5.19, RF_max_diff_pct=1.67, cells_OK=24/24 |
| `section_5_2_cmu_papergrid` | §5.2 Fig 9 right (paper-correct) | ✅ pass | cells=20 PATHWISE + 20 REINFORCE = 40, max_RF_PW_diff_pct=3.1, gap_0_1_added=alpha=0.5 gap=0.1: PW=14.64, RF=14.61 (diff 0.2%) |
| `section_5_2_cmu_5class` | §5.2 5-class | ✅ pass | PW_diff_pct=0.78, RF_diff_pct=0.19 |
| `section_5_2_rho_ablation` | §5.2 ρ robustness | ✅ pass | max_diff_pct=2.62, cells_OK=8/8 |
| `section_5_2_T_ablation` | §5.2 T robustness | ✅ pass | max_diff_pct=2.41, cells_OK=8/8 |
| `section_5_2_queue_class_ablation` | §5.2 queue_class robustness | ✅ pass | max_diff_pct=1.62, cells_OK=6/6 |
| `section_5_2_num_iter_ablation` | §5.2 num_iter robustness | ✅ pass | max_diff_pct=1.73, cells_OK=6/6 |
| `section_5_3_admission_control` | §5.3 Fig 11 | ✅ pass | reentrant_5={'PW': 31.59, 'SPSA_B1000': 65.44, 'ratio': 2.07}, reentrant_6={'PW': 38.59, 'SPSA_B1000': 66.31, 'ratio': 1.72}, reentrant_7={'PW': 44.08, 'SPSA_B1000': 106.02, 'ratio': 2.4} |
| `section_6_wc_vs_vanilla` | §6 | ✅ pass | WC_min=15.2, Vanilla_min=17.21, improvement_pct=13.2 |
| `section_7_ste_training` | §7 Tables 1-5 (training data) | ✅ pass_last_iterate | criss_cross=15.2, reentrant_2=14.71, reentrant_3=21.99 |
| `section_7_ste_vs_ppo_speed` | §7 STE/PPO speed comparison | ✅ pass | STE_walltime_h=2.5, PPO_walltime_h=67, speedup=27 |
| `section_4_3_1_gpu_benchmark` | §4.3.1 | ✅ pass | crossover_batch=1024, cpu_throughput_max=561K events/s @ B=1024, gpu_throughput_at_65536=47M events/s |
| `section_8_theorem_2` | §8 Theorem 2 variance scaling | ❌ methodology_mismatch |  |

## Open follow-ups

- **avg_iterate** — Add Polyak averaging to train_policy.py per paper §7 ; rerun training to produce best/last/avg checkpoints
- **section_5_1_canonical** — Run §5.1 with 100×100=10⁴ samples on GPU
- **fig_12_ppo_variants** — Compare vanilla PPO, PPO+BC, PPO-WC on criss-cross

## Blockers resolved

- **gpu_access** (2026-06-07) — researchgpu07 GPU 3 available
