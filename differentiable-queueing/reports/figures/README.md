# Reproduction Figures

Generated from cached JSON outputs in `loss/` and `results/`. To rebuild:

```bash
python reports/build_figures.py
```

Each figure has both `.png` (preview) and `.pdf` (publication) versions.

| File | Paper section | What it shows | Source data |
|---|---|---|---|
| `fig_section52_cmu_grid` | §5.2 main result | Heatmap of avg cost across (α × gap) for PATHWISE vs REINFORCE, plus % difference | `results/reproduction/reproduction_cmu_{pathwise,reinforce}.json` |
| `fig_section52_ablations` | §5.2 ablations | 4 panels: T, queue_class, num_iter, ρ — error bars on PATHWISE vs REINFORCE | `results/reproduction/{T,queue_class,num_iter,rho}_ablation_*.json` |
| `fig_section53_admission` | §5.3 | PATHWISE_B1 vs SPSA_{B10,B100,B1000} on 12 networks; SPSA collapses at K≥15 | `results/admission_control_summary.json` |
| `fig_section6_wc_vs_vanilla` | §6 | Criss-cross training curves: WC softmax (min 15.20) vs Vanilla (min 17.21) | `loss/criss_cross_bh_ppg_{softmax,vanilla}.json` |
| `fig_section7_training_curves` | §7 | 2×5 grid of test/train curves for criss-cross + reentrant_2..10, best epoch marked | `loss/<env>_ppg_softmax.json` |
| `fig_section7_min_cost_summary` | §7 | Bar chart: initial / best / final cost across all 10 networks | same as above |
| `fig_section8_theorem2` | §8 | Log-log estimator variance vs (1−ρ); fitted slope vs predicted | `results/reproduction/theorem2_validation_v2.json` |

All figures are reproducible from the JSON cache — no extra simulation needed.
