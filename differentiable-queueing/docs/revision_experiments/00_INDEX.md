# Revision Experiment Implementation Specs

Paper: OPRE-2025-02-1714 "Differentiable Discrete Event Simulation for Queuing Network Control"
Decision: Major Revision (2025-06-27)

## Documents

| File | Experiment | Reviewer | Priority |
|------|-----------|----------|----------|
| [E1_ste_bias_variance.md](E1_ste_bias_variance.md) | STE bias-variance beyond M/M/1 | AE-M1 | P0 |
| [E2_glr_comparison.md](E2_glr_comparison.md) | GLR comparison (M/M/1) | AE-M3 | P0 |
| [E3_gpu_benchmarks.md](E3_gpu_benchmarks.md) | GPU wall-clock benchmark | AE-M4, R2 | P1 |
| [E4_nonwc_criss_cross.md](E4_nonwc_criss_cross.md) | Non-work-conserving criss-cross | R1 | P1 |
| [E5_heavy_traffic.md](E5_heavy_traffic.md) | Heavy-traffic rho->1 curve | R1 | P0 |
| [E6_ablation_3way.md](E6_ablation_3way.md) | 3-way factorial ablation | R2 | P0 |
| [E7_hyperparam_sensitivity.md](E7_hyperparam_sensitivity.md) | Hyperparameter sensitivity | R2 | P1 |
| [E8_parameter_table.md](E8_parameter_table.md) | Parameter specification (writing) | R1, AE | P0 |

## Execution Schedule

- **Week 1**: E8, E3, E5, E7 (quick wins, parallel)
- **Week 2**: E1, E4, E6 (core experiments, parallel)
- **Week 3**: E2, rerun, figures
- **Week 4**: Paper writing + response letter

## Codebase Reference

```
differentiable-queueing/
  queuetorch/env.py          # Core simulator, STE at line 191
  queuetorch/policies.py     # sPR, sMW, sMP policies
  queuetorch/routing.py      # Sinkhorn routing
  queuetorch/ppo.py          # PPO buffer + loss
  train/train_policy.py      # STE training (L405-451), PPO training (L352-403)
  experiments/
    gradient_comparison.py    # Ground truth + cosine similarity framework
    cmu_rule_REINFORCE.py     # REINFORCE + value baseline CMU optimization
    cmu_rule_PATHWISE.py      # Pathwise CMU optimization
    cmu_step_rules_PATHWISE.py # Multi-axis ablation framework with build_env_config()
    admission_control.py      # SPSA + pathwise for buffer control
    step_rules.py             # Pluggable optimizers (Adam, SPSA, etc.)
  configs/env/                # YAML environment configs
  env_data/                   # NumPy arrays for network topology
  PPO/train.py                # Stable Baselines 3 PPO baseline
```
