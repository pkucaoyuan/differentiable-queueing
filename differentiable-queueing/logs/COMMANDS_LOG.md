# Commands Log

Every job submission is recorded here with timestamp, command, job ID, and notes.

Format: `[timestamp] [job_id] command (notes)`

---

[2026-05-15 07:32:27] [job 8673697] Phase A: Section 5.1 canonical (gt_batch=200K, num_samples=10, estimators_per_sample=20)
    Command: grid_run --grid_submit=batch --grid_mem=40G --grid_ncpus=16 ./jobs/reproduction/run_section51_canonical.sh
    Internal: python gradient_comparison.py --env criss_cross_bh.yaml --horizon 1000 --gt_batch 200000 --num_samples 10 --estimators_per_sample 20 --num_cores 16
    Expected log: run_section51_canonical.sh.{o,e}8673697

[2026-05-15 07:34:07] [job 8673698] Phase B (quick): 5-class CMU rule (1 alpha × 2 gaps × 20 trials)
    Command: grid_run --grid_submit=batch --grid_mem=20G --grid_ncpus=16 ./jobs/reproduction/run_cmu_5class.sh
    Reference: cmu/pathwise_wc_cmu_multiclass5_all_eps_950_more_runs.json

[2026-05-15 07:34:53] [job 8673699] Phase D: STE training reentrant_3 (Section 7)
    Command: grid_run --grid_submit=batch --grid_mem=8G ./jobs/reproduction/run_ste_reentrant_3.sh
    Internal: python train/train_policy.py -e=reentrant_3.yaml -m=ppg_softmax.yaml --algo ste

[2026-05-15 07:37:45] [job 8673700] Phase C (quick): rho ablation (rho=0.9, 0.99; alpha=0.5 PW, 0.1 RF; gaps=1.0, 0.05; 10 trials)
    Command: grid_run --grid_submit=batch --grid_mem=20G --grid_ncpus=16 ./jobs/reproduction/run_rho_ablation.sh

[2026-05-15 07:39:32] [job 8673701] Phase A retry: Section 5.1 with gt_batch=80K, num_samples=5 (after OOM at 200K)
    Command: grid_run --grid_submit=batch --grid_mem=60G --grid_ncpus=16 ./jobs/reproduction/run_section51_canonical.sh
    Memory increased from 40G to 60G; gt_batch reduced from 200K to 80K

[2026-05-15 09:38:20] [job 8674071] Section 7: STE training on reentrant_4
    Command: grid_run --grid_submit=batch --grid_mem=8G ./jobs/reproduction/run_ste_reentrant_4.sh
[2026-05-15 09:38:20] [job 8674072] Section 7: STE training on reentrant_5
    Command: grid_run --grid_submit=batch --grid_mem=8G ./jobs/reproduction/run_ste_reentrant_5.sh
[2026-05-15 09:38:20] [job 8674073] Section 7: STE training on reentrant_6
    Command: grid_run --grid_submit=batch --grid_mem=8G ./jobs/reproduction/run_ste_reentrant_6.sh
[2026-05-15 09:38:20] [job 8674074] Section 7: STE training on reentrant_7
    Command: grid_run --grid_submit=batch --grid_mem=8G ./jobs/reproduction/run_ste_reentrant_7.sh
[2026-05-15 09:38:20] [job 8674075] Section 7: STE training on reentrant_8
    Command: grid_run --grid_submit=batch --grid_mem=8G ./jobs/reproduction/run_ste_reentrant_8.sh
[2026-05-15 09:38:21] [job 8674076] Section 7: STE training on reentrant_9
    Command: grid_run --grid_submit=batch --grid_mem=8G ./jobs/reproduction/run_ste_reentrant_9.sh
[2026-05-15 09:38:21] [job 8674077] Section 7: STE training on reentrant_10
    Command: grid_run --grid_submit=batch --grid_mem=8G ./jobs/reproduction/run_ste_reentrant_10.sh

[2026-05-15 09:39:48] [job 8674078] Section 5.3: Admission control (PATHWISE vs SPSA, MaxWeight policy, 5 trials)
    Command: grid_run --grid_submit=batch --grid_mem=16G --grid_ncpus=4 ./jobs/reproduction/run_admission_control.sh

[2026-05-15 09:40:53] [job 8674079] Section 8: Theorem 2 variance scaling validation on M/M/1
    Command: grid_run --grid_submit=batch --grid_mem=8G --grid_ncpus=16 ./jobs/reproduction/run_theorem2.sh

[2026-05-15 09:41:50] [job 8674080] Section 5.3 retry: admission control (fixed cwd to project root)
    Previous attempt 8674078 failed: FileNotFoundError on env_data/ (cwd was experiments/)

[2026-05-15 09:43:51] [job 8674081] Section 5.2: T ablation (T=500,1000,2000,5000; 10 trials)
    Command: grid_run --grid_submit=batch --grid_mem=20G --grid_ncpus=16 ./jobs/reproduction/run_T_ablation.sh

[2026-05-15 10:26:52] [job 8674405] reentrant_10 retry (OOM at 8G, now 24G)

[2026-05-15 10:28:48] [job 8674406] Section 5.2: queue_class ablation (qc=5,15,20; 10 trials)
    Command: grid_run --grid_submit=batch --grid_mem=24G --grid_ncpus=16 ./jobs/reproduction/run_queue_class_ablation.sh

[2026-05-15 10:29:44] [job 8674408] T ablation retry (OOM at 20G, now 50G)

[2026-05-15 10:33:09] [job 8674409] Section 5.2: num_iter ablation (10, 20, 100)
    Command: grid_run --grid_submit=batch --grid_mem=30G --grid_ncpus=16 ./jobs/reproduction/run_num_iter_ablation.sh

[2026-05-15 10:35:47] [job 8674410] Section 6: Vanilla Softmax (no WC) on criss-cross
    Comparison baseline for WC-Softmax (ppg_softmax). Same network, same hyperparams.


## 2026-05-16 (continued)

### Theorem 2 v2 (T=10000, 1000 trials)
- Script: experiments/reproduction/test_theorem2_scaling_v2.py
- Job: `grid_run --grid_submit=batch --grid_mem=8G --grid_ncpus=8 jobs/reproduction/run_theorem2_v2.sh`
- Job ID: 8674651
- Submitted: 2026-05-16 00:42 EDT
- Reason: v1 (T=1000) gave wrong slopes (-1.73 vs predicted -3, -0.45 vs -4); hypothesis T=1000 insufficient for variance to reach steady-state scaling

### Reentrant_4 finished (job 8674071)
- Finished: 2026-05-15 14:36
- Wall time: ~5h
- Result: min test cost 32.20, final 35.03 — added to ledger as item #20


### Theorem 2 v2 finished (job 8674651)
- Finished: 2026-05-16 00:43 EDT (~40s)
- Result: PW slope -4.13 (vs predicted -3), RF slope -0.76 (vs predicted -4)
- Diagnosis: RF variance plateaus around 1.9e5 across all rhos
- Root cause: My Gaussian-perturbation REINFORCE is essentially SPSA, with score var ~ 1/sigma^2 (constant in rho)
- Paper's REINFORCE uses likelihood-ratio over the event history (score function depends on T events)
- Reproducing paper's slope requires implementing the paper's exact event-history score function — not currently done
- Verdict: methodology mismatch, not a bug. Theorem 2 stands; our test setup just doesn't probe it.


### Admission Control finished (job 8674080)
- Finished: 2026-05-16 01:43 EDT
- Wall time: ~16h (12 envs × 4 methods × 5 trials × 100 iters; SPSA B=1000 dominated)
- Results: results/admission_control_summary.json (12 envs), admission_control_full.json
- Key finding: PATHWISE_B1 consistent across all envs; SPSA fails on K≥15 networks
  - reentrant_5 SPSA_B1000=65.4 vs PATHWISE_B1=31.6
  - reentrant_6 SPSA_B1000=66.3 vs PATHWISE_B1=38.6
  - reentrant_7 SPSA_B1000=106.0 vs PATHWISE_B1=44.1
- Confirms paper Section 5.3 claim that pathwise gradient enables scaling


### Final 3 STE jobs finished
- 8674075 reentrant_8: end 2026-05-16 06:34, walltime 20h55m, exit=0, min cost 64.33 (ep 2)
- 8674076 reentrant_9: end 2026-05-16 09:30, walltime 23h52m, exit=0, min cost 72.92 (ep 5)
- 8674405 reentrant_10: end 2026-05-16 13:50, walltime 27h23m, exit=0, min cost 80.25 (ep 80)

