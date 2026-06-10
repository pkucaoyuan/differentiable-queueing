# How to Verify These Reproduction Results

A guide for collaborators / senior students who want to independently check the
reproduction conclusions of OPRE-2025-02-1714 ("Differentiable Discrete Event
Simulation for Queuing Network Control" by Che, Dong, Namkoong).

If you have **5 minutes** → read §1 + look at one figure.
If you have **30 minutes** → §1 + spot-check 2-3 cells (§3).
If you have **a few hours** → §1 + re-run one experiment (§5).

---

## §1. Headline conclusions to verify

| Claim | Files to inspect | Pass criterion |
|---|---|---|
| **C1.** §5.2 CμRule: PATHWISE / REINFORCE both converge to cμ-optimal cost on 10-class network | `results/reproduction/cmu_papergrid_*.json`, `reports/figures/fig_section52_cmu_papergrid.png` | All 20 PATHWISE / 20 REINFORCE cells differ by ≤5% from one another and ≤7% from paper reference in `cmu/pathwise_results_10_0.95.json` |
| **C2.** §5.3 Admission: PATHWISE scales to K=21, SPSA collapses at K≥15 | `results/admission_control_summary.json`, `reports/figures/fig_section53_admission.png` | reentrant_7: PATHWISE_B1=44.1, SPSA_B1000=106.0 (ratio ≈2.4×) |
| **C3.** §6: WC softmax beats vanilla softmax on criss-cross | `loss/criss_cross_bh_ppg_*.json`, `reports/figures/fig_section6_wc_vs_vanilla.png` | min cost WC=15.20 < Vanilla=17.21 (≥10% improvement) |
| **C4.** §7 Tables 1-5: STE matches or beats cμ baseline on majority of envs | `results/reproduction/ste_vs_cmu_benchmark.json`, `reports/figures/fig_section7_ste_vs_cmu.png` | STE wins on 7/10, ties on 3, loses on 0; mean improvement +3.4% |
| **C5.** §7 STE ≈27× faster than PPO | `logs/COMMANDS_LOG.md`, SGE `qacct` records (Job IDs 8556850 & 8556856) | STE walltime / PPO walltime ≈ 1/27 |
| **C6.** §4.3.1 GPU 84× speedup at large batch | `results/reproduction/gpu_benchmark{,_large}.json`, `reports/figures/fig_section4_3_1_gpu_benchmark.png` | GPU throughput @ B=65536: 47M ev/s vs CPU @ B=1024: 0.56M ev/s |

If those all check out, the paper's empirical narrative reproduces.

---

## §2. Verify the code we ran is the paper's code

The driver scripts under `experiments/reproduction/` are wrappers that import
the upstream library code unmodified. Confirm with:

```bash
# Clone upstream (requires collaborator access to the private namkoong-lab repo)
git clone https://github.com/namkoong-lab/differentiable-queueing.git /tmp/upstream

# Compare core library files — should be IDENTICAL
diff -q queuetorch/policies.py     /tmp/upstream/queuetorch/policies.py
diff -q queuetorch/routing.py      /tmp/upstream/queuetorch/routing.py
diff -q queuetorch/ppo.py          /tmp/upstream/queuetorch/ppo.py
diff -q train/train_policy.py      /tmp/upstream/train/train_policy.py
diff -q experiments/cmu_step_rules_PATHWISE.py   /tmp/upstream/experiments/cmu_step_rules_PATHWISE.py
diff -q experiments/cmu_rule_REINFORCE.py        /tmp/upstream/experiments/cmu_rule_REINFORCE.py
diff -q experiments/admission_control.py         /tmp/upstream/experiments/admission_control.py
diff -q experiments/gradient_comparison.py       /tmp/upstream/experiments/gradient_comparison.py
diff -q configs/env/criss_cross_bh.yaml          /tmp/upstream/configs/env/criss_cross_bh.yaml
# (every diff above should print nothing)

# Initial commit of our fork (937ac2f) matches upstream HEAD 0c21ed7
diff -rq <(git -C $(pwd) archive 937ac2f | tar -t) /tmp/upstream  # may show only docs/, literature/ as ours-only
```

The **only** source-code modification in this fork is `queuetorch/env.py` (added
a `gpu_native_sampling` branch behind `device.type == 'cuda'`). All CPU runs
hit the unchanged code path.

---

## §3. Verify experiments actually ran on the cluster (no fakes)

Every numerical claim corresponds to a real SGE job whose accounting record
lives on the CBS Grid master.

```bash
source /opt/n1ge/default/common/settings.sh

# These are independent server-side records — exit_status=0, walltime, hostname
# We don't write them; they cannot be fabricated locally.
qacct -j 8631656   # §5.2 full CμRule reproduction (5.8h walltime, exit=0)
qacct -j 8674071   # §7 STE training reentrant_4   (5h walltime, exit=0)
qacct -j 8674080   # §5.3 admission control        (16h walltime, exit=0)
qacct -j 8674408   # §5.2 T ablation               (32m walltime, exit=0)
```

`logs/COMMANDS_LOG.md` lists every `grid_run` submission with its Job ID and
timestamp.

---

## §4. Verify the figures match the JSON

Every figure under `reports/figures/` has a same-name CSV with the underlying
numbers. To rebuild any figure from raw data:

```bash
python reports/build_figures.py             # core §5/§6/§7/§8 figures
python reports/build_fig_papergrid.py       # §5.2 paper-grid heatmap
python reports/build_fig_ste_vs_cmu.py      # §7 STE vs cμ
python reports/build_fig_heavy_traffic.py   # E5 heavy traffic
python reports/build_fig_benchmark_summary.py # 6-panel summary
python reports/export_csv.py                # regenerate same-name CSVs
```

After running, all PNG/PDF and CSV outputs should be byte-identical to what's
committed (modulo matplotlib version skew on text rendering).

Spot-check example:

```bash
# JSON ↔ stdout cross-check for §5.2 T=2000 PATHWISE
python -c "
import json, statistics
d = json.load(open('results/reproduction/T_ablation_pathwise.json'))
costs = [r['avg_cost'] for r in d['T_2000']['0.5']['0.5']]
print(f'mean = {statistics.mean(costs):.4f}')   # should print 11.441
"
grep '\[OK\]' logs/reproduction/run_T_ablation.sh.o8674408 | head -1
# Expected: "T= 500   PATHWISE: Ours 11.507 | Ref 11.629 | 1.05% [OK]"
```

---

## §5. Re-run any experiment in under an hour

Each driver script runs standalone:

```bash
# Set up env (one-time)
conda create -n queuetorch python=3.11 numpy=1.26.4 scipy=1.11.4 cvxpy=1.4.2 -c conda-forge
conda activate queuetorch
pip install torch==2.2.0 pyyaml tqdm pathos pandas matplotlib

# Re-run examples (in order of increasing time):
python experiments/reproduction/test_mm1.py                    #  ~1 min
python experiments/reproduction/test_queue_class_ablation.py   #  ~9 min on 16 cores
python experiments/reproduction/test_num_iter_ablation.py      # ~13 min
python experiments/reproduction/test_T_ablation.py             # ~32 min
python experiments/reproduction/test_cmu_5class.py             # ~18 min
python experiments/reproduction/test_cmu_baseline.py           # ~17 min  (§7 STE vs cμ)
# (set NSLOTS env var to specify the number of CPU workers)
```

Compare your numbers to ours in the JSON; should agree within ±2% (multiprocess
nondeterminism + seed-dependent variance).

---

## §6. Critical caveat to verify (the eval protocol fix)

The §7 STE-vs-cμ benchmark initially used `argmax` for eval. This **wildly
underestimates** STE on harder envs because the trained network's softmax
output is not yet peaked enough for argmax to be representative.

**Correct protocol** (matches `train_policy.py:264-285` default
`test_policy='softmax'`, `randomize=True`):

```python
pr = net(queues, time)
pr = pr * dq.network
pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))
pr += 1*torch.all(pr == 0., dim=2).reshape(B, s, 1).repeat(1, 1, q) * dq.network
pr /= torch.sum(pr, dim=-1).reshape(B, s, 1)
action = one_hot_sample.OneHotCategorical(probs=pr).sample()
```

Self-check this is the protocol being used:

```bash
grep -A2 "OneHotCategorical" experiments/reproduction/test_cmu_baseline.py
# Expected: shows softmax sample inside eval_ste, NOT argmax
```

The difference is dramatic — for `reentrant_2 best epoch 13`:
- argmax protocol: STE cost = **35.95** → "STE catastrophically loses to cμ"
- softmax+sample protocol: STE cost = **15.95** → "STE matches training-time test_loss (14.71); beats cμ (17.79) by 7%"

This was caught by re-running the same checkpoint under both protocols
(`/tmp/test_eval_protocols.py`).

---

## §7. What to be skeptical about

| Conclusion | How strongly verified | What I'd ask if I were the reviewer |
|---|---|---|
| **C1-C6 above** | Strong — multiple independent SGE jobs, JSON + stdout + qacct cross-check | "Did you use the paper's eval horizon (T=200K)?" — answer: we used T=50K (4× shorter, results still hold) |
| §5.1 cossim heatmap | **Weak** — quick run with 100× less data than paper | "Why no full Figure 4?" — CPU budget; needs paper's 10⁴ samples × 12 cells |
| §6 Figure 12 PPO 3 variants | Only 1/3 done (vanilla PPO killed for time) | We have STE-WC vs STE-Vanilla; not full PPO comparison |
| §7 PPO column in Tables 1-5 | **Missing** entirely | PPO 67h/env × 11 envs = 700+ hours; out of compute budget |
| §8 Theorem 2 numerical | ❌ Doesn't match | Our REINFORCE is Gaussian perturb, not the paper's likelihood-ratio score function. Theorem stands mathematically. |

---

## §8. One-glance summary

```
✅ Paper main empirical claim (PATHWISE beats baselines on real networks): REPRODUCED
   ├── §5.2 CμRule benchmark              (Figure 9 right)
   ├── §5.3 admission scaling             (Figure 11)
   ├── §6 WC > Vanilla                    (partial; STE not PPO)
   ├── §7 Tables 1-5 STE vs cμ            (7/10 wins, 3/10 ties, 0/10 losses)
   ├── §7 STE 27× faster than PPO         (confirmed via walltime)
   └── §4.3.1 GPU 84× speedup             (confirmed on A100)

⚠️ Paper auxiliary figures: PARTIAL
   ├── §5.1 cossim Figure 4               (defensible quick check; canonical missing)
   ├── §5.3 small admission Figure 10     (subsumed by Figure 11)
   └── §6 Figure 12 PPO variants          (1/3 done)

❌ Missing from this reproduction:
   ├── §7 PPO column in Tables 1-5        (compute budget)
   └── §8 Theorem 2 numerical (our impl differs from paper's)
```

---

## §9. If you find a discrepancy

Numbers in the JSON files should reproduce within ±2% modulo multiprocessing
nondeterminism. If you find a larger gap:

1. Check you're using the **same env config** (`configs/env/<env>.yaml`) and
   **same seeds** (`cmu/seeds_cmu_*.json`) we did.
2. Check `NSLOTS` env var matches (we used 16 cores typically).
3. Verify the eval protocol — softmax+sample, not argmax (see §6 above).
4. If still off, please open an issue with: env name, alpha/gap setting, your
   number, our number, and your `pip freeze` output.

---

## §10. Files at a glance

- **`INDEX.md`** — full file map per paper section
- **`PROFESSOR_PACKAGE.md`** — TL;DR summary
- **`REPRODUCTION_LEDGER.md`** — 25-entry per-experiment status
- **`repro/status.json`** + **`STATUS_SUMMARY.md`** — machine-readable artifact contract
- **`logs/COMMANDS_LOG.md`** — every cluster submission with Job ID
- **`reports/figures/fig_benchmark_summary.png`** — single-glance overview
