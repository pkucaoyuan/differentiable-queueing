# Master Revision Guide

Paper: **Differentiable Discrete Event Simulation for Queuing Network Control**
Journal: Operations Research (OPRE-2025-02-1714)
Decision: Major Revision (2025-06-27)
Deadline: 2026-06-27 (12 months from decision)

---

## 1. Reviewer Verdict Summary

| | Tone | Core Concern | Key Quote |
|---|------|-------------|-----------|
| **AE** | Positive, constructive | Theory too thin; methods borrowed from ML | "potential for significant impact on the simulation literature" |
| **Referee 1** | Positive | Missing experiment details; extend to non-WC and heavy traffic | "well-written, and I enjoyed reading it" |
| **Referee 2** | Strong positive | Need ablation to disentangle improvements; GPU claims unsupported | "a strong paper that ultimately merits publication" |

**Bottom line**: Nobody questioned the method's validity. All concerns are "show more evidence" and "position better in literature." This is a winnable revision.

---

## 2. Complete Reviewer Comment Tracker

### AE Major Comments

| # | Comment | Response Strategy | Experiment | Writing |
|---|---------|-------------------|------------|---------|
| M1 | Position method as "generalized IPA"; extend bias-variance analysis beyond M/M/1 | Extended numerical bias-variance study across 4 environments, 5 horizons, 5 temperatures | E1 | Rewrite Section 4.2 intro to connect STE↔IPA explicitly |
| M2 | Gap between finite-horizon theory and long-run average cost practice | Discuss convergence of $J_N/N$; E5 heavy-traffic validates long-horizon behavior | E5 | Add paragraph in Section 4 or Appendix |
| M3 | Misattributed GLR; need comparative analysis with Peng et al. 2018 | Implement GLR on M/M/1; textual argument for scalability advantage | E2 | Fix citation [39]; add GLR discussion subsection |
| M4 | Explain how STE enables PyTorch auto-diff and GPU parallelism | Profile all ops; GPU vs CPU benchmarks; document no custom kernels | E3 | Add implementation details subsection |

### AE Minor Comments

| # | Comment | Fix | Location |
|---|---------|-----|----------|
| m1 | [39] is classic LR, not GLR | Change to "likelihood-ratio estimation [39]"; add Peng et al. 2018 cite | p6 para1 |
| m2 | IPA limitation unclear — is it because $f$ must be differentiable? | Add sentence: "IPA requires the transition function $f$ to be differentiable in $\theta$, which fails for queueing networks due to the argmin event selection." | p12 |
| m3 | Difference between fractional deterministic policy vs randomized policy? | Add clarifying paragraph: during training, $u = \pi_\theta(x) \in \bar{\mathcal{U}}$ is deterministic and fractional; during evaluation, $u \sim \pi_\theta(x)$ is sampled (stochastic, discrete). WC-Softmax in Section 6 is randomized. | p14 |
| m4 | Duplicate "with" in Proposition 2 | Delete one "with" | p26 |
| m5 | "a infinitesimal" | → "an infinitesimal" | p33 L38 |
| m6 | Inconsistent "reinforcement learning" / "RL" | Use "RL" consistently after first definition | Global |
| m7 | Inconsistent "discrete event" / "discrete-event" | Standardize to "discrete-event" (hyphenated) | Global |
| m8 | Core method (Section 4.2) delayed to page 14; Sections 1-3 too long | Shorten Sections 1-2; streamline Section 3 to keep only notation used in Section 4 | Structural |
| m9 | Section 6 (WC-Softmax) not central to gradient estimation theme | Consider moving to appendix (but see R1's comment that it's useful) | Structural |
| m10 | "non-stationary" mentioned in Abstract/Intro but never discussed | Either remove claim OR add brief experiment with time-varying arrivals (configs exist: `lam_type: step/sawtooth`) | Abstract, Intro |

### Referee 1 Comments

| # | Comment | Response Strategy | Experiment | Writing |
|---|---------|-------------------|------------|---------|
| 1 | Methods not new (fractional allocation standard; STE from Bengio 2013) | Acknowledge; emphasize novelty is in the APPLICATION to DES + queueing + the empirical demonstration at scale | — | Reposition in Intro/Related Work |
| 2a | Missing experiment parameters (λ, μ, h, γ, baseline) | Extract all parameters from configs into appendix tables | E8 | Appendix Tables A1-A3 |
| 2b | Non-work-conserving policies optimal in criss-cross IIB/IID (Martins 1996, Budhiraja 2017) | Train with and without WC constraint; show method discovers idling | E4 | New subsection in experiments |
| 2c | Heavy-traffic regime: how does performance degrade as ρ→1? | Sweep ρ from 0.80 to 0.99; validate Theorem 2 | E5 | New figure + discussion |
| 2d | Compare with heavy-traffic diffusion control (Ata et al. 2024, Ata & Kaşıkaralar 2023) | Qualitative discussion; note different modeling paradigm (Brownian motion vs discrete simulation) | — | 1-2 paragraphs in Related Work |
| 3 | Move criss-cross example earlier (before Figs 1-2) | Restructure Section 3 | — | Structural edit |

### Referee 2 Comments

| # | Comment | Response Strategy | Experiment | Writing |
|---|---------|-------------------|------------|---------|
| GPU | GPU parallelism underdeveloped; open-source simulator; wall-clock benchmarks | Benchmarks + code release | E3 | Subsection + GitHub link |
| Ablation | 3-way confound: continuous/deterministic/first-order. Need ablation. | 6-combination factorial experiment | E6 | New ablation table + figure |
| Hyperparams | Sensitivity to hyperparameters — likely a strength of PATHWISE | Sweep 5 axes; compare PATHWISE vs REINFORCE robustness | E7 | Appendix figure panels |
| NormSGD | Why normalized SGD? Why cosine similarity not Euclidean distance? | Justify: cosine similarity measures directional accuracy, which matters more than magnitude for SGD (magnitude adjusted by learning rate). Normalized SGD ensures fair comparison across methods with different gradient scales. | — | 1 paragraph in Section 5.1 |
| Section 8 | How does M/M/1 theory relate to actual STE method? Is pathwise estimator = SmoothBackprop? | Clarify: M/M/1 uses IPA (Lindley recursion), which is the special case of PATHWISE when dynamics are differentiable. General PATHWISE adds STE for non-differentiable dynamics. | — | Add clarifying paragraph in Section 8 |
| Notation | Eq (5) subscript confusion; Eq (Vanilla Softmax) index override; p33 L3 missing text | Fix notation | p21, p25, p33 |

---

## 3. Execution Roadmap

### Phase 0: Immediate Writing Fixes (Day 1-2)

Zero compute. Pure text editing on the LaTeX source.

```
□ Fix citation [39]: "generalized" → "classic" LR; add Peng et al. 2018    (AE m1)
□ Fix "a infinitesimal" → "an infinitesimal"                                (AE m5)
□ Fix duplicate "with" in Proposition 2                                      (AE m4)
□ Standardize "RL" after first definition                                    (AE m6)
□ Standardize "discrete-event" (hyphenated)                                  (AE m7)
□ Fix Eq (5) subscript notation                                              (R2)
□ Fix Eq (Vanilla Softmax) index override                                    (R2)
□ Fix p33 L3 missing sentence                                               (R2)
□ Add IPA limitation clarification on p12                                    (AE m2)
□ Add fractional vs randomized policy clarification on p14                   (AE m3)
```

### Phase 1: Quick Wins (Week 1)

Run in parallel. All independent.

```
□ E8: Extract parameter tables from configs/env_data     → Appendix Tables
□ E5: Heavy-traffic curve (extend existing ablation code) → Figure + Table
□ E7: Hyperparameter sensitivity (extend existing ablation) → 4-panel Figure
□ E3: GPU benchmarks (requires env.py modification first) → Table + Figure
□ Non-stationary demo: run existing reentrant_varying config → 1 Appendix figure
```

**Code modifications needed before E3**:
- `queuetorch/env.py`: Add `torch.distributions` option in `draw_service()`/`draw_inter_arrivals()` for GPU-native random sampling

### Phase 2: Core New Experiments (Week 2)

Run in parallel. E1 is compute-heaviest.

```
□ E1: STE bias-variance analysis (extend gradient_comparison.py)  → 2 Figures + Table
□ E6: 3-way ablation (6 methods, extend cmu_rule framework)       → Table + Bar Chart
□ E4: Non-WC criss-cross (requires policy architecture change)    → Table + 2 Figures
```

**Code modifications needed before E4**:
- `train/train_policy.py`: Add `--work_conserving` flag
- Create `configs/env/criss_cross_IIB.yaml` and `criss_cross_IID.yaml` with Martins et al. parameters

### Phase 3: GLR + Polish (Week 3)

```
□ E2: GLR comparison on M/M/1 (new script, closed-form GLR)      → Table + Figure
□ Rerun any experiments that need more trials
□ Generate all final figures (matplotlib/seaborn)
□ Compile all results into LaTeX tables
```

### Phase 4: Paper Rewrite (Week 4)

```
□ Restructure Sections 1-3 (shorten; move criss-cross example earlier)
□ Add new subsections for E1-E7 results
□ Update Related Work (GLR, heavy-traffic diffusion control references)
□ Update Conclusion (non-WC capability, heavy-traffic validation, robustness)
□ Consider moving Section 6 (WC-Softmax) to appendix (AE m9)
□ Write point-by-point response letter
```

---

## 4. New Paper Sections / Figures Map

### Main Paper Additions

| Location | Content | Source Experiment |
|----------|---------|-------------------|
| Section 4.2 (after STE intro) | "Connection to IPA" paragraph: PATHWISE = generalized IPA; STE handles non-diff sample paths | E2 text |
| Section 4.2 (after Theorem 1) | Extended bias-variance: "These results extend beyond M/M/1..." + reference to appendix | E1 |
| Section 5.1 (new subsection) | "Comparison with GLR" — M/M/1 results + scalability discussion | E2 |
| Section 5.1 (add paragraph) | Justify cosine similarity and normalized SGD | R2 response |
| Section 5.2 (add figure) | Heavy-traffic performance curve (rho vs normalized cost) | E5 |
| Section 5 (new subsection) | Ablation: "Disentangling the Sources of Improvement" | E6 |
| Section 6 or Appendix | Non-WC criss-cross IIB/IID results | E4 |
| Section 8 (add paragraph) | Clarify IPA↔PATHWISE relationship in M/M/1 case study | R2 response |
| Related Work | GLR (Peng et al. 2018); heavy-traffic diffusion control (Ata et al.) | AE M3, R1 |

### Appendix Additions

| Location | Content | Source |
|----------|---------|-------|
| Appendix (new) | Table A1: Environment parameters | E8 |
| Appendix (new) | Table A2: Training hyperparameters | E8 |
| Appendix (new) | Table A3: REINFORCE baseline details | E8 |
| Appendix (new) | Extended bias-variance figures (all envs, all T, all beta) | E1 |
| Appendix (new) | GPU benchmark table and speedup plot | E3 |
| Appendix (new) | Hyperparameter sensitivity panels | E7 |
| Appendix (new) | Non-stationary arrivals demo figure | Phase 1 |

### New Figures Inventory

| Fig # | Content | Experiment | Paper / Appendix |
|-------|---------|------------|-----------------|
| F-new1 | Bias and variance vs horizon T (4 environments) | E1 | Appendix |
| F-new2 | Bias and variance vs temperature beta (4 environments) | E1 | Paper (extends Fig 5) |
| F-new3 | GLR vs PATHWISE vs REINFORCE MSE curves on M/M/1 | E2 | Paper |
| F-new4 | GPU speedup vs batch size | E3 | Appendix |
| F-new5 | Heavy-traffic: normalized cost vs rho | E5 | Paper |
| F-new6 | Ablation bar chart (6 methods) | E6 | Paper |
| F-new7 | Hyperparameter sensitivity (4 panels) | E7 | Appendix |
| F-new8 | Non-WC policy heatmaps (criss-cross IIB/IID) | E4 | Paper or Appendix |
| F-new9 | Non-WC learning curves | E4 | Paper or Appendix |
| F-new10 | Non-stationary arrivals demo | Phase 1 | Appendix |

---

## 5. Response Letter Structure

```
Dear Editor and Reviewers,

We thank the AE and the two referees for their detailed and constructive feedback.
Below we address each comment point by point. All changes are highlighted in blue
in the revised manuscript.

============================================================
RESPONSE TO AE
============================================================

Major Comment 1: [Connection to IPA / Extended bias-variance analysis]
> "The authors should develop a stronger theoretical framework..."

We thank the AE for this suggestion. We have made two changes:
(1) Added a paragraph in Section 4.2 explicitly connecting PATHWISE to IPA...
(2) Extended the bias-variance analysis to 4 network types, 5 horizons...
[Reference: new Figure X, new Table Y in Appendix Z]

Major Comment 2: [Finite-horizon vs long-run average cost]
> "A theoretical gap exists..."

We acknowledge this gap. We have added a discussion in Section X...
Additionally, our new heavy-traffic experiment (Figure Y) demonstrates...

Major Comment 3: [GLR comparison]
> "The paper's treatment of likelihood ratio methods requires..."

We have corrected the citation and added a comparison with GLR on M/M/1...
[Reference: new Table X, new Figure Y]

Major Comment 4: [Auto-diff and GPU]
> "The statement on page 17 regarding SmoothBackprop..."

We have added a detailed breakdown of all operations in env.step()...
[Reference: new Table X in Appendix, GPU benchmark in Appendix Figure Y]

Minor Comments 1-10: [See detailed fixes listed in revision]

============================================================
RESPONSE TO REFEREE 1
============================================================

Comment 1: [Novelty positioning]
> "The gradient estimation methods proposed in this paper are not new..."

We agree that the individual techniques are not new. Our contribution is...

Comment 2a: [Experiment parameters]
> "I am surprised that the parameters are not specified..."

We apologize for this omission. All parameters are now listed in Appendix Tables A1-A3.

Comment 2b: [Non-work-conserving policies]
> "Non-work-conserving policies can also be optimal..."

Excellent suggestion. We have conducted experiments on criss-cross Cases IIB and IID...
[Reference: new Figure X, Table Y]

Comment 2c: [Heavy traffic]
> "The theoretical analysis suggests that the heavy-traffic regime..."

We now include a comprehensive heavy-traffic performance curve...
[Reference: new Figure X]

Comment 2d: [Diffusion control comparison]
> "Several recent works have explored diffusion control..."

We have added a discussion in Related Work comparing our approach with...

Comment 3: [Criss-cross example ordering]
> "I recommend moving the criss-cross example earlier..."

Done. The criss-cross example now appears before Figures 1 and 2.

============================================================
RESPONSE TO REFEREE 2
============================================================

GPU Parallelism:
> "GPU parallelism is mentioned early on in the abstract..."

We have clarified that GPU parallelizability is a feature of the simulator...
[Reference: GPU benchmarks in Appendix, simulator released at [URL]]

Ablation:
> "The proposed method actually introduces several changes..."

We now include a systematic 6-combination ablation study...
[Reference: new Table X, Figure Y]

Hyperparameters:
> "It would be useful to see how sensitive..."

We have added hyperparameter sensitivity analysis...
[Reference: Appendix Figure X]

Normalized SGD / Cosine Similarity:
> "I don't quite understand why normalized gradient descent..."

We use cosine similarity because... Normalized SGD ensures...

Section 8:
> "I'm confused about how this section relates..."

We have added a clarifying paragraph...

Notation Fixes:
> [Specific fixes listed]

Done. See revised equations (5), (Vanilla Softmax), and page 33.
```

---

## 6. Structural Revision Plan

The AE (m8) and both referees suggest tightening the paper. Here is the proposed restructured outline:

### Current Structure (35+ pages)
```
1. Introduction (5 pages)
2. Related Work (2 pages)  
3. Model: DES for Queueing Networks (7 pages) ← AE says too long
4. Gradient Estimation (7 pages) ← core contribution, keep
5. Empirical Evaluation (5 pages) ← add new results
6. Policy Parameterization (2 pages) ← AE says consider moving to appendix
7. Benchmarks (3 pages) ← add new results
8. Theoretical Case Study M/M/1 (3 pages)
9. Conclusion (1 page)
```

### Proposed Revised Structure
```
1. Introduction (4 pages) — shortened; move some motivation to Related Work
2. Related Work (2.5 pages) — add GLR, diffusion control; consolidate
3. Model (5 pages) — streamlined; criss-cross example moved before Fig 1-2
   - Remove commented-out DEDS general formalism
   - Keep only notation used in Sections 4-7
4. Gradient Estimation (8 pages) — add IPA connection, GLR comparison
   4.1 REINFORCE (existing)
   4.2 Our approach (existing + IPA positioning)
   4.3 Bias-variance analysis (existing + E1 extended results)
   4.4 Comparison with GLR [NEW] (E2)
5. Empirical Evaluation (7 pages) — add E5, E6, E7
   5.1 Gradient estimation quality (existing + E1 cross-reference)
   5.2 Learning the cmu rule (existing + E5 heavy-traffic figure)
   5.3 Admission control (existing)
   5.4 Ablation study [NEW] (E6)
6. Policy Optimization Benchmarks (4 pages) — merge current Sections 6+7
   6.1 Work-conserving softmax (shortened from current Section 6)
   6.2 Benchmark results (current Section 7)
   6.3 Non-work-conserving policies [NEW] (E4)
7. Theoretical Case Study (3 pages) — existing + clarification per R2
8. Conclusion (1 page) — updated
Appendix: Parameters (E8), GPU benchmarks (E3), Hyperparameter sensitivity (E7),
          Extended bias-variance (E1), Non-stationary demo
```

**Net change**: ~+5 pages new content, -3 pages from streamlining = ~+2 pages total.

---

## 7. Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Martins et al. (1996) IIB/IID parameters unclear | E4 shows no non-WC improvement | Read paper carefully; also try Budhiraja et al. 2017 parameters |
| GLR implementation on M/M/1 has subtle correctness issues | E2 results unreliable | Validate against analytical gradient; cross-check with IPA |
| GPU benchmarks show minimal speedup at small batch | Undermines GPU claim | Emphasize batch=1000+ regime; honest about small-batch limitations |
| 3-way ablation: some combinations unstable (e.g., Gumbel-STE) | Missing rows in table | Report honestly; instability is itself informative |
| Bias grows with horizon T in E1 | Undermines STE reliability | Report honestly; contrast with much larger REINFORCE variance. Bias/variance tradeoff is the point |
| AE not satisfied with M/M/1-only GLR comparison | Requests general network GLR | Argue scalability: GLR requires per-network derivation, which is the fundamental limitation |

---

## 8. Checklist Before Submission

```
EXPERIMENTS
□ All 8 experiments completed with results saved as JSON
□ All figures generated (matplotlib/seaborn, publication quality)
□ All tables compiled in LaTeX
□ Sanity checks passed (existing results reproduced)

PAPER
□ All AE major comments addressed (M1-M4)
□ All AE minor comments fixed (m1-m10)
□ All Referee 1 comments addressed (1, 2a-2d, 3)
□ All Referee 2 comments addressed (GPU, Ablation, Hyperparams, NormSGD, Section 8, Notation)
□ Response letter complete with cross-references to revised text
□ Changes highlighted in blue in revised manuscript
□ Bibliography updated (Peng et al. 2018, Ata et al. 2024, Suh et al. 2022, etc.)

CODE
□ Simulator open-sourced (Referee 2 request)
□ Experiment scripts included for reproducibility
□ README updated with instructions

FORMAT
□ INFORMS Operations Research LaTeX style
□ Page limit respected (or justified)
□ All figures have proper captions and are referenced in text
□ Supplementary material properly organized
```
