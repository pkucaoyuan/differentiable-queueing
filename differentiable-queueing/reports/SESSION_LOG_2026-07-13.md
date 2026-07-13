# Session Log — 2026-07-13

导师 Ask①（§5.1 cossim 随 batch-size / 拓扑变化）执行记录。记录人：Claude。

## 交付物

- `reports/ASK1_REPORT.md` — 可汇报正文（TL;DR + 主表 + 拓扑依赖新发现）
- `reports/ASK1_CODE_ISSUES.md` — 官方代码 5 个问题详录（位置/机制/实测证据）
- `reports/ask1_figs/fig1_cossim_grid.{png,pdf}` — 论文 Fig.8 同构热图（9 网 × ρ × 3 策略 × PW/RF）
- `reports/ask1_figs/fig2_gt_reliability.{png,pdf}` — GT 可靠性（LOO vs 无 baseline）
- `reports/ask1_figs/fig3_theta_dist.{png,pdf}` — per-θ 双峰分布
- 代码：`experiments/ask1/`（common/run_cossim/validate/analyze_stage1/build_figures/pilot_benchmark）
- 数据：`results/ask1/stage1/`（criss 24 格全规格双GT）、`results/ask1/stage2quick/`（8 reentrant 网 48 格快速规格）

## 关键结论（详见 ASK1_REPORT.md）

1. 定量复现论文 Fig.8（PW≈1）不可能：官方代码 ≠ 论文装置（sMP≡sMW、缺 value baseline、图代码丢失——官方 repo 笔记自认）。criss-cross 上测得 PW 0.16–0.58 / RF 0.06–0.13，与上游内部复现吻合。
2. 无 baseline 的 REINFORCE GT 是噪声（分半自洽 0.55–0.76）；加 LOO baseline（严格无偏）修至 0.89–1.00。正式测量以 LOO-GT 为准、无 baseline GT 并行留证（用户选方案 C）。
3. 新发现：per-θ 双峰（部分 θ 的 PW=1.00，部分强负至 −0.83 = STE 方向偏差）；拓扑依赖（6–9 类网上 RF(B=1000) 追平/反超 PW，大网 PW 重新领先）。
4. 修复历史 bug：ρ≥0.95 nan（fp32 masking backward 溢出 → fp64）；sMW/sMP 同值（同代码+同 seed → 独立 seed）。
5. ρ 缩放 A/B（24 格）：sPR 上"真实强度=标签"(λ×ρ/0.9) 系统性更好，正式采用；官方 val×ρ 实际得 0.9ρ 且对 reentrant 不生效。

## 运行备忘

- 机器：8×A800-80G，96 核；**必须设 OMP_NUM_THREADS=4**（8 worker × 默认 96 线程会互踩，慢 10 倍）
- python：/opt/conda/bin/python（torch 2.6.0+cu124；matplotlib 已装）
- worker 用 setsid 脱离进程组启动，claim 目录 results/ask1/claims 支持断点续跑
- 单格耗时：criss 全规格 ~15min，15 类快速规格 ~12min，6 类快速 ~4min

## 未完成 / 下一步

1. **Stage-3 batch-size 扫描**（导师 Ask① 的 B 轴）：`run_cossim.py --stage sweep`，复用已有 GT/θ，机制就绪未跑
2. reentrant 48 格从快速规格升级全规格（100θ×100draws、GT=1e5、补 ρ 0.8/0.95）：`--stage stage2`
3. （可选）实现论文 value-function baseline，检验能否恢复 PW≈1
4. Ask②（PPO-WC 调稳）未动
