# Session Log — 2026-07-11

复现进度汇报 + 导师反馈 → 下一步任务分解。记录人：Claude（与用户/Hanming 学长核对）。

---

## 1. 本次会话做了什么

- 核对 `pkucaoyuan/differentiable-queueing` 对论文 arXiv:2409.03740（Che, Dong & Namkoong,
  *Differentiable Discrete Event Simulation for Queuing Network Control*）的复现现状。
- 产出**「我们复现 vs 原论文」并排可视化 + 报告**，用于向 Professor Hong / Professor Dong 汇报：
  - `reports/PAPER_VS_REPRO.md` — 汇报正文（各节对照表 + 记分卡 + 范围说明）
  - `reports/build_paper_vs_repro.py` — 生成脚本（用 `gptsovits` 环境 Python，含 matplotlib）
  - `reports/paper_vs_repro/` — fig1–fig5（png+pdf）+ CSV
- 起草了发给两位导师的英文汇报邮件（已由用户发出）。

## 2. 复现状态结论（一切以论文为准）

**已复现（cμ + PATHWISE 两列、Exp 版本）：**

| 论文 artifact | 复现内容 | 吻合度 |
|---|---|---|
| Table 1（Criss-Cross, Exp） | cμ + PATHWISE | ✅ ≤1.5% |
| Table 2 / Fig 14 左（Re-entrant-1, Exp） | cμ + PATHWISE，9 个规模(6–30 类) | ✅ 6–21 类 ≤5%；27/30 类偏低(训练预算截断) |
| Fig 9（学 cμ 规则, §5.2） | PW/RF 成本 + 队列序 + 4 ablation | ✅ ≤3.1% |
| Fig 11（准入控制, §5.3） | PATHWISE vs SPSA 规模化 | ✅ 定性(SPSA 15类起崩溃) |
| Fig 12（策略架构, §6） | **STE 训练器下** WC vs vanilla softmax | ✅ 定性 |
| §7 速度 | STE vs PPO 墙钟 27× | ✅ |

**关键映射：** repo 的 `reentrant_N` = 论文 Re-entrant-1 族的 **3N 个 job class**
（reentrant_2→6 … reentrant_10→30），逐行对上 Table 2。

**范围/caveat：**
- Tables 1–5 每张还含 MaxWeight/MaxPressure/Fluid/PPO-DG/**PPO-WC** 列 + HyperExp 版本 —— **未跑**（算力）。
- Re-entrant-1 最大两网（27/30 类，reentrant_9/10）long-training 失败、用了早期 checkpoint，
  cμ 基线与 STE 同步偏低（30 类：我们 cμ=70.1 vs 论文 87.7，−20%）→ 评估配置差异非学坏。

## 3. 导师反馈 → 两个新任务（2026-07 收到）

### Ask ① — gradient cosine-similarity 随 **batch size** 与 **网络拓扑** 的变化（PW & REINFORCE）
- **对应实验：** 论文 §5.1 Gradient Estimation Efficiency = `fig:gradient_comparison`（PDF 里是 **Figure 8**；
  repo 文档误标为 "Fig 4"）。代码：`experiments/gradient_comparison.py` + `experiments/reproduction/test_gradient_gpu.py`。
- **现状：部分有，且是唯一没干净复现的一块**（status.json 标 ⚠️ noisy）：
  - 拓扑轴：有 criss_cross + reentrant_2/3/4 × 3 策略（`reports/figures/fig_section51_gradient_full.csv`，21 格）
  - ρ 轴：criss_cross 有 0.8/0.9/0.95/0.99；reentrant 只有 0.95
  - **batch-size 轴：完全没扫**（只有固定 PW B=1 vs RF B=1000）
- **数据不能直接汇报，三个问题：**
  1. 我们 PW cossim 只有 0.3–0.6，论文 ≈1.0；
  2. 21 格里只有 14 格 PW>RF，论文 94.5%；
  3. **sMW 与 sMP 每行数值完全相同**（quick 脚本 bug，两策略没跑开）；GPU canonical ρ≥0.95 出 `nan`。
  - 根因：quick 版样本量/GT 轨迹数不足（我们 n=300；论文 100θ×100draw、GT=10⁶、N=1000）。
- **补法（便宜，梯度评估非 67h 级）：**
  - A. 修 sMW/sMP bug，GPU 按论文规格重跑拓扑轴（9 网 × 4 ρ），让 PW 回到 ≈1。
  - B. 新增 batch-size 扫描 B∈{1,2,5,10,50,100,1000,10000}，PW & RF 各画 cossim(B) 曲线（2–3 网 × 2 ρ）。

### Ask ② — 让 **PPO + work-conservation** 稳定跑起来
- **对应实验：** Tables 1–5 的 **PPO-WC 列** + Fig 14「大网络 PATHWISE 超过 PPO-WC」线 + Fig 12 策略参数化。
  repo open follow-up：`fig_12_ppo_variants`。代码：`PPO/train.py`(SB3) + `queuetorch/ppo.py`。
- **现状：有代码，无稳定结果。** 这正是导师说「历来折磨、耗时巨大」的部分——不是缺脚本，是**跑不稳**。
- **⚠️ 澄清：** 我们汇报的 §6 fig4（WC vs vanilla）用的是 **STE/pathwise 训练器**，**不是 SB3 PPO**。
  导师要的 PPO-WC ≠ fig4，这块仍是空的，别把 fig4 当 PPO-WC 交。
- **做法（多天工程，从小网起步）：**
  1. 只在 criss-cross 上把 PPO-WC 训到稳定收敛；难点 = WC 约束进 SB3 动作空间（空队列 action masking）。
  2. 调稳：reward norm / entropy coef / clip range / lr schedule / GAE λ / 多 seed 看方差。
  3. 稳后对上 vanilla PPO 复现 Fig 12 → scale 到大网 → 补 Fig 14 PPO-WC 线。

## 4. 下一步优先级
- **① 先做（quick win，GPU 几小时）**：补 §5.1 batch-size 扫描 + 拓扑重跑（修 sMW/sMP bug），
  正好回答导师问题 + 补上唯一没复现干净的一节。
- **② 单独排期（硬骨头，多天）**：PPO-WC 调稳。

## 5. 环境/工具备忘
- 论文 PDF：`/tmp/paper_2409.pdf`；论文 LaTeX 源：`literature/arxiv_sources/che2024differentiable/`
  （gradient_eval.tex 是 §5.1 权威描述）。
- PDF 文本抽取用 PyMuPDF(fitz)，非 poppler。
- matplotlib 只在 `gptsovits` conda env：`/home/lenovo/miniforge3/envs/gptsovits/bin/python`。
- 代码 `queuetorch/ train/ PPO/` 与作者上游 `namkoong-lab/differentiable-queueing@0c21ed7` 逐字节相同
  → 本仓是「重跑复现」，非独立重实现。
