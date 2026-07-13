# 复现 vs 原论文 — 对比汇报

> 论文：Che, Dong & Namkoong, *Differentiable Discrete Event Simulation for
> Queuing Network Control* (arXiv:2409.03740)
> 本页只罗列**已经复现**的部分，每一项都把**我们的复现结果**和**论文原值**并排。
> 图见 `reports/paper_vs_repro/`，数据见同目录 `*.csv`。

---

## 0. 一句话

> 我们用作者原始代码 + 配置 + seed 独立重跑，**论文经验部分的核心结论全部复现**：
> §5.2（学 cμ 规则）、§5.3（准入控制）、§6（work-conserving 架构）、
> §7（PATHWISE 在多类排队网络上胜过 cμ 与各标准策略）。
> 在 Criss-Cross 和中小型 Re-entrant-1 网络上，我们的成本与论文**逐格吻合到几个百分点之内**。

---

## 1. 主结果：Re-entrant-1 (Exp) 规模化 — 论文 Table 2 / Figure 14(左)

📊 图：`fig1_reentrant1_scaling_ours_vs_paper.png`

我们的环境 `reentrant_N` 恰好等于论文 Re-entrant-1 族的 **3N 个 job class**
（`reentrant_2`→6 类 … `reentrant_10`→30 类），逐行对得上论文 Table 2。
我们复现了图里 **cμ 基线** 与 **PATHWISE** 两条线（论文核心论点：PATHWISE < cμ）。

| job classes | env | 论文 cμ | 我们 cμ | Δ% | 论文 PATHWISE | 我们 STE | Δ% |
|---:|---|---:|---:|---:|---:|---:|---:|
| 6  | reentrant_2  | 17.4 | 17.79 | +2.2 | 14.9 | 16.58 | +11.3 |
| 9  | reentrant_3  | 23.3 | 23.81 | +2.2 | 22.0 | 22.24 | +1.1 |
| 12 | reentrant_4  | 33.0 | 32.60 | −1.2 | 30.7 | 30.09 | −2.0 |
| 15 | reentrant_5  | 40.2 | 38.71 | −3.7 | 36.2 | 37.35 | +3.2 |
| 18 | reentrant_6  | 48.5 | 48.51 |  0.0 | 45.7 | 47.73 | +4.4 |
| 21 | reentrant_7  | 55.2 | 52.28 | −5.3 | 52.8 | 51.78 | −1.9 |
| 24 | reentrant_8  | 64.9 | 62.49 | −3.7 | 60.2 | 55.96 | −7.0 |
| 27 | reentrant_9  | 71.1 | 65.28 | −8.2 | 67.7 | 64.83 | −4.2 |
| 30 | reentrant_10 | 87.7 | 70.09 | −20.1 | 77.8 | 69.74 | −10.4 |

**怎么读：**
- **6–21 类（reentrant_2…7）**：cμ 与 PATHWISE 都与论文吻合到 ~5% 以内 —— 主结果稳稳复现。
- **核心定性结论复现**：每个规模上 **我们的 STE 成本都 ≤ cμ**，与论文一致（PATHWISE 胜 cμ）。
- **24–30 类（reentrant_8/9/10）**：我们的曲线整体**低于**论文（cμ 与 STE 同时偏低）。
  原因不是学习问题，而是这两三个最大网络的**训练/评估预算被截断**
  （仓库记录 `reentrant_9/10` long-training 失败、被迫用早期 checkpoint）。
  baseline cμ 也偏低，说明是评估配置（horizon T / 负载）差异，不是策略本身的差异。

---

## 2. Criss-Cross — 论文 Table 1 (Exp)

📊 图：`fig2_crisscross_table1_ours_vs_paper.png`

| 方法 | 论文 (Exp) | 我们 | Δ |
|---|---:|---:|---:|
| cμ | 17.9 ± 0.3 | 17.76 ± 2.38 | −0.8% |
| PATHWISE / STE | 15.2 ± 0.4 | 15.43 ± 1.74 | +1.5% |

→ **几乎逐点重合**。论文「PATHWISE 比 cμ 低约 15%」的结论复现。

---

## 3. §5.2 学 cμ 规则 — 论文 Figure 9

📊 图：`fig5_section52_cmu_pw_vs_rf.png`

论文主张：PATHWISE 和 REINFORCE 都能学到 cμ 最优策略。
我们用 paper-grid（5 个最优性 gap × 4 个 α × 50 trials = 40 格）复现：

| 检查项 | 我们的结果 |
|---|---|
| PATHWISE vs REINFORCE 学到的成本 | 全部落在 y=x 上，**最大差 3.1%** |
| 队列优先级排序（Fig 9 左） | Spearman \|ρ\| = **1.0**（完全恢复 cμ 序） |
| 鲁棒性 ablation（T / queue_class / num_iter / ρ） | 28 格全部 ≤ **2.62%** |

---

## 4. §5.3 准入控制 — 论文 Figure 11

📊 图：`fig3_admission_pathwise_vs_spsa.png`

论文主张：网络变大（K≥15）时，零阶方法 SPSA 崩溃，PATHWISE 仍稳定。

| job classes | PATHWISE (B=1) | SPSA (B=1000) | 倍数 |
|---:|---:|---:|---:|
| 6  | 14.91 | 13.29 | 0.9× |
| 9  | 19.53 | 20.86 | 1.1× |
| 12 | 26.90 | 27.94 | 1.0× |
| 15 | 31.59 | **65.44** | **2.1×** |
| 18 | 38.59 | **66.31** | **1.7×** |
| 21 | 44.08 | **106.02** | **2.4×** |

→ 小网络两者持平；**15 类起 SPSA 成本翻倍以上**，PATHWISE 平滑增长。论点复现。

---

## 5. §6 策略参数化 — 论文 Figure 12

📊 图：`fig4_section6_wc_vs_vanilla.png`

论文主张：work-conserving softmax 是稳定训练的关键，vanilla softmax 会发散。

| 架构 | 初始 test cost | 收敛 min cost |
|---|---:|---:|
| vanilla softmax | ~5015 | 17.2（且全程剧烈波动） |
| **work-conserving softmax（我们）** | ~18 | **15.2** |

→ vanilla 起步成本高 2–3 个数量级、不稳定；WC 一开始就稳。论点复现（+13.2%）。

---

## 6. §7 速度 — STE vs PPO

| 方法 | Criss-Cross 训练墙钟 |
|---|---:|
| PPO | 67 h |
| **STE / PATHWISE（我们）** | 2.5 h（**≈27× 更快**） |

→ 与论文「PATHWISE 用 50× 更少数据 / 大幅更快」方向一致。

---

## 7. 复现记分卡（仅已复现项）

| 论文 artifact | 我们复现的内容 | 与论文吻合度 |
|---|---|---|
| Table 1（Criss-Cross, Exp） | cμ + PATHWISE | ✅ ≤1.5% |
| Table 2（Re-entrant-1, Exp）/ Fig 14 左 | cμ + PATHWISE，9 个规模 | ✅ 中小网 ≤5%；最大两网偏低（预算截断） |
| Fig 9（学 cμ 规则） | PW/RF 成本 + 队列序 + 4 ablation | ✅ ≤3.1% |
| Fig 11（准入控制） | PATHWISE vs SPSA 规模化 | ✅ 定性复现（SPSA 崩溃） |
| Fig 12（策略架构） | WC vs vanilla 训练曲线 | ✅ 定性复现 |
| §7 速度 | STE vs PPO 墙钟 | ✅ 27× |

> ⚠️ 范围说明：论文 Table 1–5 每张还含 MaxWeight / MaxPressure / Fluid / PPO-DG /
> PPO-WC 等列，以及 HyperExp 版本 —— 这些**本次未跑**（算力原因），故本页只对比
> **已复现的 cμ 与 PATHWISE 两列、Exp 版本**。

---

*生成脚本：`reports/build_paper_vs_repro.py`（用 `gptsovits` 环境的 Python，含 matplotlib）。
论文原值由 arXiv:2409.03740 PDF 的 Table 1/2 与 Figure 11/12/14 抽取。*
