# Ask① 汇报：§5.1 梯度 cosine-similarity 复现（拓扑轴 + 估计器对比）

日期：2026-07-13 ｜ 数据与脚本：`results/ask1/`、`experiments/ask1/` ｜ 代码问题详录：`reports/ASK1_CODE_ISSUES.md`

## TL;DR

1. **定性结论部分复现**：criss-cross 上 PATHWISE（1 条轨迹）方向精度一致优于 REINFORCE（1000 条轨迹）3–5 倍；但**拓扑轴上优势不是均匀的**（见第 3 点后的"拓扑依赖"）。
2. **定量数值复现不出，且可归因**：论文 Figure 8 的 PW≈1.0 无法用官方发布代码得到（我们测得 0.16–0.58）。原因已定位：官方代码与论文装置不一致——策略参数化不符（sMP 与 sMW 代码相同）、REINFORCE 缺论文使用的 value baseline、生成论文图的代码已丢失（官方 repo 自带笔记承认，其内部复现同样只到 ~0.5）。
3. **测量本身可信**：我们给 GT 加严格无偏的 LOO baseline（分半自洽性 0.55–0.76 → 0.89–1.00，见 fig2），并排跑无 baseline 对照 GT 作为证据；修复了历史 ρ≥0.95 nan（fp32 溢出）与 sMP/sMW 同值两个 bug。
4. **新发现 A（per-θ 双峰）**：一部分 θ 的 PW 精确到 1.00（论文故事在这些点成立），另一部分 ≈0 甚至强负（STE β=1 的方向性偏差，最低 −0.83），见 fig3。
5. **新发现 B（拓扑依赖，直接回应导师问题）**：PW 的优势随拓扑显著变化——criss-cross 上压倒性（约 4 倍）；**6–9 类 reentrant 网上 REINFORCE(B=1000) 追平甚至反超**（60 格中 18 格 RF≥PW，集中在中等规模网）；12–15 类大网上 PW 重新领先（RF 随维度衰减更快）。按规模聚合（PW / RF 均值）：

| 网络规模 | sMP | sMW | sPR |
|---|---|---|---|
| criss-cross (q=3) | 0.24 / 0.09 | 0.29 / 0.09 | 0.45 / 0.10 |
| 6 类 | 0.15 / **0.25** | 0.20 / **0.22** | 0.38 / **0.44** |
| 9 类 | 0.19 / 0.21 | 0.18 / 0.19 | 0.42 / 0.32 |
| 12 类 | 0.18 / 0.17 | 0.18 / 0.16 | 0.41 / 0.27 |
| 15 类 | 0.20 / 0.15 | 0.18 / 0.15 | 0.43 / 0.23 |

（加粗 = RF 均值 ≥ PW。caveat：reentrant 行为快速规格 25θ×25draws、GT=5e4、ρ∈{0.9,0.99}，正式版将升级到全规格复核。）

## 图

- `ask1_figs/fig1_cossim_grid.png` — 论文 Figure 8 同构热图：2 行（PW/RF）× 3 列（sMP/sMW/sPR），每格 9 网络 × ρ。criss-cross 行为全规格（100θ×100draws×4ρ，GT=1e5），reentrant 8 网为快速规格（25θ×25draws×ρ∈{0.9,0.99}，GT=5e4）
- `ask1_figs/fig2_gt_reliability.png` — GT 可靠性：LOO baseline vs 官方无 baseline 的分半自洽性对比（方案 C 证据）
- `ask1_figs/fig3_theta_dist.png` — per-θ 分布的双峰结构

## 主要数字（criss-cross，全规格）

| 指标 | 论文 Fig. 8 | 本测量（可靠 GT） | 上游内部复现 |
|---|---|---|---|
| PATHWISE (B=1) | ≈1.0 | 0.16–0.58 | "peaks ≈0.50" |
| REINFORCE (B=1000) | 0.2–0.6 | 0.06–0.13 | "≈0–0.15" |
| PW>RF 占比（per-θ） | 94.5% | 70.5% | 未报告 |
| cossim 随 ρ | 退化 | sPR 反而上升 | 未报告 |

## 实验设定（与官方代码的差异，共 4 处，均有实测依据）

策略/估计器/masking/超参（γ=0.999、β=1、N=1000、PW B=1、RF B=1000）与官方代码逐位一致（前向等价性验证至 1e-15）。差异：
1. GT 加 leave-one-out baseline（严格无偏；官方无 baseline 的 GT 是噪声主导，见 fig2）；同时保留无 baseline 对照 GT；
2. 策略与 masking 用 float64（修 fp32 backward 溢出 nan，语义不变，修复后 ρ=0.99 零 nan）；
3. λ 缩放 = ρ/0.9（使真实流量强度=标签；官方脚本的 val×ρ 实际得到 0.9ρ、且对 reentrant 配置不生效；A/B 实验 24 格支持此选择）；
4. sMP 用独立 seed 采样（官方代码中 sMP 与 sMW 类代码相同，故 sMP 实为 sMW 家族的另一次抽样，图中已标注 *）。

## 对导师两个问题的回答现状

- **拓扑轴**：fig1 覆盖 criss-cross + Reentrant/Reentrant-2 各 4 个规模（6/9/12/15 类）。
- **batch-size 轴**：cossim(B) 扫描（B∈{1,…,10⁴}，PW & RF）机制已就绪（复用已有 GT 与 θ），为下一步任务（Stage-3）。

## 下一步

1. Stage-3 batch-size 扫描（3 网 × 2ρ × 3 策略 × 8 个 B 值）——回答导师问题的另一半轴
2. 把 reentrant 快速格升级到全规格（25θ→100θ，GT 5e4→1e5，补 ρ∈{0.8,0.95}）
3. （可选）实现论文的 value-function baseline，检验论文原装置能否恢复 PW≈1
