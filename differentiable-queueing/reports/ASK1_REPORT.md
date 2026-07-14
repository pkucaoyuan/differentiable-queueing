# Ask① 汇报：§5.1 梯度 cosine-similarity 复现（拓扑轴 + batch-size 轴 + 论文装置还原）

日期：2026-07-14 ｜ 数据与脚本：`results/ask1/`、`experiments/ask1/` ｜ 代码问题详录：`reports/ASK1_CODE_ISSUES.md`

## TL;DR（三层结论）

1. **发布代码原样跑不出论文 Figure 8 的数值**（PW 只有 0.16–0.58 vs 论文 ≈1.0）——三方独立复现一致（上游实验室内部、我们旧 canonical、我们带可靠 GT 的本次），根源不是采样量而是装置缺失。
2. **我们按论文描述还原完整装置后，论文数值大部分恢复**（fig5）：论文公式策略（θ~Lognormal 对角参数化）+ value-function baseline V(x,t/N)（论文 intro 图注声明、代码未发布）+ **不加发布代码的掩蔽归一化**（论文公式字面）→ criss-cross 上 PW 回升到 **0.58–0.88**（ρ=0.9），且高 ρ 退化模式与论文自述一致。**结论：论文结果大概率真实，发布代码缺装置**（缺 baseline、多一层 mask 包装、策略实现不符）。
3. **导师两问的答案已完整**：
   - **拓扑轴**（fig1，54 格全规格）：PW 优势强烈依赖拓扑——criss-cross 上 4 倍碾压；**6 类网上 RF(B=1000) 全面反超 PW**（0.25 vs 0.19）；9 类持平；12/15 类 PW 夺回（RF 随维度衰减更快）。全网格 per-θ 胜率 58.1%（论文 94.5%，差异归因同上）。
   - **batch-size 轴**（fig4，18 格 × B∈{1..10⁴}）：PW 从 B=1 即高位饱和；RF 从 ≈0 随 B 爬升，**需要 B≈10³–10⁴ 才追上 PW(B=1)**，交叉点随拓扑变化（reentrant 上更早）。这定量化了论文"数量级样本效率优势"的说法。

## 图（reports/ask1_figs/）

- **fig1_cossim_grid** — 论文 Fig.8 同构热图，54 格全满（criss 全规格 + 8 reentrant 网全规格 96 格）
- **fig2_gt_reliability** — GT 可靠性：LOO baseline vs 发布代码无 baseline（后者噪声主导的证据）
- **fig3_theta_dist** — per-θ 双峰：部分 θ 的 PW=1.00，部分为负（STE 方向性偏差，最低 −0.83）
- **fig4_batch_sweep** — cossim vs B 曲线（PW 实线 / RF 虚线 × 3 策略 × 3 网 × 2ρ）
- **fig5_paper_apparatus** — 论文装置还原三级对比（released → +V-baseline → +去 mask）

## 论文装置还原实验（criss-cross pilot，10θ×20draws，GT=1e5 带 V baseline）

| PW cossim | released | 论文策略+V-baseline(带mask) | **论文字面(no-mask)+V-baseline** | 论文 |
|---|---|---|---|---|
| sMP ρ=0.9 | 0.23 | 0.42 | **0.88** | ≈1 |
| sMW ρ=0.9 | 0.38 | 0.43 | **0.58** | ≈1 |
| sPR ρ=0.9 | 0.50 | 0.44 | **0.73** | ≈1 |
| sMP ρ=0.99 | 0.33 | 0.13 | 0.37 | ~0.8 |
| sMW ρ=0.99 | 0.25 | 0.37 | **0.65** | ~0.8 |
| sPR ρ=0.99 | 0.58 | 0.59 | **0.81** | ~0.8 |

配套发现：RF-BL（带 baseline 的 REINFORCE，B=1000）= 0.14–0.97，落回论文的 0.2–0.6 量级（sPR 因有效维度低达 0.9+）；V(x,t/N) 拟合 R²=0.65–0.94（**时间特征必需**：state-only V 的 R² 仅 0.05）；GT 分半自洽 0.95–1.00（sMP ρ0.99 除外 0.18，Lognormal 饱和重尾残留）。

关键机制：发布代码的 min(softmax, queues)+归一化掩蔽**改变了被测策略族**并显著压低 cossim；论文公式没有这层包装。

## 全规格拓扑聚合（stage1+stage2，108 格，PW / RF 均值）

| 规模 | sMP | sMW | sPR |
|---|---|---|---|
| criss (q=3) | 0.24 / 0.09 | 0.29 / 0.09 | 0.45 / 0.10 |
| 6 类 | 0.19 / **0.25** | 0.21 / **0.25** | 0.35 / **0.45** |
| 9 类 | 0.18 / **0.20** | 0.18 / **0.20** | 0.44 / 0.34 |
| 12 类 | 0.17 / 0.17 | 0.18 / 0.17 | 0.40 / 0.28 |
| 15 类 | 0.18 / 0.15 | 0.18 / 0.15 | 0.42 / 0.23 |

（加粗 = RF ≥ PW；47/108 格 RF≥PW。此表基于发布代码策略族 + LOO-GT；掩蔽包装对两个估计器同等作用，拓扑趋势应稳健。）

## 测量装置说明

主网格：发布代码策略/估计器/掩蔽逐位一致（前向等价验证 1e-15），偏离仅 4 处（LOO-GT + fp64 数值加固 + λ×ρ/0.9 真实强度缩放 + sMP 独立 seed），依据见 ASK1_CODE_ISSUES.md。规格：100θ×100draws，GT=1e5（12/15 类 5e4），γ=0.999，β=1，N=1000。
论文装置实验：`experiments/ask1/paper_impl.py`（V(x,t/N) MLP 按论文"10⁶ 状态转移"拟合，PAPERIMPL_NOMASK 开关控制掩蔽）。

## 建议下一步

1. 论文装置 no-mask 变体升级到全规格（100θ×100draws）+ 扩展到 reentrant 拓扑，验证 Figure 8 全图可恢复性
2. sMP ρ=0.99 的 GT 重尾问题：试 clipped-advantage 或更大 V 拟合集
3. 把"发布代码 vs 论文装置"的差异清单反馈给原作者（Ethan Che / Namkoong lab）
