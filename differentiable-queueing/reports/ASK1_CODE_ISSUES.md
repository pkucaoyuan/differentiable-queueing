# Ask① §5.1 复现：官方代码问题记录（2026-07-13）

对象：`namkoong-lab/differentiable-queueing@0c21ed7`（导师提供的官方发布 repo，本仓 queuetorch/ 等与其逐字节相同）。
目标实验：论文 arXiv:2409.03740 §5.1 Figure 8（gradient cosine-similarity，PATHWISE vs REINFORCE，9 网络 × 4ρ × 3 策略）。

**总结论：论文的定性结论（PATHWISE 方向精度远超 REINFORCE）可以用官方代码复现；定量数值（PW≈1、94.5% 胜率）复现不出，且是官方代码与论文装置不一致导致的结构性问题——官方 repo 自带的复现笔记也承认这一点。**

---

## 一、官方 repo 自己承认的问题（`experiments/reports/report_subtleties.md`）

1. **策略实现与论文不符**：论文给出 sPR/sMW/sMP 的公式（θ∈R⁺ⁿ 对角参数化），发布代码 `queuetorch/policies.py` 用 `nn.Linear` 实现，"seems inconsistent with the paper's description"。
2. **cossim 度量未实现**："this metric does not appear to be implemented in the provided codebase"。
3. **论文图的生成代码丢失**："the specific code used to generate the figures in the paper could not be located"，且 "the obtained results differ significantly from those reported in the paper"。
4. 上游自己复现的数值（`Section 5/Section 5.md`）：PATHWISE 峰值约 0.50（sPR）、REINFORCE 0–0.15 —— 与论文的 PW≈1.0 差距显著。**我们的测量与上游内部复现逐点吻合。**

## 二、我们实测定位的代码缺陷

### P1（关键）：REINFORCE 缺 value baseline → “真值梯度”本身是噪声
- 位置：`experiments/gradient_comparison.py::_compute_reinforce_grad_core`（GT 与估计器共用），无任何 baseline/advantage。
- 论文证据：intro 图注明确写 REINFORCE "equipped with a value function baseline, which is fitted using 10⁶ state transitions"；§3 给出 BASELINE 估计器公式 (eq. reinforce_baseline)。该组件未随代码发布。
- 实测：GT=10⁵ 轨迹时分半自洽性（split-half cosine，同一 θ 两半 GT 的夹角）
  - 无 baseline（官方原样）：sMW/sMP 0.55–0.76，sPR 出现随机 ±1 方向翻转 → **GT 噪声主导，任何与它的 cossim 都被系统性压低**；
  - 加严格无偏的 leave-one-out baseline（组内留一均值，仅用于 GT）：0.89–1.00。
- 影响：这是官方代码测不出论文数值的首要原因。我们的正式测量用 LOO-GT，并同时保留无 baseline 对照 GT 作为证据（方案 C）。

### P2（关键）：sMP 与 sMW 是同一份代码
- 位置：`queuetorch/policies.py::SoftMaxPressurePolicy`，与 `SoftMaxWeightPolicy` 逐行相同（docstring 自认："behaves structurally like sMW"）。真正的 MaxPressure 需要路由矩阵 R 的下游项，未实现。
- 影响：历史跑批中 sMP 与 sMW 两行数值完全相同（同结构 + 同 seed 初始化）。我们在报告中把 sMP 标注为"sMW 家族的另一次随机抽样"，并注明与论文公式的差异。

### P3：float32 下 masking 归一化的 backward 溢出 → ρ≥0.95 出 nan
- 位置：`gradient_comparison.py` 的掩蔽逻辑 `probs = probs / sum_probs`（以及 env 内同构操作）。
- 机制：logits 随队列长度增长而饱和 → 兼容队列的 softmax 质量下溢到 ~1e-38 → backward 中 1/ssum ≈ 1e38 溢出为 inf → nan。ρ 越高队列越长越易触发（与历史 "GPU canonical ρ≥0.95 nan" 完全对应）。
- 修复：策略与 masking 前后向用 float64（语义不变），nan 块计数并丢弃。修复后全部 ρ（含 0.99）0 丢弃。

### P4：ρ 缩放不一致且对 reentrant 配置不生效
- 位置：`gradient_comparison.py::run_experiment`，`val × ρ` 仅当 `lam_params.val` 非 null 时执行。
- 事实：所有基线配置的真实流量强度是 **0.9**（criss_cross_bh: λ=0.9, 服务容量 1.0；reentrant: 14×0.0643=0.9），所以 `val×ρ` 得到的真实强度是 0.9ρ（标签 0.99 → 实际 0.891）；而 reentrant 系列 `val: null`（λ 从 npy 读取），**缩放代码根本不会执行**，无论标签写多少实际都跑在 ρ=0.9。
- 处理：A/B 实测两种缩放（criss-cross 24 格）：sMP/sMW 差异在噪声内，sPR 上"真实强度=标签"（λ×ρ/0.9）系统性更高。正式跑采用 λ×ρ/0.9 并对 reentrant 显式生效。

### P5：论文的 θ~Lognormal(0,1) 装置在无 value baseline 下不可测（解释为何不能"按论文公式直接跑"）
- 按论文公式实现对角参数化策略后实测：θ 为正且与队列长度、μ 相乘 → logits 达数百 → 策略近确定性 → REINFORCE score function 呈"几乎全零 + 稀有巨大尖峰"的重尾分布，标量 baseline 无法挽救，GT=10⁵ 时方向随机翻转（实测两次独立 GT 的 cos = −1）。
- 结论：复现论文 Figure 8 的原貌需要论文未开源的 value function baseline 装置；在只有发布代码的前提下，可靠的测量对象只能是发布代码的策略参数化。

## 三、我们的测量装置（与官方代码的差异清单）

全部策略/估计器/masking/超参与官方代码逐位一致（前向等价性经数值验证，最大偏差 9.4e-16），差异仅：
1. GT 加 leave-one-out baseline（严格无偏，只改方差）；同时保留无 baseline 对照 GT；
2. 策略与 masking 用 float64（修 P3，语义不变）；
3. λ×ρ/0.9 缩放使真实强度=标签（修 P4，A/B 实验支持）；
4. sMP 与 sMW 用不同随机 seed 采样（缓解 P2 的完全同值问题）；
5. 工程层：分组批处理（每 θ×draw 独立参数副本，梯度无交叉泄漏，已验证）、8 卡并行、nan 计数丢弃。

## 四、关键数字（criss-cross 全 24 格，100θ×100draws，GT=1e5）

| 指标 | 论文 | 本次测量（LOO-GT） |
|---|---|---|
| PATHWISE (B=1) cossim | ≈1.0 | 0.16–0.58（per-θ 双峰：sPR 54% 的 θ >0.8 甚至=1.00；sMW/sMP 13–19%） |
| REINFORCE (B=1000) cossim | 0.2–0.6 | 0.06–0.13 |
| PW>RF 占比 | 94.5% | 70.5% |
| 随 ρ 变化 | 退化 | sPR 反而上升（0.16→0.58） |
| GT 分半自洽性 | —（未报告） | LOO 0.76–1.00；无 baseline 对照 0.55–0.76（sPR ±1 翻转） |

**新发现**：部分 θ 的 PATHWISE 方向为负（sMW 最低 −0.83，q10=−0.81）——STE (β=1) 对状态依赖策略存在方向性偏差，论文"bias 小"的断言在发布代码参数化下不成立。

数据：`results/ask1/stage1/*.npz`（每格含 gt/gt_nb/per-θ×draw 余弦/分半诊断/参数）；
复现脚本：`experiments/ask1/`（common.py / run_cossim.py / validate.py / analyze_stage1.py）。

---

## 追加（2026-07-14）：P6 与论文装置还原实验的定论

### P6：掩蔽包装（min(softmax, queues) + 归一化）显著压低 cossim 且改变被测策略族
- 位置：`gradient_comparison.py` 的掩蔽逻辑（也见 env.step 内部同构操作）。
- 实测（criss-cross，论文策略 + V-baseline，同 θ 同 GT 规格）：带掩蔽 PW=0.42–0.44（ρ=0.9），去掉掩蔽（论文公式字面）PW=0.58–0.88。论文 §5.1 的公式没有这层包装。
- 影响：这是发布代码测不出论文数值的第二大因素（第一是 P1 缺 value baseline）。

### 定论：论文 Figure 8 大概率真实，"无法复现"是发布代码装置缺失
按论文描述完整还原（论文公式策略 + Lognormal θ + V(x,t/N) baseline [10⁶ 转移拟合，时间特征必需] + 无掩蔽）后：PW = 0.58–0.88（ρ=0.9）/ 0.37–0.81（ρ=0.99，退化模式与论文一致），RF-BL 回到论文的 0.2–0.6 量级，GT 分半自洽 0.95–1.00。剩余与 ≈1 的差距可归于 pilot 规模（10θ×20draws vs 论文 100×100）与 GT=1e5（论文 1e6）。
详见 `experiments/ask1/paper_impl.py` 与 `reports/ask1_figs/fig5_paper_apparatus.png`。

### 外部校验（2026-07-14）：有限差分黄金标准确认恢复结论
用与 REINFORCE 家族无关的中心有限差分（CRN，B=2e5/点，h_rel∈{0.05,0.1} 两档自洽）验证 no-mask GT 方向：
9 个可靠 θ 中 8 个 cos(FD, GT)=0.998–1.000，其上 PW per-θ 达 0.95–1.00（= 论文 Fig.8 水平）。
例外 1 例（sMW θ9：GT 自洽 0.99 但与 FD 方向 −0.46）暴露论文自身 γ=0.999 折扣信用分配的方向偏差——个别 θ 上 REINFORCE-γ 的期望方向偏离真实 ∇J_N；论文 GT 同样定义，复现意义一致，但属论文装置的固有局限。
脚本：`experiments/ask1/fd_check.py`。
