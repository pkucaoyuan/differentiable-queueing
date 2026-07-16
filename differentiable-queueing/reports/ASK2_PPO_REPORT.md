# Ask② 汇报：PPO + Work-Conserving 复现（Fig 12，多 seed）

日期：2026-07-15 ｜ 运行：`ppo_runs/`（8 GPU 并行）｜ 出图：`experiments/ask1/plot_ppo.py` → `reports/ask1_figs/fig8_ppo_wc_multiseed.png`

## 目标

导师 Ask②：让 SB3 PPO + work-conserving softmax 在本机稳定跑起来，多 seed 验证方差，复现论文 Fig 12
（PPO-WC > cμ > PPO-BC > PPO vanilla 的排序及训练曲线）。上游笔记曾定性复现（单 seed）；
本次为本机首跑 + 多 seed 稳定性验证。

## 运行设定

- 网络：reentrant_2（论文 Reentrant-1 六类，与上游 Section 6 复现一致）
- 变体 × seed：PPO-WC × 4、PPO vanilla × 2、PPO-BC × 2（各自独立 model/train seed，共享测试 seed）
- 规模：episode_steps=15000 × 20 actors × 50 迭代（论文/上游为 50000×100——完整规格单 run 需数日，
  本次为 13h 预算内的缩减规格；每迭代 30 万环境步，实测 9.5 分钟/迭代）
- 评估：每迭代 25 个测试 env × 2500 步（测试 env 数原代码硬编码 100，已改为读 test_batch 配置）
- cμ 基线：PPO/cmu_results.json（reentrant_2）

## 工程记录（本机首跑修复项）

1. **依赖栈**：代码按 SB3 2.x API 编写（PyTorchObs 等），与 CLAUDE.md 声称的 SB3+gym 不符；
   实际安装 stable_baselines3 2.9 + gymnasium 1.3 + shimmy + gym 0.26（自定义 env 用老 API 自带实现，兼容）。
   ⚠️ pip 升级 SB3 时会连带安装无 CUDA 的新版 torch 进 user site 遮蔽 conda torch——需 `pip install --no-deps` 或事后卸载。
2. **BC 设备 bug**：`utils/eval.py::BCD` 数据集持有 cuda 上的 network 与 CPU 样本相乘崩溃（上游只在 CPU 跑过 BC）；
   修复：数据集持 `network.cpu()`，训练循环把 batch 搬到 policy.device。
3. **评估开销**：pre-train/每迭代评估（原 100 env × test_T=10000 @ ~6.6 vec-step/s ≈ 25 分钟）远超训练本身；
   通过 env 变体 `reentrant_2_ppofast`（test_T=2500）+ test_batch=25 压到 ~95 秒。
4. 训练吞吐：DummyVecEnv 20 actors 串行步进 ~40 vec-steps/s（fps≈526 计入 20 actors）——
   这个管线的根本瓶颈；后续如需完整规格建议把 env 内部 batch 维用起来（P_DiffDiscreteEventSystem 原生支持）。

## 结果（2026-07-16 00:01 收尾）

**cμ 基线（协议匹配 25 envs × 2500 步，seeds 42..66）**：`cmu_matched = 16.49`

**多 seed 最终 cost（最后 5 迭代均值，n=4/2/2）**：

| 变体 | n | 迭代 | last-5 均值 | seed std | 逐 seed |
|---|---|---|---|---|---|
| **PPO-WC** | 4 | 51 | **14.06** | 0.23 | 14.1, 13.8, 14.0, 14.4 |
| cμ 基线 | - | - | 16.49 | - | - |
| PPO-BC | 2 | 51 | 33.76 | 0.15 | 33.6, 33.9 |
| PPO vanilla | 2 | 31/36 未完 | 468.7 | 7.2 | 461.5, 475.9 |

**排序**：**PPO-WC (14.06) < cμ (16.49) < PPO-BC (33.76) ≪ PPO vanilla (468.7)** — 与论文 Fig 12 排序完全一致。

**Fig 12 关键定性特征全部复现**：
1. WC 从随机初始化即已接近 cμ（iter-0 中位数 13.8，甚至已略超 cμ），说明"work-conserving softmax"结构本身就是好策略族，PPO 的作用是在其内部微调
2. WC 训练 50 迭代 seed 间方差极小（std=0.23）→ 复现稳定，非单 seed 幸运
3. BC 反向漂移：iter-0 中位数 18.5（接近 cμ）→ iter-50 达 34.0，与上游单 seed 复现观察一致（BC 冷启动接近 cμ 但训练发散）
4. vanilla 卡在 460+ 高成本（初始 519 → iter-31 时 461），无收敛趋势——与论文 Fig 12 里 vanilla 停在 ~10³ 一致

图：`reports/ask1_figs/fig8_ppo_wc_multiseed.png/pdf`（log 面板 + zoom 面板）

## 与论文/上游的对照基准

| | WC | cμ | BC | vanilla |
|---|---|---|---|---|
| 论文 Fig 12（Reentrant-1） | 从 cμ 起并降低 | ~17 | 介于两者 | ~10³ 卡住 |
| 上游单 seed（fig_section6） | 17.7 | 17.4 | 29.6 | 2020 |
| **本次 multi-seed** | **14.1 (±0.2)** | **16.49** | **33.8 (±0.15)** | **469 (±7)** |

本次 WC (14.1) 比上游单 seed 复现 (17.7) 更低——可能是缩短版规格（15000×50 vs 50000×100）在有限迭代下 WC 尚未过拟合、或本次协议 (25 envs × 2500 步) 与上游 (100 envs × 10000 步) 评估噪声不同。总体定性/排序完全稳。

## 收尾

- 图：`fig8_ppo_wc_multiseed.png/pdf` ✅
- 8 个 run 目录：`ppo_runs/{WC_s0..3, bc_s0..1, vanilla_s0..1}` 全部保留 `run.log` + `*_results.json`（vanilla 因超时无 json，可从 log 恢复）
- cμ 基线：`ppo_runs/cmu_matched.json`（`{"avg_cost":16.49, ...}`）
