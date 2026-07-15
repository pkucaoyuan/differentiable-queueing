# Ask① — 参考梯度稳定性检查计划(GT 样本量 ×100)

日期：2026-07-15 ｜ 对象：released-code 的 cossim 测量（`experiments/ask1/run_cossim.py`，stage2 拓扑网格）

## 1. 背景（导师反馈）

导师看到 released-code 策略族的梯度 cosine-similarity 结果后指出：

> "REINFORCE actually does quite well here, which is quite surprising given its usual unreliability.
>  Just so that we reduce confounding, can you make sure that the reference gradient you're
>  computing is stable by increasing the sample size 100x?"

即：REINFORCE(B=1000) 在若干拓扑上意外地追平/反超 PATHWISE(B=1)。导师担心这可能是**参考梯度（ground-truth）本身不稳（噪声）导致的假象**，要求把 GT 的样本量提高 100×，确认参考梯度稳定、该结论不是 GT 噪声造成的。

## 2. 当前测量设置（原代码）

- 参考梯度 `GT` = LOO-baselined REINFORCE 在 `GT_TRAJS` 条轨迹上的均值（每个策略初始化 θ）。
- stage2 全规格：6–9 类网 `GT_TRAJS=1e5`；12–15 类网 `GT_TRAJS=5e4`。`n_theta=100`，`n_draws=100`，`RF_B=1000`，`γ=0.999`，`β=1`。
- 已内建 GT 稳定性诊断：`gt_split_cos`（把 GT 的轨迹分两半、各算一次梯度、量两半的夹角；1.0 = 完全自洽/无噪声）。

## 3. "REINFORCE 反超" 的区域，以及现有 GT 稳定性

全 96 格中 **RF ≥ PW 共 47 格**，集中在小网，且此处 GT 已经很稳：

| 网络 | RF≥PW 格数 | GT 分半自洽（均值） | 现用 GT_TRAJS |
|---|---|---|---|
| reentrant_2 (6 类) | 12/12 | 0.962 | 1e5 |
| re-reentrant_2 (6 类) | 12/12 | 0.956 | 1e5 |
| reentrant_3 (9 类) | 8/12 | 0.950 | 1e5 |
| re-reentrant_3 (9 类) | 6/12 | 0.951 | 1e5 |
| reentrant_4 (12 类) | 7/12 | 0.892 | **5e4** |
| reentrant_5 (15 类) | 1/12 | 0.883 | **5e4** |
| re-reentrant_4/5 | 1/12, 0/12 | 0.88, 0.87 | **5e4** |

代表数字（reentrant_2, ρ=0.9）：sMP PW+0.157 / RF+0.279；sMW PW+0.202 / RF+0.257；sPR PW+0.343 / RF+0.488；分半自洽 0.95–1.00。

**读出的两点**：
1. **强反超（6–9 类小网）的 GT 在 1e5 下已 0.94–1.0，几乎无噪声** → RF 反超大概率是真的，非 GT 假象。
2. **边缘反超（12–15 类大网）当时只用了 gt=5e4、分半自洽掉到 0.83–0.88，且 RF 优势很薄（margin ~0.01）** → 这才是对导师担忧最敏感、最需要 100× 复核的地方。

## 4. 实验设计（100×）

- `GT_TRAJS`：`1e5`（已有）→ `1e6` → `1e7`（= 100×，且已超过论文使用的 1e6）。
- `n_theta=20`（稳定性检查不需要 100 θ），`n_draws=100`，`RF_B=1000`，LOO baseline，固定 seed。
- 目标网：
  - **验证组**（预期不变）：`reentrant_2`、`re-reentrant_2`
  - **压力测试组**（GT 最不稳、最可能翻盘）：`reentrant_4`、`reentrant_5`
- **输出按 GT 值分目录，不覆盖现有 1e5/5e4 结果**：`results/ask1/gtstab/gt{trajs}/...`

## 5. 交付物

每个目标格一张收敛表：`GT_TRAJS ∈ {1e5, 1e6, 1e7}` × {分半自洽, PW·cos, RF·cos}，显示数值随 GT 收敛、以及 RF≥PW 的排序是否稳定。

给导师的结论应形如：
> 参考梯度在 1e7 条轨迹（超过论文 1e6）下已收敛，分半自洽 → ~1.0；小网上 REINFORCE(B=1000) 反超 PATHWISE(B=1) 的结论稳定成立 /（或大网边缘格在 GT 加强后翻回 PW）。

## 6. 需要的代码改动（原代码，最小）

`experiments/ask1/run_cossim.py`：加一个输出命名参数（如 `--out-tag` 或直接按 `GT_TRAJS` 自动加后缀），使 npz 写入含 GT 值的独立目录，避免与现有 stage2 的 `env__rho__pol__paper.npz` 同名冲突、并可与 1e5 结果并排比较。

## 7. 运行（8×A800）

```bash
# 先 1e6，再 1e7；claims 系统支持断点续跑；OOM 格保持 claim 稍后重试
OMP_NUM_THREADS=4 /opt/conda/bin/python experiments/ask1/run_cossim.py \
  --stage stage2 --scaling paper --no-control-gt \
  --n-theta 20 --n-draws 100 --gt-trajs 1000000 \
  --nets reentrant_2,re-reentrant_2,reentrant_4,reentrant_5

OMP_NUM_THREADS=4 /opt/conda/bin/python experiments/ask1/run_cossim.py \
  --stage stage2 --scaling paper --no-control-gt \
  --n-theta 20 --n-draws 100 --gt-trajs 10000000 \
  --nets reentrant_2,re-reentrant_2,reentrant_4,reentrant_5
```

成本：GT ∝ n_theta × GT_TRAJS。减到 20 θ 后，每格约为原 stage2 全规格格的 ~20×；建议 8 卡分摊、过夜。
