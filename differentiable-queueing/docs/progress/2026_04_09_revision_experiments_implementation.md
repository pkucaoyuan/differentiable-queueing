# Revision Experiments Implementation - 2026-04-09

## 状态
🚧 进行中

## 概要
实现了 OPRE-2025-02-1714 论文大修所需的全部 8 个实验脚本（E1-E8），修改了核心模拟器以支持 GPU 原生采样，正在复现主要实验结果以验证正确性。

## 完成内容

### 环境搭建
- **GPU 集群探索**: 确认 researchgpu04/05 各有 8×A40 (48GB) 可用
- **Conda 环境**: 在共享 NFS home 安装 miniconda3 + gpu 环境 (Python 3.11, PyTorch 2.5.1+CUDA 12.1)
- **queuetorch 安装**: 编辑模式安装（torch 2.2.0 + numpy 1.26.4 per pyproject.toml）
- **Git 配置**: SSH key 添加到 GitHub，clone DJ_OR repo

### 核心代码修改
- **`queuetorch/env.py`**: 新增 GPU 原生采样路径
  - `draw_service()`: CUDA 设备下使用 `torch.distributions.Exponential/LogNormal`
  - `draw_inter_arrivals()`: CUDA 设备下使用 `torch.distributions.Exponential`
  - 新增 `gpu_native_sampling` 和 `_service_type` 属性
  - `load_env()`: 自动为 CUDA+constant arrival 启用 GPU 采样

### 新增实验脚本（8 个）

| 实验 | 文件 | 优先级 | 对应审稿意见 |
|------|------|--------|-------------|
| E1 | `experiments/ste_bias_variance.py` | P0 | AE-M1: STE bias-variance 扩展到多环境 |
| E2 | `experiments/glr_comparison.py` | P0 | AE-M3: GLR vs PATHWISE vs REINFORCE (M/M/1) |
| E3 | `experiments/gpu_benchmarks.py` | P1 | AE-M4, R2: GPU wall-clock benchmark |
| E4 | `experiments/criss_cross_nonwc.py` | P1 | R1: 非工作守恒策略 (criss-cross IIB) |
| E5 | `experiments/heavy_traffic_curve.py` | P0 | R1: 重交通 rho→1 性能曲线 |
| E6 | `experiments/ablation_3way.py` | P0 | R2: 3因素消融 (6种方法) |
| E7 | `experiments/hyperparam_sensitivity.py` | P1 | R2: 超参数敏感性分析 |
| E8 | `experiments/extract_parameters.py` | P0 | R1: 参数表提取 (已完成运行) |

### 新增配置
- **`configs/env/criss_cross_IIB.yaml`**: Martins et al. (1996) Case IIB 参数

### 复现验证（进行中）
- **`experiments/reproduce_main.py`**: 复现论文主要结果
  - ✅ Test 1: M/M/1 模拟器验证 (误差 0.028, PASS)
  - ✅ Test 5: cmu-rule 最优策略参考成本 (gap=1.0 改善 76.8%)
  - 🚧 Test 2: PATHWISE CMU 优化 (20 trials × 3 gaps, 运行中)
  - ⏳ Test 3: REINFORCE CMU 优化
  - ⏳ Test 4: 梯度比较 cosine similarity

## 代码变更

| 类型 | 数量 |
|------|------|
| 新增文件 | 12 |
| 修改文件 | 2 |
| 删除文件 | 0 |

### 关键文件变更

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `queuetorch/env.py` | 修改 | +43 行: GPU 原生采样 (torch.distributions) |
| `configs/env/criss_cross_IIB.yaml` | 修改 | 更新为 Martins (1996) Case IIB 参数 |
| `experiments/ste_bias_variance.py` | 新增 | E1: 4环境×5 horizon×5 temp bias-variance |
| `experiments/glr_comparison.py` | 新增 | E2: GLR/IPA/REINFORCE 梯度 MSE |
| `experiments/gpu_benchmarks.py` | 新增 | E3: GPU vs CPU wall-clock benchmark |
| `experiments/criss_cross_nonwc.py` | 新增 | E4: WC vs non-WC 策略 + 热力图 |
| `experiments/heavy_traffic_curve.py` | 新增 | E5: rho 0.80→0.99 性能曲线 |
| `experiments/ablation_3way.py` | 新增 | E6: 6种方法3因素消融 |
| `experiments/hyperparam_sensitivity.py` | 新增 | E7: 温度/学习率/horizon/batch 扫描 |
| `experiments/extract_parameters.py` | 新增 | E8: 参数表提取 (已运行) |
| `experiments/reproduce_main.py` | 新增 | 论文主要结果复现验证 |
| `results/E8_*.json` | 新增 | 25环境+4模型配置参数 |

## 测试情况

- [x] queuetorch 安装成功 (editable mode)
- [x] M/M/1 模拟器验证通过 (E[Q]=9.03 vs 理论值 9.00)
- [x] E8 参数提取运行成功 (25 environments)
- [x] E2 GLR 梯度估计器烟雾测试通过
- [x] 所有 8 个实验脚本语法检查通过
- [ ] 论文主要结果完整复现（运行中）
- [ ] GPU benchmark 验证（需在 gpu04/05 上运行）

## 下一步计划

- [ ] 等待复现结果完成，对比已有 cmu/ 数据
- [ ] 确认复现无误后，提交集群运行 Week 1 实验 (E5, E7)
- [ ] 在 GPU 节点运行 E3 benchmark
- [ ] 启动 Week 2 实验 (E1, E4, E6)
- [ ] Week 3: E2 + 图表生成

## 计算资源规划

| 实验 | 预计耗时 | 核心数 | 运行节点 |
|------|---------|--------|---------|
| E5 Heavy Traffic | ~4h | 80 CPU | researchint01 |
| E7 Hyperparams | ~3.5h | 60 CPU | researchint01 |
| E3 GPU Benchmark | ~2h | 1 GPU | researchgpu04 |
| E1 Bias-Variance | ~80h | 100 CPU | researchint01 (nohup) |
| E6 Ablation | ~6h | 80 CPU | researchint01 |
| E4 Non-WC | ~150 core-h | 80 CPU | researchint01 |
| E2 GLR | ~20 core-min | 80 CPU | researchint01 |

---

**作者**: Claude Code (自动生成)
**审核**: 待审核
