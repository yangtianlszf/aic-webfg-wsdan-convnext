# 实验记录

## 当前最终版本

- 主文件：`src/wsdan_convnext_plus.py`
- 主干：ConvNeXt / ConvNeXtV2
- 默认模型：`convnext_base`
- 图像尺寸：384
- 默认损失：ELR+
- 关键策略：WS-DAN、ArcFace、MixUp、CutMix、RandAugment、自清洗、类别均衡、一致性正则

## 后续实验建议

### 实验 1：基础模型对比

| 实验 | Backbone | img_size | loss | 备注 |
|---|---|---:|---|---|
| A1 | convnext_tiny | 384 | CE | 快速 baseline |
| A2 | convnext_base | 384 | CE | backbone 提升 |
| A3 | convnext_base | 384 | ELR+ | 噪声鲁棒 |

### 实验 2：细粒度模块消融

| 实验 | WS-DAN | ArcFace | 目的 |
|---|---|---|---|
| B1 | 否 | 否 | 普通分类头 |
| B2 | 是 | 否 | 验证注意力擦除 |
| B3 | 是 | 是 | 最终配置 |

### 实验 3：鲁棒训练消融

| 实验 | 自清洗 | 类别均衡 | 一致性正则 | 目的 |
|---|---|---|---|---|
| C1 | 否 | 否 | 否 | 基础鲁棒损失 |
| C2 | 是 | 否 | 否 | 样本噪声处理 |
| C3 | 是 | 是 | 是 | 最终配置 |

## 记录模板

| 日期 | 数据集 | 配置 | checkpoint | 分数 | 备注 |
|---|---|---|---|---:|---|
| YYYY-MM-DD | WebFG-400 | convnext_base + ELR+ | best.pt | - | - |
