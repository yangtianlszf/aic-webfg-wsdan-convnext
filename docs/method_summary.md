# 方法方案摘要

## 1. 问题分析

网络监督细粒度图像识别的主要难点包括：

1. 网络爬取数据存在标签噪声；
2. 细粒度类别之间差异小，需要关注局部判别区域；
3. 训练集存在类别长尾分布；
4. 测试集为人工精确标注数据，要求模型具备较强泛化能力。

## 2. 总体方案

最终方案采用单模型分类框架：

```text
Image → Strong Augmentation → ConvNeXt Backbone → WS-DAN Attention → ArcFace Head → Class Prediction
```

训练阶段引入噪声鲁棒学习、类别重加权、自清洗与一致性正则；推理阶段使用单模型权重和轻量 TTA 输出 CSV。

## 3. 核心模块

### 3.1 ConvNeXt Backbone

使用 ConvNeXt / ConvNeXtV2 作为主干网络，并限制为 ImageNet-1k 预训练权重，符合比赛约束。

### 3.2 WS-DAN 注意力机制

在 backbone feature map 上预测多个注意力图，引导模型关注细粒度判别区域。训练时可对高响应区域进行 erase，迫使模型学习更多互补局部特征。

### 3.3 ArcFace 分类头

使用 margin-based 分类头扩大类间间隔，缓解细粒度类别相似导致的边界模糊问题。

### 3.4 噪声鲁棒损失

代码支持 ELR+、SCE、GCE、EQL 等损失函数。默认推荐 ELR+，用于抑制模型对错误标签的记忆。

### 3.5 类别长尾处理

采用 class-balanced sampler 和 class-aware loss reweighting，使尾部类别在训练中获得更充分的优化。

### 3.6 自清洗机制

利用预测置信度 EMA 估计样本可靠性，对持续低置信样本降低训练权重，以减弱疑似噪声样本的影响。

### 3.7 伪标签机制

支持生成 Top-K soft pseudo label，并在后续训练中以较低权重混入训练集，进一步利用模型自身高置信预测。

## 4. 单模型约束说明

Uniform Model Soup 只对同一架构的多个 checkpoint 做权重平均，最终推理时仍加载一个 checkpoint，属于单模型推理，不是多模型 ensemble。
