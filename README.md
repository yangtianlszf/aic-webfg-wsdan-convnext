# AIC 网络监督细粒度图像识别项目记录

本仓库用于记录“全球校园人工智能算法精英大赛·算法挑战赛·网络监督细粒度图像识别”项目的核心代码、训练方案、推理脚本和提交材料整理。

## 项目定位

任务目标是在仅使用比赛提供的网络弱监督训练数据、并遵守单模型提交限制的前提下，训练一个对标签噪声、类别长尾和细粒度类别差异更鲁棒的图像分类模型。

本公开仓库重点展示：

- 以 ConvNeXt / ConvNeXtV2 为 backbone，基于 ImageNet-1k 预训练权重构建细粒度图像分类模型；
- 围绕网络弱监督数据中的标签噪声、类别长尾和细粒度差异问题进行训练配置与增强策略优化；
- 整理训练脚本、推理脚本、提交文件说明和方法记录，便于回顾项目流程。

更完整的方法探索记录见 [docs/method_summary.md](docs/method_summary.md)，实验记录模板见 [docs/experiment_log.md](docs/experiment_log.md)。

## 仓库结构

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── wsdan_convnext_plus.py
├── scripts/
│   ├── train_web400.sh
│   ├── train_web5000.sh
│   ├── predict_web400.sh
│   └── predict_web5000.sh
├── docs/
│   ├── method_summary.md
│   ├── experiment_log.md
│   └── submission_checklist.md
├── submissions/
│   └── README.md
├── checkpoints/
│   └── .gitkeep
└── data/
    └── .gitkeep
```

## 环境安装

建议使用 Python 3.10+ 和 PyTorch CUDA 环境。

```bash
pip install -r requirements.txt
```

## 数据目录约定

训练数据建议按 `ImageFolder` 格式组织：

```text
data/
├── WebFG-400/
│   ├── train/
│   │   ├── 0000/
│   │   ├── 0001/
│   │   └── ...
│   └── test/
└── WebiNat-5000/
    ├── train/
    │   ├── 0000/
    │   ├── 0001/
    │   └── ...
    └── test/
```

测试集可以是扁平目录，也可以是多层目录，脚本会递归扫描图片文件。

## 训练示例

WebFG-400：

```bash
bash scripts/train_web400.sh
```

WebiNat-5000：

```bash
bash scripts/train_web5000.sh
```

## 推理示例

WebFG-400：

```bash
bash scripts/predict_web400.sh
```

WebiNat-5000：

```bash
bash scripts/predict_web5000.sh
```

## 提交文件

推理后应生成：

```text
pred_results_web400.csv
pred_results_web5000.csv
```

最终将两个 CSV 压缩为一个 zip 文件提交。

## 当前代码

主代码文件：`src/wsdan_convnext_plus.py`

它是单文件训练与推理脚本，主要支持训练和预测流程；部分实验辅助模式用于伪标签生成与权重平均尝试。

```bash
python src/wsdan_convnext_plus.py --mode train   ...
python src/wsdan_convnext_plus.py --mode predict ...
python src/wsdan_convnext_plus.py --mode pseudo  ...
python src/wsdan_convnext_plus.py --mode soup    ...
```

## 说明

- 本仓库为比赛项目整理版，不包含竞赛数据集、训练权重和最终提交压缩包。
- README 仅保留稳定概述，避免把所有实验性策略都表述为最终结论；具体尝试以 `docs/` 中记录为准。
