## 项目简介
本项目基于多模态学习，探索文本与图像特征的融合方法，用于微地图分类任务。支持多种融合策略（如简单拼接、中间融合、早期融合、晚期融合）以及多种模型（BERT-ResNet、CLIP、ViLT）。通过消融实验和多模型比较，验证不同融合策略的性能。
## 功能特点
支持多种模型的训练与评估，包括：
    MultiModalModel（中间融合，BERT + ResNet）
    TextOnlyModel（仅文本，BERT）
    ImageOnlyModel（仅图像，ResNet）
    NoFusionLayerModel（无融合层）
    EarlyFusionModel（早期融合）
    LateFusionModel（晚期融合）
    CLIP（跨模态对比模型）
    ViLT（基于 Transformer 的视觉-文本模型）
自动化消融实验流程，输出模型性能指标（Accuracy、Precision、Recall、F1-Score）。
支持自定义数据集和动态模型选择。
## 文件结构
project/
│
├── Ablation_experiment.py      # 主程序，包含训练和评估逻辑
├── class_definition.py         # 模型和数据集的定义
├── eval_all_model.py           # CLIP 和 ViLT 的评估整合逻辑
├── train_data.csv              # 样本数据（文本、图像路径和标签）
├── result/                     # 评估结果输出目录
│   └── model_evaluation_results.csv  # 评估结果保存文件
└── model_data/                 # 模型权重文件和预训练模型存储路径
