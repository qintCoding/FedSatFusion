# FedSatFusion

联邦卫星多模态影像融合与资源优化平台（FedSatFusion）

A Federated Learning Platform for Multi-modal Satellite Image Fusion and Resource Optimization

---

## 📖 项目简介 | Project Introduction
FedSatFusion 是一个基于联邦学习的多卫星多模态影像融合与资源分配优化平台，支持动态资源建模、深度强化学习优化、端到端遥感分类等功能。适用于遥感影像智能处理、分布式学习、资源受限场景下的多卫星协同。

FedSatFusion is a federated learning platform for multi-satellite, multi-modal image fusion and resource allocation optimization. It supports dynamic resource modeling, deep reinforcement learning optimization, and end-to-end remote sensing classification. It is suitable for intelligent remote sensing image processing, distributed learning, and resource-constrained multi-satellite collaboration.

---

## ✨ 主要特性 | Main Features
- 多卫星多模态影像融合（如光学、热红外等）
- 联邦学习与参数聚合机制
- 卫星动态资源（能量、计算、负载）建模
- 深度强化学习（DQN）驱动的资源分配优化
- 灵活的数据集与模型配置
- 支持断点训练与模型保存

---

## 🛠️ 环境依赖 | Environment Requirements
- Python >= 3.8
- PyTorch >= 1.10
- numpy, pyyaml, tqdm
- 推荐使用 Anaconda 环境

```bash
conda create -n fedsatfusion python=3.8
conda activate fedsatfusion
pip install torch torchvision numpy pyyaml tqdm
```

---

## 🚀 快速开始 | Quick Start
1. **准备数据集**（如 EuroSAT，或自定义多模态数据）
2. 配置 `config.yaml` 文件
3. 运行主训练脚本：

```bash
python federated_fedavg_train.py
```

---

## 📂 项目结构 | Project Structure
```
FedSatFusion/
├── config.yaml                # 配置文件
├── federated_fedavg_train.py  # 主训练入口
├── data/
│   └── eurosat_dataset.py     # 数据集定义
├── model/
│   └── fusion_model.py        # 融合模型
├── federated/
│   ├── client.py              # 客户端逻辑
│   ├── server.py              # 服务器与联邦控制
│   ├── environment.py         # 卫星环境与资源建模
│   └── drl_agent.py           # 深度强化学习代理
└── ...
```

---

## ⚙️ 配置说明 | Configuration
- `config.yaml` 包含数据、模型、训练、联邦学习等参数
- 主要参数：
  - `data_dir`：数据集路径
  - `input_channels`：各模态输入通道数
  - `feature_dim`：特征维度
  - `batch_size`、`epochs`、`learning_rate` 等训练参数
  - `num_clients`、`local_epochs`、`global_rounds` 等联邦参数

---

## 🧩 主要功能 | Main Functionalities
- **多模态影像融合**：支持光学、热红外等多源数据融合
- **联邦学习训练**：多客户端本地训练+全局参数聚合
- **动态资源建模**：每轮资源状态变化，模拟真实卫星
- **深度强化学习优化**：DQN智能分配计算与能量

---

## 📝 使用示例 | Usage Example
```python
# 只需运行主脚本
python federated_fedavg_train.py
```

---

## ⚠️ 注意事项 | Notes
- 数据集需提前准备并解压到指定目录
- 推荐使用GPU加速
- 配置文件参数可根据实际需求调整

---

## 📄 许可证 | License
MIT License

---

## 🤝 贡献指南 | Contributing
欢迎提交 issue、PR 或建议！

Welcome to submit issues, PRs, or suggestions! 