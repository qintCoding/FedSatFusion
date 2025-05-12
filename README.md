# FedSatFusion

基于联邦学习的卫星图像融合系统

## 环境要求

- Python 3.8+
- CUDA 11.0+ (如果使用GPU)
- 至少16GB RAM
- 至少100GB磁盘空间（用于存储数据集）

## 快速开始

1. 克隆项目：
```bash
git clone https://github.com/yourusername/FedSatFusion.git
cd FedSatFusion
```

2. 运行安装脚本：
```bash
chmod +x setup.sh
./setup.sh
```

3. 激活虚拟环境：
```bash
source venv/bin/activate
```

4. 开始训练：
```bash
python federated_fedavg_train.py
```

## 项目结构

```
FedSatFusion/
├── data/
│   └── sen12ms/          # Sen12MS数据集
├── models/               # 模型定义
├── federated/           # 联邦学习相关代码
├── logs/                # 训练日志
├── checkpoints/         # 模型检查点
├── config.yaml          # 配置文件
├── setup.sh            # 安装脚本
└── requirements.txt     # 项目依赖
```

## 数据集

本项目使用Sen12MS数据集，包含：
- Sentinel-1 SAR图像
- Sentinel-2光学图像
- 土地覆盖分类标签

数据集下载链接：https://dataserv.ub.tum.de/s/m1524895

## 配置说明

在`config.yaml`中可以配置：
- 训练参数
- 模型参数
- 联邦学习参数
- 资源管理参数

## 训练过程

1. DRL代理训练：
   - 训练深度强化学习代理优化资源分配
   - 训练轮数：100轮
   - 每轮步数：100步

2. 联邦学习训练：
   - 全局轮数：10轮
   - 本地训练轮数：5轮
   - 客户端数量：3个

## 监控和可视化

使用TensorBoard查看训练过程：
```bash
tensorboard --logdir=logs
```

## 注意事项

1. 确保有足够的磁盘空间存储数据集
2. 建议使用GPU进行训练
3. 可以根据需要调整配置文件中的参数
4. 定期备份模型检查点

## 许可证

MIT License 