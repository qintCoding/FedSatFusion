#!/bin/bash

# 创建项目目录
mkdir -p FedSatFusion
cd FedSatFusion

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib
pip install pyyaml tqdm
pip install scikit-learn
pip install tensorboard

# 创建项目目录结构
mkdir -p data/sen12ms
mkdir -p models
mkdir -p federated
mkdir -p logs
mkdir -p checkpoints

# 创建requirements.txt
cat > requirements.txt << EOL
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.2
pandas>=1.2.4
matplotlib>=3.4.3
pyyaml>=5.4.1
tqdm>=4.62.3
scikit-learn>=0.24.2
tensorboard>=2.6.0
EOL

echo "项目环境设置完成！" 