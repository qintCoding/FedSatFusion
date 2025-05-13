#!/bin/bash

# Create project directory
mkdir -p FedSatFusion
cd FedSatFusion

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib
pip install pyyaml tqdm
pip install scikit-learn
pip install tensorboard
pip install opencv-python
pip install albumentations
pip install Pillow

# Create project directory structure
mkdir -p data/eurosat
mkdir -p model
mkdir -p federated
mkdir -p logs
mkdir -p checkpoints

# Create requirements.txt
cat > requirements.txt << EOL
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.2
scikit-learn>=0.24.2
tqdm>=4.62.3
matplotlib>=3.4.3
Pillow>=8.3.2
pyyaml>=5.4.1
tensorboard>=2.6.0
pandas>=1.2.4
opencv-python>=4.5.3
albumentations>=1.0.3
EOL

echo "Project environment setup completed!" 