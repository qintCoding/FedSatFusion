import torch
import yaml
from pathlib import Path
from data.eurosat_dataset import EuroSATDataset
from model.fusion_model import FusionModel
from federated.client import SatelliteClient
from federated.server import FederatedServer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import numpy as np

# 读取配置
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    config = load_config('config.yaml')
    device = torch.device(config['training']['device'])
    num_clients = config.get('federated', {}).get('num_clients', 3)
    local_epochs = config.get('federated', {}).get('local_epochs', 5)
    global_epochs = config.get('federated', {}).get('global_epochs', 10)
    
    # 创建数据集
    dataset = EuroSATDataset(
        root_dir=config['data']['data_dir'],
        transform=None,
        split='train'
    )
    
    # 创建模型
    model = FusionModel(
        num_classes=config['model']['num_classes'],
        d=config['model']['feature_dim'],  # 特征维度
        k=32,  # 门控维度
        optical_in_channels=config['model']['input_channels']['optical'],
        thermal_in_channels=config['model']['input_channels']['thermal']
    ).to(device)
    
    # 创建联邦学习服务器
    server = FederatedServer(
        model=model,
        device=device,
        num_clients=num_clients,
        local_epochs=local_epochs,
        global_epochs=global_epochs
    )
    
    # 创建并注册客户端
    data_len = len(dataset)
    lengths = [data_len // num_clients] * num_clients
    lengths[-1] += data_len - sum(lengths)  # 保证总数不变
    client_datasets = random_split(dataset, lengths)
    
    for i in range(num_clients):
        client_model = FusionModel(
            num_classes=config['model']['num_classes'],
            d=config['model']['feature_dim'],
            k=32,
            optical_in_channels=config['model']['input_channels']['optical'],
            thermal_in_channels=config['model']['input_channels']['thermal']
        ).to(device)
        
        client = SatelliteClient(
            client_id=i,
            dataset=client_datasets[i],
            model=client_model,
            device=device,
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers']
        )
        server.register_client(client)
    
    # 训练DRL代理
    print("开始训练深度强化学习代理...")
    server.train_drl()
    print("DRL代理训练完成\n")
    
    # 开始联邦学习
    print("开始联邦学习训练...")
    for epoch in range(global_epochs):
        print(f"\n开始第 {epoch + 1} 轮联邦训练")
        server.federated_round(
            dataset=dataset,
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers']
        )
        print(f"完成第 {epoch + 1} 轮联邦训练")
    
    print("\n联邦学习训练完成！")

if __name__ == '__main__':
    main() 