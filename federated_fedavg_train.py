import torch
import yaml
from pathlib import Path
from data.sen12ms_dataset import Sen12MSDataset
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
    local_epochs = config.get('federated', {}).get('local_epochs', 1)
    global_rounds = config.get('federated', {}).get('global_rounds', 10)

    # 加载数据集
    full_dataset = Sen12MSDataset(
        root_dir=config['data']['data_dir'],
        transform=None,
        split='train'
    )
    data_len = len(full_dataset)
    lengths = [data_len // num_clients] * num_clients
    lengths[-1] += data_len - sum(lengths)  # 保证总数不变
    client_datasets = random_split(full_dataset, lengths)

    # 创建全局模型和服务器
    global_model = FusionModel(num_classes=config['model']['num_classes'])
    server = FederatedServer(global_model, device, num_satellites=num_clients)

    # 创建客户端
    clients = []
    for i in range(num_clients):
        model = FusionModel(num_classes=config['model']['num_classes'])
        client = SatelliteClient(
            client_id=i,
            dataset=client_datasets[i],
            model=model,
            device=device,
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers']
        )
        clients.append(client)
        server.register_client(client)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 首先训练DRL代理
    print("开始训练深度强化学习代理...")
    server.train_drl()
    print("DRL代理训练完成")

    # 联邦训练主循环
    energy_threshold = config.get('federated', {}).get('energy_threshold', 0.2)
    gamma = config.get('federated', {}).get('gamma', 0.1)
    beta = config.get('federated', {}).get('beta', 0.1)
    for round in range(global_rounds):
        print(f'\n--- Global Round {round+1} ---')
        
        # 1. 使用DRL优化资源分配
        server.optimize_resources()
        
        # 2. 领导者选举
        leader_id = server.elect_leader()
        print(f'当前领导者卫星编号: {leader_id}')
        
        # 3. 分发全局参数
        server.distribute_global_params()
        
        # 4. 各客户端本地训练
        for client in clients:
            optimizer = optim.Adam(
                client.model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['optimizer']['weight_decay']
            )
            client.local_train(criterion, optimizer, epochs=local_epochs,
                              energy_threshold=energy_threshold, gamma=gamma, beta=beta)
        
        # 5. 聚合参数
        server.aggregate_params()
        print('Global aggregation done.')
        
        # 6. 打印所有客户端状态
        print("各客户端状态：")
        for client in clients:
            state = client.get_resource_state()
            print(f"  Client {client.client_id}: Energy={state['energy']:.3f}, "
                  f"Compute={state['compute']:.3f}, Load={state['load']:.3f}")
        
        # 7. 保存检查点
        if (round + 1) % 5 == 0:
            checkpoint_dir = Path(config['training']['checkpoint_dir'])
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save({
                'round': round,
                'model_state_dict': server.get_global_model().state_dict(),
                'drl_state_dict': server.drl_agent.policy_net.state_dict(),
            }, checkpoint_dir / f'checkpoint_round_{round+1}.pth')
    
    # 保存最终模型
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(server.get_global_model().state_dict(), checkpoint_dir / 'fedavg_final_model.pth')
    print('Final global model saved.')

if __name__ == '__main__':
    main() 