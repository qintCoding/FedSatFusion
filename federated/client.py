import torch
from torch.utils.data import DataLoader
import random
import numpy as np

class SatelliteClient:
    def __init__(self, client_id, dataset, model, device, batch_size=16, num_workers=2,
                 energy=None, compute=None, load=None):
        self.client_id = client_id
        self.dataset = dataset
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # 初始化数据加载器
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        
        # 动态资源状态
        self.energy = energy if energy is not None else np.random.uniform(0.5, 0.8)  # 提高初始能量
        self.compute = compute if compute is not None else np.random.uniform(0.5, 0.8)  # 提高初始计算资源
        self.load = load if load is not None else np.random.uniform(0.1, 0.3)  # 降低初始负载
        
        # 资源阈值
        self.energy_threshold = 0.2  # 降低能量阈值
        self.compute_threshold = 0.2  # 降低计算资源阈值
        self.max_load = 0.8  # 最大负载阈值
        
        # 资源消耗系数
        self.energy_consumption_rate = 0.05  # 降低能量消耗率
        self.compute_consumption_rate = 0.05  # 降低计算资源消耗率
        self.load_increase_rate = 0.01  # 降低负载增加率
        
    def _update_resources(self):
        """更新资源状态"""
        # 能量消耗
        self.energy = max(0.0, self.energy - self.energy_consumption_rate)
        
        # 计算资源消耗
        self.compute = max(0.0, self.compute - self.compute_consumption_rate)
        
        # 负载增加
        self.load = min(self.max_load, self.load + self.load_increase_rate)
        
    def local_train(self, criterion, optimizer, epochs=1, energy_threshold=0.2, gamma=0.1, beta=0.1):
        # 能量安全约束
        if self.energy < energy_threshold:
            print(f"[Client {self.client_id}] 能量不足，跳过本轮本地训练 (energy={self.energy:.3f})")
            return
            
        self.model.train()
        self.model.to(self.device)
        
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(self.dataloader):
                # 检查资源是否足够
                if self.energy < energy_threshold:
                    print(f"[Client {self.client_id}] 训练过程中能量不足，停止训练 (energy={self.energy:.3f})")
                    break
                    
                s1_data = batch['sar'].to(self.device)
                s2_data = batch['optical'].to(self.device)
                labels = batch['label'].to(self.device)
                labels = labels - 1  # 标签归一化
                labels = labels.view(-1)
                
                optimizer.zero_grad()
                outputs = self.model(s1_data, s2_data)
                outputs = outputs.unsqueeze(1).unsqueeze(1)
                outputs = outputs.expand(-1, 256, 256, -1)
                outputs = outputs.reshape(-1, outputs.size(-1))
                
                task_loss = criterion(outputs, labels)
                
                # 多目标损失：任务损失 + 负载正则 + 能量正则
                load_reg = gamma * self.load
                energy_reg = beta * (1.0 - self.energy)  # 能量越低惩罚越大
                loss = task_loss + load_reg + energy_reg
                
                loss.backward()
                optimizer.step()
                
                # 更新资源状态
                self._update_resources()
                
                if batch_idx % 5 == 0:
                    print(f"[Client {self.client_id}] Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f} "
                          f"Energy: {self.energy:.3f} Load: {self.load:.3f}")
                          
    def update_dynamic_state(self, energy=None, compute=None, load=None):
        """更新动态状态"""
        if energy is not None:
            self.energy = np.clip(energy, 0, 1)
        if compute is not None:
            self.compute = np.clip(compute, 0, 1)
        if load is not None:
            self.load = np.clip(load, 0, self.max_load)
            
    def apply_resource_action(self, action):
        """应用资源分配动作"""
        # 更新计算资源
        compute_change = action['compute'] * self.compute_consumption_rate
        self.compute = np.clip(self.compute + compute_change, 0, 1)
        
        # 更新能量
        energy_consumption = (
            action['compute'] * self.energy_consumption_rate +
            action['transmit_power'] * 0.02  # 降低传输功率消耗
        )
        self.energy = np.clip(self.energy - energy_consumption, 0, 1)
        
        # 更新负载
        load_increase = action['compute'] * 0.1  # 降低负载增加率
        self.load = np.clip(self.load + load_increase, 0, self.max_load)
            
    def get_resource_state(self):
        """获取当前资源状态"""
        return {
            'energy': self.energy,
            'compute': self.compute,
            'load': self.load
        }

    def get_model_params(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def set_model_params(self, params):
        self.model.load_state_dict(params)

    def calc_reward(self, accuracy, time_cost, energy_cost, w_acc=1.0, w_time=1.0, w_energy=1.0):
        # 奖励函数示例，可根据实际需求调整
        reward = w_acc * accuracy - w_time * time_cost - w_energy * energy_cost
        return reward 