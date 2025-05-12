import numpy as np
import torch
import torch.nn as nn

class SatelliteEnvironment:
    def __init__(self, num_satellites, state_dim=6, action_dim=3):
        self.num_satellites = num_satellites
        self.state_dim = state_dim  # [energy, compute, load, position_x, position_y, position_z]
        self.action_dim = action_dim  # [compute_allocation, energy_allocation, transmission_power]
        
        # 初始化卫星状态
        self.states = np.zeros((num_satellites, state_dim))
        self.initialize_states()
        
        # 通信拓扑矩阵
        self.communication_matrix = np.zeros((num_satellites, num_satellites))
        
        # 动态参数 - 调整恢复率和消耗率
        self.energy_recovery_rate = 0.05  # 增加能量恢复率
        self.compute_recovery_rate = 0.03  # 增加计算资源恢复率
        self.load_decay_rate = 0.02  # 增加负载衰减率
        self.position_update_rate = 0.01  # 位置更新率
        
        # 资源消耗系数
        self.compute_consumption_rate = 0.05  # 降低计算资源消耗率
        self.energy_consumption_rate = 0.03  # 降低能量消耗率
        self.transmission_consumption_rate = 0.02  # 降低传输功率消耗率
        
    def initialize_states(self):
        # 随机初始化卫星状态
        for i in range(self.num_satellites):
            self.states[i, 0] = np.random.uniform(0.5, 1.0)  # energy
            self.states[i, 1] = np.random.uniform(0.5, 1.0)  # compute
            self.states[i, 2] = np.random.uniform(0.0, 0.5)  # load
            # 随机初始化位置（简化版）
            self.states[i, 3:] = np.random.uniform(-1, 1, 3)  # position
            
    def update_communication_matrix(self):
        # 基于卫星位置更新通信拓扑
        for i in range(self.num_satellites):
            for j in range(self.num_satellites):
                if i != j:
                    distance = np.linalg.norm(self.states[i, 3:] - self.states[j, 3:])
                    # 通信概率随距离衰减
                    self.communication_matrix[i, j] = np.exp(-distance)
                    
    def step(self, actions):
        """
        执行动作并更新环境
        actions: shape (num_satellites, action_dim)
        """
        # 确保actions是二维数组
        if len(actions.shape) == 1:
            actions = actions.reshape(1, -1)
            
        rewards = np.zeros(self.num_satellites)
        done = False
        
        # 更新卫星状态
        for i in range(self.num_satellites):
            # 更新计算资源
            compute_change = actions[i, 0] * self.compute_consumption_rate
            self.states[i, 1] = np.clip(self.states[i, 1] + compute_change, 0, 1)
            
            # 更新能量
            energy_consumption = (
                actions[i, 0] * self.energy_consumption_rate +  # 计算消耗
                actions[i, 2] * self.transmission_consumption_rate  # 通信消耗
            )
            energy_recovery = self.energy_recovery_rate
            self.states[i, 0] = np.clip(self.states[i, 0] - energy_consumption + energy_recovery, 0, 1)
            
            # 更新负载
            load_increase = actions[i, 0] * 0.1  # 降低负载增加率
            load_decay = self.load_decay_rate
            self.states[i, 2] = np.clip(self.states[i, 2] + load_increase - load_decay, 0, 1)
            
            # 更新位置（模拟卫星运动）
            position_noise = np.random.normal(0, self.position_update_rate, 3)
            self.states[i, 3:] = np.clip(self.states[i, 3:] + position_noise, -1, 1)
            
            # 计算奖励
            rewards[i] = self.calculate_reward(i, actions[i])
            
        # 更新通信拓扑
        self.update_communication_matrix()
        
        # 检查是否结束
        if np.any(self.states[:, 0] < 0.1):  # 如果有卫星能量过低
            done = True
            
        return self.states.copy(), rewards, done
        
    def calculate_reward(self, satellite_id, action):
        """
        计算单个卫星的奖励
        """
        energy = self.states[satellite_id, 0]
        compute = self.states[satellite_id, 1]
        load = self.states[satellite_id, 2]
        
        # 计算通信效率
        comm_efficiency = np.mean(self.communication_matrix[satellite_id])
        
        # 奖励计算
        reward = (
            0.3 * energy +  # 能量奖励
            0.2 * compute +  # 计算能力奖励
            0.2 * (1 - load) +  # 负载惩罚
            0.2 * comm_efficiency -  # 通信效率奖励
            0.1 * action[2]  # 通信功率惩罚
        )
        
        return reward
        
    def get_state(self):
        return self.states.copy()
        
    def reset(self):
        self.initialize_states()
        self.update_communication_matrix()
        return self.states.copy() 