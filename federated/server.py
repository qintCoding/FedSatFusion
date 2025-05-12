import copy
import torch
import numpy as np
from .environment import SatelliteEnvironment
from .drl_agent import DRLAgent

class FederatedServer:
    def __init__(self, global_model, device, num_satellites=3):
        self.global_model = global_model
        self.device = device
        self.clients = []
        self.leader_id = None
        self.leader_score = None
        self.delta = 0.9  # 衰减因子
        
        # 初始化环境和DRL代理
        self.env = SatelliteEnvironment(num_satellites)
        self.drl_agent = DRLAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            device=device
        )
        
        # 训练参数
        self.drl_episodes = 100
        self.drl_steps = 100
        
    def register_client(self, client):
        self.clients.append(client)
        
    def distribute_global_params(self):
        global_params = self.global_model.state_dict()
        for client in self.clients:
            client.set_model_params(global_params)
            
    def aggregate_params(self):
        # 只聚合W_pub和b_pub参数
        global_dict = copy.deepcopy(self.global_model.state_dict())
        pub_keys = [k for k in global_dict.keys() if 'W_pub' in k or 'b_pub' in k]
        for k in pub_keys:
            global_dict[k] = torch.stack([client.get_model_params()[k] for client in self.clients], dim=0).mean(dim=0)
        # 其余参数保持原值
        self.global_model.load_state_dict(global_dict)
        
    def get_global_model(self):
        return self.global_model
        
    def score_client(self, client, lambda_=0.7, nu=0.3):
        # 只用能量和负载，后续可扩展
        # Score = λ * energy + (1-λ) * (1/load) - ν * load
        energy_score = client.energy  # 假设已归一化
        load_score = 1.0 / (client.load + 1e-6)  # 防止除0
        score = lambda_ * energy_score + (1 - lambda_) * load_score - nu * client.load
        return score
        
    def train_drl(self):
        """训练深度强化学习代理"""
        for episode in range(self.drl_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(self.drl_steps):
                # 选择动作
                action = self.drl_agent.select_action(state)
                
                # 执行动作
                next_state, rewards, done = self.env.step(action)
                
                # 存储经验
                for i in range(len(self.clients)):
                    self.drl_agent.store_transition(
                        state[i], action[i], rewards[i], next_state[i], done
                    )
                
                # 训练网络
                loss = self.drl_agent.train()
                
                # 更新状态和奖励
                state = next_state
                episode_reward += np.mean(rewards)
                
                # 更新目标网络
                if step % self.drl_agent.target_update == 0:
                    self.drl_agent.update_target_network()
                    
                if done:
                    break
                    
            # 更新探索率
            self.drl_agent.update_epsilon()
            
            print(f"Episode {episode+1}, Reward: {episode_reward:.2f}, Epsilon: {self.drl_agent.epsilon:.2f}")
            
    def optimize_resources(self):
        """使用训练好的DRL代理优化资源分配"""
        state = self.env.get_state()
        actions = self.drl_agent.select_action(state, training=False)
        
        # 将动作应用到客户端
        for i, client in enumerate(self.clients):
            # 更新客户端资源状态
            client.update_dynamic_state(
                energy=state[i, 0],
                compute=state[i, 1],
                load=state[i, 2]
            )
            
            # 应用资源分配动作
            client.apply_resource_action({
                'compute': actions[i, 0],
                'energy': actions[i, 1],
                'transmit_power': actions[i, 2]
            })
            
    def federated_round(self):
        """执行一轮联邦学习，包含资源优化"""
        # 1. 使用DRL优化资源分配
        self.optimize_resources()
        
        # 2. 分发全局参数
        self.distribute_global_params()
        
        # 3. 客户端本地训练
        for client in self.clients:
            client.local_train()
            
        # 4. 聚合参数
        self.aggregate_params()
        
        # 5. 更新环境状态
        self.env.step(np.zeros((len(self.clients), self.env.action_dim)))

    def elect_leader(self):
        scores = [self.score_client(c) for c in self.clients]
        max_score = max(scores)
        max_idx = scores.index(max_score)
        # 衰减机制，防止频繁切换
        if self.leader_id is not None and self.leader_score is not None:
            if self.leader_score > self.delta * max_score:
                # 保持原领导者
                return self.leader_id
        # 否则切换
        self.leader_id = max_idx
        self.leader_score = max_score
        return self.leader_id 