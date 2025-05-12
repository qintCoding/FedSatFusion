import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModel(nn.Module):
    def __init__(self, num_classes=17, d=256, k=32):
        super(FusionModel, self).__init__()
        # SAR特征提取器
        self.sar_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, d, kernel_size=3, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 输出(B, d, 1, 1)
        )
        # Optical特征提取器
        self.optical_encoder = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, d, kernel_size=3, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 输出(B, d, 1, 1)
        )
        # DGA门控注意力
        self.Wg = nn.Linear(d, k)
        self.bg = nn.Parameter(torch.zeros(k))
        self.Wm_sar = nn.Linear(d, d)
        self.Wm_optical = nn.Linear(d, d)
        # 跨模态关联矩阵A（2模态，参数可学习）
        self.A = nn.Parameter(torch.ones(2, 2))
        # 特征对齐
        self.W_pub = nn.Linear(d, d)
        self.b_pub = nn.Parameter(torch.zeros(d))
        self.priv_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    def dga_attention(self, h_list):
        # h_list: [(B, d), ...]
        # 门控权重
        gate_logits = [self.Wg(h) + self.bg for h in h_list]  # [(B, k), ...]
        gate_logits = [torch.sigmoid(g) for g in gate_logits]  # GLU门控
        gate_scores = [g.mean(dim=1, keepdim=True) for g in gate_logits]  # (B, 1)
        gate_weights = torch.softmax(torch.cat(gate_scores, dim=1), dim=1)  # (B, M)
        return gate_weights
    def forward(self, sar, optical, missing_modalities=None):
        B = sar.size(0)
        d = self.Wg.in_features
        if missing_modalities is None:
            missing_modalities = []
        # 提取特征
        h_sar = self.sar_encoder(sar).view(B, -1)      # (B, d)
        h_optical = self.optical_encoder(optical).view(B, -1)  # (B, d)
        h_list = [h_sar, h_optical]
        # 缺失模态补偿
        # 0: sar, 1: optical
        for idx, name in enumerate(['sar', 'optical']):
            if name in missing_modalities:
                # 用A矩阵补偿
                # 只用另一个模态（K=1）
                other_idx = 1 - idx
                h_list[idx] = self.A[idx, other_idx] * h_list[other_idx]
        # DGA门控
        gate_weights = self.dga_attention(h_list)  # (B, 2)
        h_sar_proj = self.Wm_sar(h_list[0])
        h_optical_proj = self.Wm_optical(h_list[1])
        h_proj = torch.stack([h_sar_proj, h_optical_proj], dim=1)  # (B, 2, d)
        gate_weights = gate_weights.unsqueeze(-1)  # (B, 2, 1)
        h_fusion = (h_proj * gate_weights).sum(dim=1)  # (B, d)
        # 特征对齐
        h_pub = self.W_pub(h_fusion) + self.b_pub  # (B, d)
        h_priv = self.priv_mlp(h_fusion)           # (B, d)
        # 分类（只用公共特征）
        output = self.classifier(h_pub)
        return output 