# 文件名: model_dominant_best.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN

# --- 辅助模块 ---

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)

class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values

class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values

class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out

class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)
        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))
        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))
        logits = torch.cat(tuple(scs))
        return logits

# --- 主模型 ---

class Model_dominant(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout):
        super(Model_dominant, self).__init__()
        self.read_mode = readout
        
        # 结构编码器 (GCN)
        self.dense_stru = nn.Linear(n_in, n_h)
        self.gcn = GCN(n_h, n_h, num_layers=2) # GCN层输入输出维度都是n_h
        
        # 属性编码器 (MLP)
        self.dense_attr_1 = nn.Linear(n_in, n_h)
        self.dense_attr_2 = nn.Linear(n_h, n_in)
        
        self.dropout = 0.
        self.act = nn.ReLU()

        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()
        
        self.disc = Discriminator(n_h, negsamp_round)

    def double_recon_loss(self, x, x_, weight=1.0):
        # 属性重构损失
        diff_attr = torch.pow(x - x_, 2)
        attr_error = torch.sqrt(torch.sum(diff_attr, 1))
        score = weight * attr_error
        return score

    def model_enc(self, x, edge_index):
        # 结构编码
        h = self.act(self.dense_stru(x))
        h = F.dropout(h, self.dropout)
        h_emb = self.gcn(h, edge_index)

        # 属性重构
        x_attr = self.act(self.dense_attr_1(x))
        x_attr = F.dropout(x_attr, self.dropout)
        x_reconstructed = self.dense_attr_2(x_attr)
        
        # 在这个模型中，结构重构 s_ 并未被有效使用，所以我们不返回它
        # s_reconstructed = torch.sigmoid(h_emb @ h_emb.T)
        
        return x_reconstructed, h_emb

    def forward(self, seq1, edge_index, idx_train, idx_test):
        seq1 = torch.squeeze(seq1)
        
        # 直接使用传入的edge_index进行编码和重构
        x_reconstructed, h_emb = self.model_enc(seq1, edge_index)

        # 计算训练集的损失
        score_train = self.double_recon_loss(seq1[idx_train, :], x_reconstructed[idx_train, :])
        loss = torch.mean(score_train)
        
        # 计算测试集的异常分数（用于评估）
        score_test = self.double_recon_loss(seq1[idx_test, :], x_reconstructed[idx_test, :])

        return loss, score_train, score_test, h_emb, seq1, x_reconstructed