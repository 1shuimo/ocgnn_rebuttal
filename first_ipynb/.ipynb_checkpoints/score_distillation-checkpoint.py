import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_ocgnn import Model
from utils import *

from sklearn.metrics import roc_auc_score
import random
import os
import dgl
from sklearn.metrics import average_precision_score
import argparse
from tqdm import tqdm

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2]))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set argument parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default='tf_finace')
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)

args = parser.parse_args()

if args.lr is None:
    if args.dataset in ['Amazon']:
        args.lr = 1e-3
    elif args.dataset in ['tf_finace']:
        args.lr = 5e-4
    elif args.dataset in ['reddit']:
        args.lr = 1e-3
    elif args.dataset in ['elliptic']:
        args.lr = 1e-3
    elif args.dataset in ['photo']:
        args.lr = 1e-3

if args.num_epoch is None:

    if args.dataset in ['reddit']:
        args.num_epoch = 500
    elif args.dataset in ['tf_finace']:
        args.num_epoch = 1500
    elif args.dataset in ['Amazon']:
        args.num_epoch = 800
    if args.dataset in ['elliptic']:
        args.num_epoch = 500
    elif args.dataset in ['photo']:
        args.num_epoch = 600

batch_size = args.batch_size
subgraph_size = args.subgraph_size

print('Dataset: ', args.dataset)

# Set random seed
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define loss function for OCGNN
def loss_func(emb):
    r = 0
    beta = 0.5
    warmup = 2
    eps = 0.001
    c = torch.zeros(args.embedding_dim)
    dist = torch.sum(torch.pow(emb.cpu() - c, 2), 1)
    score = dist - r ** 2
    loss = r ** 2 + 1 / beta * torch.mean(torch.relu(score))

    if warmup > 0:
        with torch.no_grad():
            warmup -= 1
            r = torch.quantile(torch.sqrt(dist), 1 - beta)
            c = torch.mean(emb, 0)
            c[(abs(c) < eps) & (c < 0)] = -eps
            c[(abs(c) < eps) & (c > 0)] = eps

    return loss, score

# Define score distillation loss function
def score_distillation_loss(score_t, score_s):
    loss = F.mse_loss(score_s, score_t)
    return loss

# Load and preprocess data
adj, features, labels, all_idx, idx_train, idx_val, \
idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(args.dataset)

if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
    features, _ = preprocess_features(features)
else:
    features = features.todense()

# Convert adjacency matrix to DGL graph
dgl_graph = adj_to_dgl_graph(adj)

# Prepare input tensors
nb_nodes = features.shape[0]
ft_size = features.shape[1]
adj = normalize_adj(adj)
adj = (adj + sp.eye(adj.shape[0])).todense()
features = torch.FloatTensor(features[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])

# Initialize models
model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
model_s = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)  # Independent Student model

# Initialize MLP for distillation
print('Initializing MLP for distillation...')
mlp = nn.Linear(args.embedding_dim, 1)  # Output a single score

# Initialize optimizers
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_s = torch.optim.Adam(model_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_mlp = torch.optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Move models to GPU if available
if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    model_s.cuda()
    mlp.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()

b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda() if torch.cuda.is_available() else torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()

import time
# 训练模型
print('Starting training loop...')
# 打开文件，以追加模式写入 (每次运行会追加新的内容)
output_file = "training_log.txt"

with open(output_file, "a") as f:
    # Train models
    with tqdm(total=args.num_epoch) as pbar:
        total_time = 0
        pbar.set_description('Training')
        for epoch in range(args.num_epoch):
            start_time = time.time()

            # 设置训练模式
            model.train()
            model_s.train()
            mlp.train()

            # 清除所有模型的梯度
            optimiser.zero_grad()      # 清除教师模型的梯度
            optimiser_s.zero_grad()    # 清除学生模型的梯度
            optimiser_mlp.zero_grad()  # 清除 MLP 的梯度

            # 教师模型的嵌入表示 (只用于正常节点)
            emb_t_normal = torch.squeeze(model(features, adj))
            emb_t_normal = emb_t_normal[normal_label_idx]

            # 计算教师模型的损失 (仅使用正常节点的嵌入表示)
            loss, _ = loss_func(emb_t_normal)

            # 教师模型的嵌入表示 (用于蒸馏损失计算 - 所有节点)
            emb_t_all = torch.squeeze(model(features, adj))

            # 学生模型的嵌入表示 (用于蒸馏损失计算 - 所有节点)
            emb_s_all = torch.squeeze(model_s(features, adj))

            _, teacher_score = loss_func(emb_t_all)

            student_score = mlp(emb_s_all).squeeze(dim=-1)

            # 计算蒸馏损失 (使用所有节点的嵌入)
            distillation_loss_value = score_distillation_loss(teacher_score, student_score)



            # 合并损失 (教师损失 + 蒸馏损失)，用于更新教师和学生模型
            total_loss = loss + distillation_loss_value

            # 对合并后的总损失进行反向传播并更新教师和学生模型
            total_loss.backward()  # 对合并损失进行反向传播
            optimiser.step()       # 更新教师模型的参数
            optimiser_s.step()     # 更新学生模型的参数

            # 重新计算蒸馏损失用于更新 MLP
            # 注意：此时计算图已经被释放，因此我们重新计算嵌入和蒸馏损失
            optimiser_mlp.zero_grad()  # 清除 MLP 的梯度

            # 重新计算教师和学生模型的嵌入 (用于更新 MLP)
            emb_t_all = torch.squeeze(model(features, adj))
            emb_s_all = torch.squeeze(model_s(features, adj))

            # 重新计算蒸馏损失
            teacher_loss, teacher_score = loss_func(emb_t_all)

            student_score = mlp(emb_s_all).squeeze(dim=-1)

            # 计算蒸馏损失 (使用所有节点的嵌入)
            distillation_loss_value = score_distillation_loss(teacher_score, student_score)

            # 对蒸馏损失进行反向传播并更新 MLP
            distillation_loss_value.backward()  # 反向传播蒸馏损失，用于更新 MLP
            optimiser_mlp.step()  # 更新 MLP 的参数



            if epoch % 5 == 0:

                log_message = (
                    f"Epoch {epoch}: Total Loss = {total_loss.item()}, Distillation Loss = { distillation_loss_value.item()}\n"
                )

                # Evaluate Student model
                model_s.eval()
                emb = torch.squeeze(model_s(features, adj))
                student_score = mlp(emb)  # Use MLP to compute scores for Student model
                logits = np.squeeze(student_score[idx_test].cpu().detach().numpy())
                auc = roc_auc_score(ano_label[idx_test], logits)
                log_message += f'Testing {args.dataset} AUC: {auc:.4f}\n'
                AP = average_precision_score(ano_label[idx_test], logits, average='macro', pos_label=1)
                log_message += f'Testing AP: {AP}\n'
                log_message += f'Total time is: {total_time}\n'

                            # 打印日志
                print(log_message)

                # 写入日志到文件
                f.write(log_message)
                f.flush()  # 确保内容立即写入文件

            end_time = time.time()
            total_time += end_time - start_time
            pbar.update(1)

            
print("Training completed.")
