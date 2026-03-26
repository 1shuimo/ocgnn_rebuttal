import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models.model_dominant_edge import Model_dominant
from utils.utils import *
import pickle

from sklearn.metrics import roc_auc_score
import random
import os
import dgl
from sklearn.metrics import average_precision_score
import argparse
from tqdm import tqdm
import time

# 🚨 为了在没有GPU的环境下也能运行，您可以注释掉下面这行
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2])) 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set argument
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

# 根据数据集设置学习率
if args.lr is None:
    if args.dataset in ['Amazon']:
        args.lr = 1e-3
    elif args.dataset in ['tf_finace']:
        args.lr = 1e-4
    elif args.dataset in ['reddit']:
        args.lr = 1e-3
    elif args.dataset in ['elliptic']:
        args.lr = 1e-3
    elif args.dataset in ['photo']:
        args.lr = 1e-3
    elif args.dataset in ['tolokers']:
        args.lr = 1e-3
    elif args.dataset in ['YelpChi-all']:
        args.lr = 1e-3

# 根据数据集设置训练轮数
if args.num_epoch is None:
    if args.dataset in ['reddit']:
        args.num_epoch = 500
    elif args.dataset in ['tf_finace']:
        args.num_epoch = 1500
    elif args.dataset in ['Amazon']:
        args.num_epoch = 800
    elif args.dataset in ['elliptic']:
        args.num_epoch = 500
    elif args.dataset in ['photo']:
        args.num_epoch = 600
    elif args.dataset in ['tolokers']:
        args.num_epoch = 1500
    elif args.dataset in ['YelpChi-all']:
        args.num_epoch = 1500

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

# 🚨 修改开始: 这是代码的核心修改区域
# -------------------------------------------------------------------

# Load and preprocess data
adj, features, labels, all_idx, idx_train, idx_val, \
idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(args.dataset)

# 1. 处理特征矩阵 (features)
print("正在处理特征矩阵...")
if args.dataset in ['reddit', 'elliptic']:
    # 对特定数据集进行行归一化并转为密集格式
    features, _ = preprocess_features(features)
else:
    # 对于其他数据集，如果特征是稀疏的，需要转为密集数组
    # 这里的 .todense() 或 .toarray() 是必要的
    features = features.toarray()

# dgl_graph = adj_to_dgl_graph(adj) # 如果不使用DGL模型，此行可以注释掉

nb_nodes = features.shape[0]
ft_size = features.shape[1]

# 2. 处理邻接矩阵 (adj) - 核心内存优化
print("正在处理邻接矩阵 (内存优化)...")
# 标准做法是先加自环，再归一化
adj_with_selfloops = adj + sp.eye(adj.shape[0])
adj_normalized_sparse = normalize_adj(adj_with_selfloops)

# 从归一化后的稀疏矩阵创建 GNN 需要的 edge_index
coo = adj_normalized_sparse.tocoo()
edge_index = torch.LongTensor(np.vstack((coo.row, coo.col)))

# 3. 转换为 PyTorch Tensors
print("正在将数据转换为 PyTorch Tensors...")
features = torch.FloatTensor(features) # 直接转为2D Tensor
labels = torch.FloatTensor(labels[np.newaxis])

# 注意：我们不再创建和使用 adj 和 raw_adj 的密集张量，从而节省大量内存

# -------------------------------------------------------------------
# 🚨 修改结束

# Initialize model and optimiser
model = Model_dominant(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout, args.dataset)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    # 🚨 修改: 将 edge_index 移到 GPU，而不是 adj
    edge_index = edge_index.cuda()
    labels = labels.cuda()

# Train model
output_file = f"{args.dataset}_dominant_github.txt"
best_auc_window = 0.0
best_epoch_window = -1
target_epoch = args.num_epoch//2
window = args.num_epoch//2
weight_save_path = f"./dominant_teacher_best_pth/{args.dataset}_dominant_teacher_best.pth"

with open(output_file, "a") as f:
    with tqdm(total=args.num_epoch) as pbar:
        total_time = 0
        pbar.set_description('Training')
        for epoch in range(args.num_epoch):
            start_time = time.time()
            model.train()
            optimiser.zero_grad()

            # 🚨 修改: Train model，传入 edge_index 而不是 adj
            loss, score_train, score_test, emb, seq1, x_ = model(features, edge_index, idx_train, idx_test)

            loss.backward()
            optimiser.step()
            if epoch >= target_epoch + window:
                break
            
            log_message = (
                f"Epoch {epoch}: Total Loss = {loss.item()}\n"
            )
            model.eval()

            # 重新计算测试分数用于评估
            with torch.no_grad():
                # 🚨 修改: 传入 edge_index 而不是 adj
                _, _, score_test, _, _, _ = model(features, edge_index, idx_train, idx_test)
                logits = score_test.cpu().detach().numpy()
                auc = roc_auc_score(ano_label[idx_test], logits)
                log_message += f"Testing {args.dataset} AUC: {auc}\n"
                AP = average_precision_score(ano_label[idx_test], logits, average='macro', pos_label=1)
                log_message += f"Testing AP: {AP}\n"

            # 在window区间内，每次有更好AUC就保存并覆盖权重文件
            if (epoch >= target_epoch - window) and (epoch <= target_epoch + window):
                if auc > best_auc_window:
                    best_auc_window = auc
                    best_epoch_window = epoch
                    torch.save(model.state_dict(), weight_save_path)
                    print(f"✅ New best model saved at epoch {epoch} with AUC {auc:.4f}.")
                
            print(log_message)
            f.write(log_message)
            f.flush()

            end_time = time.time()
            total_time += end_time - start_time
            pbar.update(1)

print("Training completed.")
if best_epoch_window != -1:
    print(f"✅ Best model in [{target_epoch-window}, {target_epoch+window}] saved at epoch {best_epoch_window} with AUC {best_auc_window:.4f}.")
else:
    print("No model was saved in the specified window.")