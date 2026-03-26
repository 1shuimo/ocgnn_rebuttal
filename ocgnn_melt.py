# train_student.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import random
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from model_ocgnn import Model_ocgnn
from utils import *
import os
import pickle

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2]))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===========================
# 参数设置
# ===========================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='reddit')
parser.add_argument('--teacher_path', type=str, default='reddit_ocgnn_teacher_final.pth')
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

# 根据数据集设置学习率
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
    elif args.dataset in ['tolokers']:
        args.lr = 1e-3
    elif args.dataset in ['YelpChi-all']:
        args.lr = 1e-3

# 根据数据集设置训练轮数
if args.num_epoch is None:
    if args.dataset in ['reddit']:
        args.num_epoch = 1000
    elif args.dataset in ['tf_finace']:
        args.num_epoch = 2500
    elif args.dataset in ['Amazon']:
        args.num_epoch = 1300
    elif args.dataset in ['elliptic']:
        args.num_epoch = 2000
    elif args.dataset in ['photo']:
        args.num_epoch = 2000
    elif args.dataset in ['tolokers']:
        args.num_epoch = 1500
    elif args.dataset in ['YelpChi-all']:
        args.num_epoch = 1500
        
# 噪声参数设置
if args.dataset in ['reddit', 'Photo']:
    args.mean = 0.02
    args.var = 0.01
else:
    args.mean = 0.0
    args.var = 0.0

batch_size = args.batch_size
subgraph_size = args.subgraph_size

print('Dataset: ', args.dataset) 

# ===========================
# 设置随机种子
# ===========================
print('Setting random seeds...')
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

# ===========================
# 辅助函数定义
# ===========================

def distillation_loss_emb(emb_t, emb_s):
    """计算embedding之间的蒸馏损失 (MSE)"""
    loss = F.mse_loss(emb_s, emb_t, reduction='mean')
    return loss

def score_distillation_loss(score_t, score_s):
    """计算分数之间的蒸馏损失 (MSE)"""
    loss = F.mse_loss(score_s, score_t, reduction='mean')
    return loss

def min_max_normalize(tensor):
    """最小-最大归一化"""
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    if max_val == min_val:
        return torch.zeros_like(tensor)
    else:
        return (tensor - min_val) / (max_val - min_val)

def loss_func(emb):
    """OCGNN的损失函数 (One-Class目标)"""
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

    return loss, score, c, r

def kl_loss_student_teacher(student_score, teacher_score, eps=1e-8):
    """计算KL散度损失"""
    student_prob = torch.clamp(student_score, min=eps, max=1)
    teacher_prob = torch.clamp(teacher_score, min=eps, max=1)
    student_prob = student_prob / (student_prob.sum() + eps)
    teacher_prob = teacher_prob / (teacher_prob.sum() + eps)
    kl = F.kl_div(student_prob.log(), teacher_prob, reduction='batchmean')
    return kl

def to_hard_label_exclude_normal(score, normal_idx, top_percent=0.05):
    """
    生成硬标签：只在非normal_idx的节点中选择top_percent分数最高的标记为异常(1)
    
    Args:
        score: [N] 节点异常分数
        normal_idx: 已知正常节点的索引
        top_percent: 选择前百分之多少标记为异常
    
    Returns:
        hard_label: [N] 0/1硬标签，1表示异常
    """
    N = score.shape[0]
    all_idx = set(range(N))
    normal_set = set(normal_idx)
    candidate_idx = list(all_idx - normal_set)  # 候选异常节点：所有节点 - 已知正常节点
    candidate_scores = score[candidate_idx]
    top_k = int(len(all_idx) * top_percent)  # 计算top5%的节点数量
    if top_k < 1:
        top_k = 1
    _, indices = torch.topk(candidate_scores, top_k)  # 选择top_k个分数最高的
    hard_label = torch.zeros_like(score)
    selected_idx = torch.tensor(candidate_idx, device=score.device)[indices]
    hard_label[selected_idx] = 1.0  # 标记为异常
    return hard_label

def get_all_normal_nodes(emb_s_all, teacher_hard):
    """
    获取所有normal节点的索引
    
    Args:
        emb_s_all: student模型的embedding [N, D]
        teacher_hard: teacher生成的硬标签 [N], 1表示异常
    
    Returns:
        all_normal_idx: 所有normal节点的索引列表
        
    逻辑：
        - emb_s_all包含原始图的所有N个节点
        - teacher_hard标记了其中top5%的异常节点
        - 所有normal节点 = 全部节点 - top5%异常节点
        - 包括: labeled normal + unlabeled normal
        - 伪异常节点是单独concat的，不在emb_s_all中
    """
    total_nodes = emb_s_all.shape[0]  # 原始图的节点数N
    all_idx = set(range(total_nodes))
    
    # 找出top5%异常节点的索引
    top5_abnormal_idx = set(torch.where(teacher_hard == 1)[0].cpu().numpy())
    
    # 所有normal节点 = 全部节点 - top5%异常节点
    all_normal_idx = list(all_idx - top5_abnormal_idx)
    
    return all_normal_idx

# ===========================
# 数据加载和预处理
# ===========================
print('Loading and preprocessing data...')
adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(args.dataset)

# 预处理特征
if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
    print('Preprocessing features...')
    features, _ = preprocess_features(features)
else:
    features = features.todense()

# 转换邻接矩阵为DGL图（如果需要的话）
print('Converting adjacency matrix to DGL graph...')
# 转换邻接矩阵为DGL图 - 带缓存
dgl_graph_path = f"./cache/{args.dataset}_dgl_graph.pkl"
os.makedirs("./cache", exist_ok=True)

if os.path.exists(dgl_graph_path):
    print(f'Loading cached DGL graph from {dgl_graph_path}...')
    with open(dgl_graph_path, 'rb') as f:
        dgl_graph = pickle.load(f)
    print('DGL graph loaded from cache.')
else:
    print('Converting adjacency matrix to DGL graph...')
    dgl_graph = adj_to_dgl_graph(adj)
    print('Saving DGL graph to cache...')
    with open(dgl_graph_path, 'wb') as f:
        pickle.dump(dgl_graph, f)
    print(f'DGL graph saved to {dgl_graph_path}')

# 准备输入张量
print('Preparing input tensors...')
nb_nodes = features.shape[0]
ft_size = features.shape[1]
raw_adj = adj
adj = normalize_adj(adj)

raw_adj = (raw_adj + sp.eye(raw_adj.shape[0])).todense()
adj = (adj + sp.eye(adj.shape[0])).todense()
features = torch.FloatTensor(features[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
raw_adj = torch.FloatTensor(raw_adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])

# ===========================
# 模型初始化
# ===========================
print('Initializing models...')

# Teacher模型 (ocgnn) - 加载权重
model_t = Model_ocgnn(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
model_t.load_state_dict(torch.load(args.teacher_path))
model_t.eval()
for p in model_t.parameters():
    p.requires_grad = False

# Student模型 (OCGNN) - 需要训练
model_s = Model_ocgnn(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)  

# 蒸馏用的MLP
print('Initializing MLP for distillation...')
mlp_s = nn.Linear(args.embedding_dim, 1)  # 将embedding映射到异常分数

# ===========================
# 优化器初始化
# ===========================
print('Initializing optimizers...')
optimiser = torch.optim.Adam(model_t.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_s = torch.optim.Adam(model_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_mlp_s = torch.optim.Adam(mlp_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# 损失函数
b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda() if torch.cuda.is_available() else torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()

# ===========================
# 训练循环
# ===========================
print("\n🔁 Starting Student Training...")
output_file = f"./ocgnn_melt_data_enhance/{args.dataset}_melt.txt"
with open(output_file, "a") as f:
    with tqdm(total=args.num_epoch) as pbar:
        total_time = 0
        pbar.set_description('Training')
        for epoch in range(args.num_epoch):
            start_time = time.time()
            
            # 设置训练模式
            model_s.train()
            mlp_s.train()
            optimiser_s.zero_grad()
            optimiser_mlp_s.zero_grad()

            with torch.no_grad():
                # Teacher模型输出
                _, emb_t_all_raw = model_t(features, adj)
                emb_t_all = emb_t_all_raw.squeeze(0)  # [N, D]
                # 用loss_func得到teacher分数
                _, score_from_ocgnn_non_normalize, _, _ = loss_func(emb_t_all)  # [N]

                # 计算teacher AUC
                logits_teacher = np.squeeze(score_from_ocgnn_non_normalize[idx_test].cpu().detach().numpy())
                auc_teacher = roc_auc_score(ano_label[idx_test], logits_teacher)
                log_message = f"Testing {args.dataset} Teacher AUC: {auc_teacher:.4f}\n"
                teacher_score = min_max_normalize(score_from_ocgnn_non_normalize)  # [N] 归一化分数

            # Student模型输出
            _, emb_s_all_raw = model_s(features, adj)
            emb_s_all = emb_s_all_raw.squeeze(0)  # [N, D]
            student_score_non_normalize = mlp_s(emb_s_all).squeeze(dim=-1)  # [N]
            student_score = min_max_normalize(student_score_non_normalize)  # [N] 归一化分数

            # # === 生成硬标签 ===
            # teacher_hard = to_hard_label_exclude_normal(score_from_ocgnn_non_normalize, normal_label_idx, top_percent=0.05)  # [N]
            # all_normal_idx = get_all_normal_nodes(emb_s_all, teacher_hard)

            # # === 数据增强部分 ===
            # all_normal_features = features.squeeze()[all_normal_idx]  # [N_normal, D]
            # mask = torch.rand_like(all_normal_features) > 0.3
            # masked_normal_features = all_normal_features * mask
            # masked_features = features.clone().squeeze()
            # masked_features[all_normal_idx] = masked_normal_features
            # masked_features = masked_features.unsqueeze(0)  # [1, N, D]
            # _, emb_s_augmented_raw = model_s(masked_features, adj)
            # emb_s_augmented = emb_s_augmented_raw.squeeze(0)  # [N, D]
            # emb_s_normal_original = emb_s_all[all_normal_idx]
            # emb_s_normal_augmented = emb_s_augmented[all_normal_idx]
            # reg2_mse = F.mse_loss(emb_s_normal_augmented, emb_s_normal_original, reduction='mean')

            # === 分数蒸馏MSE ===
            mse_loss = score_distillation_loss(teacher_score, student_score)

            # === 总损失 ===
            total_loss = mse_loss 

            # 反向传播和参数更新
            total_loss.backward()
            optimiser_s.step()
            optimiser_mlp_s.step()

            # === 评估和日志 (每5个epoch) ===
            if epoch % 5 == 0:
                log_message += (
                    f"Epoch {epoch}: Total Loss = {total_loss.item()}\n"
                    f"MSE Loss = {mse_loss.item()}\n"
                )
                
                # 切换到评估模式
                model_s.eval()

                _, emb_s_all_raw = model_s(features, adj)
                emb_s_all = emb_s_all_raw.squeeze(0)  # [N, D]
                student_score_non_normalize = mlp_s(emb_s_all).squeeze(dim=-1)
                student_score = min_max_normalize(student_score_non_normalize)
                
                score_from_ocgnn = torch.sigmoid(score_from_ocgnn_non_normalize)

                # 记录平均分数
                log_message += f'student_score: {torch.mean(student_score)}\n'
                log_message += f'ocgnn_score: {torch.mean(score_from_ocgnn)}\n'
                log_message += f'student_score_non_normalize: {torch.mean(student_score_non_normalize)}\n'
                log_message += f'ocgnn_score_non_normalize: {torch.mean(score_from_ocgnn_non_normalize)}\n'

                # 计算组合分数
                student_minus_ocgnn_score = abs(student_score - score_from_ocgnn)
                stu_add_ocgnn_score = student_score + score_from_ocgnn
                stu_add_ocgnn_non_normalize_score = student_score_non_normalize + score_from_ocgnn_non_normalize
                student_minus_ocgnn_non_normalize_score = student_score_non_normalize - score_from_ocgnn_non_normalize

                # 提取测试集分数
                logits_stu = np.squeeze(student_score[idx_test].cpu().detach().numpy())
                logits_stu_non_normalize = np.squeeze(student_score_non_normalize[idx_test].cpu().detach().numpy())
                logits_stu_ocgnn = np.squeeze(stu_add_ocgnn_score[idx_test].cpu().detach().numpy())
                logits_stu_ocgnn_non_normalize = np.squeeze(stu_add_ocgnn_non_normalize_score[idx_test].cpu().detach().numpy())
                logits_stu_minus_ocgnn = np.squeeze(student_minus_ocgnn_score[idx_test].cpu().detach().numpy())
                logits_stu_minus_ocgnn_non_normalize = np.squeeze(student_minus_ocgnn_non_normalize_score[idx_test].cpu().detach().numpy())

                # 计算AUC
                auc_stu = roc_auc_score(ano_label[idx_test], logits_stu)
                auc_stu_non_normalize = roc_auc_score(ano_label[idx_test], logits_stu_non_normalize)
                auc_stu_ocgnn = roc_auc_score(ano_label[idx_test], logits_stu_ocgnn)
                auc_stu_ocgnn_non_normalize = roc_auc_score(ano_label[idx_test], logits_stu_ocgnn_non_normalize)
                auc_stu_minus_ocgnn = roc_auc_score(ano_label[idx_test], logits_stu_minus_ocgnn)
                auc_stu_minus_ocgnn_non_normalize = roc_auc_score(ano_label[idx_test], logits_stu_minus_ocgnn_non_normalize)

                # 记录AUC
                log_message += f'Testing {args.dataset} AUC_student_mlp_s: {auc_stu:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_mlp_s_non_normalize: {auc_stu_non_normalize:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_ocgnn: {auc_stu_ocgnn:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_ocgnn_non_normalize: {auc_stu_ocgnn_non_normalize:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_minus_ocgnn: {auc_stu_minus_ocgnn:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_minus_ocgnn_non_normalize: {auc_stu_minus_ocgnn_non_normalize:.4f}\n'

                # 计算AP
                AP_stu = average_precision_score(ano_label[idx_test], logits_stu, average='macro', pos_label=1)
                AP_stu_non_normalize = average_precision_score(ano_label[idx_test], logits_stu_non_normalize, average='macro', pos_label=1)
                AP_stu_ocgnn = average_precision_score(ano_label[idx_test], logits_stu_ocgnn, average='macro', pos_label=1)
                AP_stu_ocgnn_non_normalize = average_precision_score(ano_label[idx_test], logits_stu_ocgnn_non_normalize, average='macro', pos_label=1)
                AP_stu_minus_ocgnn = average_precision_score(ano_label[idx_test], logits_stu_minus_ocgnn, average='macro', pos_label=1)
                AP_stu_minus_ocgnn_non_normalize = average_precision_score(ano_label[idx_test], logits_stu_minus_ocgnn_non_normalize, average='macro', pos_label=1)

                # 记录AP
                log_message += f'Testing AP_student_mlp_s: {AP_stu:.4f}\n'
                log_message += f'Testing AP_student_mlp_s_non_normalize: {AP_stu_non_normalize:.4f}\n'
                log_message += f'Testing AP_student_ocgnn: {AP_stu_ocgnn:.4f}\n'
                log_message += f'Testing AP_student_ocgnn_non_normalize: {AP_stu_ocgnn_non_normalize:.4f}\n'
                log_message += f'Testing AP_student_minus_ocgnn: {AP_stu_minus_ocgnn:.4f}\n'
                log_message += f'Testing AP_student_minus_ocgnn_non_normalize: {AP_stu_minus_ocgnn_non_normalize:.4f}\n'

                log_message += f'Total time is: {total_time:.2f}\n'

                # 输出和保存日志
                print(log_message)
                f.write(log_message)
                f.flush()

            end_time = time.time()
            total_time += end_time - start_time
            pbar.update(1)

    print("Training completed.")


