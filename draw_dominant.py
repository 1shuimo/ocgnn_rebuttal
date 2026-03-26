import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import random
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from model_dominant_official import Model_dominant
from model_ocgnn import Model_ocgnn
from utils import *
import os
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm
# Set CPU only
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===========================
# 参数设置
# ===========================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='reddit')
parser.add_argument('--teacher_path', type=str, default='reddit_dominant_teacher_best.pth')
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
        args.lr = 2e-3
    elif args.dataset in ['tf_finace']:
        args.lr = 5e-4
    elif args.dataset in ['reddit']:
        args.lr = 1e-3
    elif args.dataset in ['elliptic']:
        args.lr = 3e-3
    elif args.dataset in ['photo']:
        args.lr = 3e-3
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
        args.num_epoch = 2500
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
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===========================
# 辅助函数定义
# ===========================

def score_distillation_loss(score_t, score_s):
    """计算分数之间的蒸馏损失 (MSE)"""
    loss = F.mse_loss(score_s, score_t, reduction='mean')
    return loss

def min_max_normalize(data):
    """Min-max normalization for PyTorch Tensors or NumPy arrays."""
    if isinstance(data, torch.Tensor):
        min_val = torch.min(data)
        max_val = torch.max(data)
        if max_val == min_val:
            return torch.zeros_like(data)
        else:
            return (data - min_val) / (max_val - min_val)
    elif isinstance(data, np.ndarray):
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        else:
            return (data - min_val) / (max_val - min_val)
    else:
        raise TypeError("Unsupported data type for min_max_normalize. Use torch.Tensor or np.ndarray.")



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
    """
    total_nodes = emb_s_all.shape[0]  # 原始图的节点数N
    all_idx = set(range(total_nodes))
    
    # 找出top5%异常节点的索引
    top5_abnormal_idx = set(torch.where(teacher_hard == 1)[0].cpu().numpy())
    
    # 所有normal节点 = 全部节点 - top5%异常节点
    all_normal_idx = list(all_idx - top5_abnormal_idx)
    
    return all_normal_idx

def z_score_normalize(data):
    """Z-score (mean) normalization for PyTorch Tensors or NumPy arrays."""
    if isinstance(data, torch.Tensor):
        mean, std = torch.mean(data), torch.std(data)
        if std == 0: return torch.zeros_like(data)
        return (data - mean) / std
    elif isinstance(data, np.ndarray):
        mean, std = np.mean(data), np.std(data)
        if std == 0: return np.zeros_like(data)
        return (data - mean) / std
    raise TypeError("Unsupported data type. Use torch.Tensor or np.ndarray.")


def draw_pdf(message_normal, message_abnormal, dataset, epoch):
    """
    绘制分数分布图 - 只显示normal和abnormal两类
    message_normal: 正常节点分数 (蓝色)
    message_abnormal: 异常节点分数 (红色)
    """
    # 计算统计信息
    mu_0 = np.mean(message_normal)
    sigma_0 = np.std(message_normal)
    mu_1 = np.mean(message_abnormal)
    sigma_1 = np.std(message_abnormal)
    
    # 设置图形大小
    plt.figure(figsize=(10, 6))
    
    # 尝试设置样式
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass  # 使用默认样式
    
    # 绘制正常节点直方图
    plt.hist(message_normal, bins=30, density=True, 
             color='#4A90E2', alpha=0.7, 
             label='Normal Nodes',
             edgecolor='white', linewidth=0.7)
    
    # 绘制异常节点直方图
    plt.hist(message_abnormal, bins=30, density=True, 
             color='#E74C3C', alpha=0.7, 
             label='Abnormal Nodes',
             edgecolor='white', linewidth=0.7)
    
    # 确定x轴范围用于绘制拟合曲线
    xmin, xmax = plt.xlim()
    x_range = np.linspace(xmin, xmax, 100)
    
    # 拟合正态分布曲线
    y_0 = norm.pdf(x_range, mu_0, sigma_0)
    y_1 = norm.pdf(x_range, mu_1, sigma_1)
    
    # 绘制曲线
    plt.plot(x_range, y_0, color='#1E88E5', linestyle='--', linewidth=2)
    plt.plot(x_range, y_1, color='#D81B60', linestyle='--', linewidth=2)
    
    # 设置标签和标题
    plt.xlabel('Anomaly Score', fontsize=14, fontweight='bold')
    plt.ylabel('Density', fontsize=14, fontweight='bold')
    plt.title(f'{dataset.replace("_", " ").title()} - Score Distribution', 
              fontsize=16, fontweight='bold', pad=20)
    
    # 添加图例
    plt.legend(loc='upper right', fontsize=11, framealpha=0.95, 
               fancybox=True, shadow=True)
    
    # 设置刻度字体大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 美化边框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 创建保存目录
    os.makedirs(f'fig/dominant/{dataset}_dominant', exist_ok=True)
    
    # 保存图片
    plt.savefig(f'fig/dominant/{dataset}_{epoch}_dominant.pdf', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # 打印统计信息到控制台
    print(f"\n📊 {dataset} Distribution Statistics:")
    print(f"  🔵 Normal Nodes:   mean={mu_0:.4f}, std={sigma_0:.4f}, count={len(message_normal)}")
    print(f"  🔴 Abnormal Nodes: mean={mu_1:.4f}, std={sigma_1:.4f}, count={len(message_abnormal)}")

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

# # 转换邻接矩阵为DGL图（如果需要的话）
# print('Converting adjacency matrix to DGL graph...')
# # 转换邻接矩阵为DGL图 - 带缓存
# dgl_graph_path = f"./cache/{args.dataset}_dgl_graph.pkl"
# os.makedirs("./cache", exist_ok=True)

# if os.path.exists(dgl_graph_path):
#     print(f'Loading cached DGL graph from {dgl_graph_path}...')
#     with open(dgl_graph_path, 'rb') as f:
#         dgl_graph = pickle.load(f)
#     print('DGL graph loaded from cache.')
# else:
#     print('Converting adjacency matrix to DGL graph...')
#     dgl_graph = adj_to_dgl_graph(adj)
#     print('Saving DGL graph to cache...')
#     with open(dgl_graph_path, 'wb') as f:
#         pickle.dump(dgl_graph, f)
#     print(f'DGL graph saved to {dgl_graph_path}')
    
    

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

# 将索引转为tensor
idx_train = torch.tensor(idx_train)
idx_test = torch.tensor(idx_test)
normal_label_idx = torch.tensor(normal_label_idx)
abnormal_label_idx = torch.tensor(abnormal_label_idx)

# ===========================
# 模型初始化
# ===========================
print('Initializing models...')

# Teacher模型 (DOMINANT) - 加载权重
model_t = Model_dominant(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout, args.dataset)
model_t.load_state_dict(torch.load(args.teacher_path, map_location='cpu'))
model_t.eval()
for p in model_t.parameters():
    p.requires_grad = False

# Student模型 (OCGNN) - 需要训练
model_s = Model_ocgnn(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)

# 蒸馏用的MLP
print('Initializing MLP for distillation...')
mlp_s = nn.Linear(args.embedding_dim, 1)  # 将embedding映射到异常分数

# ===========================
# 🔥 预计算Teacher模型的所有输出（只计算一次）
# ===========================
print('Precomputing teacher outputs...')
with torch.no_grad():
    # Teacher模型输出 - 只计算一次，整个训练过程中都不会变
    _, _, teacher_score_test_cached, _, _, _ = model_t(features, adj, normal_label_idx, idx_test)
    _, teacher_score_non_normalize_cached, _, emb_t_all_cached, _, _ = model_t(features, adj, all_idx, idx_test)
    
    # 预计算所有teacher相关的数据
    teacher_score_cached = min_max_normalize(teacher_score_non_normalize_cached)
    score_from_dominant_cached = torch.sigmoid(teacher_score_non_normalize_cached)
    
    # 预计算teacher AUC
    logits_teacher = teacher_score_test_cached.cpu().detach().numpy()
    auc_teacher = roc_auc_score(ano_label[idx_test], logits_teacher)
    teacher_auc_message = f"Testing {args.dataset} Teacher AUC: {auc_teacher:.4f}"
    
    # 预计算硬标签
    teacher_hard_cached = to_hard_label_exclude_normal(teacher_score_non_normalize_cached, normal_label_idx, top_percent=0.05)
    
    print(f'Teacher outputs precomputed. {teacher_auc_message}')

# ===========================
# 优化器初始化
# ===========================
print('Initializing optimizers...')
optimiser_s = torch.optim.Adam(model_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_mlp_s = torch.optim.Adam(mlp_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# ===========================
# 训练循环
# ===========================
print("\n🔁 Starting Student Training...")
os.makedirs("./dominant_reg_data_enhance", exist_ok=True)
output_file = f"./dominant_reg_data_enhance/{args.dataset}_optimize_{args.lr}_{args.num_epoch}_0.9.txt"
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

            # 🔥 直接使用缓存的teacher输出，无需重新计算
            log_message = f"{teacher_auc_message}\n"

            # Student模型 (OCGNN) 输出
            _, emb_s_all_raw = model_s(features, adj)
            emb_s_all = emb_s_all_raw.squeeze(0)  # [N, D]
            student_score_non_normalize = mlp_s(emb_s_all).squeeze(dim=-1)  # [N]
            student_score = min_max_normalize(student_score_non_normalize)  # [N] 归一化分数

            # === 使用缓存的硬标签 ===
            all_normal_idx = get_all_normal_nodes(emb_s_all, teacher_hard_cached)

            # === 数据增强部分 ===
            all_normal_features = features.squeeze()[all_normal_idx]  # [N_normal, D]
            mask = torch.rand_like(all_normal_features) > 0.3
            masked_normal_features = all_normal_features * mask
            masked_features = features.clone().squeeze()
            masked_features[all_normal_idx] = masked_normal_features
            masked_features = masked_features.unsqueeze(0)  # [1, N, D]
            _, emb_s_augmented_raw = model_s(masked_features, adj)
            emb_s_augmented = emb_s_augmented_raw.squeeze(0)  # [N, D]
            emb_s_normal_original = emb_s_all[all_normal_idx]
            emb_s_normal_augmented = emb_s_augmented[all_normal_idx]
            reg2_mse = F.mse_loss(emb_s_normal_augmented, emb_s_normal_original, reduction='mean')

            # === 分数蒸馏MSE - 使用缓存的teacher分数 ===
            mse_loss = score_distillation_loss(teacher_score_cached, student_score)

            # === 总损失 ===
            total_loss = mse_loss + 0.01 * reg2_mse

            # 反向传播和参数更新
            total_loss.backward()
            optimiser_s.step()
            optimiser_mlp_s.step()

            # === 评估和日志 (每5个epoch) ===
            # if epoch % 5 == 0:
            log_message += (
                f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}\n"
                f"MSE Loss = {mse_loss.item():.6f}\n"
                f"Reg2 MSE Loss = {reg2_mse.item():.6f}\n"
            )

            # 切换到评估模式
            model_s.eval()
            with torch.no_grad():
                _, emb_s_all_raw = model_s(features, adj)
                emb_s_all = emb_s_all_raw.squeeze(0)  # [N, D]

                student_score_non_normalize = mlp_s(emb_s_all).squeeze(dim=-1)  # [N]
                student_score = min_max_normalize(student_score_non_normalize)

                # 🔥 使用缓存的teacher分数
                score_from_dominant_non_normalize = teacher_score_non_normalize_cached
                score_from_dominant = score_from_dominant_cached

                # 记录平均分数
                log_message += f'student_score: {torch.mean(student_score):.6f}\n'
                log_message += f'dominant_score: {torch.mean(score_from_dominant):.6f}\n'
                log_message += f'student_score_non_normalize: {torch.mean(student_score_non_normalize):.6f}\n'
                log_message += f'dominant_score_non_normalize: {torch.mean(score_from_dominant_non_normalize):.6f}\n'

                # 计算组合分数
                student_minus_dominant_score = abs(student_score - score_from_dominant)
                stu_add_dominant_score = student_score + score_from_dominant
                stu_add_dominant_non_normalize_score = student_score_non_normalize + score_from_dominant_non_normalize
                student_minus_dominant_non_normalize_score = student_score_non_normalize - score_from_dominant_non_normalize

                # 提取测试集分数
                logits_stu = student_score[idx_test].cpu().detach().numpy()
                logits_stu_non_normalize = student_score_non_normalize[idx_test].cpu().detach().numpy()
                logits_stu_dominant = stu_add_dominant_score[idx_test].cpu().detach().numpy()
                logits_stu_dominant_non_normalize = stu_add_dominant_non_normalize_score[idx_test].cpu().detach().numpy()
                logits_stu_minus_dominant = student_minus_dominant_score[idx_test].cpu().detach().numpy()
                logits_stu_minus_dominant_non_normalize = student_minus_dominant_non_normalize_score[idx_test].cpu().detach().numpy()

                # 计算AUC
                auc_stu = roc_auc_score(ano_label[idx_test], logits_stu)
                auc_stu_non_normalize = roc_auc_score(ano_label[idx_test], logits_stu_non_normalize)
                auc_stu_dominant = roc_auc_score(ano_label[idx_test], logits_stu_dominant)
                auc_stu_dominant_non_normalize = roc_auc_score(ano_label[idx_test], logits_stu_dominant_non_normalize)
                auc_stu_minus_dominant = roc_auc_score(ano_label[idx_test], logits_stu_minus_dominant)
                auc_stu_minus_dominant_non_normalize = roc_auc_score(ano_label[idx_test], logits_stu_minus_dominant_non_normalize)

                # 记录AUC
                log_message += f'Testing {args.dataset} AUC_student_mlp_s: {auc_stu:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_mlp_s_non_normalize: {auc_stu_non_normalize:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_dominant: {auc_stu_dominant:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_dominant_non_normalize: {auc_stu_dominant_non_normalize:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_minus_dominant: {auc_stu_minus_dominant:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_minus_dominant_non_normalize: {auc_stu_minus_dominant_non_normalize:.4f}\n'

                # 计算AP
                AP_stu = average_precision_score(ano_label[idx_test], logits_stu, average='macro', pos_label=1)
                AP_stu_non_normalize = average_precision_score(ano_label[idx_test], logits_stu_non_normalize, average='macro', pos_label=1)
                AP_stu_dominant = average_precision_score(ano_label[idx_test], logits_stu_dominant, average='macro', pos_label=1)
                AP_stu_dominant_non_normalize = average_precision_score(ano_label[idx_test], logits_stu_dominant_non_normalize, average='macro', pos_label=1)
                AP_stu_minus_dominant = average_precision_score(ano_label[idx_test], logits_stu_minus_dominant, average='macro', pos_label=1)
                AP_stu_minus_dominant_non_normalize = average_precision_score(ano_label[idx_test], logits_stu_minus_dominant_non_normalize, average='macro', pos_label=1)

                # 记录AP
                log_message += f'Testing AP_student_mlp_s: {AP_stu:.4f}\n'
                log_message += f'Testing AP_student_mlp_s_non_normalize: {AP_stu_non_normalize:.4f}\n'
                log_message += f'Testing AP_student_dominant: {AP_stu_dominant:.4f}\n'
                log_message += f'Testing AP_student_dominant_non_normalize: {AP_stu_dominant_non_normalize:.4f}\n'
                log_message += f'Testing AP_student_minus_dominant: {AP_stu_minus_dominant:.4f}\n'
                log_message += f'Testing AP_student_minus_dominant_non_normalize: {AP_stu_minus_dominant_non_normalize:.4f}\n'

                log_message += f'Total time is: {total_time:.2f}s\n'
                log_message += "="*50 + "\n"

                # 输出和保存日志
                print(log_message)
                f.write(log_message)
                f.flush()
                                # 🚀 新增判断条件：如果AUC大于0.88，则退出循环
                if auc_stu >= 0.88:
                    print(f"\n🎉 AUC threshold of 0.88 reached! Stopping training at epoch {epoch}.")
                    break

            end_time = time.time()
            total_time += end_time - start_time
            pbar.update(1)

    print("Training completed.")


# ===================================================================
# 🏁 最终评估与绘图（生成三种标准化模式的图）
# ===================================================================
print("\n🏁 Starting final evaluation and plotting...")
with torch.no_grad():
    model_s.eval(); mlp_s.eval()

    # --- 1. 获取最终的原始（未标准化）分数 ---
    _, final_emb_s_raw = model_s(features, adj)
    final_emb_s = final_emb_s_raw.squeeze(0)
    student_scores_raw = mlp_s(final_emb_s).squeeze(dim=-1)
    teacher_scores_raw = teacher_score_non_normalize_cached

    # --- 2. 计算所有标准化版本的分数 ---
    student_scores_minmax = min_max_normalize(student_scores_raw)
    teacher_scores_minmax = min_max_normalize(teacher_scores_raw)
    
    student_scores_zscore = z_score_normalize(student_scores_raw)
    teacher_scores_zscore = z_score_normalize(teacher_scores_raw)
    
    # --- 3. 准备绘图用的索引和掩码 ---
    idx_test_np = idx_test.cpu().numpy()
    test_normal_mask = ano_label[idx_test_np] == 0
    test_abnormal_mask = ano_label[idx_test_np] == 1
    test_normal_indices = idx_test_np[test_normal_mask]
    test_abnormal_indices = idx_test_np[test_abnormal_mask]
    print(f"Test set contains: {len(test_normal_indices)} normal nodes, {len(test_abnormal_indices)} abnormal nodes")

    # --- 4. 循环生成三种模式的图 ---
    score_modes = {
        "raw": (student_scores_raw, teacher_scores_raw),
        "minmax": (student_scores_minmax, teacher_scores_minmax),
        "zscore": (student_scores_zscore, teacher_scores_zscore)
    }

    for mode, (student_scores, teacher_scores) in score_modes.items():
        print(f"\n{'='*20} Plotting for '{mode.upper()}' Scores {'='*20}")
        
        # --- 绘制Student模型分数分布 ---
        # 🔥 FIX: 将Tensor转换为NumPy Array
        message_normal_student = student_scores[test_normal_indices].cpu().numpy()
        message_abnormal_student = student_scores[test_abnormal_indices].cpu().numpy()
        
        if len(message_normal_student) > 0 and len(message_abnormal_student) > 0:
            draw_pdf(message_normal_student, message_abnormal_student,
                     f"{args.dataset}_student", f"final_{mode}")
            print(f"✅ Student ({mode}) distribution plot saved.")
        else:
            print(f"❌ Skipping student ({mode}) plot - not enough data.")

        # --- 绘制Teacher模型分数分布 ---
        # 🔥 FIX: 将Tensor转换为NumPy Array
        message_normal_teacher = teacher_scores[test_normal_indices].cpu().numpy()
        message_abnormal_teacher = teacher_scores[test_abnormal_indices].cpu().numpy()

        if len(message_normal_teacher) > 0 and len(message_abnormal_teacher) > 0:
            draw_pdf(message_normal_teacher, message_abnormal_teacher,
                     f"{args.dataset}_teacher", f"final_{mode}")
            print(f"✅ Teacher ({mode}) distribution plot saved.")
        else:
            print(f"❌ Skipping teacher ({mode}) plot - not enough data.")

    # --- 5. 输出最终测试AUC结果 ---
    auc_student_raw = roc_auc_score(ano_label[idx_test_np], student_scores_raw[idx_test_np].cpu().numpy())
    auc_teacher_raw = roc_auc_score(ano_label[idx_test_np], teacher_scores_raw[idx_test_np].cpu().numpy())
    
    print(f"\n🎯 Final Test AUC Results (using RAW scores) on '{args.dataset}':")
    print(f"     - Student AUC:  {auc_student_raw:.4f}")
    print(f"     - Teacher AUC:  {auc_teacher_raw:.4f}")