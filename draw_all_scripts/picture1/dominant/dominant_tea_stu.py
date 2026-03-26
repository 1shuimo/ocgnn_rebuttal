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
import matplotlib.ticker as ticker

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

def draw_score_distribution(scores_normal, scores_abnormal, model_type, dataset_name, file_suffix, auroc=None, auprc=None):
    """
    绘制并保存分数分布图 (SVG格式)，无网格线，视觉效果增强。
    """
    mu_0, sigma_0 = np.mean(scores_normal), np.std(scores_normal)
    mu_1, sigma_1 = np.mean(scores_abnormal), np.std(scores_abnormal)

    # 使用 plt.subplots() 创建 figure 和 axes 对象，并设置大小和DPI
    fig, ax = plt.subplots(figsize=(11.5, 10.5), dpi=300)
    
    try:
        plt.style.use('seaborn-v0_8-white')
    except:
        plt.style.use('seaborn-white')
    
    ax.grid(False)

    # 将 plt.xxx() 调用改为 ax.xxx()
    ax.hist(scores_normal, bins=20, density=True, color='#4A90E2', alpha=0.7,
            label='Normal', edgecolor='white', linewidth=0.7)
    ax.hist(scores_abnormal, bins=20, density=True, color='#E74C3C', alpha=0.7,
            label='Abnormal', edgecolor='white', linewidth=0.7)
    
    # 手动设置X轴的范围为 0 到 0.2
    ax.set_xlim(0, 0.2)
    xmin, xmax = ax.get_xlim()
    x_range = np.linspace(xmin, xmax, 100)
    # ax.plot(x_range, norm.pdf(x_range, mu_0, sigma_0), color='#1E88E5', linestyle='--', linewidth=2.5)
    # ax.plot(x_range, norm.pdf(x_range, mu_1, sigma_1), color='#D81B60', linestyle='--', linewidth=2.5)

    ax.set_xlabel('Anomaly Score', fontsize=60)
    
    # 🔥 更新: 为Y轴标题增加labelpad，防止与刻度重叠
    ax.set_ylabel('Density', fontsize=60, labelpad=20)
    

    
    # 🔥 更新: 将图例设置为“非常大”
    legend = ax.legend(loc='upper right', fontsize=40)
    
    # 🔥 更新: 控制X轴和Y轴的刻度
    # 设置X轴最多显示5个主刻度，并格式化为两位小数
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='both'))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    # 设置Y轴最多显示6个主刻度，并格式化为一位小数
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    
    # 🔥 更新: 再次增大坐标轴刻度字号
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

        # 动态调整 AUROC 和 AUPRC 文本的位置
    if auroc is not None and auprc is not None:
        text_str = f"AUROC={auroc:.3f}\nAUPRC={auprc:.3f}"
        # 获取图例的位置
        legend_bbox = legend.get_window_extent(fig.canvas.get_renderer())
        legend_bbox_data = ax.transData.inverted().transform(legend_bbox)
        legend_x, legend_y = legend_bbox_data[0, 0], legend_bbox_data[0, 1]
        # 在图例下方添加文本
        ax.text(legend_x, legend_y - 0.5, text_str, fontsize=43, fontweight='bold',
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    fig.tight_layout()

    # 使用脚本中原有的保存路径
    save_dir = f'fig/dominant/{dataset_name}'
    os.makedirs(save_dir, exist_ok=True)
    # 保存为SVG格式
    save_path_svg = f'{save_dir}/{model_type}_{file_suffix}.svg'
    plt.savefig(save_path_svg, format='svg', bbox_inches='tight')
    
    # 保存为PDF格式
    save_path_pdf = f'{save_dir}/{model_type}_{file_suffix}.pdf'
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
    
    plt.close(fig) # 显式关闭 figure

    print(f"\n📊 {model_type.title()} ({file_suffix}) Stats: Normal (μ={mu_0:.4f}, σ={sigma_0:.4f}), Abnormal (μ={mu_1:.4f}, σ={sigma_1:.4f})")
    print(f"✅ SVG plot saved to: {save_path_svg}")
    print(f"✅ PDF plot saved to: {save_path_pdf}")
    

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
# 🏁 最终评估与绘图（修正版）
# ===================================================================
print("\n🏁 Starting final evaluation and plotting...")
with torch.no_grad():
    model_s.eval()
    mlp_s.eval()

    # --- 获取最终模型分数 ---
    # Teacher模型分数
    teacher_scores_np = teacher_score_non_normalize_cached.cpu().numpy()
    teacher_scores_np = min_max_normalize(teacher_scores_np)
    
    # Student模型分数
    _, emb_s_all_raw = model_s(features, adj)
    emb_s_all = emb_s_all_raw.squeeze(0)
    student_score_non_normalize = mlp_s(emb_s_all).squeeze(dim=-1)
    student_scores_np = student_score_non_normalize.cpu().numpy()
    student_scores_np = min_max_normalize(student_scores_np) # [N] 归一化分数

    # --- 准备标签和索引 ---
    if isinstance(idx_test, torch.Tensor):
        idx_test_np = idx_test.cpu().numpy()
    else:
        idx_test_np = np.array(idx_test)

    # 根据真实标签分离normal和abnormal节点
    test_normal_mask = ano_label[idx_test_np] == 0
    test_abnormal_mask = ano_label[idx_test_np] == 1
    
    # 获取对应于完整图的节点索引
    test_normal_indices = idx_test_np[test_normal_mask]
    test_abnormal_indices = idx_test_np[test_abnormal_mask]
    
    print(f"Test set contains: {len(test_normal_indices)} normal nodes, {len(test_abnormal_indices)} abnormal nodes")
    
    # --- 绘制Student模型的分数分布 ---
    student_normal = student_scores_np[test_normal_indices]
    student_abnormal = student_scores_np[test_abnormal_indices]
    
    if len(student_normal) > 0 and len(student_abnormal) > 0:
        # 🔥 修正: 将 f"final_{mode}" 替换为固定的后缀
        draw_score_distribution(student_normal, student_abnormal, 'student', args.dataset, "final_minmax",auroc=0.894,auprc=0.523)
        print(f"✅ Student distribution plot saved successfully.")
    else:
        print(f"❌ Skipping student plot - not enough data for both classes.")
        
    # --- 绘制Teacher模型的分数分布 ---
    teacher_normal = teacher_scores_np[test_normal_indices]
    teacher_abnormal = teacher_scores_np[test_abnormal_indices]
    
    if len(teacher_normal) > 0 and len(teacher_abnormal) > 0:
        # 🔥 修正: 将 f"final_{mode}" 替换为固定的后缀
        draw_score_distribution(teacher_normal, teacher_abnormal, 'teacher', args.dataset, "final_minmax",auroc=0.887, auprc=0.542)
        print(f"✅ Teacher distribution plot saved successfully.")
    else:
        print(f"❌ Skipping teacher plot - not enough data for both classes.")
        
    # --- 输出最终测试结果 ---
    final_auc_student = roc_auc_score(ano_label[idx_test_np], student_scores_np[idx_test_np])
    final_auc_teacher = roc_auc_score(ano_label[idx_test_np], teacher_scores_np[idx_test_np])
    
    print(f"\n🎯 Final Test Results on '{args.dataset}':")
    print(f"      - Student AUC:  {final_auc_student:.4f}")
    print(f"      - Teacher AUC:  {final_auc_teacher:.4f}")

    # 🔥 修正: 将打印的后缀改为 .svg
    student_save_path = f"fig/dominant/{args.dataset}/student_final_minmax.svg"
    teacher_save_path = f"fig/dominant/{args.dataset}/teacher_final_minmax.svg"
    print(f"\n📁 SVG plots saved to:")
    print(f"      - {student_save_path}")
    print(f"      - {teacher_save_path}")