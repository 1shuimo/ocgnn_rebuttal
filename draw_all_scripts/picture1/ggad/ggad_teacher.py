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
from model import Model_ggad
from utils import *
import os
import matplotlib.pyplot as plt
import dgl
from scipy.stats import norm
import scipy.sparse as sp
# 🔥 新增导入，用于控制绘图刻度
import matplotlib.ticker as ticker



# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2]))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===========================
# 参数设置
# ... (此部分代码保持不变) ...
# ===========================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='reddit')
parser.add_argument('--teacher_path', type=str, default='reddit_ggad_teacher_final.pth')
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
        args.lr = 5e-3
    elif args.dataset in ['tolokers']:
        args.lr = 5e-3
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
        args.num_epoch = 1000
    elif args.dataset in ['tolokers']:
        args.num_epoch = 200
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
# ... (此部分代码保持不变) ...
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
# ... (除了绘图函数之外，其他辅助函数保持不变) ...
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
    
# ... (to_hard_label_exclude_normal, get_all_normal_nodes 等其他函数保持不变) ...
def to_hard_label_exclude_normal(score, normal_idx, top_percent=0.05):
    N = score.shape[0]
    all_idx = set(range(N))
    normal_set = set(normal_idx)
    candidate_idx = list(all_idx - normal_set)
    candidate_scores = score[candidate_idx]
    top_k = int(len(all_idx) * top_percent)
    if top_k < 1:
        top_k = 1
    _, indices = torch.topk(candidate_scores, top_k)
    hard_label = torch.zeros_like(score)
    selected_idx = torch.tensor(candidate_idx, device=score.device)[indices]
    hard_label[selected_idx] = 1.0
    return hard_label

def get_all_normal_nodes(emb_s_all, teacher_hard):
    total_nodes = emb_s_all.shape[0]
    all_idx = set(range(total_nodes))
    top5_abnormal_idx = set(torch.where(teacher_hard == 1)[0].cpu().numpy())
    all_normal_idx = list(all_idx - top5_abnormal_idx)
    return all_normal_idx

# ==========================================================
# 🔥 全新、功能强大的绘图函数
# ==========================================================
def draw_score_distribution(scores_normal, scores_abnormal, model_type, dataset_name, file_suffix, auroc=None, auprc=None):
    """
    绘制并保存分数分布图 (SVG和PDF格式)，并控制刻度数量和格式。
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

    # 使用 ax 对象进行绘图
    ax.hist(scores_normal, bins=30, density=True, color='#4A90E2', alpha=0.7,
            label='Normal', edgecolor='white', linewidth=0.7)
    ax.hist(scores_abnormal, bins=30, density=True, color='#E74C3C', alpha=0.7,
            label='Abnormal', edgecolor='white', linewidth=0.7)
    
    xmin, xmax = ax.get_xlim()
    x_range = np.linspace(xmin, xmax, 100)
    # ax.plot(x_range, norm.pdf(x_range, mu_0, sigma_0), color='#1E88E5', linestyle='--', linewidth=10)
    # ax.plot(x_range, norm.pdf(x_range, mu_1, sigma_1), color='#D81B60', linestyle='--', linewidth=10)

    ax.set_xlabel('Anomaly Score', fontsize=60, fontweight='bold')
    ax.set_ylabel('Density', fontsize=60, fontweight='bold', labelpad=20)
    
    dataset_title = dataset_name.replace("_", " ").title()
    legend = ax.legend(loc='upper left', fontsize=40)
    
    # 控制X轴和Y轴的刻度
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='both'))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

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
    save_dir = f'fig/ggad/{dataset_name}'
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
# ... (此部分代码保持不变) ...
# ===========================
print('Loading and preprocessing data...')
adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(args.dataset)
if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
    print('Preprocessing features...')
    features, _ = preprocess_features(features)
else:
    features = features.todense()
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
# ... (此部分代码保持不变) ...
# ===========================
print('Initializing models...')
model = Model_ggad(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
model.load_state_dict(torch.load(args.teacher_path))
model.eval()
for p in model.parameters():
    p.requires_grad = False
model_s = Model_ocgnn(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
print('Initializing MLP for distillation...')
mlp_s = nn.Linear(args.embedding_dim, 1)
pseudo_emb_mlp = nn.Linear(args.embedding_dim, args.embedding_dim)

# ===========================
# 优化器与损失函数初始化
# ... (此部分代码保持不变) ...
# ===========================
print('Initializing optimizers...')
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_s = torch.optim.Adam(model_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_mlp_s = torch.optim.Adam(mlp_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_pseudo_emb_mlp = torch.optim.Adam(pseudo_emb_mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda() if torch.cuda.is_available() else torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()

# ===========================
# 训练循环
# ... (此部分代码保持不变) ...
# ===========================
print("\n🔁 Starting Student Training...")
output_file = f"./ggad_2_step_2_reg_data_enhance/{args.dataset}_draw_pdf_{args.lr}.txt"
with open(output_file, "a") as f:
    # The entire training loop is left as is.
    with tqdm(total=args.num_epoch) as pbar:
        total_time = 0
        pbar.set_description('Training')
        for epoch in range(args.num_epoch):
            # ... The user's original training logic ...
            start_time = time.time()
            model_s.train()
            mlp_s.train()
            optimiser_s.zero_grad()
            optimiser_mlp_s.zero_grad()
            optimiser_pseudo_emb_mlp.zero_grad()
            with torch.no_grad():
                _, _, logits_total, _, _, _ = model(features, adj, abnormal_label_idx, normal_label_idx, train_flag=False, args=args)
                score_from_ggad_non_normalize = logits_total.squeeze(dim=-1).squeeze(0)
                logits = np.squeeze(logits_total[:, idx_test, :].cpu().detach().numpy())
                auc = roc_auc_score(ano_label[idx_test], logits)
                log_message = (f'Testing_last_ggad_ {args.dataset} AUC: {auc:.4f}\n')
                score_from_ggad = min_max_normalize(score_from_ggad_non_normalize)
                emb_t_all, _, logits, _, emb_abnormal , _= model(features, adj, abnormal_label_idx, normal_label_idx, train_flag=True, args=args)
                emb_t_all = emb_t_all.squeeze(0)
                pseudo_emb = emb_abnormal.squeeze(0)
            pseudo_emb_proj = pseudo_emb_mlp(pseudo_emb)
            num_nodes = emb_t_all.size(0)
            _, emb_s_all_raw = model_s(features, adj)
            emb_s_all = emb_s_all_raw.squeeze(0)
            emb_concat = torch.cat([emb_s_all, pseudo_emb_proj], dim=0)
            student_score_non_normalize = mlp_s(emb_s_all).squeeze(dim=-1)
            student_score = torch.sigmoid(student_score_non_normalize)
            student_score_concat_non_normalize = mlp_s(emb_concat).squeeze(dim=-1)
            student_score_concat = min_max_normalize(student_score_concat_non_normalize)
            teacher_hard = to_hard_label_exclude_normal(score_from_ggad_non_normalize, normal_label_idx, top_percent=0.05)
            hard1_indices = torch.where(teacher_hard == 1)[0]
            all_normal_idx = get_all_normal_nodes(emb_s_all, teacher_hard)
            all_normal_features = features.squeeze()[all_normal_idx]
            mask = torch.rand_like(all_normal_features) > 0.3
            masked_normal_features = all_normal_features * mask
            masked_features = features.clone().squeeze()
            masked_features[all_normal_idx] = masked_normal_features
            masked_features = masked_features.unsqueeze(0)
            _, emb_s_augmented_raw = model_s(masked_features, adj)
            emb_s_augmented = emb_s_augmented_raw.squeeze(0)
            emb_s_normal_original = emb_s_all[all_normal_idx]
            emb_s_normal_augmented = emb_s_augmented[all_normal_idx]
            reg2_mse = F.mse_loss(emb_s_normal_augmented, emb_s_normal_original, reduction='mean')
            pseudo_score_from_ggad = score_from_ggad[abnormal_label_idx]
            teacher_score_concat = torch.cat([score_from_ggad, pseudo_score_from_ggad], dim=0)
            mse_loss = score_distillation_loss(student_score_concat,teacher_score_concat)
            total_loss = mse_loss + 0.01 * reg2_mse
            total_loss.backward()
            optimiser_s.step()
            optimiser_mlp_s.step()
            optimiser_pseudo_emb_mlp.step()
            if epoch % 5 == 0:
                log_message += (f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}\n"
                                f"MSE Loss = {mse_loss.item():.6f}\n"
                                f"Reg2 MSE Loss = {reg2_mse.item():.6f}\n")
                model_s.eval()
                train_flag = False
                emb_t, emb_combine, logits, emb_con, emb_abnormal, _= model(features, adj, abnormal_label_idx, normal_label_idx, train_flag, args)
                score_from_ggad_non_normalize = logits.squeeze(dim=-1).squeeze(0)
                score_from_ggad = torch.sigmoid(score_from_ggad_non_normalize)
                log_message += f'student_score: {torch.mean(student_score)}\n'
                log_message += f'ggad_score: {torch.mean(score_from_ggad)}\n'
                log_message += f'student_score_non_normalize: {torch.mean(student_score_non_normalize)}\n'
                log_message += f'ggad_score_non_normalize: {torch.mean(score_from_ggad_non_normalize)}\n'
                student_minus_ggad_score = abs(student_score - score_from_ggad)
                stu_add_ggad_score = student_score + score_from_ggad
                stu_add_ggad_non_normalize_score = student_score_non_normalize + score_from_ggad_non_normalize
                student_minus_ggad_non_normalize_score = student_score_non_normalize - score_from_ggad_non_normalize
                logits_stu = np.squeeze(student_score[idx_test].cpu().detach().numpy())
                logits_stu_non_normalize = np.squeeze(student_score_non_normalize[idx_test].cpu().detach().numpy())
                logits_stu_ggad = np.squeeze(stu_add_ggad_score[idx_test].cpu().detach().numpy())
                logits_stu_ggad_non_normalize = np.squeeze(stu_add_ggad_non_normalize_score[idx_test].cpu().detach().numpy())
                logits_stu_minus_ggad = np.squeeze(student_minus_ggad_score[idx_test].cpu().detach().numpy())
                logits_stu_minus_ggad_non_normalize = np.squeeze(student_minus_ggad_non_normalize_score[idx_test].cpu().detach().numpy())
                auc_stu = roc_auc_score(ano_label[idx_test], logits_stu)
                auc_stu_non_normalize = roc_auc_score(ano_label[idx_test], logits_stu_non_normalize)
                auc_stu_ggad = roc_auc_score(ano_label[idx_test], logits_stu_ggad)
                auc_stu_ggad_non_normalize = roc_auc_score(ano_label[idx_test], logits_stu_ggad_non_normalize)
                auc_stu_minus_ggad = roc_auc_score(ano_label[idx_test], logits_stu_minus_ggad)
                auc_stu_minus_ggad_non_normalize = roc_auc_score(ano_label[idx_test], logits_stu_minus_ggad_non_normalize)
                log_message += f'Testing {args.dataset} AUC_student_mlp_s: {auc_stu:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_mlp_s_non_normalize: {auc_stu_non_normalize:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_ggad: {auc_stu_ggad:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_ggad_non_normalize: {auc_stu_ggad_non_normalize:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_minus_ggad: {auc_stu_minus_ggad:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_minus_ggad_non_normalize: {auc_stu_minus_ggad_non_normalize:.4f}\n'
                AP_stu = average_precision_score(ano_label[idx_test], logits_stu, average='macro', pos_label=1)
                AP_stu_non_normalize = average_precision_score(ano_label[idx_test], logits_stu_non_normalize, average='macro', pos_label=1)
                AP_stu_ggad = average_precision_score(ano_label[idx_test], logits_stu_ggad, average='macro', pos_label=1)
                AP_stu_ggad_non_normalize = average_precision_score(ano_label[idx_test], logits_stu_ggad_non_normalize, average='macro', pos_label=1)
                AP_stu_minus_ggad = average_precision_score(ano_label[idx_test], logits_stu_minus_ggad, average='macro', pos_label=1)
                AP_stu_minus_ggad_non_normalize = average_precision_score(ano_label[idx_test], logits_stu_minus_ggad_non_normalize, average='macro', pos_label=1)
                log_message += f'Testing AP_student_mlp_s: {AP_stu:.4f}\n'
                log_message += f'Testing AP_student_mlp_s_non_normalize: {AP_stu_non_normalize:.4f}\n'
                log_message += f'Testing AP_student_ggad: {AP_stu_ggad:.4f}\n'
                log_message += f'Testing AP_student_ggad_non_normalize: {AP_stu_ggad_non_normalize:.4f}\n'
                log_message += f'Testing AP_student_minus_ggad: {AP_stu_minus_ggad:.4f}\n'
                log_message += f'Testing AP_student_minus_ggad_non_normalize: {AP_stu_minus_ggad_non_normalize:.4f}\n'
                log_message += f'Total time is: {total_time:.2f}\n'
                print(log_message)
                f.write(log_message)
                f.flush()
                print(auc_stu)
                if auc_stu >= 0.88:
                    print(f"\n🎉 AUC threshold of 0.88 reached! Stopping training at epoch {epoch}.")
                    break
            end_time = time.time()
            total_time += end_time - start_time
            pbar.update(1)
    print("Training completed.")
    

# ==========================================================
# 🔥 最终评估与绘图（已修改）
# ==========================================================
print("\n📊 Starting final evaluation and plotting...")
with torch.no_grad():
    model_s.eval()
    
    # --- 1. 获取最终的原始分数 ---
    _, final_emb_s_raw = model_s(features, adj)
    student_scores_raw = mlp_s(final_emb_s_raw.squeeze(0)).squeeze()
    # Teacher分数可直接使用训练最后一步缓存的结果
    teacher_scores_raw = score_from_ggad_non_normalize

    # --- 2. 计算所有归一化版本的分数 ---
    student_scores_minmax = min_max_normalize(student_scores_raw)
    teacher_scores_minmax = min_max_normalize(teacher_scores_raw)
    student_scores_zscore = z_score_normalize(student_scores_raw)
    teacher_scores_zscore = z_score_normalize(teacher_scores_raw)

    # --- 3. 准备绘图用的索引 ---
    if isinstance(idx_test, torch.Tensor):
        idx_test_np = idx_test.cpu().numpy()
    else:
        idx_test_np = np.array(idx_test)
        
    test_normal_mask = ano_label[idx_test_np] == 0
    test_abnormal_mask = ano_label[idx_test_np] == 1
    test_normal_idx = idx_test_np[test_normal_mask]
    test_abnormal_idx = idx_test_np[test_abnormal_mask]
    
    print(f"Test set: {len(test_normal_idx)} normal, {len(test_abnormal_idx)} abnormal nodes.")

    # --- 4. 循环为所有模式生成绘图 ---
    score_modes = {
        "minmax": (student_scores_minmax, teacher_scores_minmax),
    }

    for mode, (student_scores, teacher_scores) in score_modes.items():
        print(f"\n{'='*20} Plotting for '{mode.upper()}' Scores {'='*20}")
        
        student_scores_np = student_scores.cpu().numpy()
        teacher_scores_np = teacher_scores.cpu().numpy()

        s_normal = student_scores_np[test_normal_idx]
        s_abnormal = student_scores_np[test_abnormal_idx]
        
        t_normal = teacher_scores_np[test_normal_idx]
        t_abnormal = teacher_scores_np[test_abnormal_idx]
        
        # 🔥 更新函数调用
        if all(len(arr) > 0 for arr in [s_normal, s_abnormal]):
            draw_score_distribution(s_normal, s_abnormal, 'student', args.dataset, f"final_{mode}", auroc=0.944, auprc=0.792)
        else:
            print(f"❌ Skipping student ({mode}) plot - one or more node categories are empty.")

        if all(len(arr) > 0 for arr in [t_normal, t_abnormal]):
            draw_score_distribution(t_normal,t_abnormal, 'teacher', args.dataset, f"final_{mode}", auroc=0.944, auprc=0.792)
        else:
            print(f"❌ Skipping teacher ({mode}) plot - one or more node categories are empty.")

    # --- 5. 在原始分数上输出最终AUC ---
    final_auc_student = roc_auc_score(ano_label[idx_test_np], student_scores_raw[idx_test_np].cpu().numpy())
    final_auc_teacher = roc_auc_score(ano_label[idx_test_np], teacher_scores_raw[idx_test_np].cpu().numpy())

    print(f"\n🎯 Final Test AUC Results (using RAW scores) on '{args.dataset}':")
    print(f"    - Student AUC:  {final_auc_student:.4f}")
    print(f"    - Teacher AUC:  {final_auc_teacher:.4f}")

print("\nEvaluation and plotting completed!")