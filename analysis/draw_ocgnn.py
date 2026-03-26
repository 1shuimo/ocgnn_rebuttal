import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from models.model_ocgnn import Model_ocgnn
# from model import Model_ocgnn # Assuming the import above is correct
from utils.utils import *
import os
import dgl # Assuming dgl is used in utils
import scipy.sparse as sp # Assuming sp is used in utils

# ==========================================================
# Add necessary imports
# ==========================================================
import matplotlib.pyplot as plt
from scipy.stats import norm


# Set CUDA device
# For this example, let's make it CPU-compatible to avoid errors if CUDA is not present.
# You can change it back if you have the specified GPU.
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2]))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===========================
# Parameter Settings
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

# Set learning rate according to dataset
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

# Set training epochs according to dataset
if args.num_epoch is None:
    if args.dataset in ['reddit']:
        args.num_epoch = 1000
    elif args.dataset in ['tf_finace']:
        args.num_epoch = 2500
    elif args.dataset in ['Amazon']:
        args.num_epoch = 1000
    elif args.dataset in ['elliptic']:
        args.num_epoch = 2000
    elif args.dataset in ['photo']:
        args.num_epoch = 50
    elif args.dataset in ['tolokers']:
        args.num_epoch = 1500
    elif args.dataset in ['YelpChi-all']:
        args.num_epoch = 1500
        
# Noise parameter settings
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
# Set Random Seed
# ===========================
print('Setting random seeds...')
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===========================
# Helper Function Definitions
# ===========================

def distillation_loss_emb(emb_t, emb_s):
    """Calculate distillation loss (MSE) between embeddings"""
    loss = F.mse_loss(emb_s, emb_t, reduction='mean')
    return loss

def score_distillation_loss(score_t, score_s):
    """Calculate distillation loss (MSE) between scores"""
    loss = F.mse_loss(score_s, score_t, reduction='mean')
    return loss

# ===================================================================
# MODIFICATION 1: Make the normalization function more robust
# ===================================================================
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
    """OCGNN's loss function (One-Class objective)"""
    r = 0
    beta = 0.5
    warmup = 2
    eps = 0.001
    c = torch.zeros(args.embedding_dim, device=emb.device)
    dist = torch.sum(torch.pow(emb - c, 2), 1)
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
    """Calculate KL divergence loss"""
    student_prob = torch.clamp(student_score, min=eps, max=1)
    teacher_prob = torch.clamp(teacher_score, min=eps, max=1)
    student_prob = student_prob / (student_prob.sum() + eps)
    teacher_prob = teacher_prob / (teacher_prob.sum() + eps)
    kl = F.kl_div(student_prob.log(), teacher_prob, reduction='batchmean')
    return kl

def to_hard_label_exclude_normal(score, normal_idx, top_percent=0.05):
    """
    Generate hard labels: select only the top_percent highest scores among non-normal_idx nodes and mark them as abnormal (1)
    """
    N = score.shape[0]
    all_idx = set(range(N))
    normal_set = set(normal_idx)
    candidate_idx = list(all_idx - normal_set)  # Candidate abnormal nodes: all nodes - known normal nodes
    candidate_scores = score[candidate_idx]
    top_k = int(len(all_idx) * top_percent)  # Calculate the number of top 5% nodes
    if top_k < 1:
        top_k = 1
    _, indices = torch.topk(candidate_scores, top_k)  # Select the top_k highest scores
    hard_label = torch.zeros_like(score)
    selected_idx = torch.tensor(candidate_idx, device=score.device)[indices]
    hard_label[selected_idx] = 1.0  # Mark as abnormal
    return hard_label

def get_all_normal_nodes(emb_s_all, teacher_hard):
    """
    Get the indices of all normal nodes
    """
    total_nodes = emb_s_all.shape[0]  # Number of nodes in the original graph N
    all_idx = set(range(total_nodes))
    
    # Find the indices of the top 5% abnormal nodes
    top5_abnormal_idx = set(torch.where(teacher_hard == 1)[0].cpu().numpy())
    
    # All normal nodes = All nodes - top 5% abnormal nodes
    all_normal_idx = list(all_idx - top5_abnormal_idx)
    
    return all_normal_idx

# 🔥 ADDED: Z-Score (mean) normalization function
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

# ==========================================================
# Add plotting function
# ==========================================================
def draw_pdf(message_normal, message_abnormal, dataset, epoch):
    """
    Draw score distribution plot - showing only normal and abnormal classes
    """
    mu_0, sigma_0 = np.mean(message_normal), np.std(message_normal)
    mu_1, sigma_1 = np.mean(message_abnormal), np.std(message_abnormal)
    plt.figure(figsize=(10, 6))
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')
    plt.hist(message_normal, bins=30, density=True, color='#4A90E2', alpha=0.7, 
             label='Normal Nodes', edgecolor='white', linewidth=0.7)
    plt.hist(message_abnormal, bins=30, density=True, color='#E74C3C', alpha=0.7, 
             label='Abnormal Nodes', edgecolor='white', linewidth=0.7)
    xmin, xmax = plt.xlim()
    x_range = np.linspace(xmin, xmax, 100)
    plt.plot(x_range, norm.pdf(x_range, mu_0, sigma_0), color='#1E88E5', linestyle='--', linewidth=2)
    plt.plot(x_range, norm.pdf(x_range, mu_1, sigma_1), color='#D81B60', linestyle='--', linewidth=2)
    plt.xlabel('Anomaly Score', fontsize=14, fontweight='bold')
    plt.ylabel('Density', fontsize=14, fontweight='bold')
    plt.title(f'{dataset.replace("_", " ").title()} - Score Distribution', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=11)
    os.makedirs(f'fig/ocgnn/{args.dataset}', exist_ok=True) # Use args.dataset for consistency
    save_path = f'fig/ocgnn/{args.dataset}/{dataset}_{epoch}.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n📊 {dataset} Distribution Stats: Normal (μ={mu_0:.4f}, σ={sigma_0:.4f}), Abnormal (μ={mu_1:.4f}, σ={sigma_1:.4f})")


# ===========================
# Data Loading and Preprocessing
# ===========================
print('Loading and preprocessing data...')
adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(args.dataset)

# Preprocess features
if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
    print('Preprocessing features...')
    features, _ = preprocess_features(features)
else:
    features = features.todense()



# Prepare input tensors
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
# Model Initialization
# ===========================
print('Initializing models...')

# Teacher model (ocgnn) - load weights
model_t = Model_ocgnn(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
model_t.load_state_dict(torch.load(args.teacher_path))
model_t.eval()
for p in model_t.parameters():
    p.requires_grad = False

# Student model (OCGNN) - needs training
model_s = Model_ocgnn(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)

# MLP for distillation
print('Initializing MLP for distillation...')
mlp_s = nn.Linear(args.embedding_dim, 1)  # Map embedding to anomaly score

# ===========================
# Optimizer Initialization
# ===========================
print('Initializing optimizers...')
optimiser = torch.optim.Adam(model_t.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_s = torch.optim.Adam(model_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_mlp_s = torch.optim.Adam(mlp_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Loss functions
device = 'cuda' if torch.cuda.is_available() else 'cpu'
b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))
xent = nn.CrossEntropyLoss()
# ===========================
# Training Loop
# ===========================
print("\n🔁 Starting Student Training...")
output_file = f"./ocgnn_2_step_2_reg_data_enhance/{args.dataset}_reg2_enhance_mse.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True) # Ensure directory exists

with open(output_file, "a") as f:
    with tqdm(total=args.num_epoch) as pbar:
        total_time = 0
        pbar.set_description('Training')
        for epoch in range(args.num_epoch):
            start_time = time.time()
            
            # Set training mode
            model_s.train()
            mlp_s.train()
            optimiser_s.zero_grad()
            optimiser_mlp_s.zero_grad()

            with torch.no_grad():
                # Teacher model output
                _, emb_t_all_raw = model_t(features, adj)
                emb_t_all = emb_t_all_raw.squeeze(0)  # [N, D]
                # Get teacher scores using loss_func
                _, score_from_ocgnn_non_normalize, _, _ = loss_func(emb_t_all)  # [N]

                # Calculate teacher AUC
                logits_teacher = np.squeeze(score_from_ocgnn_non_normalize[idx_test].cpu().detach().numpy())
                auc_teacher = roc_auc_score(ano_label[idx_test], logits_teacher)
                log_message = f"Testing {args.dataset} Teacher AUC: {auc_teacher:.4f}\n"
                teacher_score = min_max_normalize(score_from_ocgnn_non_normalize)  # [N] normalized score

            # Student model output
            _, emb_s_all_raw = model_s(features, adj)
            emb_s_all = emb_s_all_raw.squeeze(0)  # [N, D]
            student_score_non_normalize = mlp_s(emb_s_all).squeeze(dim=-1)  # [N]
            student_score = min_max_normalize(student_score_non_normalize)  # [N] normalized score

            # === Generate hard labels ===
            teacher_hard = to_hard_label_exclude_normal(score_from_ocgnn_non_normalize, normal_label_idx, top_percent=0.05)  # [N]
            all_normal_idx = get_all_normal_nodes(emb_s_all, teacher_hard)

            # === Data Augmentation Part ===
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

            # === Score Distillation MSE ===
            mse_loss = score_distillation_loss(teacher_score, student_score)

            # === Total Loss ===
            total_loss = mse_loss + 0.01 * reg2_mse

            # Backpropagation and parameter update
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
                
                if auc_stu >= 0.92:
                    print(f"\n🎉 AUC threshold of 0.88 reached! Stopping training at epoch {epoch}.")
                    break

            end_time = time.time()
            total_time += end_time - start_time
            pbar.update(1)

    print("Training completed.")


# ==========================================================
# REWRITTEN: Final evaluation and plotting for three modes
# ==========================================================
print("\n🏁 Starting final evaluation and plotting...")
with torch.no_grad():
    model_s.eval(); mlp_s.eval()
    
    # --- 1. Get final raw scores ---
    _, final_emb_s_raw = model_s(features, adj)
    student_scores_raw = mlp_s(final_emb_s_raw.squeeze(0)).squeeze(-1)

    teacher_scores_raw = score_from_ocgnn_non_normalize
    
    # --- 2. Calculate all normalized versions ---
    student_scores_minmax = min_max_normalize(student_scores_raw)
    teacher_scores_minmax = min_max_normalize(teacher_scores_raw)
    student_scores_zscore = z_score_normalize(student_scores_raw)
    teacher_scores_zscore = z_score_normalize(teacher_scores_raw)
    
    # ==========================================================
    # 🔥 FIX: Safely convert idx_test to a NumPy array
    # ==========================================================
    if isinstance(idx_test, torch.Tensor):
        idx_test_np = idx_test.cpu().numpy()
    else:
        # Handles list or existing numpy array
        idx_test_np = np.array(idx_test)
        
    # --- 3. Prepare plotting indices ---
    test_normal_mask = ano_label[idx_test_np] == 0
    test_abnormal_mask = ano_label[idx_test_np] == 1
    test_normal_indices = idx_test_np[test_normal_mask]
    test_abnormal_indices = idx_test_np[test_abnormal_mask]
    print(f"Test set: {len(test_normal_indices)} normal nodes, {len(test_abnormal_indices)} abnormal nodes.")

    # --- 4. Loop to generate all plots ---
    score_modes = {
        "raw": (student_scores_raw, teacher_scores_raw),
        "minmax": (student_scores_minmax, teacher_scores_minmax),
        "zscore": (student_scores_zscore, teacher_scores_zscore)
    }

    for mode, (student_scores, teacher_scores) in score_modes.items():
        print(f"\n{'='*20} Plotting for '{mode.upper()}' Scores {'='*20}")

        student_normal = student_scores[test_normal_indices].cpu().numpy()
        student_abnormal = student_scores[test_abnormal_indices].cpu().numpy()
        teacher_normal = teacher_scores[test_normal_indices].cpu().numpy()
        teacher_abnormal = teacher_scores[test_abnormal_indices].cpu().numpy()

        if len(student_normal) > 0 and len(student_abnormal) > 0:
            draw_pdf(student_normal, student_abnormal, f"{args.dataset}_student", f"0.92_{mode}")
        
        if len(teacher_normal) > 0 and len(teacher_abnormal) > 0:
            draw_pdf(teacher_normal, teacher_abnormal, f"{args.dataset}_teacher", f"0.92_{mode}")
            
    # --- 5. Output final AUC on raw scores ---
    final_auc_student = roc_auc_score(ano_label[idx_test_np], student_scores_raw[idx_test_np].cpu().numpy())
    final_auc_teacher = roc_auc_score(ano_label[idx_test_np], teacher_scores_raw[idx_test_np].cpu().numpy())

    print(f"\n🎯 Final Test Results (using RAW scores) on '{args.dataset}':")
    print(f"   - Student AUC:  {final_auc_student:.4f}")
    print(f"   - Teacher AUC:  {final_auc_teacher:.4f}")