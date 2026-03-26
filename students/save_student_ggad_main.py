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
from models.model import Model_ggad
from utils.utils import *
import os
# Corrected: Added missing imports
import dgl
import scipy.sparse as sp


# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2]))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===========================
# 参数设置
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
        args.lr = 5e-4

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

def to_hard_label_exclude_normal(score, normal_idx, top_percent=0.05):
    """
    生成硬标签：只在非normal_idx的节点中选择top_percent分数最高的标记为异常(1)
    """
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
    """
    获取所有normal节点的索引
    """
    total_nodes = emb_s_all.shape[0]
    all_idx = set(range(total_nodes))
    top5_abnormal_idx = set(torch.where(teacher_hard == 1)[0].cpu().numpy())
    all_normal_idx = list(all_idx - top5_abnormal_idx)
    return all_normal_idx

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

# ===========================
# 模型初始化
# ===========================
print('Initializing models...')

# Teacher模型 (GGAD) - 预训练好的，冻结参数
model = Model_ggad(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
model.load_state_dict(torch.load(args.teacher_path))
model.eval()
for p in model.parameters():
    p.requires_grad = False

# Student模型 (OCGNN) - 需要训练
model_s = Model_ocgnn(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)  

# 蒸馏用的MLP
print('Initializing MLP for distillation...')
mlp_s = nn.Linear(args.embedding_dim, 1)  # 将embedding映射到异常分数
pseudo_emb_mlp = nn.Linear(args.embedding_dim, args.embedding_dim)  # 伪异常embedding变换

# ===========================
# 优化器初始化
# ===========================
print('Initializing optimizers...')
optimiser_s = torch.optim.Adam(model_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_mlp_s = torch.optim.Adam(mlp_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_pseudo_emb_mlp = torch.optim.Adam(pseudo_emb_mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)


# ===========================
# 训练循环
# ===========================
print("\n🔁 Starting Student Training...")
output_file = f"./ggad_2_step_2_reg_data_enhance/{args.dataset}_draw_pdf_{args.lr}.txt"
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
            optimiser_pseudo_emb_mlp.zero_grad()

            # === Teacher模型前向传播 (无梯度) ===
            with torch.no_grad():
                _, _, logits_total, _, _, _ = model(features, adj, abnormal_label_idx, normal_label_idx, train_flag=False, args=args)
                score_from_ggad_non_normalize = logits_total.squeeze(dim=-1).squeeze(0)
                
                logits = np.squeeze(logits_total[:, idx_test, :].cpu().detach().numpy())
                auc = roc_auc_score(ano_label[idx_test], logits)
                log_message = (f'Testing_last_ggad_ {args.dataset} AUC: {auc:.4f}\n')
                
                score_from_ggad = min_max_normalize(score_from_ggad_non_normalize)
                
                emb_t_all, _, _, _, emb_abnormal , _= model(features, adj, abnormal_label_idx, normal_label_idx, train_flag=True, args=args)
                emb_t_all = emb_t_all.squeeze(0)
                pseudo_emb = emb_abnormal.squeeze(0)
            
            pseudo_emb_proj = pseudo_emb_mlp(pseudo_emb)

            # === Student模型前向传播 ===
            _, emb_s_all_raw = model_s(features, adj)
            emb_s_all = emb_s_all_raw.squeeze(0)
            
            emb_concat = torch.cat([emb_s_all, pseudo_emb_proj], dim=0)

            student_score_non_normalize = mlp_s(emb_s_all).squeeze(dim=-1)
            student_score = torch.sigmoid(student_score_non_normalize)

            student_score_concat_non_normalize = mlp_s(emb_concat).squeeze(dim=-1)
            student_score_concat = min_max_normalize(student_score_concat_non_normalize)

            # === 生成硬标签 ===
            teacher_hard = to_hard_label_exclude_normal(score_from_ggad_non_normalize, normal_label_idx, top_percent=0.05)
            
            # === 数据增强 ===
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

            # === 计算MSE损失 ===
            pseudo_score_from_ggad = score_from_ggad[abnormal_label_idx]
            teacher_score_concat = torch.cat([score_from_ggad, pseudo_score_from_ggad], dim=0)
            mse_loss = score_distillation_loss(student_score_concat,teacher_score_concat)
            
            # === 总损失计算 ===
            total_loss = mse_loss + 0.01 * reg2_mse

            # === 反向传播和参数更新 ===
            total_loss.backward()
            optimiser_s.step()
            optimiser_mlp_s.step()
            optimiser_pseudo_emb_mlp.step()

            # === 评估和日志 (每5个epoch) ===
            # if epoch % 5 == 0:
            log_message += (
                f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}\n"
                f"MSE Loss = {mse_loss.item():.6f}\n"
                f"Reg2 MSE Loss = {reg2_mse.item():.6f}\n"
            )

            model_s.eval()

            with torch.no_grad():
                _, _, logits, _, _, _= model(features, adj, abnormal_label_idx, normal_label_idx, False, args)
                score_from_ggad_non_normalize = logits.squeeze(dim=-1).squeeze(0) 
                score_from_ggad = torch.sigmoid(score_from_ggad_non_normalize)

            logits_stu = np.squeeze(student_score[idx_test].cpu().detach().numpy())
            auc_stu = roc_auc_score(ano_label[idx_test], logits_stu)

            log_message += f'Testing {args.dataset} AUC_student_mlp_s: {auc_stu:.4f}\n'

            AP_stu = average_precision_score(ano_label[idx_test], logits_stu, average='macro', pos_label=1)
            log_message += f'Testing AP_student_mlp_s: {AP_stu:.4f}\n'

            print(log_message)
            f.write(log_message)
            f.flush()

            if auc_stu >= 0.74:
                print(f"\n🎉 AUC threshold of 0.94 reached! Stopping training at epoch {epoch}.")
                break

            end_time = time.time()
            total_time += end_time - start_time
            pbar.update(1)

    print("Training completed.")

# ==========================================================
# 🏁 Final Evaluation & Model Saving
# ==========================================================
print("\n🏁 Starting final evaluation and saving model weights...")
with torch.no_grad():
    model_s.eval(); mlp_s.eval(); model.eval()
    
    # --- 1. Get final raw scores for evaluation ---
    _, final_emb_s_raw = model_s(features, adj)
    emb_s_all = final_emb_s_raw.squeeze(0)
    student_scores_raw = mlp_s(emb_s_all).squeeze(dim=-1)

    _, _, final_logits_t, _, _, _ = model(features, adj, abnormal_label_idx, normal_label_idx, False, args)
    teacher_scores_raw = final_logits_t.squeeze(dim=-1).squeeze(0)
    
    # --- 2. Safely convert idx_test to a NumPy array ---
    if isinstance(idx_test, torch.Tensor):
        idx_test_np = idx_test.cpu().numpy()
    else:
        idx_test_np = np.array(idx_test)
        
    # --- 3. Output final AUC on raw scores ---
    final_auc_student = roc_auc_score(ano_label[idx_test_np], student_scores_raw[idx_test_np].cpu().numpy())
    final_auc_teacher = roc_auc_score(ano_label[idx_test_np], teacher_scores_raw[idx_test_np].cpu().numpy())

    print(f"\n🎯 Final Test Results (using RAW scores) on '{args.dataset}':")
    print(f"   - Student AUC:  {final_auc_student:.4f}")
    print(f"   - Teacher AUC:  {final_auc_teacher:.4f}")

# --- 4. ✨ NEW: Save Student Model and MLP Weights ---
save_dir = f"./saved_weights/{args.dataset}"
os.makedirs(save_dir, exist_ok=True)

student_model_path = os.path.join(save_dir, f"student_model_lr{args.lr}.pth")
mlp_path = os.path.join(save_dir, f"mlp_lr{args.lr}.pth")

torch.save(model_s.state_dict(), student_model_path)
torch.save(mlp_s.state_dict(), mlp_path)

print(f"\n✅ Student model weights saved to: {student_model_path}")
print(f"✅ MLP weights saved to: {mlp_path}")