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

# 转换邻接矩阵为DGL图
print('Converting adjacency matrix to DGL graph...')
dgl_graph = adj_to_dgl_graph(adj)

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
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_s = torch.optim.Adam(model_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_mlp_s = torch.optim.Adam(mlp_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_pseudo_emb_mlp = torch.optim.Adam(pseudo_emb_mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# 损失函数
b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda() if torch.cuda.is_available() else torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()

# ===========================
# 训练循环
# ===========================
print("\n🔁 Starting Student Training...")
output_file = f"./ggad_2_step_2_reg_data_enhance/{args.dataset}_reg2_enhance_mse.txt"
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
                # 获取teacher模型输出
                _, _, logits_total, _, _, _ = model(features, adj, abnormal_label_idx, normal_label_idx, train_flag=False, args=args)
                score_from_ggad_non_normalize = logits_total.squeeze(dim=-1).squeeze(0)  # [N] teacher异常分数
                
                # 计算测试集AUC (用于监控)
                logits = np.squeeze(logits_total[:, idx_test, :].cpu().detach().numpy())
                auc = roc_auc_score(ano_label[idx_test], logits)
                log_message = (f'Testing_last_ggad_ {args.dataset} AUC: {auc:.4f}\n')
                
                # 归一化teacher分数
                score_from_ggad = min_max_normalize(score_from_ggad_non_normalize)
                
                # 获取teacher embedding和伪异常embedding
                emb_t_all, _, logits, _, emb_abnormal , _= model(features, adj, abnormal_label_idx, normal_label_idx, train_flag=True, args=args)
                emb_t_all = emb_t_all.squeeze(0)  # [N, D] teacher embedding
                pseudo_emb = emb_abnormal.squeeze(0)  # [M, D] 伪异常embedding
            
            # 对伪异常embedding进行变换
            pseudo_emb_proj = pseudo_emb_mlp(pseudo_emb)  # [M, D]
            num_nodes = emb_t_all.size(0)

            # === Student模型前向传播 ===
            _, emb_s_all_raw = model_s(features, adj)
            emb_s_all = emb_s_all_raw.squeeze(0)  # [N, D] student embedding
            
            # 拼接原始节点和伪异常节点的embedding
            emb_concat = torch.cat([emb_s_all, pseudo_emb_proj], dim=0)   # [N+M, D]

            # 计算student异常分数
            student_score_non_normalize = mlp_s(emb_s_all).squeeze(dim=-1)  # [N] 原始分数
            student_score = torch.sigmoid(student_score_non_normalize)  # [N] sigmoid后的分数

            # 计算拼接后的异常分数
            student_score_concat_non_normalize = mlp_s(emb_concat).squeeze(dim=-1)  # [N+M]
            student_score_concat = min_max_normalize(student_score_concat_non_normalize) # [N+M] 归一化分数

            # === 生成硬标签 ===
            teacher_hard = to_hard_label_exclude_normal(score_from_ggad_non_normalize, normal_label_idx, top_percent=0.05)  # [N]
            hard1_indices = torch.where(teacher_hard == 1)[0]  # top5%异常节点索引 [K]

            # === 计算第一个正则化项 (reg_loss) ===
            # 让伪异常embedding接近top5%异常节点的中心
            hard1_emb = emb_t_all[hard1_indices]  # [K, D] top5%异常节点的embedding
            if hard1_emb.shape[0] > 0:
                hard1_emb_center = hard1_emb.mean(dim=0)  # [D] 异常节点中心
            else:
                hard1_emb_center = torch.zeros_like(pseudo_emb[0])
            reg_loss = torch.norm(pseudo_emb_proj - hard1_emb_center, p=2, dim=1).mean()  # L2距离

            # === 计算第二个正则化项 (reg2_mse) - 数据增强 ===
            # Step 1: 获取所有normal节点索引
            all_normal_idx = get_all_normal_nodes(emb_s_all, teacher_hard)

            # Step 2: 对所有normal节点的输入特征进行随机mask增强
            all_normal_features = features.squeeze()[all_normal_idx]  # [N_normal, D] 正常节点特征
            mask = torch.rand_like(all_normal_features) > 0.3  # 30%概率被mask (置0)
            masked_normal_features = all_normal_features * mask  # mask后的特征

            # Step 3: 构建完整的masked特征矩阵
            masked_features = features.clone().squeeze()  # 复制原始特征
            masked_features[all_normal_idx] = masked_normal_features  # 只替换normal节点特征
            masked_features = masked_features.unsqueeze(0)  # 恢复batch维度 [1, N, D]

            # Step 4: 将masked特征输入student网络
            _, emb_s_augmented_raw = model_s(masked_features, adj)
            emb_s_augmented = emb_s_augmented_raw.squeeze(0)  # [N, D] 增强后的embedding

            # Step 5: 计算增强前后normal节点embedding的MSE损失
            emb_s_normal_original = emb_s_all[all_normal_idx]  # 原始normal embedding
            emb_s_normal_augmented = emb_s_augmented[all_normal_idx]  # 增强后normal embedding
            reg2_mse = F.mse_loss(emb_s_normal_augmented, emb_s_normal_original, reduction='mean')

            # # === 计算BCE损失 ===
            # pseudo_score_from_ggad = score_from_ggad[abnormal_label_idx]  # [M] 伪异常的teacher分数
            # pseudo_hard = torch.ones_like(pseudo_score_from_ggad)  # [M] 伪异常硬标签全为1
            # label_concat = torch.cat([teacher_hard, pseudo_hard], dim=0)  # [N+M] 合并标签
            # bce_loss = F.binary_cross_entropy(student_score_concat, label_concat, reduction='mean')

            # === 计算MSE损失 ===
            pseudo_score_from_ggad = score_from_ggad[abnormal_label_idx]           # [M]
            teacher_score_concat = torch.cat([score_from_ggad, pseudo_score_from_ggad], dim=0)  # [N+M]
            mse_loss = score_distillation_loss(student_score_concat,teacher_score_concat)
            # === 总损失计算 ===
            total_loss = mse_loss + 0.1 * reg_loss + 0.01 * reg2_mse
            #            ^^^^^^^^   ^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^
            #            MSE损失    第一个正则化项       第二个正则化项(数据增强MSE)

            # === 反向传播和参数更新 ===
            total_loss.backward()
            optimiser_s.step()
            optimiser_mlp_s.step()
            optimiser_pseudo_emb_mlp.step()

            # === 评估和日志 (每5个epoch) ===
            if epoch % 5 == 0:
                log_message += (
                    f"Epoch {epoch}: Total Loss = {total_loss.item()}\n"
                    f"MSE Loss = {mse_loss.item()}\n"
                    f"Reg1 Loss = {reg_loss.item()}\n"
                    f"Reg2 MSE Loss = {reg2_mse.item()}\n"
                )
                
                # 切换到评估模式
                model_s.eval()
                
                # 重新计算teacher分数 (用于评估)
                train_flag = False
                emb_t, emb_combine, logits, emb_con, emb_abnormal, _= model(features, adj, abnormal_label_idx, normal_label_idx, train_flag, args)
                score_from_ggad_non_normalize = logits.squeeze(dim=-1).squeeze(0) 
                score_from_ggad = torch.sigmoid(score_from_ggad_non_normalize)

                # 记录平均分数
                log_message += f'student_score: {torch.mean(student_score)}\n'
                log_message += f'ggad_score: {torch.mean(score_from_ggad)}\n'
                log_message += f'student_score_non_normalize: {torch.mean(student_score_non_normalize)}\n'
                log_message += f'ggad_score_non_normalize: {torch.mean(score_from_ggad_non_normalize)}\n'

                # 计算组合分数
                student_minus_ggad_score = abs(student_score - score_from_ggad)
                stu_add_ggad_score = student_score + score_from_ggad
                stu_add_ggad_non_normalize_score = student_score_non_normalize + score_from_ggad_non_normalize
                student_minus_ggad_non_normalize_score = student_score_non_normalize - score_from_ggad_non_normalize

                # 提取测试集分数
                logits_stu = np.squeeze(student_score[idx_test].cpu().detach().numpy())
                logits_stu_non_normalize = np.squeeze(student_score_non_normalize[idx_test].cpu().detach().numpy())
                logits_stu_ggad = np.squeeze(stu_add_ggad_score[idx_test].cpu().detach().numpy())
                logits_stu_ggad_non_normalize = np.squeeze(stu_add_ggad_non_normalize_score[idx_test].cpu().detach().numpy())
                logits_stu_minus_ggad = np.squeeze(student_minus_ggad_score[idx_test].cpu().detach().numpy())
                logits_stu_minus_ggad_non_normalize = np.squeeze(student_minus_ggad_non_normalize_score[idx_test].cpu().detach().numpy())

                # 计算AUC
                auc_stu = roc_auc_score(ano_label[idx_test], logits_stu)
                auc_stu_non_normalize = roc_auc_score(ano_label[idx_test], logits_stu_non_normalize)
                auc_stu_ggad = roc_auc_score(ano_label[idx_test], logits_stu_ggad)
                auc_stu_ggad_non_normalize = roc_auc_score(ano_label[idx_test], logits_stu_ggad_non_normalize)
                auc_stu_minus_ggad = roc_auc_score(ano_label[idx_test], logits_stu_minus_ggad)
                auc_stu_minus_ggad_non_normalize = roc_auc_score(ano_label[idx_test], logits_stu_minus_ggad_non_normalize)

                # 记录AUC
                log_message += f'Testing {args.dataset} AUC_student_mlp_s: {auc_stu:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_mlp_s_non_normalize: {auc_stu_non_normalize:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_ggad: {auc_stu_ggad:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_ggad_non_normalize: {auc_stu_ggad_non_normalize:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_minus_ggad: {auc_stu_minus_ggad:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_minus_ggad_non_normalize: {auc_stu_minus_ggad_non_normalize:.4f}\n'

                # 计算AP
                AP_stu = average_precision_score(ano_label[idx_test], logits_stu, average='macro', pos_label=1)
                AP_stu_non_normalize = average_precision_score(ano_label[idx_test], logits_stu_non_normalize, average='macro', pos_label=1)
                AP_stu_ggad = average_precision_score(ano_label[idx_test], logits_stu_ggad, average='macro', pos_label=1)
                AP_stu_ggad_non_normalize = average_precision_score(ano_label[idx_test], logits_stu_ggad_non_normalize, average='macro', pos_label=1)
                AP_stu_minus_ggad = average_precision_score(ano_label[idx_test], logits_stu_minus_ggad, average='macro', pos_label=1)
                AP_stu_minus_ggad_non_normalize = average_precision_score(ano_label[idx_test], logits_stu_minus_ggad_non_normalize, average='macro', pos_label=1)

                # 记录AP
                log_message += f'Testing AP_student_mlp_s: {AP_stu:.4f}\n'
                log_message += f'Testing AP_student_mlp_s_non_normalize: {AP_stu_non_normalize:.4f}\n'
                log_message += f'Testing AP_student_ggad: {AP_stu_ggad:.4f}\n'
                log_message += f'Testing AP_student_ggad_non_normalize: {AP_stu_ggad_non_normalize:.4f}\n'
                log_message += f'Testing AP_student_minus_ggad: {AP_stu_minus_ggad:.4f}\n'
                log_message += f'Testing AP_student_minus_ggad_non_normalize: {AP_stu_minus_ggad_non_normalize:.4f}\n'

                log_message += f'Total time is: {total_time:.2f}\n'

                # 输出和保存日志
                print(log_message)
                f.write(log_message)
                f.flush()

            end_time = time.time()
            total_time += end_time - start_time
            pbar.update(1)

    print("Training completed.")


