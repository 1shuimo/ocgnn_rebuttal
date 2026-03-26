import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import random
import argparse
from sklearn.metrics import roc_auc_score

# 确保导入与训练时完全相同的模型和工具函数
from models.model import Model_ggad
from utils.utils import *

def debug_test():
    parser = argparse.ArgumentParser(description="用于GGAD教师模型的隔离环境测试")
    parser.add_argument('--dataset', type=str, default='tolokers', help="数据集名称") 
    # 重要：请确保这里的路径指向您要测试的最佳模型文件
    parser.add_argument('--teacher_path', type=str, default='tolokers_ggad_teacher_best_435.pth', help="教师模型权重文件路径")
    parser.add_argument('--embedding_dim', type=int, default=300, help="嵌入维度")
    parser.add_argument('--readout', type=str, default='avg', help="Readout方式")
    parser.add_argument('--negsamp_ratio', type=int, default=1, help="负采样率")
    parser.add_argument('--seed', type=int, default=0, help="随机种子")
    args = parser.parse_args()
    
    # 设置模型forward函数需要的其他参数，必须与原始训练时一致
    if args.dataset in ['reddit', 'Photo']:
        args.mean = 0.02
        args.var = 0.01
    else:
        args.mean = 0.0
        args.var = 0.0

    # 1. 设置随机种子，与原脚本保持一致
    print("正在设置随机种子...")
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    
    # 2. 使用完全相同的函数加载数据
    print("正在加载数据...")
    adj, features, _, _, _, _, idx_test, ano_label, _, _, normal_label_idx, abnormal_label_idx = load_mat(args.dataset)

    # 3. 使用完全相同的步骤预处理数据
    print("正在预处理数据...")
    if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
        features, _ = preprocess_features(features)
    else:
        features = features.todense()

    ft_size = features.shape[1]
    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()
    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    
    # 4. 初始化模型并加载权重
    print(f"正在初始化模型并从 {args.teacher_path} 加载权重...")
    model = Model_ggad(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
    model.load_state_dict(torch.load(args.teacher_path))
    
    # 5. 关键步骤：切换到评估模式
    model.eval()
    
    # 6. 执行一次干净的、无干扰的前向传播
    print("正在执行推理...")
    with torch.no_grad():
        # 使用 train_flag=False，与原始评估逻辑完全一致
        _, _, logits, _, _, _ = model(features, adj, abnormal_label_idx, normal_label_idx, train_flag=False, args=args)

    # 7. 计算并打印最终结果
    logits_test = np.squeeze(logits[:, idx_test, :].cpu().numpy())
    auc = roc_auc_score(ano_label[idx_test], logits_test)
    
    print("\n--- 调试测试结果 ---")
    print(f"在测试集上的AUC: {auc:.4f}")
    print("------------------------\n")


if __name__ == '__main__':
    debug_test()
