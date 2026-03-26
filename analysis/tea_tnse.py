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
import matplotlib.pyplot as plt
# Corrected: Added missing imports
import dgl
from scipy.stats import norm
import scipy.sparse as sp
import seaborn as sns
from sklearn.manifold import TSNE


# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2]))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def plot_tsne_embeddings(embeddings, labels, title, filename, seed=0):
    """
    使用 t-SNE 对节点嵌入进行降维并绘制散点图
    
    Args:
        embeddings (np.array): 节点嵌入, 形状为 [N, D]
        labels (np.array): 节点的真实标签 (0 for normal, 1 for abnormal)
        title (str): 图像标题
        filename (str): 保存图像的文件路径
        seed (int): 随机种子
    """
    print(f"🔬 Running t-SNE for '{title}'...")
    # 初始化 t-SNE 模型
    tsne = TSNE(n_components=2, perplexity=30, n_iter=500, random_state=seed)
    
    # 执行降维
    tsne_results = tsne.fit_transform(embeddings)
    
    # 绘图
    plt.figure(figsize=(12, 10))
    # 使用 seaborn 绘制散点图
    sns.scatterplot(
        x=tsne_results[:, 0], 
        y=tsne_results[:, 1],
        hue=labels,
        palette=['#4A90E2', '#E74C3C'], # 0: 蓝色 (Normal), 1: 红色 (Abnormal)
        legend='full',
        alpha=0.7,
        s=50  # s控制点的大小
    )
    
    # 设置图表信息
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.xticks([]) # 隐藏坐标轴刻度
    plt.yticks([])
    
    # 确保保存目录存在
    save_dir = os.path.dirname(filename)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存图像
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ t-SNE plot saved to '{filename}'")

def plot_tsne_test_nodes(embeddings, labels, model_name, dataset_name, seed=0):
    """
    专门为测试节点绘制t-SNE图
    """
    title = f't-SNE of {model_name} Embeddings ({dataset_name})'
    filename = f'fig/tsne/{dataset_name}/tsne_{model_name.lower().replace(" ", "_")}.svg'
    plot_tsne_embeddings(embeddings, labels, title, filename, seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and compare t-SNE embeddings of student models.")
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset (e.g., reddit, YelpChi-all)')
    
    # --- Arguments for Model 1 (Student Model) ---
    parser.add_argument('--student_path_1', type=str, help='Path to the first student model .pth file (optional)')
    parser.add_argument('--name_1', type=str, default='Student_Model_main', help='Descriptive name for the first model')
    
    # --- Arguments for Model 2 (Optional second student model) ---
    parser.add_argument('--student_path_2', type=str, help='Path to the second student model .pth file (optional)')
    parser.add_argument('--name_2', type=str, default='Student_Model_melt', help='Descriptive name for the second model')

    # --- Teacher Model Arguments ---
    parser.add_argument('--teacher_path', type=str, help='Path to the teacher model .pth file (optional)')
    parser.add_argument('--include_teacher', action='store_true', help='Include teacher model visualization')

    # ✨ --- 采样参数 ---
    parser.add_argument('--num_normal_samples', type=int, default=200, help='Number of normal test nodes to sample.')
    parser.add_argument('--num_abnormal_samples', type=int, default=150, help='Number of abnormal test nodes to sample.')

    # --- Model Architecture Arguments ---
    parser.add_argument('--embedding_dim', type=int, default=300, help='Dimension of the node embeddings')
    parser.add_argument('--negsamp_ratio', type=int, default=1, help='Negative sampling ratio')
    parser.add_argument('--readout', type=str, default='avg', help='Readout method')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    
    # --- Training Arguments (for compatibility) ---
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--drop_prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--batch_size', type=int, default=300, help='Batch size')
    parser.add_argument('--num_epoch', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--subgraph_size', type=int, default=4, help='Subgraph size')
    parser.add_argument('--auc_test_rounds', type=int, default=256, help='AUC test rounds')
    
    args = parser.parse_args()

    # --- Set Seed ---
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

    # 设置数据集特定参数
    if args.dataset in ['reddit', 'Photo']:
        args.mean = 0.02
        args.var = 0.01
    else:
        args.mean = 0.0
        args.var = 0.0

    # --- 1. Load Data ---
    print(f"Loading dataset: {args.dataset}...")
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(args.dataset)

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

    # --- 2. Initialize Models and Load Weights ---
    models_to_visualize = []
    
    # Student Model 1
    if args.student_path_1:
        print(f"Loading first student model from: {args.student_path_1}")
        model_s1 = Model_ocgnn(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
        model_s1.load_state_dict(torch.load(args.student_path_1, map_location=torch.device('cpu')))
        model_s1.eval()
        models_to_visualize.append(('student_1', model_s1, args.name_1))

    # Student Model 2 (optional)
    if args.student_path_2:
        print(f"Loading second student model from: {args.student_path_2}")
        model_s2 = Model_ocgnn(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
        model_s2.load_state_dict(torch.load(args.student_path_2, map_location=torch.device('cpu')))
        model_s2.eval()
        models_to_visualize.append(('student_2', model_s2, args.name_2))

    # Teacher Model (optional)
    if args.include_teacher and args.teacher_path:
        print(f"Loading teacher model from: {args.teacher_path}")
        model_teacher = Model_ggad(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
        model_teacher.load_state_dict(torch.load(args.teacher_path, map_location=torch.device('cpu')))
        model_teacher.eval()
        models_to_visualize.append(('teacher', model_teacher, 'Teacher_GGAD'))

    if not models_to_visualize:
        print("❌ No models specified! Please provide at least one model path.")
        exit(1)

    # --- 3. Sample Test Nodes ---
    print("Sampling normal and abnormal nodes from the test set...")
    
    # 确保idx_test是numpy数组
    if isinstance(idx_test, torch.Tensor):
        test_indices = idx_test.cpu().numpy()
    else:
        test_indices = np.array(idx_test)
    
    test_labels = ano_label[test_indices]
    
    # 创建节点池
    normal_pool = test_indices[test_labels == 0]
    abnormal_pool = test_indices[test_labels == 1]

    print(f"Node pools in test set: {len(normal_pool)} Normal, {len(abnormal_pool)} Abnormal.")

    # 从每个池中采样
    n_normal = min(args.num_normal_samples, len(normal_pool))
    n_abnormal = min(args.num_abnormal_samples, len(abnormal_pool))

    sampled_normal_indices = np.random.choice(normal_pool, n_normal, replace=False)
    sampled_abnormal_indices = np.random.choice(abnormal_pool, n_abnormal, replace=False)

    # 组合索引和标签
    sampled_indices = np.concatenate([sampled_normal_indices, sampled_abnormal_indices])
    plot_labels = np.concatenate([np.zeros(n_normal), np.ones(n_abnormal)])
    
    print(f"Final sampling: {n_normal} Normal + {n_abnormal} Abnormal = {len(sampled_indices)} total nodes")

    # --- 4. Generate Embeddings and Plot t-SNE for Each Model ---
    print("Generating embeddings and creating t-SNE visualizations...")
    
    with torch.no_grad():
        for model_type, model, model_name in models_to_visualize:
            print(f"\n🎨 Processing {model_name}...")
            
            if model_type == 'teacher':
                # Teacher model has different output format
                emb_t_all, _, _, _, _, _ = model(features, adj, abnormal_label_idx, normal_label_idx, train_flag=False, args=args)
                embeddings = emb_t_all.squeeze(0)
            else:
                # Student models
                _, emb_raw = model(features, adj)
                embeddings = emb_raw.squeeze(0)
            
            # 获取采样节点的嵌入
            sampled_embeddings = embeddings[sampled_indices].cpu().numpy()
            
            # 绘制t-SNE图
            plot_tsne_test_nodes(
                embeddings=sampled_embeddings,
                labels=plot_labels,
                model_name=model_name,
                dataset_name=args.dataset,
                seed=args.seed
            )

    # --- 5. Output Summary ---
    print(f"\n🎯 Visualization Summary:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Nodes visualized: {len(sampled_indices)} ({n_normal} Normal + {n_abnormal} Abnormal)")
    print(f"   Models processed: {len(models_to_visualize)}")
    for _, _, model_name in models_to_visualize:
        print(f"      - {model_name}")
    print(f"   Output directory: fig/tsne/{args.dataset}/")
    print("\n✅ All visualizations are complete!")