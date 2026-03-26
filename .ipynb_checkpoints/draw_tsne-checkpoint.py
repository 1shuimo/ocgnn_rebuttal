import torch
import argparse
import numpy as np
import os
import random

# Visualization and ML libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Import necessary components from your project
from model_ocgnn import Model_ocgnn
from utils import load_mat, preprocess_features, normalize_adj
import scipy.sparse as sp

# ===========================
# Plotting Function (无需改动)
# ===========================
def plot_tsne_test_nodes(embeddings, labels, model_name, dataset_name, seed, run_identifier, keep_left_percent=None):
    # ... (这个函数内部完全不需要修改)
    print(f"🔬 正在为'{model_name}'在'{dataset_name}'测试集上运行 t-SNE...")
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=500, random_state=seed, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(embeddings)
    
    title_suffix = f"({run_identifier})"
    if keep_left_percent is not None and 0 < keep_left_percent < 100:
        print(f"🔍 根据X轴坐标，正在裁剪数据以只保留最左侧 {keep_left_percent}% 的点...")
        x_coords = tsne_results[:, 0]
        threshold = np.percentile(x_coords, keep_left_percent)
        mask = x_coords <= threshold
        original_count = len(tsne_results)
        tsne_results = tsne_results[mask]
        labels = labels[mask]
        print(f"   移除了 {original_count - len(tsne_results)} 个点。保留 {len(tsne_results)} 个点用于绘图。")
        title_suffix = f"({run_identifier}, Kept Left {keep_left_percent}%)"

    plt.figure(figsize=(12, 10))
    plot = sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels,
        palette=['#4A90E2', '#E74C3C'], legend='full', alpha=0.8, s=60
    )
    title = f't-SNE of {model_name} Embeddings ({dataset_name} Test Set) {title_suffix}'
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.xticks([]); plt.yticks([])
    handles, legend_labels = plot.get_legend_handles_labels()
    if legend_labels == ['0.0', '1.0'] or legend_labels == ['0', '1']:
        legend_labels = ['Normal', 'Abnormal']
    plot.legend(handles, legend_labels, title='Node Type', fontsize=12, title_fontsize=13)

    save_dir = f'./fig_melt/{dataset_name}/'
    os.makedirs(save_dir, exist_ok=True)
    crop_suffix = f"_kept_left_{keep_left_percent}" if keep_left_percent else ""
    filename = os.path.join(save_dir, f"tsne_{model_name.replace(' ', '_')}_{run_identifier}{crop_suffix}.svg")
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ t-SNE 图像已成功保存至 '{filename}'")

# ===========================
# Main Execution Block (已更新)
# ===========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and compare t-SNE embeddings of two student models.")
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称')
    
    # 模型
    parser.add_argument('--student_path_1', type=str, required=True, help='模型1 .pth 文件路径')
    parser.add_argument('--name_1', type=str, default='Student_Model_main', help='模型1名称')
    parser.add_argument('--student_path_2', type=str, required=True, help='模型2 .pth 文件路径')
    parser.add_argument('--name_2', type=str, default='Student_Model_melt', help='模型2名称')

    # 采样与保存/加载参数
    parser.add_argument('--num_normal_samples', type=int, default=200, help='采样正常节点数')
    parser.add_argument('--num_abnormal_samples', type=int, default=150, help='要采样的异常节点数')
    parser.add_argument('--save_abnormal_indices', type=str, help='将本次采样的异常节点ID保存到指定文件路径')
    parser.add_argument('--load_abnormal_indices', type=str, help='从指定文件加载固定的异常节点ID')
    
    # ✨ --- 新增：用于保存正常节点列表的参数 ---
    parser.add_argument('--save_normal_indices', type=str, help='将本次采样的正常节点ID保存到指定文件路径')

    parser.add_argument('--run_identifier', type=str, default='single_run', help='用于文件名的唯一运行标识符')
    
    # 裁剪
    parser.add_argument('--keep_left_percent_model1', type=int, help='[模型1专用] 只保留X轴最左侧N%%的点')
    parser.add_argument('--keep_left_percent_model2', type=int, help='[模型2专用] 只保留X轴最左侧N%%的点')

    # 架构
    parser.add_argument('--embedding_dim', type=int, default=300, help='嵌入维度')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    
    args = parser.parse_args()

    # ... (数据加载和模型初始化的代码与之前相同) ...
    # --- 1. Load Data ---
    print(f"正在加载数据集: {args.dataset}...")
    adj, features, _, _, _, _, idx_test, ano_label, _, _, _, _ = load_mat(args.dataset)
    if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
        features, _ = preprocess_features(features)
    else:
        features = features.todense()
    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()
    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    ft_size = features.shape[2]
    
    # --- 2. Initialize Models and Load Weights ---
    print("正在初始化模型并加载权重...")
    model_1 = Model_ocgnn(ft_size, args.embedding_dim, 'prelu', 1, 'avg')
    model_2 = Model_ocgnn(ft_size, args.embedding_dim, 'prelu', 1, 'avg')
    model_1.load_state_dict(torch.load(args.student_path_1, map_location=torch.device('cpu')))
    model_2.load_state_dict(torch.load(args.student_path_2, map_location=torch.device('cpu')))
    model_1.eval()
    model_2.eval()

    # --- 3. Generate Embeddings ---
    print("正在为两个模型生成嵌入...")
    with torch.no_grad():
        _, emb_raw_1 = model_1(features, adj)
        embeddings_1 = emb_raw_1.squeeze(0)
        _, emb_raw_2 = model_2(features, adj)
        embeddings_2 = emb_raw_2.squeeze(0)
        
    # --- 4. 节点采样逻辑 (支持保存和加载) ---
    print("正在从测试集中采样节点...")
    
    test_indices = np.array(idx_test)
    test_labels = ano_label[test_indices]
    
    normal_pool = test_indices[test_labels == 0]
    abnormal_pool = test_indices[test_labels == 1]

    print(f"测试集中的节点池: {len(normal_pool)} 个正常, {len(abnormal_pool)} 个异常.")

    n_normal = min(args.num_normal_samples, len(normal_pool))
    sampled_normal_indices = np.random.choice(normal_pool, n_normal, replace=False)

    # ✨ --- 新增：保存采样的正常节点 ---
    if args.save_normal_indices:
        print(f"--- 将当前采样的 {n_normal} 个正常节点保存至 '{args.save_normal_indices}' ---")
        np.save(args.save_normal_indices, sampled_normal_indices)

    if args.load_abnormal_indices:
        print(f"--- 从文件 '{args.load_abnormal_indices}' 加载固定的异常节点 ---")
        sampled_abnormal_indices = np.load(args.load_abnormal_indices)
        n_abnormal = len(sampled_abnormal_indices)
        actual_labels = ano_label[sampled_abnormal_indices]
        if np.any(actual_labels == 0):
            print("警告：加载的节点列表中包含非异常节点！")
    else:
        print(f"--- 正在随机采样 {args.num_abnormal_samples} 个异常节点 ---")
        n_abnormal = min(args.num_abnormal_samples, len(abnormal_pool))
        sampled_abnormal_indices = np.random.choice(abnormal_pool, n_abnormal, replace=False)
        
        if args.save_abnormal_indices:
            print(f"--- 将当前采样的 {n_abnormal} 个异常节点保存至 '{args.save_abnormal_indices}' ---")
            np.save(args.save_abnormal_indices, sampled_abnormal_indices)

    sampled_indices = np.concatenate([sampled_normal_indices, sampled_abnormal_indices])
    plot_labels = np.concatenate([np.zeros(n_normal), np.ones(n_abnormal)])
    
    print(f"采样完成。总共 {len(sampled_indices)} 个节点用于可视化 ({n_normal} 正常, {n_abnormal} 异常)。")

    sampled_embeddings_1 = embeddings_1[sampled_indices].cpu().numpy()
    sampled_embeddings_2 = embeddings_2[sampled_indices].cpu().numpy()

    # --- 5. 为每个模型绘图 ---
    plot_tsne_test_nodes(embeddings=sampled_embeddings_1, labels=plot_labels.copy(), model_name=args.name_1, dataset_name=args.dataset, seed=args.seed, run_identifier=args.run_identifier, keep_left_percent=args.keep_left_percent_model1)
    plot_tsne_test_nodes(embeddings=sampled_embeddings_2, labels=plot_labels.copy(), model_name=args.name_2, dataset_name=args.dataset, seed=args.seed, run_identifier=args.run_identifier, keep_left_percent=args.keep_left_percent_model2)
    
    print("\n所有可视化任务已完成。")