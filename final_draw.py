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
# Plotting Function (已修改)
# ===========================
def plot_tsne_test_nodes(embeddings, labels, model_name, dataset_name, seed, run_identifier, keep_left_percent=None, subcaption=None):
    """
    生成并保存 t-SNE 图像，并对图例进行精细化自定义调整。
    """
    print(f"🔬 正在为'{model_name}'在'{dataset_name}'测试集上运行 t-SNE...")
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=500, random_state=seed, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(embeddings)
    
    if keep_left_percent is not None and 0 < keep_left_percent < 100:
        print(f"🔍 根据X轴坐标，正在裁剪数据以只保留最左侧 {keep_left_percent}% 的点...")
        x_coords = tsne_results[:, 0]
        threshold = np.percentile(x_coords, keep_left_percent)
        mask = x_coords <= threshold
        original_count = len(tsne_results)
        tsne_results = tsne_results[mask]
        labels = labels[mask]
        print(f"    移除了 {original_count - len(tsne_results)} 个点。保留 {len(tsne_results)} 个点用于绘图。")

    # 使用 plt.subplots() 创建 figure 和 axes 对象，并设置 DPI
    fig, ax = plt.subplots(figsize=(11.5, 10.5), dpi=300)

    # 修改散点颜色为大红和大蓝，并增大点的大小
    plot = sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, ax=ax,  # 将 ax=ax 传递给 seaborn
        palette=['#0000FF', '#FF0000'], legend='full', alpha=0.9, s=180  # 修改颜色代码和点大小
    )
    
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(''); ax.set_ylabel('') # 移除 x 和 y 标签以保持绘图简洁
    
    # --- 🔥 核心修改部分：自定义图例 ---
    # 1. 获取原始的图例句柄和标签
    handles, legend_labels = plot.get_legend_handles_labels()
    
    # 2. 修改图例的标签文本
    if legend_labels == ['0.0', '1.0'] or legend_labels == ['0', '1']:
        legend_labels = ['Normal Node', 'Abnormal Node']
        
        # 3. 单独修改 'Normal Node' 句柄的标记点大小
        #    假设 'Normal Node' 对应第一个句柄 (handles[0])
        #    我们将它的尺寸设置为一个更大的值，例如 200
        handles[0]._sizes = [400]
        
        #    为了保持一致性，可以明确设置 'Abnormal Node' 的标记点大小
        handles[1]._sizes = [400]

    # 4. 使用修改后的句柄和标签，并调整间距来生成最终图例
    #    handletextpad: 控制标记点和文字之间的距离
    ax.legend(handles, legend_labels,
              fontsize=30,
              handletextpad=0.1, # 减小这个值，让文字更靠近标记点
              title_fontsize=19,
              bbox_to_anchor=(1, 1), loc='upper right')

    # --- 视觉修改结束 ---
    
    if subcaption:
        # 使用 fig.figtext 将文本放置在整个 figure 上
        fig.text(0.5, 0.05, subcaption, wrap=True, horizontalalignment='center', fontsize=22, fontweight='bold')

    save_dir = f'./fig_melt/{dataset_name}/'
    os.makedirs(save_dir, exist_ok=True)
    crop_suffix = f"_kept_left_{keep_left_percent}" if keep_left_percent else ""
    
    # 保存为 SVG 格式
    filename_svg = os.path.join(save_dir, f"tsne_{model_name.replace(' ', '_')}_{run_identifier}{crop_suffix}.svg")
    plt.savefig(filename_svg, bbox_inches='tight', dpi=300)
    
    # 保存为 PDF 格式
    filename_pdf = os.path.join(save_dir, f"tsne_{model_name.replace(' ', '_')}_{run_identifier}{crop_suffix}.pdf")
    plt.savefig(filename_pdf, bbox_inches='tight', dpi=300)
    
    plt.close(fig) # 显式关闭 figure
    print(f"✅ t-SNE 图像已成功保存至 '{filename_svg}' 和 '{filename_pdf}'")

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

    # --- UPDATED: 加载节点列表参数 ---
    parser.add_argument('--load_normal_indices', type=str, required=True, help='从指定 .npy 文件加载固定的正常节点ID')
    parser.add_argument('--load_abnormal_indices', type=str, required=True, help='从指定 .npy 文件加载固定的异常节点ID')
    
    parser.add_argument('--run_identifier', type=str, default='single_run', help='用于文件名的唯一运行标识符')
    
    # 裁剪
    parser.add_argument('--keep_left_percent_model1', type=int, help='[模型1专用] 只保留X轴最左侧N%%的点')
    parser.add_argument('--keep_left_percent_model2', type=int, help='[模型2专用] 只保留X轴最左侧N%%的点')

    # 架构
    parser.add_argument('--embedding_dim', type=int, default=300, help='嵌入维度')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    
    args = parser.parse_args()

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
        
    # --- 4. 节点加载逻辑 (已修改) ---
    print("--- 正在从文件加载固定的节点索引... ---")
    
    # 检查文件是否存在
    if not os.path.exists(args.load_normal_indices):
        print(f"❌ 错误: 找不到正常节点文件: '{args.load_normal_indices}'")
        exit()
    if not os.path.exists(args.load_abnormal_indices):
        print(f"❌ 错误: 找不到异常节点文件: '{args.load_abnormal_indices}'")
        exit()

    # 加载节点
    print(f"--- 从 '{args.load_normal_indices}' 加载正常节点 ---")
    sampled_normal_indices = np.load(args.load_normal_indices)
    n_normal = len(sampled_normal_indices)

    print(f"--- 从 '{args.load_abnormal_indices}' 加载异常节点 ---")
    sampled_abnormal_indices = np.load(args.load_abnormal_indices)
    n_abnormal = len(sampled_abnormal_indices)

    # 组合节点并创建标签
    sampled_indices = np.concatenate([sampled_normal_indices, sampled_abnormal_indices])
    plot_labels = np.concatenate([np.zeros(n_normal), np.ones(n_abnormal)])
    
    print(f"加载完成。总共 {len(sampled_indices)} 个节点用于可视化 ({n_normal} 正常, {n_abnormal} 异常)。")

    sampled_embeddings_1 = embeddings_1[sampled_indices].cpu().numpy()
    sampled_embeddings_2 = embeddings_2[sampled_indices].cpu().numpy()

    # --- 5. 为每个模型绘图 (已修改) ---
    # 为模型1绘图，并添加子图标题
    plot_tsne_test_nodes(
        embeddings=sampled_embeddings_1, labels=plot_labels.copy(), 
        model_name=args.name_1, dataset_name=args.dataset, seed=args.seed, 
        run_identifier=args.run_identifier, keep_left_percent=args.keep_left_percent_model1,
        # subcaption='(a) DoLA'
    )
    
    # 为模型2绘图，并添加子图标题
    plot_tsne_test_nodes(
        embeddings=sampled_embeddings_2, labels=plot_labels.copy(), 
        model_name=args.name_2, dataset_name=args.dataset, seed=args.seed, 
        run_identifier=args.run_identifier, keep_left_percent=args.keep_left_percent_model2,
        # subcaption='(b) SoAD'
    )
    
    print("\n所有可视化任务已完成。")