import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

def parse_log_file(log_file_path):
    """
    解析log文件，提取各种指标
    """
    data = {
        'epoch': [],
        'total_loss': [],
        'reg_loss': [],
        'bce_loss': [],
        'reg2_loss': [],
        'auc_student': [],
        'auc_student_non_norm': [],
        'auc_student_ggad': [],
        'auc_student_ggad_non_norm': [],
        'auc_student_minus_ggad': [],
        'auc_student_minus_ggad_non_norm': [],
        'ap_student': [],
        'ap_student_non_norm': [],
        'ap_student_ggad': [],
        'ap_student_ggad_non_norm': [],
        'ap_student_minus_ggad': [],
        'ap_student_minus_ggad_non_norm': []
    }
    
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # 按epoch分割
    epoch_blocks = re.split(r'Epoch (\d+):', content)[1:]  # 去掉第一个空元素
    
    for i in range(0, len(epoch_blocks), 2):
        if i + 1 < len(epoch_blocks):
            epoch_num = int(epoch_blocks[i])
            epoch_content = epoch_blocks[i + 1]
            
            data['epoch'].append(epoch_num)
            
            # 提取Loss
            total_loss = re.search(r'Total Loss = ([\d.]+)', epoch_content)
            reg_loss = re.search(r'Reg Loss = ([\d.]+)', epoch_content)
            bce_loss = re.search(r'BCE Loss = ([\d.]+)', epoch_content)
            reg2_loss = re.search(r'Reg2 Loss = ([\d.]+)', epoch_content)
            
            data['total_loss'].append(float(total_loss.group(1)) if total_loss else np.nan)
            data['reg_loss'].append(float(reg_loss.group(1)) if reg_loss else np.nan)
            data['bce_loss'].append(float(bce_loss.group(1)) if bce_loss else np.nan)
            data['reg2_loss'].append(float(reg2_loss.group(1)) if reg2_loss else np.nan)
            
            # 提取AUC
            auc_patterns = [
                (r'AUC_student_mlp_s: ([\d.]+)', 'auc_student'),
                (r'AUC_student_mlp_s_non_normalize: ([\d.]+)', 'auc_student_non_norm'),
                (r'AUC_student_ggad: ([\d.]+)', 'auc_student_ggad'),
                (r'AUC_student_ggad_non_normalize: ([\d.]+)', 'auc_student_ggad_non_norm'),
                (r'AUC_student_minus_ggad: ([\d.]+)', 'auc_student_minus_ggad'),
                (r'AUC_student_minus_ggad_non_normalize: ([\d.]+)', 'auc_student_minus_ggad_non_norm')
            ]
            
            for pattern, key in auc_patterns:
                match = re.search(pattern, epoch_content)
                data[key].append(float(match.group(1)) if match else np.nan)
            
            # 提取AP
            ap_patterns = [
                (r'AP_student_mlp_s: ([\d.]+)', 'ap_student'),
                (r'AP_student_mlp_s_non_normalize: ([\d.]+)', 'ap_student_non_norm'),
                (r'AP_student_ggad: ([\d.]+)', 'ap_student_ggad'),
                (r'AP_student_ggad_non_normalize: ([\d.]+)', 'ap_student_ggad_non_norm'),
                (r'AP_student_minus_ggad: ([\d.]+)', 'ap_student_minus_ggad'),
                (r'AP_student_minus_ggad_non_normalize: ([\d.]+)', 'ap_student_minus_ggad_non_norm')
            ]
            
            for pattern, key in ap_patterns:
                match = re.search(pattern, epoch_content)
                data[key].append(float(match.group(1)) if match else np.nan)
    
    return data

def plot_training_curves(data, save_path=None):
    """
    画训练曲线
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Curves', fontsize=16, fontweight='bold')
    
    epochs = data['epoch']
    
    # 1. 总损失
    axes[0, 0].plot(epochs, data['total_loss'], 'b-', linewidth=2, label='Total Loss')
    axes[0, 0].set_title('Total Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 2. 各个损失分量
    axes[0, 1].plot(epochs, data['reg_loss'], 'r-', linewidth=2, label='Reg Loss')
    axes[0, 1].plot(epochs, data['bce_loss'], 'g-', linewidth=2, label='BCE Loss')
    axes[0, 1].plot(epochs, data['reg2_loss'], 'orange', linewidth=2, label='Reg2 Loss')
    axes[0, 1].set_title('Loss Components', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 3. 主要AUC指标
    axes[0, 2].plot(epochs, data['auc_student'], 'b-', linewidth=2, label='Student')
    axes[0, 2].plot(epochs, data['auc_student_ggad'], 'r-', linewidth=2, label='Student+GGAD')
    axes[0, 2].plot(epochs, data['auc_student_ggad_non_norm'], 'g-', linewidth=2, label='Student+GGAD(non-norm)')
    axes[0, 2].set_title('Main AUC Curves', fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('AUC')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    axes[0, 2].set_ylim([0, 1])
    
    # 4. 所有AUC指标
    auc_keys = ['auc_student', 'auc_student_non_norm', 'auc_student_ggad', 
                'auc_student_ggad_non_norm', 'auc_student_minus_ggad', 'auc_student_minus_ggad_non_norm']
    auc_labels = ['Student', 'Student(non-norm)', 'Student+GGAD', 
                  'Student+GGAD(non-norm)', 'Student-GGAD', 'Student-GGAD(non-norm)']
    colors = ['blue', 'lightblue', 'red', 'lightcoral', 'green', 'lightgreen']
    
    for key, label, color in zip(auc_keys, auc_labels, colors):
        axes[1, 0].plot(epochs, data[key], color=color, linewidth=2, label=label, alpha=0.8)
    axes[1, 0].set_title('All AUC Curves', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_ylim([0, 1])
    
    # 5. 主要AP指标
    axes[1, 1].plot(epochs, data['ap_student'], 'b-', linewidth=2, label='Student')
    axes[1, 1].plot(epochs, data['ap_student_ggad'], 'r-', linewidth=2, label='Student+GGAD')
    axes[1, 1].plot(epochs, data['ap_student_ggad_non_norm'], 'g-', linewidth=2, label='Student+GGAD(non-norm)')
    axes[1, 1].set_title('Main AP Curves', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AP')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0, 1])
    
    # 6. 所有AP指标
    ap_keys = ['ap_student', 'ap_student_non_norm', 'ap_student_ggad', 
               'ap_student_ggad_non_norm', 'ap_student_minus_ggad', 'ap_student_minus_ggad_non_norm']
    ap_labels = ['Student', 'Student(non-norm)', 'Student+GGAD', 
                 'Student+GGAD(non-norm)', 'Student-GGAD', 'Student-GGAD(non-norm)']
    
    for key, label, color in zip(ap_keys, ap_labels, colors):
        axes[1, 2].plot(epochs, data[key], color=color, linewidth=2, label=label, alpha=0.8)
    axes[1, 2].set_title('All AP Curves', fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('AP')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_summary(data):
    """
    打印训练总结
    """
    print("="*60)
    print("Training Summary")
    print("="*60)
    print(f"Total Epochs: {len(data['epoch'])}")
    print(f"Epoch Range: {min(data['epoch'])} - {max(data['epoch'])}")
    
    print("\nFinal Values:")
    final_idx = -1
    print(f"Total Loss: {data['total_loss'][final_idx]:.4f}")
    print(f"Reg Loss: {data['reg_loss'][final_idx]:.4f}")
    print(f"BCE Loss: {data['bce_loss'][final_idx]:.4f}")
    print(f"Reg2 Loss: {data['reg2_loss'][final_idx]:.4f}")
    
    print("\nBest AUC:")
    print(f"Student: {max(data['auc_student']):.4f}")
    print(f"Student+GGAD: {max(data['auc_student_ggad']):.4f}")
    print(f"Student+GGAD(non-norm): {max(data['auc_student_ggad_non_norm']):.4f}")
    
    print("\nBest AP:")
    print(f"Student: {max(data['ap_student']):.4f}")
    print(f"Student+GGAD: {max(data['ap_student_ggad']):.4f}")
    print(f"Student+GGAD(non-norm): {max(data['ap_student_ggad_non_norm']):.4f}")

# 使用示例
if __name__ == "__main__":
    # 替换为你的log文件路径
    log_file_path = "photo_reg2_bce.txt"
    
    # 解析log文件
    print("Parsing log file...")
    data = parse_log_file(log_file_path)
    
    # 打印总结
    print_summary(data)
    
    # 画图
    print("Plotting curves...")
    plot_training_curves(data, save_path="training_curves.png")
    
    print("Done! Graph saved as 'training_curves.png'")