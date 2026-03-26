import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ===========================
# 配置文件路径
# ===========================
# 替换为你的 .npy 文件路径
logits_files = [
    "./teacher_find_fp/photo_best_logits.npy",  # GGAD 模型的 logits
    "./student_only_distill_find_fp/photo_best_logits.npy",  # NormD 模型的 logits
    "./student_find_fp/photo_best_logits.npy",  # NormDR 模型的 logits
]
labels_file = "./teacher_find_fp/photo_test_labels.npy"  # 测试集真实标签

# 模型名称（用于图例）
model_names = ["GGAD", "NormD", "NormDR"]

# ===========================
# 加载数据
# ===========================
# 加载测试集真实标签
labels = np.load(labels_file)

# 加载每个模型的 logits
logits_list = [np.load(file) for file in logits_files]

# ===========================
# 绘制 AUC 曲线
# ===========================
plt.figure(figsize=(8, 6))

for logits, model_name in zip(logits_list, model_names):
    # 计算 ROC 曲线
    fpr, tpr, _ = roc_curve(labels, logits)
    roc_auc = auc(fpr, tpr)
    if model_name == "GGAD":
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = 0.6476)")
    else:
    # 绘制 ROC 曲线
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.4f})")

# 图形美化
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)", fontsize=12)
plt.ylabel("True Positive Rate (TPR)", fontsize=12)
plt.title("ROC Curve Comparison", fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)

# 保存图像
plt.savefig("./auc_comparison_curve.svg", dpi=300)
plt.show()