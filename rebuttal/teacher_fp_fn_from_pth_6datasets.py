import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import random

import dgl
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from models.model import Model_ggad
from utils.log_paths import add_log_subdir_argument, get_log_file
from utils.utils_old import *


DATASETS = [
    "Amazon",
    "tolokers",
    "tf_finace",
    "YelpChi-all",
    "reddit",
    "photo",
]


def confusion_counts(labels, predicted_abnormal_mask):
    labels = labels.astype(int)
    pred = predicted_abnormal_mask.astype(int)
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    return tp, fp, tn, fn


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def predict_by_ratio(logits, ratio):
    num_total = logits.shape[0]
    predicted_abnormal_count = max(1, int(ratio * num_total))
    sorted_indices = np.argsort(logits)[::-1]
    predicted_abnormal_indices = sorted_indices[:predicted_abnormal_count]
    mask = np.zeros(num_total, dtype=bool)
    mask[predicted_abnormal_indices] = True
    return mask, predicted_abnormal_indices


def dataset_noise_args(dataset):
    if dataset in ["reddit", "photo"]:
        return 0.02, 0.01
    return 0.0, 0.0


def load_teacher_scores(dataset, teacher_path, args):
    mean, var = dataset_noise_args(dataset)
    args.mean = mean
    args.var = var

    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(
        dataset
    )

    if dataset in ["Amazon", "tf_finace", "reddit", "elliptic"]:
        features, _ = preprocess_features(features)
    else:
        features = features.todense()

    ft_size = features.shape[1]
    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()
    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])

    model = Model_ggad(ft_size, args.embedding_dim, "prelu", args.negsamp_ratio, args.readout)
    model.load_state_dict(torch.load(teacher_path))
    model.eval()

    with torch.no_grad():
        _, _, logits, _, _, _ = model(
            features, adj, abnormal_label_idx, normal_label_idx, train_flag=False, args=args
        )
        logits_test = np.squeeze(logits[:, idx_test, :].cpu().detach().numpy())
        labels_test = np.asarray(ano_label[idx_test])
        auc = roc_auc_score(labels_test, logits_test)
        ap = average_precision_score(labels_test, logits_test, average="macro", pos_label=1)

    return logits_test, labels_test, auc, ap


parser = argparse.ArgumentParser(description="Compute teacher FP/FN for 6 datasets directly from teacher pth.")
parser.add_argument("--teacher_dir", type=str, default="../ggad_new_best_pth")
parser.add_argument("--base_ratio", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--embedding_dim", type=int, default=300)
parser.add_argument("--readout", type=str, default="avg")
parser.add_argument("--negsamp_ratio", type=int, default=1)
add_log_subdir_argument(parser, "teacher_fp_fn_from_pth")
args = parser.parse_args()

dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ratios = [args.base_ratio - 0.1, args.base_ratio, args.base_ratio + 0.1]
ratios = [min(0.99, max(0.01, ratio)) for ratio in ratios]

summary_lines = [
    f"Teacher dir: {args.teacher_dir}",
    f"Base ratio: {args.base_ratio:.4f}",
    f"Datasets: {', '.join(DATASETS)}",
    "",
]

for dataset in DATASETS:
    teacher_path = str(Path(args.teacher_dir) / f"{dataset}_ggad_teacher_final.pth")
    logits_test, labels_test, auc, ap = load_teacher_scores(dataset, teacher_path, args)

    lines = [
        f"Dataset: {dataset}",
        f"Teacher path: {teacher_path}",
        f"Teacher AUC: {auc:.4f}",
        f"Teacher AP: {ap:.4f}",
        "",
    ]

    summary_lines.extend(
        [
            f"Dataset: {dataset}",
            f"Teacher AUC: {auc:.4f}",
            f"Teacher AP: {ap:.4f}",
        ]
    )

    for ratio in ratios:
        predicted_mask, predicted_indices = predict_by_ratio(logits_test, ratio)
        tp, fp, tn, fn = confusion_counts(labels_test, predicted_mask)
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        false_positive_indices = np.where((predicted_mask == 1) & (labels_test == 0))[0]
        false_negative_indices = np.where((predicted_mask == 0) & (labels_test == 1))[0]

        lines.extend(
            [
                f"Ratio: {ratio:.4f}",
                f"Predicted abnormal count: {predicted_mask.sum()}",
                f"TP: {tp}",
                f"FP: {fp}",
                f"TN: {tn}",
                f"FN: {fn}",
                f"Precision: {precision:.4f}",
                f"Recall: {recall:.4f}",
                f"F1: {f1:.4f}",
                f"FP indices: {false_positive_indices.tolist()}",
                f"FN indices: {false_negative_indices.tolist()}",
                f"Top-ranked abnormal indices: {predicted_indices.tolist()}",
                "",
            ]
        )

        summary_lines.append(
            f"Ratio {ratio:.2f}: TP={tp} FP={fp} TN={tn} FN={fn} "
            f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}"
        )

    lines.append("")
    summary_lines.append("")

    report = "\n".join(lines)
    print(report)

    output_file = get_log_file(args, f"{dataset}_teacher_fp_fn.txt")
    with open(output_file, "w") as f:
        f.write(report + "\n")

summary_report = "\n".join(summary_lines)
summary_file = get_log_file(args, "teacher_fp_fn_6datasets_summary.txt")
with open(summary_file, "w") as f:
    f.write(summary_report + "\n")

print(summary_report)
