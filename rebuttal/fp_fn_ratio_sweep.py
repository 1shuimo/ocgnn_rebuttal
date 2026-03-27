import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import numpy as np

from utils.log_paths import add_log_subdir_argument, get_log_file


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


parser = argparse.ArgumentParser(description="Sweep FP/FN counts with ratio thresholds.")
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--logits_path", type=str, required=True)
parser.add_argument("--labels_path", type=str, required=True)
parser.add_argument("--base_ratio", type=float, default=0.2)
add_log_subdir_argument(parser, "fp_fn_ratio_sweep")
args = parser.parse_args()

logits = np.load(args.logits_path)
labels = np.load(args.labels_path)

ratios = [args.base_ratio - 0.1, args.base_ratio, args.base_ratio + 0.1]
ratios = [min(0.99, max(0.01, ratio)) for ratio in ratios]

output_file = get_log_file(args, f"{args.dataset}_fp_fn_ratio_sweep.txt")

lines = [
    f"Dataset: {args.dataset}",
    f"Logits path: {args.logits_path}",
    f"Labels path: {args.labels_path}",
    f"Base ratio: {args.base_ratio:.4f}",
    "",
]

for ratio in ratios:
    predicted_mask, predicted_indices = predict_by_ratio(logits, ratio)
    tp, fp, tn, fn = confusion_counts(labels, predicted_mask)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    false_positive_indices = np.where((predicted_mask == 1) & (labels == 0))[0]
    false_negative_indices = np.where((predicted_mask == 0) & (labels == 1))[0]

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

report = "\n".join(lines)
print(report)

with open(output_file, "w") as f:
    f.write(report + "\n")
