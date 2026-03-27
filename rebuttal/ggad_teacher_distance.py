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


def _to_index_tensor(index_list, device):
    return torch.as_tensor(index_list, dtype=torch.long, device=device)


def compute_center_distance(embeddings, abnormal_idx, normal_idx):
    abnormal_idx = _to_index_tensor(abnormal_idx, embeddings.device)
    normal_idx = _to_index_tensor(normal_idx, embeddings.device)
    abnormal_center = embeddings[abnormal_idx].mean(dim=0)
    normal_center = embeddings[normal_idx].mean(dim=0)
    return torch.norm(abnormal_center - normal_center, p=2).item()


def compute_distance_metrics(embeddings, ano_label, labeled_normal_idx):
    real_abnormal_idx = np.where(ano_label == 1)[0]
    all_normal_idx = np.where(ano_label == 0)[0]
    return {
        "real_abnormal_to_labeled_normal_center": compute_center_distance(
            embeddings, real_abnormal_idx, labeled_normal_idx
        ),
        "real_abnormal_to_all_normal_center": compute_center_distance(
            embeddings, real_abnormal_idx, all_normal_idx
        ),
    }


parser = argparse.ArgumentParser(description="Compute teacher-only center distances.")
parser.add_argument("--dataset", type=str, default="reddit")
parser.add_argument("--teacher_path", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--embedding_dim", type=int, default=300)
parser.add_argument("--readout", type=str, default="avg")
parser.add_argument("--negsamp_ratio", type=int, default=1)
parser.add_argument("--mean", type=float, default=0.0)
parser.add_argument("--var", type=float, default=0.0)
add_log_subdir_argument(parser, "ggad_teacher_distance")
args = parser.parse_args()

if args.dataset in ["reddit", "photo"]:
    args.mean = 0.02
    args.var = 0.01
else:
    args.mean = 0.0
    args.var = 0.0

print("Dataset:", args.dataset)

dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(
    args.dataset
)

if args.dataset in ["Amazon", "tf_finace", "reddit", "elliptic"]:
    features, _ = preprocess_features(features)
else:
    features = features.todense()

ft_size = features.shape[1]
raw_adj = adj
adj = normalize_adj(adj)

raw_adj = (raw_adj + sp.eye(raw_adj.shape[0])).todense()
adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
raw_adj = torch.FloatTensor(raw_adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])

model = Model_ggad(ft_size, args.embedding_dim, "prelu", args.negsamp_ratio, args.readout)
model.load_state_dict(torch.load(args.teacher_path))
model.eval()

with torch.no_grad():
    emb_t_all, _, logits, _, _, _ = model(
        features, adj, abnormal_label_idx, normal_label_idx, train_flag=False, args=args
    )
    emb_t_all = emb_t_all.squeeze(0)
    score = logits.squeeze(dim=-1).squeeze(0)
    logits_test = np.squeeze(logits[:, idx_test, :].cpu().detach().numpy())
    auc = roc_auc_score(ano_label[idx_test], logits_test)
    ap = average_precision_score(ano_label[idx_test], logits_test, average="macro", pos_label=1)
    distance_metrics = compute_distance_metrics(emb_t_all, ano_label, normal_label_idx)

report = (
    f"Dataset: {args.dataset}\n"
    f"Teacher path: {args.teacher_path}\n"
    f"Teacher AUC: {auc:.4f}\n"
    f"Teacher AP: {ap:.4f}\n"
    f"Teacher real_abnormal_center -> labeled_normal_center: "
    f"{distance_metrics['real_abnormal_to_labeled_normal_center']:.4f}\n"
    f"Teacher real_abnormal_center -> all_normal_center: "
    f"{distance_metrics['real_abnormal_to_all_normal_center']:.4f}\n"
)

print(report)

output_file = get_log_file(args, f"{args.dataset}_teacher_distance.txt")
with open(output_file, "w") as f:
    f.write(report)
