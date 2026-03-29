import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from models.model_ocgnn import Model_ocgnn
from models.model_rho import Model_RHO
from utils.log_paths import add_log_subdir_argument, get_log_file
from utils.utils_old import load_mat, preprocess_features


def fix_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def build_laplacian(adj):
    adj = sp.csr_matrix(adj)
    adj = adj.maximum(adj.T)
    adj = adj + sp.eye(adj.shape[0], dtype=adj.dtype)
    degrees = np.array(adj.sum(axis=1)).flatten()
    with np.errstate(divide="ignore"):
        deg_inv_sqrt = np.power(degrees, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    d_inv_sqrt = sp.diags(deg_inv_sqrt)
    lap = sp.eye(adj.shape[0], dtype=np.float32) - d_inv_sqrt @ adj @ d_inv_sqrt
    return sparse_mx_to_torch_sparse_tensor(lap)


def init_center_c(lap, inputs, net, device, eps=0.1):
    c_global = torch.zeros(net.rep_dim, device=device)
    c_local = torch.zeros(net.rep_dim, device=device)
    net.eval()
    with torch.no_grad():
        outputs_global, outputs_local, _ = net(lap, inputs)
        n_samples = outputs_global.shape[0]
        c_global = torch.sum(outputs_global, dim=0) / n_samples
        c_local = torch.sum(outputs_local, dim=0) / n_samples
    c_local[(abs(c_local) < eps) & (c_local < 0)] = -eps
    c_local[(abs(c_local) < eps) & (c_local > 0)] = eps
    c_global[(abs(c_global) < eps) & (c_global < 0)] = -eps
    c_global[(abs(c_global) < eps) & (c_global > 0)] = eps
    return c_local, c_global


def rho_scores(outputs_global, outputs_local, center_global, center_local):
    dist_global = torch.sum((outputs_global - center_global) ** 2, dim=1)
    dist_local = torch.sum((outputs_local - center_local) ** 2, dim=1)
    return 0.5 * dist_global + 0.5 * dist_local


def min_max_normalize(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    if max_val == min_val:
        return torch.zeros_like(tensor)
    return (tensor - min_val) / (max_val - min_val)


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


parser = argparse.ArgumentParser(description="RHO teacher -> OCGNN student distillation.")
parser.add_argument("--dataset", type=str, default="Amazon")
parser.add_argument("--teacher_path", type=str, required=True)
parser.add_argument("--lr", type=float)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_epoch", type=int)
parser.add_argument("--embedding_dim", type=int, default=300)
parser.add_argument("--readout", type=str, default="avg")
parser.add_argument("--negsamp_ratio", type=int, default=1)
parser.add_argument("--use_normreg", type=int, choices=[0, 1], default=1)
parser.add_argument("--normreg_weight", type=float, default=0.01)
parser.add_argument("--rho_hidden1", type=int, default=1024)
parser.add_argument("--rho_hidden2", type=int, default=64)
parser.add_argument("--rho_layers", type=int, default=2)
parser.add_argument("--rho_batch_size", type=int, default=0)
parser.add_argument("--rho_tau", type=float, default=0.2)
add_log_subdir_argument(parser, "rho_labeledNormal_our")
args = parser.parse_args()

if args.lr is None:
    if args.dataset in ["Amazon", "tf_finace", "YelpChi-all"]:
        args.lr = 5e-4
    elif args.dataset in ["reddit", "photo", "tolokers"]:
        args.lr = 1e-3
    else:
        args.lr = 5e-4

if args.num_epoch is None:
    if args.dataset in ["Amazon"]:
        args.num_epoch = 1200
    elif args.dataset in ["tf_finace"]:
        args.num_epoch = 1500
    elif args.dataset in ["reddit", "photo"]:
        args.num_epoch = 1500
    elif args.dataset in ["tolokers", "YelpChi-all"]:
        args.num_epoch = 1500
    else:
        args.num_epoch = 1200

fix_seed(args.seed)

adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(
    args.dataset
)

if args.dataset in ["Amazon", "tf_finace", "reddit", "elliptic", "questions"]:
    features, _ = preprocess_features(features)
else:
    features = features.todense()

lap = build_laplacian(adj)
adj_dense = adj
adj_dense = (adj_dense + sp.eye(adj_dense.shape[0])).todense()
adj_dense = torch.FloatTensor(np.asarray(adj_dense)[np.newaxis])
features = torch.FloatTensor(np.asarray(features))
features_batched = features.unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lap = lap.to(device)
features = features.to(device)
features_batched = features_batched.to(device)
adj_dense = adj_dense.to(device)

teacher = Model_RHO(
    in_features=features.shape[1],
    hidden1=args.rho_hidden1,
    hidden2=args.rho_hidden2,
    layers=args.rho_layers,
    batch_size=args.rho_batch_size,
    tau=args.rho_tau,
).to(device)
teacher.load_state_dict(torch.load(args.teacher_path))
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

student = Model_ocgnn(features.shape[1], args.embedding_dim, "prelu", args.negsamp_ratio, args.readout).to(device)
student_score_head = nn.Linear(args.embedding_dim, 1).to(device)

optimizer_student = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer_head = torch.optim.Adam(student_score_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

with torch.no_grad():
    center_local, center_global = init_center_c(lap, features, teacher, device)
    teacher_global, teacher_local, _ = teacher(lap, features)
    teacher_scores_raw = rho_scores(teacher_global, teacher_local, center_global, center_local)
    teacher_scores = min_max_normalize(teacher_scores_raw)
    teacher_distance_metrics = compute_distance_metrics(teacher_local, ano_label, normal_label_idx)

run_tag = "with_normreg" if args.use_normreg else "without_normreg"
output_file = get_log_file(args, f"{args.dataset}_{run_tag}.txt")
best_file = get_log_file(args, f"{args.dataset}_{run_tag}_best.txt")
best_auc = float("-inf")
best_summary = ""

with open(output_file, "a") as f:
    f.write(
        f"Teacher baseline distances\n"
        f"Teacher real_abnormal_center -> labeled_normal_center: "
        f"{teacher_distance_metrics['real_abnormal_to_labeled_normal_center']:.4f}\n"
        f"Teacher real_abnormal_center -> all_normal_center: "
        f"{teacher_distance_metrics['real_abnormal_to_all_normal_center']:.4f}\n"
    )
    with tqdm(total=args.num_epoch) as pbar:
        pbar.set_description("Training RHO Student")
        for epoch in range(args.num_epoch):
            student.train()
            student_score_head.train()
            optimizer_student.zero_grad()
            optimizer_head.zero_grad()

            _, student_emb = student(features_batched, adj_dense)
            student_emb = student_emb.squeeze(0)
            student_scores_raw = student_score_head(student_emb).squeeze(-1)
            student_scores = min_max_normalize(student_scores_raw)

            mse_loss = F.mse_loss(student_scores, teacher_scores, reduction="mean")

            normal_features = features[normal_label_idx]
            mask = torch.rand_like(normal_features) > 0.3
            masked_features = features.clone()
            masked_features[normal_label_idx] = normal_features * mask
            _, student_emb_aug = student(masked_features.unsqueeze(0), adj_dense)
            student_emb_aug = student_emb_aug.squeeze(0)
            reg2_mse = F.mse_loss(student_emb_aug[normal_label_idx], student_emb[normal_label_idx], reduction="mean")

            normreg_term = args.normreg_weight * reg2_mse if args.use_normreg else torch.zeros_like(reg2_mse)
            total_loss = mse_loss + normreg_term
            total_loss.backward()
            optimizer_student.step()
            optimizer_head.step()

            student.eval()
            student_score_head.eval()
            with torch.no_grad():
                _, student_emb = student(features_batched, adj_dense)
                student_emb = student_emb.squeeze(0)
                student_scores_raw = student_score_head(student_emb).squeeze(-1)
                logits_test = student_scores_raw[idx_test].detach().cpu().numpy()
                labels_test = np.asarray(ano_label[idx_test])
                auc = roc_auc_score(labels_test, logits_test)
                ap = average_precision_score(labels_test, logits_test, average="macro", pos_label=1)
                distance_metrics = compute_distance_metrics(student_emb, ano_label, normal_label_idx)

            message = (
                f"Epoch {epoch}: Total Loss = {total_loss.item():.4f}\n"
                f"MSE Loss = {mse_loss.item():.4f}\n"
                f"Reg2 MSE Loss = {reg2_mse.item():.4f}\n"
                f"Testing {args.dataset} AUC_student: {auc:.4f}\n"
                f"Testing {args.dataset} AP_student: {ap:.4f}\n"
                f"Student real_abnormal_center -> labeled_normal_center: "
                f"{distance_metrics['real_abnormal_to_labeled_normal_center']:.4f}\n"
                f"Student real_abnormal_center -> all_normal_center: "
                f"{distance_metrics['real_abnormal_to_all_normal_center']:.4f}\n"
            )
            print(message)
            f.write(message)
            f.flush()

            if auc > best_auc:
                best_auc = auc
                best_summary = (
                    f"Best epoch: {epoch}\n"
                    f"Best AUC: {auc:.4f}\n"
                    f"Best AP: {ap:.4f}\n"
                    f"NormReg enabled: {bool(args.use_normreg)}\n"
                    f"Teacher path: {args.teacher_path}\n"
                )

            pbar.update(1)

with open(best_file, "w") as f:
    f.write(best_summary)

print(best_summary)
