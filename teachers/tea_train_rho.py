import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from models.model_rho import RHO
from utils.log_paths import add_log_subdir_argument, get_log_file
from utils.utils_old import load_mat, preprocess_features


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_lap(adj):
    adj = sp.csr_matrix(adj)
    adj = adj.maximum(adj.T)
    adj = adj + sp.eye(adj.shape[0])
    degrees = np.array(adj.sum(axis=1)).flatten()
    with np.errstate(divide="ignore"):
        deg_inv_sqrt = np.power(degrees, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    d_inv_sqrt = sp.diags(deg_inv_sqrt)
    lap = sp.eye(adj.shape[0], dtype=np.float32) - d_inv_sqrt @ adj @ d_inv_sqrt
    return sparse_mx_to_torch_sparse_tensor(lap)


def init_center_c(adj, inputs, net, device, eps=0.1):
    c_global = torch.zeros(net.rep_dim).to(device)
    c_local = torch.zeros(net.rep_dim).to(device)
    net.eval()
    with torch.no_grad():
        outputs_global, outputs_local, _ = net(adj, inputs)
        n_samples = outputs_global.shape[0]
        c_global = torch.sum(outputs_global, dim=0)
        c_local = torch.sum(outputs_local, dim=0)

    c_global /= n_samples
    c_local /= n_samples

    c_local[(abs(c_local) < eps) & (c_local < 0)] = -eps
    c_local[(abs(c_local) < eps) & (c_local > 0)] = eps
    c_global[(abs(c_global) < eps) & (c_global < 0)] = -eps
    c_global[(abs(c_global) < eps) & (c_global > 0)] = eps

    return c_local, c_global


def get_default_lr(dataset):
    if dataset in ["Amazon", "tf_finace", "reddit", "photo", "tolokers"]:
        return 5e-3
    if dataset in ["elliptic", "questions"]:
        return 5e-4
    return 5e-4


def get_default_epochs(dataset):
    if dataset in ["Amazon", "tolokers"]:
        return 500
    return 100


def get_default_batch_size(dataset):
    if dataset in ["tf_finace", "elliptic", "questions"]:
        return 1024
    return 0


def main_worker(args):
    fix_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(model, optimizer, lap, inputs, idx_train, center_local, center_global):
        model.train()
        optimizer.zero_grad()
        outputs_global, outputs_local, nce_loss = model(lap, inputs)
        dist_global = torch.sum((outputs_global[idx_train] - center_global) ** 2, dim=1)
        dist_local = torch.sum((outputs_local[idx_train] - center_local) ** 2, dim=1)
        dist = 0.5 * dist_global + 0.5 * dist_local
        loss = torch.mean(dist) + args.alpha * nce_loss
        loss.backward()
        optimizer.step()
        return loss

    def test(model, lap, inputs, labels, idx, center_local, center_global):
        with torch.no_grad():
            model.eval()
            outputs_global, outputs_local, _ = model(lap, inputs)
            scores = (
                torch.sum((outputs_global[idx] - center_global) ** 2, dim=1)
                + torch.sum((outputs_local[idx] - center_local) ** 2, dim=1)
            ) / 2
            scores = np.array(scores.cpu().data.numpy())
            labels = np.array(labels.cpu().data.numpy())
            auroc = roc_auc_score(labels[idx], scores)
            auprc = average_precision_score(labels[idx], scores, average="macro", pos_label=1)
            return auprc, auroc

    adj, features, labels, all_idx, idx_train_all, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(
        args.dataset
    )

    if args.dataset in ["Amazon", "tf_finace", "reddit", "elliptic", "questions"]:
        features, _ = preprocess_features(features)
    else:
        features = features.todense()

    Lap = get_lap(adj)
    features = torch.FloatTensor(np.asarray(features))
    labels = torch.as_tensor(ano_label, dtype=torch.long)

    in_feats = features.shape[1]
    idx_train = normal_label_idx

    net = RHO(in_feats, args.hidden1, args.hidden2, args.nlayers, args.batch_size, args.tau).to(device)
    Lap = Lap.to(device)
    features = features.to(device)
    labels = labels.to(device)
    net.apply(init_params)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_file = get_log_file(args, f"{args.dataset}_rho_teacher.txt")
    if args.weight_save_path is None:
        weight_dir = ROOT.parent / "rebuttal_ckpt"
        weight_dir.mkdir(parents=True, exist_ok=True)
        args.weight_save_path = str(weight_dir / f"{args.dataset}_rho_teacher_best.pth")

    best_auroc = 0.0
    best_summary = ""
    with open(output_file, "a") as f:
        with tqdm(total=args.epochs) as pbar:
            pbar.set_description("Training RHO Teacher")
            for epoch in range(args.epochs):
                c_local, c_global = init_center_c(Lap, features, net, device)
                losses_train = train(net, optimizer, Lap, features, idx_train, c_local, c_global)
                auprc_test, auroc_test = test(net, Lap, features, labels, idx_test, c_local, c_global)
                message = (
                    f"{epoch:>4} epochs trained. "
                    f"Current auprc: {round(auprc_test, 4):>7} "
                    f"Current auroc:  {round(auroc_test, 4):>7} "
                    f"Total loss: {round(float(losses_train), 4):>7}\n"
                )
                print(message, end="")
                f.write(message)
                f.flush()
                if auroc_test > best_auroc:
                    best_auroc = auroc_test
                    torch.save(net.state_dict(), args.weight_save_path)
                    best_summary = (
                        f"Best epoch: {epoch}\n"
                        f"Best AUROC: {auroc_test:.4f}\n"
                        f"Best AUPRC: {auprc_test:.4f}\n"
                        f"Weight path: {args.weight_save_path}\n"
                    )
                pbar.update(1)

    best_file = get_log_file(args, f"{args.dataset}_rho_teacher_best.txt")
    with open(best_file, "w") as f:
        f.write(best_summary)
    print(best_summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="Amazon")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--hidden1", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=64)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--weight_save_path", type=str, default=None)
    add_log_subdir_argument(parser, "tea_train_rho")
    args = parser.parse_args()

    if args.lr is None:
        args.lr = get_default_lr(args.dataset)
    if args.epochs is None:
        args.epochs = get_default_epochs(args.dataset)
    if args.batch_size is None:
        args.batch_size = get_default_batch_size(args.dataset)

    main_worker(args)
