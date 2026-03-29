import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import random
from collections import Counter

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

from models.model_rho import RHO
from utils.log_paths import add_log_subdir_argument, get_log_file
from utils.utils_old import load_mat, preprocess_features


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()


def get_split(num_node, label, train_rate=0.3, val_rate=0.1):
    all_labels = np.squeeze(np.array(label))
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[:num_train]
    idx_val = all_idx[num_train : num_train + num_val]
    idx_test = all_idx[num_train + num_val :]
    print("Training", Counter(np.squeeze(all_labels[idx_train])))
    print("Test", Counter(np.squeeze(all_labels[idx_test])))
    all_normal_label_idx = [i for i in idx_train if all_labels[i] == 0]
    rate = 0.5
    normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * rate)]
    print("Training rate", rate * train_rate)
    return normal_label_idx, idx_val, idx_test


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def build_lap(adj):
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


def official_rho_dataset_name(dataset):
    mapping = {
        "Amazon": "amazon",
        "tf_finace": "tfinance",
        "reddit": "reddit",
        "photo": "photo",
        "elliptic": "elliptic",
        "tolokers": "tolokers",
        "questions": "questions",
    }
    return mapping.get(dataset)


def load_lap(dataset, rho_root):
    official_name = official_rho_dataset_name(dataset)
    if rho_root and official_name is not None:
        lap_path = Path(rho_root) / "datasets" / f"Lap_matrix_{official_name}.npz"
        if lap_path.exists():
            return sparse_mx_to_torch_sparse_tensor(sp.load_npz(lap_path)), str(lap_path)
    return None, None


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main_worker(args):
    fix_seed(args.seed)
    device = torch.device("cuda:" + str(args.cuda) if torch.cuda.is_available() else "cpu")
    time_run = []

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

    def train(model, optimizer, adj, idx_train, center_local, center_global):
        model.train()
        optimizer.zero_grad()
        outputs_global, outputs_local, nce_loss = model(adj, features)
        dist_global = torch.sum((outputs_global[idx_train] - center_global) ** 2, dim=1)
        dist_local = torch.sum((outputs_local[idx_train] - center_local) ** 2, dim=1)
        dist = 0.5 * dist_global + 0.5 * dist_local
        loss = torch.mean(dist) + args.alpha * nce_loss
        loss.backward()
        optimizer.step()
        return loss

    def test(model, adj, labels, idx, center_local, center_global):
        with torch.no_grad():
            model.eval()
            outputs_global, outputs_local, _ = model(adj, features)
            scores = (
                torch.sum((outputs_global[idx] - center_global) ** 2, dim=1)
                + torch.sum((outputs_local[idx] - center_local) ** 2, dim=1)
            ) / 2
            labels_np = np.array(labels.cpu().data.numpy())
            scores_np = np.array(scores.cpu().data.numpy())
            precision, recall, _ = precision_recall_curve(labels_np[idx], scores_np)
            auprc = auc(recall, precision)
            auroc = roc_auc_score(labels_np[idx], scores_np)
            return auprc, auroc

    adj, feat, _, all_idx, idx_train_loaded, idx_val_loaded, idx_test_loaded, ano_label, _, _, _, _ = load_mat(
        args.dataset, train_rate=args.train_ratio, val_rate=args.val_ratio
    )

    if args.dataset in ["Amazon", "tf_finace", "reddit", "elliptic", "questions"]:
        feat, _ = preprocess_features(feat)
    else:
        feat = feat.todense()

    labels = torch.as_tensor(ano_label, dtype=torch.long)
    features = torch.FloatTensor(np.asarray(feat))
    in_feats = features.shape[1]
    num_node = features.shape[0]
    idx_train, idx_val, idx_test = get_split(num_node, labels, args.train_ratio)

    Lap, lap_path = load_lap(args.dataset, args.rho_root)
    if Lap is None:
        Lap = build_lap(adj)

    net = RHO(in_feats, args.hidden1, args.hidden2, args.nlayers, args.batch_size, args.tau).to(device)
    Lap = Lap.to(device)
    features = features.to(device)
    labels = labels.to(device)
    net.apply(init_params)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_file = get_log_file(args, f"{args.dataset}_rho_official_minimal.txt")
    if args.save_path is None:
        save_dir = ROOT.parent / "rho_teacher_best_pth_official_minimal"
        save_dir.mkdir(parents=True, exist_ok=True)
        args.save_path = str(save_dir / f"{args.dataset}_rho_teacher_official_minimal_last.pth")
    if args.best_save_path is None:
        save_dir = Path(args.save_path).resolve().parent
        save_dir.mkdir(parents=True, exist_ok=True)
        args.best_save_path = str(save_dir / f"{args.dataset}_rho_teacher_official_minimal_best.pth")

    best_auroc = -1.0
    best_auprc = -1.0
    best_epoch = -1
    with open(output_file, "a") as f:
        if lap_path is not None:
            f.write(f"Using official RHO Lap: {lap_path}\n")
        else:
            f.write("Using locally constructed Lap from current adjacency.\n")

        for epoch in range(args.epochs):
            c_local, c_global = init_center_c(Lap, features, net, device)
            losses_train = train(net, optimizer, Lap, idx_train, c_local, c_global)
            auprc_test, auroc_test = test(net, Lap, labels, idx_test, c_local, c_global)
            line = (
                f"{epoch:>4} epochs trained. "
                f"Current auprc: {round(auprc_test, 4):>7} "
                f"Current auroc:  {round(auroc_test, 4):>7} "
                f"Total loss: {round(float(losses_train), 4):>7}"
            )
            print(line)
            f.write(line + "\n")
            f.flush()

            if auroc_test > best_auroc:
                best_auroc = auroc_test
                best_auprc = auprc_test
                best_epoch = epoch
                torch.save(net.state_dict(), args.best_save_path)

        torch.save(net.state_dict(), args.save_path)

    print(f"AUROC:{best_auroc}")
    print(f"AUPRC:{best_auprc}")
    print(f"Best epoch:{best_epoch}")
    print(f"Best save path:{args.best_save_path}")
    print(f"Last save path:{args.save_path}")


if __name__ == "__main__":
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument(
        "--dataset",
        type=str,
        default="reddit",
        choices=["Amazon", "tf_finace", "reddit", "photo", "elliptic", "tolokers", "questions"],
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--train_ratio", type=float, default=0.3, help="Training ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Val ratio")
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--early_stopping", type=int, default=200)
    parser.add_argument("--hidden1", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=64)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--rho_root", type=str, default=str(ROOT / "rho_official"))
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--best_save_path", type=str, default=None)
    add_log_subdir_argument(parser, "tea_train_rho_official_minimal")
    args = parser.parse_args()

    if args.lr is None:
        args.lr = get_default_lr(args.dataset)
    if args.epochs is None:
        args.epochs = get_default_epochs(args.dataset)
    if args.batch_size is None:
        args.batch_size = get_default_batch_size(args.dataset)

    main_worker(args)
