import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch.nn as nn

from models.model import Model_ggad
from utils.utils_old import *

from sklearn.metrics import roc_auc_score
import random
import dgl
from sklearn.metrics import average_precision_score
import argparse
from tqdm import tqdm
import time
import scipy.io as scio

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [3]))
# os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
# Set argument
parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataset', type=str,
                    default='questions')  # 'BlogCatalog'  'Flickr'  'ACM'  'cora'  'citeseer'  'pubmed', 'Amazon
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--mean', type=float, default=0)
parser.add_argument('--var', type=float, default=0)

args = parser.parse_args()

if args.lr is None:
    if args.dataset in ['Amazon']:
        args.lr = 1e-3
    elif args.dataset in ['tf_finace']:
        args.lr = 1e-3
    elif args.dataset in ['reddit']:
        args.lr = 5e-3
    elif args.dataset in ['photo']:
        args.lr = 1e-3
    elif args.dataset in ['elliptic']:
        args.lr = 1e-3
    elif args.dataset in ['tolokers']:
        args.lr = 5e-3
    elif args.dataset in ['YelpChi-all']:
        args.lr = 1e-3

if args.num_epoch is None:
    if args.dataset in ['photo']:
        args.num_epoch = 2500
    if args.dataset in ['elliptic']:
        args.num_epoch = 150
    if args.dataset in ['reddit']:
        args.num_epoch = 2000
    elif args.dataset in ['tf_finace']:
        args.num_epoch = 500
    elif args.dataset in ['Amazon']:
        args.num_epoch = 800
    elif args.dataset in ['tolokers']:
        args.num_epoch = 1500
    elif args.dataset in ['YelpChi-all']:
        args.num_epoch = 1500
        
        
if args.dataset in ['reddit', 'Photo']:
    args.mean = 0.02
    args.var = 0.01
else:
    args.mean = 0.0
    args.var = 0.0

batch_size = args.batch_size
subgraph_size = args.subgraph_size

print('Dataset: ', args.dataset)

# Set random seed
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
# os.environ['PYTHONHASHSEED'] = str(args.seed)
# os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load and preprocess data
adj, features, labels, all_idx, idx_train, idx_val, \
idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(args.dataset)

if args.dataset in ['Amazon', 'YelpChi_no_isolate ', 'tf_finace', 'reddit',
                    'tolokers_no_isolated', 'questions', 'elliptic']:
    features, _ = preprocess_features(features)
else:
    features = features.todense()
# dgl_graph = adj_to_dgl_graph(adj)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
raw_adj = adj
print(adj.sum())
adj = normalize_adj(adj)

if args.dataset in ['questions_no_isolated', 'tolokers_no_isolated']:
    adj = adj.todense()
    raw_adj = raw_adj.todense()
else:
    raw_adj = (raw_adj + sp.eye(raw_adj.shape[0])).todense()
    adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
# adj = torch.FloatTensor(adj[np.newaxis])
features = torch.FloatTensor(features)
adj = torch.FloatTensor(adj)
# adj = adj.to_sparse_csr()
adj = torch.FloatTensor(adj[np.newaxis])
raw_adj = torch.FloatTensor(raw_adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])

# idx_train = torch.LongTensor(idx_train)
# idx_val = torch.LongTensor(idx_val)
# idx_test = torch.LongTensor(idx_test)

# Initialize model and optimiser
model = Model_ggad(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#
# if torch.cuda.is_available():
#     print('Using CUDA')
#     model.cuda()
#     features = features.cuda()
#     adj = adj.cuda()
#     labels = labels.cuda()
#     raw_adj = raw_adj.cuda()

# idx_train = idx_train.cuda()
# idx_val = idx_val.cuda()
# idx_test = idx_test.cuda()
#
# if torch.cuda.is_available():
#     b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda())
# else:
#     b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))

b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0
batch_num = nb_nodes // batch_size + 1

# Setup for training and logging
output_file = f"./with_noise_pth/{args.dataset}_training_log_0105_new_noise.txt"
weight_save_path = f"./with_noise_pth/{args.dataset}_best_model_0105_new_noise.pth"  # 保存最佳权重的路径

best_auc = 0.0  # 用于记录最佳 AUC
best_epoch = -1  # 用于记录最佳 AUC 对应的 epoch

print('Starting training loop...')
with open(output_file, "a") as f:
    with tqdm(total=args.num_epoch) as pbar:
        pbar.set_description('Training')
        for epoch in range(args.num_epoch):
            model.train()
            optimiser.zero_grad()

            # Train model step
            train_flag = True
            emb, emb_combine, logits, emb_con, emb_abnormal, emb_orign = model(features, adj, abnormal_label_idx,
                                                                               normal_label_idx, train_flag, args)

            # BCE loss
            lbl = torch.unsqueeze(torch.cat(
                (torch.zeros(len(normal_label_idx)), torch.ones(len(emb_con)))),
                1).unsqueeze(0)
            loss_bce = b_xent(logits, lbl)
            loss_bce = torch.mean(loss_bce)

            # Local affinity margin loss
            emb = torch.squeeze(emb)
            emb_inf = torch.norm(emb, dim=-1, keepdim=True)
            emb_inf = torch.pow(emb_inf, -1)
            emb_inf[torch.isinf(emb_inf)] = 0.
            emb_norm = emb * emb_inf
            sim_matrix = torch.mm(emb_norm, emb_norm.T)
            raw_adj_squeezed = torch.squeeze(raw_adj)
            similar_matrix = sim_matrix * raw_adj_squeezed
            r_inv = torch.pow(torch.sum(raw_adj_squeezed, 0), -1)
            r_inv[torch.isinf(r_inv)] = 0.
            affinity = torch.sum(similar_matrix, 0) * r_inv
            affinity_normal_mean = torch.mean(affinity[normal_label_idx])
            affinity_abnormal_mean = torch.mean(affinity[abnormal_label_idx])
            confidence_margin = 0.7
            loss_margin = (confidence_margin - (affinity_normal_mean - affinity_abnormal_mean)).clamp_min(min=0)

            # Reconstruction loss
            diff_attribute = torch.pow(emb_con - emb_abnormal, 2)
            loss_rec = torch.mean(torch.sqrt(torch.sum(diff_attribute, 1)))

            # Total loss
            loss = 1 * loss_margin + 1 * loss_bce + 1 * loss_rec
            loss.backward()
            optimiser.step()
            pbar.update(1)

            # --- Evaluation and Logging ---
            log_message = ""
            model.eval()
            train_flag = False
            emb, emb_combine, logits, emb_con, emb_abnormal, emb_orign = model(features, adj, abnormal_label_idx,
                                                                               normal_label_idx, train_flag, args)
            logits_test = np.squeeze(logits[:, idx_test, :].cpu().detach().numpy())
            auc = roc_auc_score(ano_label[idx_test], logits_test)
            AP = average_precision_score(ano_label[idx_test], logits_test, average='macro', pos_label=1)

            log_message += f"Epoch {epoch}: Loss = {loss.item():.4f}\n"
            log_message += f"Train Loss Margin = {loss_margin.item():.4f}, BCE Loss = {loss_bce.item():.4f}, Reconstruction Loss = {loss_rec.item():.4f}\n"
            log_message += f"Testing {args.dataset} AUC: {auc:.4f}, AP: {AP:.4f}\n"

            # Save the best model
            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                torch.save(model.state_dict(), weight_save_path)
                log_message += f"✅ New best model saved at epoch {epoch} with AUC {auc:.4f}.\n"

            print(log_message)
            # Write log to file
            f.write(log_message)
            f.flush()

print("Training completed.")
with open(output_file, "a") as f:
    if best_epoch != -1:
        final_message = f"✅ Best model was saved at epoch {best_epoch} with AUC {best_auc:.4f}.\n"
        final_message += f"Weights saved to: {weight_save_path}\n"
    else:
        final_message = "No model achieved better performance during training.\n"
    f.write(final_message)
    print(final_message)

