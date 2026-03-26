import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# train_student.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import random
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from models.model_ocgnn import Model_ocgnn
from utils.utils import *
import os

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2]))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='reddit')
# 'reddit_ocgnn_teacher_final.pth'
parser.add_argument('--teacher_path', type=str)
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()


# Set learning rate and number of epochs based on dataset
if args.lr is None:
    if args.dataset in ['Amazon']:
        args.lr = 1e-3
    elif args.dataset in ['tf_finace']:
        args.lr = 5e-4
    elif args.dataset in ['reddit']:
        args.lr = 1e-3
    elif args.dataset in ['elliptic']:
        args.lr = 1e-3
    elif args.dataset in ['photo']:
        args.lr = 1e-3
    elif args.dataset in ['tolokers']:
        args.lr = 1e-3
    elif args.dataset in ['YelpChi-all']:
        args.lr = 1e-3

if args.num_epoch is None:
    if args.dataset in ['reddit']:
        args.num_epoch = 2000
    elif args.dataset in ['tf_finace']:
        args.num_epoch = 3000
    elif args.dataset in ['Amazon']:
        args.num_epoch = 2300
    elif args.dataset in ['elliptic']:
        args.num_epoch = 2000
    elif args.dataset in ['photo']:
        args.num_epoch = 2100
    elif args.dataset in ['tolokers']:
        args.num_epoch = 2000
    elif args.dataset in ['YelpChi-all']:
        args.num_epoch = 2000
        
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
print('Setting random seeds...')
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def loss_func(emb):
    r = 0
    beta = 0.5
    warmup = 2
    eps = 0.001
    c = torch.zeros(args.embedding_dim)
    dist = torch.sum(torch.pow(emb.cpu() - c, 2), 1)
    score = dist - r ** 2
    loss = r ** 2 + 1 / beta * torch.mean(torch.relu(score))

    if warmup > 0:
        with torch.no_grad():
            warmup -= 1
            r = torch.quantile(torch.sqrt(dist), 1 - beta)
            c = torch.mean(emb, 0)
            c[(abs(c) < eps) & (c < 0)] = -eps
            c[(abs(c) < eps) & (c > 0)] = eps

    return loss, score, c, r

# Define distillation loss function
def distillation_loss_emb(emb_t, emb_s):
    # Calculate mean squared error between teacher and student embeddings
    loss = F.mse_loss(emb_s, emb_t, reduction='mean')
    return loss


# Define score distillation loss function
def score_distillation_loss(score_t, score_s):
    loss = F.mse_loss(score_s, score_t)
    return loss


def min_max_normalize(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    # 检查 max_val 是否等于 min_val
    if max_val == min_val:
        # 返回全零张量，或者返回原始张量
        # 返回全零张量：
        return torch.zeros_like(tensor)
        # 返回原始张量：
        # return tensor
    else:
        return (tensor - min_val) / (max_val - min_val)
    
def z_score_normalize(tensor):
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    
    # 检查 std 是否为零
    if std == 0:
        return torch.zeros_like(tensor)  # 返回全零张量
    else:
        return (tensor - mean) / std


# Load and preprocess data
print('Loading and preprocessing data...')
adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(args.dataset)

if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
    print('Preprocessing features...')
    features, _ = preprocess_features(features)
else:
    features = features.todense()

    
# Convert adjacency matrix to DGL graph
print('Converting adjacency matrix to DGL graph...')
dgl_graph = adj_to_dgl_graph(adj)

# Prepare input tensors
print('Preparing input tensors...')
nb_nodes = features.shape[0]
ft_size = features.shape[1]
raw_adj = adj
adj = normalize_adj(adj)

raw_adj = (raw_adj + sp.eye(raw_adj.shape[0])).todense()
adj = (adj + sp.eye(adj.shape[0])).todense()
features = torch.FloatTensor(features[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
raw_adj = torch.FloatTensor(raw_adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])

# Initialize models
print('Initializing models...')
model = Model_ocgnn(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
model.load_state_dict(torch.load(args.teacher_path))
model.eval()
# 冻结参数
for p in model.parameters():
    p.requires_grad = False

model_s = Model_ocgnn(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)  


# Initialize optimizers
print('Initializing optimizers...')
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_s = torch.optim.Adam(model_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)


b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda() if torch.cuda.is_available() else torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()


print("\n🔁 Starting Student Training...")
output_file = f"./ocgnn_2_step_norm_emb/{args.dataset}_ocgnn_2_step_add_1500_epoch.txt"
with open(output_file, "a") as f:
    with tqdm(total=args.num_epoch) as pbar:
        total_time = 0
        pbar.set_description('Training')
        for epoch in range(args.num_epoch):
            start_time = time.time()
            model_s.train()
            optimiser_s.zero_grad()

            with torch.no_grad():
                emb_tea_1_raw, emb_tea_2_raw = model(features, adj)
                emb_tea_1 = emb_tea_1_raw.squeeze(0)
                emb_tea_2 = emb_tea_2_raw.squeeze(0)
                emb_tea_1_norm = z_score_normalize(emb_tea_1)
                emb_tea_2_norm = z_score_normalize(emb_tea_2)

            # auc of teacher model
                loss, score, _, _ = loss_func(emb_tea_2)
                # evaluation on the valid and test node
                logits = np.squeeze(score[idx_test].cpu().detach().numpy())
                auc = roc_auc_score(ano_label[idx_test], logits)
                log_message = (f"Testing Teacher{args.dataset} AUC: {auc}\n")

            emb_stu_1_raw, emb_stu_2_raw = model_s(features, adj)
            emb_stu_1 = emb_stu_1_raw.squeeze(0)
            emb_stu_2 = emb_stu_2_raw.squeeze(0)
            emb_stu_1_norm = z_score_normalize(emb_stu_1)
            emb_stu_2_norm = z_score_normalize(emb_stu_2)


            
            mse_loss_1 = distillation_loss_emb(emb_tea_1_norm, emb_stu_1_norm)
            mse_loss_2 = distillation_loss_emb(emb_tea_2_norm, emb_stu_2_norm)

            total_loss = (mse_loss_1 + mse_loss_2) 

            total_loss.backward()
            optimiser_s.step()

            if epoch % 5 == 0:
                # Print and save log
                log_message += f"Epoch {epoch}: Total Loss = {total_loss.item()}\n"
                model_s.eval()
                model.eval()


                mse_loss_1 = F.mse_loss(emb_tea_1_norm, emb_stu_1_norm,reduction='none')
                mse_loss_2 = F.mse_loss(emb_tea_2_norm, emb_stu_2_norm,reduction='none')
                

                student_score = (mse_loss_1 + mse_loss_2).mean(dim=1)

                log_message += f'student_score: {torch.mean(student_score)}\n'
                
                logits_stu = np.squeeze(student_score[idx_test].cpu().detach().numpy())


    
                # Calculate AUC for different logits
                auc_stu = roc_auc_score(ano_label[idx_test], logits_stu)
                log_message += f'Testing {args.dataset} AUC_emb_s: {auc_stu:.4f}\n'

                # Calculate AP for different logits
                AP_stu = average_precision_score(ano_label[idx_test], logits_stu, average='macro', pos_label=1)
  
                log_message += f'Testing AP_emb_s: {AP_stu:.4f}\n'

                log_message += f'Total time is: {total_time}\n'
                
                # Print log
                print(log_message)
                # Write log to file
                f.write(log_message)
                f.flush()  # Ensure content is immediately written to file

            end_time = time.time()
            total_time += end_time - start_time
            pbar.update(1)

    print("Training completed.")


torch.save(model_s.state_dict(), f'{args.dataset}_student_final.pth')
print("\n✅ Student model saved.")
