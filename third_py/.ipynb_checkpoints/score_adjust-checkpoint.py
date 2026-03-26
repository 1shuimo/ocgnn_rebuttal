import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_ocgnn import Model
from utils import *

from sklearn.metrics import roc_auc_score
import random
import os
import dgl
from sklearn.metrics import average_precision_score
import argparse
from tqdm import tqdm
import time

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2]))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set argument parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default='tf_finace')
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)

args = parser.parse_args()

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

if args.num_epoch is None:

    if args.dataset in ['reddit']:
        args.num_epoch = 500
    elif args.dataset in ['tf_finace']:
        args.num_epoch = 1500
    elif args.dataset in ['Amazon']:
        args.num_epoch = 800
    if args.dataset in ['elliptic']:
        args.num_epoch = 500
    elif args.dataset in ['photo']:
        args.num_epoch = 600

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
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define loss function for OCGNN
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

# use fixed c and r
def loss_func_fixed(emb, c_fixed, r_fixed):
    beta = 0.5
    dist = torch.sum(torch.pow(emb - c_fixed, 2), 1)
    score = dist - r_fixed ** 2
    loss = r_fixed ** 2 + 1 / beta * torch.mean(torch.relu(score))
    return loss, score

# Define score distillation loss function
def score_distillation_loss(score_t, score_s):
    loss = F.mse_loss(score_s, score_t)
    return loss

# Load and preprocess data
print('Loading and preprocessing data...')
adj, features, labels, all_idx, idx_train, idx_val, \
idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(args.dataset)

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
adj = normalize_adj(adj)
adj = (adj + sp.eye(adj.shape[0])).todense()
features = torch.FloatTensor(features[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])

# Initialize models
print('Initializing models...')
model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
model_s = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)  # Independent Student model

# Initialize MLP for distillation
print('Initializing MLP for distillation...')
mlp = nn.Linear(args.embedding_dim, 1)

# Initialize optimizers
print('Initializing optimizers...')
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_s = torch.optim.Adam(model_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_mlp = torch.optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Move models to GPU if available
if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    model_s.cuda()
    mlp.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()

b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda() if torch.cuda.is_available() else torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()

# Train the model
print('Starting training loop...')
# Open file in append mode to write (each run appends new content)
output_file = f"{args.dataset}_log_score_adjust.txt"

with open(output_file, "a") as f:
    # Train models
    with tqdm(total=args.num_epoch) as pbar:
        total_time = 0
        pbar.set_description('Training')
        for epoch in range(args.num_epoch):
            start_time = time.time()

            # Set training mode
            model.train()
            model_s.train()
            mlp.train()
            
            # Clear all model gradients
            optimiser.zero_grad()      # Clear teacher model gradients
            optimiser_s.zero_grad()    # Clear student model gradients
            optimiser_mlp.zero_grad()  # Clear MLP gradients


            # Teacher model embeddings (only for normal nodes)
            emb_t_normal = torch.squeeze(model(features, adj))
            emb_t_normal = emb_t_normal[normal_label_idx]

            _, _, c, r = loss_func(emb_t_normal)

            # Teacher model embeddings (for distillation loss calculation - all nodes)
            emb_t_all = torch.squeeze(model(features, adj))

            # Student model embeddings (for distillation loss calculation - all nodes)
            emb_s_all = torch.squeeze(model_s(features, adj))

            # calculate teacher loss and score
            teacher_loss, teacher_score = loss_func_fixed(emb_t_all, c, r)

            # calculate student score
            student_score = mlp(emb_s_all).squeeze(dim=-1)
            # calculate distillation loss
            distillation_loss_value = score_distillation_loss(teacher_score, student_score)

            # Combine losses (teacher loss + distillation loss) for updating teacher and student models
            total_loss =  teacher_loss + distillation_loss_value

            # Backpropagate combined total loss and update teacher and student models
            total_loss.backward()  # Backpropagate combined loss
            optimiser.step()       # Update teacher model parameters
            optimiser_s.step()     # Update student model parameters
            
            # Recalculate distillation loss to update MLP
            # Note: The computation graph has been released, so we recalculate embeddings and distillation loss
            optimiser_mlp.zero_grad()  # Clear MLP gradients

            # Recalculate teacher and student model embeddings (for MLP update)
            emb_t_all = torch.squeeze(model(features, adj))
            emb_s_all = torch.squeeze(model_s(features, adj))

            # calculate teacher loss and score
            _, teacher_score = loss_func_fixed(emb_t_all, c, r)
  
            # calculate student score
            student_score = mlp(emb_s_all).squeeze(dim=-1)
            
            # calculate distillation loss
            distillation_loss_value = score_distillation_loss(teacher_score, student_score)

            # Backpropagate distillation loss and update MLP
            distillation_loss_value.backward()  # Backpropagate distillation loss to update MLP
            optimiser_mlp.step()  # Update MLP parameters



            if epoch % 5 == 0:
                # Print and save log
                log_message = (
                    f"Epoch {epoch}: Total Loss = {total_loss.item()}, Distillation Loss = {distillation_loss_value.item()}\n"
                )            
                # Evaluate Student model
                model_s.eval()
                emb = torch.squeeze(model_s(features, adj))
                student_score = mlp(emb).squeeze(dim=-1)  # Use MLP to compute scores for Student model
                # Calculate the ocgnn score
                _, ocgnn_score, _, _ = loss_func(emb)
                total_score = ocgnn_score + student_score
    
                logits = np.squeeze(student_score[idx_test].cpu().detach().numpy())
                logits_total = np.squeeze(total_score[idx_test].cpu().detach().numpy())
                
                auc = roc_auc_score(ano_label[idx_test], logits)
                auc_total = roc_auc_score(ano_label[idx_test], logits_total)

                log_message += f'Testing {args.dataset} AUC: {auc:.4f}\n'
                log_message += f'Testing {args.dataset} OCGNN AUC: {auc_total:.4f}\n'

                AP = average_precision_score(ano_label[idx_test], logits, average='macro', pos_label=1)
                AP_total = average_precision_score(ano_label[idx_test], logits_total, average='macro', pos_label=1)

                log_message += f'Testing AP: {AP}\n'
                log_message += f'Testing OCGNN AP: {AP_total}\n'
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