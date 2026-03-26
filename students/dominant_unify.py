import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_dominant import Model_dominant
from models.model_ocgnn import Model_ocgnn
from utils.utils import *

from sklearn.metrics import roc_auc_score
import random
import os
import dgl
from sklearn.metrics import average_precision_score
import argparse
from tqdm import tqdm
import time



os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print(torch.cuda.is_available())  # True 表示可用，False 表示不可用
print(torch.cuda.device_count())  # 输出可用 GPU 的数量



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

if args.num_epoch is None:
    if args.dataset in ['reddit']:
        args.num_epoch = 500
    elif args.dataset in ['tf_finace']:
        args.num_epoch = 1500
    elif args.dataset in ['Amazon']:
        args.num_epoch = 800
    elif args.dataset in ['elliptic']:
        args.num_epoch = 500
    elif args.dataset in ['photo']:
        args.num_epoch = 600
        
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


# Define distillation loss function
def distillation_loss(emb_t, emb_s, mlp_t, mlp_s):
    teacher_logits = mlp_t(emb_t)
    student_logits = mlp_s(emb_s)
    # Calculate mean squared error between teacher and student outputs
    loss = F.mse_loss(student_logits, teacher_logits, reduction='mean')
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


# Define distillation loss function
def distillation_loss_emb(emb_t, emb_s):
    # Calculate mean squared error between teacher and student embeddings
    loss = F.mse_loss(emb_s, emb_t, reduction='mean')
    return loss



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
model = Model_dominant(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
model_s = Model_ocgnn(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)  # Independent Student model

# Initialize MLP for distillation
print('Initializing MLP for distillation...')
mlp_s = nn.Linear(args.embedding_dim, 1)

# Initialize optimizers
print('Initializing optimizers...')
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_s = torch.optim.Adam(model_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_mlp_s = torch.optim.Adam(mlp_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)


b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda() if torch.cuda.is_available() else torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()






# Train the model
print('Starting training loop...')
# Open file in append mode to write (each run appends new content)
output_file = f"{args.dataset}_dominant_unify_edit_gcn.txt"
mlp_reduce = nn.Linear(ft_size, args.embedding_dim)
with open(output_file, "a") as f:
    with tqdm(total=args.num_epoch) as pbar:
        total_time = 0
        pbar.set_description('Training')
        for epoch in range(args.num_epoch):
            start_time = time.time()

            # Set training mode
            model.train()
            model_s.train()
            mlp_s.train()

            # Clear all model gradients
            optimiser.zero_grad()      # Clear teacher model gradients
            optimiser_s.zero_grad()    # Clear student model gradients
            optimiser_mlp_s.zero_grad()  # Clear MLP gradients


            dominant_loss, _, _, _ = model(features, adj, normal_label_idx, idx_test)

            _, score_from_dominant, _, emb_t_all = model(features, adj, all_idx, idx_test)
            
            # Student model embeddings (for distillation loss calculation - all nodes)
            emb_s_all = torch.squeeze(model_s(features, adj))


            student_score = mlp_s(emb_s_all).squeeze(dim=-1)

            # calculate kd loss(distance and mlp_student)
            dominant_distance_mlp_s= score_distillation_loss(score_from_dominant, student_score)
            # Calculate distillation loss (using all node embeddings)
            distillation_loss_emb_value = distillation_loss_emb(emb_t_all, emb_s_all)


            distillation_loss_value = dominant_distance_mlp_s
            
                        # Calculate the number of nodes
            num_nodes = emb_t_all.size(0)

            # Calculate the mean of distillation losses
            mean_distillation_loss_value = distillation_loss_value / num_nodes
            mean_distillation_loss_emb_value = distillation_loss_emb_value / num_nodes

            mean_distillation = mean_distillation_loss_value + mean_distillation_loss_emb_value

            # Combine losses (teacher loss + distillation loss) for updating teacher and student models
            total_loss = dominant_loss + mean_distillation_loss_value + mean_distillation_loss_emb_value
            # Backpropagate combined total loss and update teacher and student models
            total_loss.backward()  # Backpropagate combined loss
            optimiser.step()       # Update teacher model parameters
            optimiser_s.step()     # Update student model parameters
            optimiser_mlp_s.step()


            # Evaluate student model every 5 epochs
            if epoch % 5 == 0:
                # Print and save log
                log_message = (
                    f"Epoch {epoch}: Total Loss = {total_loss.item()}, Distillation Loss = {mean_distillation.item()}\n"
                )
                model_s.eval()
                model.eval()

                _, score_from_dominant, _, emb_t_all = model(features, adj, all_idx, idx_test)

                
                score_from_dominant= min_max_normalize(score_from_dominant)
                
                emb_s_all = torch.squeeze(model_s(features, adj))

                
                student_score = mlp_s(emb_s_all).squeeze(dim=-1) 

                differences = emb_s_all - emb_t_all
                emb_score =  torch.sum(torch.norm(differences, dim=1))
                emb_score = min_max_normalize(emb_score)

                stu_add_dominant_score = student_score + score_from_dominant
                
                stu_dominant_emb_score = student_score + score_from_dominant + emb_score
                # Add 0.5 * dominant score cases
                stu_add_05dominant_score = student_score + 0.5 * score_from_dominant
                # Add 0.5 * dominant score and 0.5 * emb score cases
                stu_add_05dominant_05emb_score = student_score + 0.5 * score_from_dominant + 0.5 * emb_score


                logits_stu = np.squeeze(student_score[idx_test].cpu().detach().numpy())
                logits_stu_dominant = np.squeeze(stu_add_dominant_score[idx_test].cpu().detach().numpy())
                logits_stu_dominant_emb = np.squeeze(stu_dominant_emb_score[idx_test].cpu().detach().numpy())
                logits_stu_05dominant = np.squeeze(stu_add_05dominant_score[idx_test].cpu().detach().numpy())
                logits_stu_05dominant_05emb = np.squeeze(stu_add_05dominant_05emb_score[idx_test].cpu().detach().numpy())


                # Calculate AUC for different logits
                auc_stu = roc_auc_score(ano_label[idx_test], logits_stu)
                auc_stu_dominant = roc_auc_score(ano_label[idx_test], logits_stu_dominant)
                auc_stu_dominant_emb = roc_auc_score(ano_label[idx_test], logits_stu_dominant_emb)

                # Calculate AUC for 0.5 * dominant cases
                auc_stu_05dominant = roc_auc_score(ano_label[idx_test], logits_stu_05dominant)
                auc_stu_05dominant_05emb = roc_auc_score(ano_label[idx_test], logits_stu_05dominant_05emb)


                log_message += f'Testing {args.dataset} AUC_student_mlp_s: {auc_stu:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_dominant: {auc_stu_dominant:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_dominant_emb: {auc_stu_dominant_emb:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_05dominant: {auc_stu_05dominant:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_05dominant_05emb: {auc_stu_05dominant_05emb:.4f}\n'



                # Calculate AP for different logits
                AP_stu = average_precision_score(ano_label[idx_test], logits_stu, average='macro', pos_label=1)
                AP_stu_dominant = average_precision_score(ano_label[idx_test], logits_stu_dominant, average='macro', pos_label=1)
                AP_stu_dominant_emb = average_precision_score(ano_label[idx_test], logits_stu_dominant_emb, average='macro', pos_label=1)
                # Calculate AP for 0.5 * dominant cases
                AP_stu_05dominant = average_precision_score(ano_label[idx_test], logits_stu_05dominant, average='macro', pos_label=1)
                AP_stu_05dominant_05emb = average_precision_score(ano_label[idx_test], logits_stu_05dominant_05emb, average='macro', pos_label=1)

                log_message += f'Testing AP_student_mlp_s: {AP_stu:.4f}\n'
                log_message += f'Testing AP_student_dominant: {AP_stu_dominant:.4f}\n'
                log_message += f'Testing AP_student_dominant_emb: {AP_stu_dominant_emb:.4f}\n'
                log_message += f'Testing AP_student_05dominant: {AP_stu_05dominant:.4f}\n'
                log_message += f'Testing AP_student_05dominant_05emb: {AP_stu_05dominant_05emb:.4f}\n'

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


                
