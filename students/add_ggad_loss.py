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

from models.model_ocgnn import Model_ocgnn
from models.model import Model_ggad
from utils.utils import *

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


# Max-Min normalize: (x - min) / (max - min)
def min_max_normalize(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val)



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
model = Model_ggad(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
model_s = Model_ocgnn(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)  # Independent Student model

# Initialize MLP for distillation
print('Initializing MLP for distillation...')
mlp_t = nn.Linear(args.embedding_dim, 1)  # Output a single score
mlp_s = nn.Linear(args.embedding_dim, 1)

# Initialize optimizers
print('Initializing optimizers...')
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_s = torch.optim.Adam(model_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_mlp_t = torch.optim.Adam(mlp_t.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_mlp_s = torch.optim.Adam(mlp_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)


b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda() if torch.cuda.is_available() else torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()



# Train the model
print('Starting training loop...')
# Open file in append mode to write (each run appends new content)
output_file = f"{args.dataset}_ggad.txt"

with open(output_file, "a") as f:
    with tqdm(total=args.num_epoch) as pbar:
        total_time = 0
        pbar.set_description('Training')
        for epoch in range(args.num_epoch):
            start_time = time.time()

            # Set training mode
            model.train()
            model_s.train()
            mlp_t.train()
            mlp_s.train()

            # Clear all model gradients
            optimiser.zero_grad()      # Clear teacher model gradients
            optimiser_s.zero_grad()    # Clear student model gradients
            optimiser_mlp_s.zero_grad()  # Clear MLP gradients
            optimiser_mlp_t.zero_grad()  # Clear MLP gradients
            train_flag = False
            _, _, logits_total, _, _ = model(features, adj, abnormal_label_idx, normal_label_idx, train_flag, args)
            train_flag = True
            # Calculate teacher model loss (ggad)
            emb, emb_combine, logits, emb_con, emb_abnormal = model(features, adj, abnormal_label_idx, normal_label_idx, train_flag, args)

            #BCE loss
            lbl = torch.unsqueeze(torch.cat(
                (torch.zeros(len(normal_label_idx)), torch.ones(len(emb_con)))),
                1).unsqueeze(0)

            loss_bce = b_xent(logits, lbl)
            loss_bce = torch.mean(loss_bce)
            
            # Local Affinity Margin Loss
            emb = torch.squeeze(emb)

            emb_inf = torch.norm(emb, dim=-1, keepdim=True)
            emb_inf = torch.pow(emb_inf, -1)
            emb_inf[torch.isinf(emb_inf)] = 0.
            emb_norm = emb * emb_inf

            sim_matrix = torch.mm(emb_norm, emb_norm.T)
            raw_adj = torch.squeeze(raw_adj)
            similar_matrix = sim_matrix * raw_adj

            r_inv = torch.pow(torch.sum(raw_adj, 0), -1)
            r_inv[torch.isinf(r_inv)] = 0.
            affinity = torch.sum(similar_matrix, 0) * r_inv

            affinity_normal_mean = torch.mean(affinity[normal_label_idx])
            affinity_abnormal_mean = torch.mean(affinity[abnormal_label_idx])

            confidence_margin = 0.7
            loss_margin = (confidence_margin - (affinity_normal_mean - affinity_abnormal_mean)).clamp_min(min=0)
            
            #Reconstruction Loss
            diff_attribute = torch.pow(emb_con - emb_abnormal, 2)
            loss_rec = torch.mean(torch.sqrt(torch.sum(diff_attribute, 1)))
            
            
            #Total ggad loss
            ggad_loss = 1 * loss_margin + 1 * loss_bce + 1 * loss_rec
            
      
            # Teacher model embeddings (for distillation loss calculation - all nodes)
            emb_t_all = emb.squeeze(0)
            
            # Student model embeddings (for distillation loss calculation - all nodes)
            emb_s_all = torch.squeeze(model_s(features, adj))

            # calculate student score
            student_score = mlp_t(emb_s_all).squeeze(dim=-1)

            score_from_ggad = logits_total.squeeze(dim=-1).squeeze(0) 


            # calculate kd loss(distance and mlp_student)
            ggad_distance_mlp_s= score_distillation_loss(score_from_ggad, student_score)
            # Calculate distillation loss (using all node embeddings)
            kd_double_mlp= distillation_loss(emb_t_all, emb_s_all, mlp_t, mlp_s)

            distillation_loss_value = ggad_distance_mlp_s + kd_double_mlp
            # Combine losses (teacher loss + distillation loss) for updating teacher and student models
            total_loss = ggad_loss + distillation_loss_value
            # Backpropagate combined total loss and update teacher and student models
            total_loss.backward()  # Backpropagate combined loss
            optimiser.step()       # Update teacher model parameters
            optimiser_s.step()     # Update student model parameters

            # Recalculate distillation loss to update MLP
            # Note: The computation graph has been released, so we recalculate embeddings and distillation loss
            optimiser_mlp_t.zero_grad()  # Clear MLP gradients
            optimiser_mlp_s.zero_grad()  # Clear MLP gradients

            # Recalculate teacher and student model embeddings (for MLP update)
            train_flag = False
            _, _, logits_total, _, _ = model(features, adj, abnormal_label_idx, normal_label_idx, train_flag, args)
            train_flag = True
            # Calculate teacher model loss (ggad)
            emb_t_all, emb_combine, logits, emb_con, emb_abnormal = model(features, adj, abnormal_label_idx, normal_label_idx, train_flag, args)
            emb_t_all = emb_t_all.squeeze(0)
            emb_s_all = torch.squeeze(model_s(features, adj))
  
            score_from_ggad = logits_total.squeeze(dim=-1).squeeze(0) 

            # Recalculate distillation loss
  
            # calculate student score
            student_score = mlp_t(emb_s_all).squeeze(dim=-1)

            
            # calculate kd loss(distance and mlp_student)
            ggad_distance_mlp_s= score_distillation_loss(score_from_ggad, student_score)
            

            # Calculate distillation loss (using all node embeddings)
            kd_double_mlp= distillation_loss(emb_t_all, emb_s_all, mlp_t, mlp_s)

            distillation_loss_value = ggad_distance_mlp_s + kd_double_mlp

            # Backpropagate distillation loss and update MLP
            distillation_loss_value.backward()  # Backpropagate distillation loss to update MLP
            optimiser_mlp_t.step()  # Update MLP parameters
            optimiser_mlp_s.step()


            # Evaluate student model every 5 epochs
            if epoch % 5 == 0:
                # Print and save log
                log_message = (
                    f"Epoch {epoch}: Total Loss = {total_loss.item()}, Distillation Loss = {distillation_loss_value.item()}\n"
                )
                model_s.eval()
                model.eval()
                # Teacher model embeddings for all nodes
                train_flag = False
                emb_t, emb_combine, logits, emb_con, emb_abnormal = model(features, adj, abnormal_label_idx, normal_label_idx,
                                                                    train_flag, args)
                # evaluation on the valid and test node
                # score_from_ggad = np.squeeze(logits[:, idx_test, :].cpu().detach().numpy())


                score_from_ggad = logits.squeeze(dim=-1).squeeze(0) 

                score_from_ggad = min_max_normalize(score_from_ggad)

                emb_s = torch.squeeze(model_s(features, adj))
                student_score = mlp_s(emb_s).squeeze(dim=-1) 

                student_score = min_max_normalize(student_score)
                teacher_score = mlp_t(emb_t).squeeze(dim=-1).squeeze(0) 

                teacher_score = min_max_normalize(teacher_score)
                teacher_score = teacher_score.squeeze(0)
                stu_add_tea_score = student_score + teacher_score


                stu_add_ggad_score = student_score + score_from_ggad 
                tea_add_ggad_score = teacher_score + score_from_ggad 
                stu_tea_ggad_score = stu_add_tea_score + score_from_ggad 

                # Add 0.5 * ggad score cases
                stu_add_05ggad_score = student_score + 0.5 * score_from_ggad 
                tea_add_05ggad_score = teacher_score + 0.5 * score_from_ggad 
                stu_tea_05ggad_score = stu_add_tea_score + 0.5 * score_from_ggad 
                

                logits_stu = np.squeeze(student_score[idx_test].cpu().detach().numpy())
                logits_tea = np.squeeze(teacher_score[idx_test].cpu().detach().numpy())
                logits_stu_tea = np.squeeze(stu_add_tea_score[idx_test].cpu().detach().numpy())
                logits_stu_ggad = np.squeeze(stu_add_ggad_score[idx_test].cpu().detach().numpy())
                logits_tea_ggad = np.squeeze(tea_add_ggad_score[idx_test].cpu().detach().numpy())
                logits_stu_tea_ggad = np.squeeze(stu_tea_ggad_score[idx_test].cpu().detach().numpy())
                logits_stu_05ggad = np.squeeze(stu_add_05ggad_score[idx_test].cpu().detach().numpy())
                logits_tea_05ggad = np.squeeze(tea_add_05ggad_score[idx_test].cpu().detach().numpy())
                logits_stu_tea_05ggad = np.squeeze(stu_tea_05ggad_score[idx_test].cpu().detach().numpy())

                # Calculate AUC for different logits
                auc_stu = roc_auc_score(ano_label[idx_test], logits_stu)
                auc_tea = roc_auc_score(ano_label[idx_test], logits_tea)
                auc_stu_tea = roc_auc_score(ano_label[idx_test], logits_stu_tea)
                auc_stu_ggad = roc_auc_score(ano_label[idx_test], logits_stu_ggad)
                auc_tea_ggad = roc_auc_score(ano_label[idx_test], logits_tea_ggad)
                auc_stu_tea_ggad = roc_auc_score(ano_label[idx_test], logits_stu_tea_ggad)
                # Calculate AUC for 0.5 * ggad cases
                auc_stu_05ggad = roc_auc_score(ano_label[idx_test], logits_stu_05ggad)
                auc_tea_05ggad = roc_auc_score(ano_label[idx_test], logits_tea_05ggad)
                auc_stu_tea_05ggad = roc_auc_score(ano_label[idx_test], logits_stu_tea_05ggad)

                log_message += f'Testing {args.dataset} AUC_student_mlp_s: {auc_stu:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_teacher_mlp_t: {auc_tea:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_teacher_mlp: {auc_stu_tea:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_ggad: {auc_stu_ggad:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_teacher_ggad: {auc_tea_ggad:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_teacher_ggad: {auc_stu_tea_ggad:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_05ggad: {auc_stu_05ggad:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_teacher_05ggad: {auc_tea_05ggad:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_teacher_05ggad: {auc_stu_tea_05ggad:.4f}\n'

                # Calculate AP for different logits
                AP_stu = average_precision_score(ano_label[idx_test], logits_stu, average='macro', pos_label=1)
                AP_tea = average_precision_score(ano_label[idx_test], logits_tea, average='macro', pos_label=1)
                AP_stu_tea = average_precision_score(ano_label[idx_test], logits_stu_tea, average='macro', pos_label=1)
                AP_stu_ggad = average_precision_score(ano_label[idx_test], logits_stu_ggad, average='macro', pos_label=1)
                AP_tea_ggad = average_precision_score(ano_label[idx_test], logits_tea_ggad, average='macro', pos_label=1)
                AP_stu_tea_ggad = average_precision_score(ano_label[idx_test], logits_stu_tea_ggad, average='macro', pos_label=1)
                # Calculate AP for 0.5 * ggad cases
                AP_stu_05ggad = average_precision_score(ano_label[idx_test], logits_stu_05ggad, average='macro', pos_label=1)
                AP_tea_05ggad = average_precision_score(ano_label[idx_test], logits_tea_05ggad, average='macro', pos_label=1)
                AP_stu_tea_05ggad = average_precision_score(ano_label[idx_test], logits_stu_tea_05ggad, average='macro', pos_label=1)


                log_message += f'Testing AP_student_mlp_s: {AP_stu:.4f}\n'
                log_message += f'Testing AP_teacher_mlp_t: {AP_tea:.4f}\n'
                log_message += f'Testing AP_student_teacher_mlp: {AP_stu_tea:.4f}\n'
                log_message += f'Testing AP_student_ggad: {AP_stu_ggad:.4f}\n'
                log_message += f'Testing AP_teacher_ggad: {AP_tea_ggad:.4f}\n'
                log_message += f'Testing AP_student_teacher_ggad: {AP_stu_tea_ggad:.4f}\n'
                log_message += f'Testing AP_student_05ggad: {AP_stu_05ggad:.4f}\n'
                log_message += f'Testing AP_teacher_05ggad: {AP_tea_05ggad:.4f}\n'
                log_message += f'Testing AP_student_teacher_05ggad: {AP_stu_tea_05ggad:.4f}\n'

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