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
from models.model import Model_ggad
from utils.utils import *
from utils.log_paths import add_log_subdir_argument, get_log_file
import os

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2]))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='reddit')
parser.add_argument('--teacher_path', type=str, default='reddit_ggad_teacher_final.pth')
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--seed', type=int, default=0)
add_log_subdir_argument(parser, 'ggad_score_distill')
args = parser.parse_args()


# Set learning rate and number of epochs based on dataset
if args.lr is None:
    if args.dataset in ['Amazon']:
        args.lr = 1e-3
    elif args.dataset in ['tf_finace']:
        args.lr = 1e-3
    elif args.dataset in ['reddit']:
        args.lr = 1e-3
    elif args.dataset in ['elliptic']:
        args.lr = 1e-3
    elif args.dataset in ['photo']:
        args.lr = 1e-3

if args.num_epoch is None:
    if args.dataset in ['reddit']:
        args.num_epoch = 1300
    elif args.dataset in ['tf_finace']:
        args.num_epoch = 1500
    elif args.dataset in ['Amazon']:
        args.num_epoch = 1800
    elif args.dataset in ['elliptic']:
        args.num_epoch = 1150
    elif args.dataset in ['photo']:
        args.num_epoch = 1100
        
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
model.load_state_dict(torch.load(args.teacher_path))
model.eval()
# 冻结参数
for p in model.parameters():
    p.requires_grad = False

model_s = Model_ocgnn(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)  

print(args.lr)
print(args.num_epoch)


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

_, _, logits_total, _, _ = model(features, adj, abnormal_label_idx, normal_label_idx, train_flag=False, args=args)


print("\n🔁 Starting Student Training...")
output_file = get_log_file(args, f"{args.dataset}_ggad_2_step.txt")
with open(output_file, "a") as f:
    with tqdm(total=args.num_epoch) as pbar:
        total_time = 0
        pbar.set_description('Training')
        for epoch in range(args.num_epoch):
            start_time = time.time()
            model_s.train()
            mlp_s.train()
            optimiser_s.zero_grad()
            optimiser_mlp_s.zero_grad()
            
            with torch.no_grad():
                _, _, logits_total, _, _ = model(features, adj, abnormal_label_idx, normal_label_idx, train_flag=False, args=args)
                # 输出ggad 模型分数
                logits = np.squeeze(logits_total[:, idx_test, :].cpu().detach().numpy())
                auc = roc_auc_score(ano_label[idx_test], logits)
                log_message = (f'Testing_last_ggad_ {args.dataset} AUC: {auc:.4f}\n')
                score_from_ggad_non_normalize = logits_total.squeeze(dim=-1).squeeze(0)
                score_from_ggad = min_max_normalize(score_from_ggad_non_normalize)
                emb_t_all, _, logits, _, _ = model(features, adj, abnormal_label_idx, normal_label_idx, train_flag=True, args=args)
                emb_t_all = emb_t_all.squeeze(0)


            _, emb_s_all_raw = model_s(features, adj)
            emb_s_all = emb_s_all_raw.squeeze(0)
            student_score_non_normalize = mlp_s(emb_s_all).squeeze(dim=-1)
            student_score = torch.sigmoid(student_score_non_normalize)
            mean_distillation = score_distillation_loss(score_from_ggad, student_score)


            loss_one,_,_,_ = loss_func(emb_s_all[normal_label_idx])
            # student model one class loss
            total_loss = mean_distillation

            total_loss.backward()
            optimiser_s.step()
            optimiser_mlp_s.step()

            if epoch % 5 == 0:
                # Print and save log
                log_message += f"Epoch {epoch}: Total Loss = {total_loss.item()}, Distillation Loss ={mean_distillation.item()}\n"
                model_s.eval()
                # Teacher model embeddings for all nodes
                train_flag = False
                emb_t, emb_combine, logits, emb_con, emb_abnormal = model(features, adj, abnormal_label_idx, normal_label_idx,
                                                                    train_flag, args)
                # evaluation on the valid and test node

                score_from_ggad_non_normalize = logits.squeeze(dim=-1).squeeze(0) 

                score_from_ggad = torch.sigmoid(score_from_ggad_non_normalize)



                log_message += f'student_score: {torch.mean(student_score)}\n'
                log_message += f'ggad_score: {torch.mean(score_from_ggad)}\n'
                log_message += f'student_score_non_normalize: {torch.mean(student_score_non_normalize)}\n'
                log_message += f'ggad_score_non_normalize: {torch.mean(score_from_ggad_non_normalize)}\n'


                student_minus_ggad_score = abs(student_score - score_from_ggad)
                stu_add_ggad_score = student_score + score_from_ggad

                stu_add_ggad_non_normalize_score = student_score_non_normalize + score_from_ggad_non_normalize
                student_minus_ggad_non_normalize_score = student_score_non_normalize - score_from_ggad_non_normalize

                logits_stu = np.squeeze(student_score[idx_test].cpu().detach().numpy())
                logits_stu_non_normalize = np.squeeze(student_score_non_normalize[idx_test].cpu().detach().numpy())
                logits_stu_ggad = np.squeeze(stu_add_ggad_score[idx_test].cpu().detach().numpy())
                logits_stu_ggad_non_normalize = np.squeeze(stu_add_ggad_non_normalize_score[idx_test].cpu().detach().numpy())
                logits_stu_minus_ggad = np.squeeze(student_minus_ggad_score[idx_test].cpu().detach().numpy())
                logits_stu_minus_ggad_non_normalize = np.squeeze(student_minus_ggad_non_normalize_score[idx_test].cpu().detach().numpy())

                # Calculate AUC for different logits
                auc_stu = roc_auc_score(ano_label[idx_test], logits_stu)
                auc_stu_non_normalize = roc_auc_score(ano_label[idx_test], logits_stu_non_normalize)
                auc_stu_ggad = roc_auc_score(ano_label[idx_test], logits_stu_ggad)
                auc_stu_ggad_non_normalize = roc_auc_score(ano_label[idx_test], logits_stu_ggad_non_normalize)
                auc_stu_minus_ggad = roc_auc_score(ano_label[idx_test], logits_stu_minus_ggad)
                auc_stu_minus_ggad_non_normalize = roc_auc_score(ano_label[idx_test], logits_stu_minus_ggad_non_normalize)




                log_message += f'Testing {args.dataset} AUC_student_mlp_s: {auc_stu:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_mlp_s_non_normalize: {auc_stu_non_normalize:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_ggad: {auc_stu_ggad:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_ggad_non_normalize: {auc_stu_ggad_non_normalize:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_minus_ggad: {auc_stu_minus_ggad:.4f}\n'
                log_message += f'Testing {args.dataset} AUC_student_minus_ggad_non_normalize: {auc_stu_minus_ggad_non_normalize:.4f}\n'



                # Calculate AP for different logits
                AP_stu = average_precision_score(ano_label[idx_test], logits_stu, average='macro', pos_label=1)
                AP_stu_non_normalize = average_precision_score(ano_label[idx_test], logits_stu_non_normalize, average='macro', pos_label=1)
                AP_stu_ggad = average_precision_score(ano_label[idx_test], logits_stu_ggad, average='macro', pos_label=1)
                AP_stu_ggad_non_normalize = average_precision_score(ano_label[idx_test], logits_stu_ggad_non_normalize, average='macro', pos_label=1)
                AP_stu_minus_ggad = average_precision_score(ano_label[idx_test], logits_stu_minus_ggad, average='macro', pos_label=1)
                AP_stu_minus_ggad_non_normalize = average_precision_score(ano_label[idx_test], logits_stu_minus_ggad_non_normalize, average='macro', pos_label=1)
                # Calculate AP for 0.5 * ggad cases

                log_message += f'Testing AP_student_mlp_s: {AP_stu:.4f}\n'
                log_message += f'Testing AP_student_mlp_s_non_normalize: {AP_stu_non_normalize:.4f}\n'
                log_message += f'Testing AP_student_ggad: {AP_stu_ggad:.4f}\n'
                log_message += f'Testing AP_student_ggad_non_normalize: {AP_stu_ggad_non_normalize:.4f}\n'
                log_message += f'Testing AP_student_minus_ggad: {AP_stu_minus_ggad:.4f}\n'
                log_message += f'Testing AP_student_minus_ggad_non_normalize: {AP_stu_minus_ggad_non_normalize:.4f}\n'

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
torch.save(mlp_s.state_dict(), f'{args.dataset}_mlp_final.pth')
print("\n✅ Student and MLP saved.")
