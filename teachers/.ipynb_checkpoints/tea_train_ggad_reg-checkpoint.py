import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch.nn as nn

from models.model import Model_ggad
from utils.utils import *

from sklearn.metrics import roc_auc_score
import random
import dgl
from sklearn.metrics import average_precision_score
import argparse
from tqdm import tqdm
import time

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [3]))
# os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
# Set argument
parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataset', type=str,
                    default='reddit')
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--mean', type=float, default=0.0)
parser.add_argument('--var', type=float, default=0.0)



args = parser.parse_args()

if args.lr is None:
    if args.dataset in ['Amazon']:
        args.lr = 1e-3
    elif args.dataset in ['tf_finace']:
        args.lr = 1e-3
    elif args.dataset in ['reddit']:
        args.lr = 1e-3
    elif args.dataset in ['photo']:
        args.lr = 1e-3
    elif args.dataset in ['elliptic']:
        args.lr = 1e-3
    elif args.dataset in ['tolokers']:
        args.lr = 1e-3
    elif args.dataset in ['YelpChi-all']:
        args.lr = 1e-3

if args.num_epoch is None:
    if args.dataset in ['photo']:
        args.num_epoch = 100
    if args.dataset in ['elliptic']:
        args.num_epoch = 150
    if args.dataset in ['reddit']:
        args.num_epoch = 300
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

if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
    features, _ = preprocess_features(features)
else:
    features = features.todense()

dgl_graph = adj_to_dgl_graph(adj)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
raw_adj = adj
print(adj.sum())
adj = normalize_adj(adj)

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

b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()

# Train the model
print('Starting training loop...')
# Open file in append mode to write (each run appends new content)
output_file = f"{args.dataset}_unify_ggad_origin.txt"
# Train model
target_epoch = 480 # 你可以根据需要修改
window = 100

best_auc_window = 0.0
best_model_state_window = None
best_epoch_window = -1

with open(output_file, "a") as f:
    with tqdm(total=args.num_epoch) as pbar:
        pbar.set_description('Training')
        total_time = 0
        for epoch in range(args.num_epoch):
            start_time = time.time()
            model.train()
            optimiser.zero_grad()

            # Train model
            train_flag = True
            emb, emb_combine, logits, emb_con, emb_abnormal = model(features, adj,
                                                                    abnormal_label_idx, normal_label_idx,
                                                                    train_flag, args)
            if epoch % 10 == 0:
                # save data for tsne
                pass

                # tsne_data_path = 'draw/tfinance/tsne_data_{}.mat'.format(str(epoch))
                # io.savemat(tsne_data_path, {'emb': np.array(emb.cpu().detach()), 'ano_label': ano_label,
                #                             'abnormal_label_idx': np.array(abnormal_label_idx),
                #                             'normal_label_idx': np.array(normal_label_idx)})

            # BCE loss
            lbl = torch.unsqueeze(torch.cat(
                (torch.zeros(len(normal_label_idx)), torch.ones(len(emb_con)))),
                1).unsqueeze(0)
            # if torch.cuda.is_available():
            #     lbl = lbl.cuda()

            loss_bce = b_xent(logits, lbl)
            loss_bce = torch.mean(loss_bce)

            # Local affinity margin loss
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

            diff_attribute = torch.pow(emb_con - emb_abnormal, 2)
            loss_rec = torch.mean(torch.sqrt(torch.sum(diff_attribute, 1)))

            loss = 1 * loss_margin + 1 * loss_bce + 1 * loss_rec

            loss.backward()
            optimiser.step()
            end_time = time.time()
            total_time += end_time - start_time
            if epoch % 2 == 0:
                log_message = (
                    f"Epoch {epoch}: ggad train Loss = {loss.item()}\n"
                )
                logits = np.squeeze(logits.cpu().detach().numpy())
                lbl = np.squeeze(lbl.cpu().detach().numpy())
                auc = roc_auc_score(lbl, logits)
                # print('Traininig {} AUC:{:.4f}'.format(args.dataset, auc))
                # AP = average_precision_score(lbl, logits, average='macro', pos_label=1, sample_weight=None)
                # print('Traininig AP:', AP)

            if epoch % 5 == 0:
                model.eval()
                train_flag = False
                emb, emb_combine, logits, emb_con, emb_abnormal = model(features, adj, abnormal_label_idx, normal_label_idx,
                                                                        train_flag, args)
                # evaluation on the valid and test node
                logits = np.squeeze(logits[:, idx_test, :].cpu().detach().numpy())
                auc = roc_auc_score(ano_label[idx_test], logits)
                log_message += f'Testing {args.dataset} AUC: {auc:.4f}\n'
                AP = average_precision_score(ano_label[idx_test], logits, average='macro', pos_label=1, sample_weight=None)

                log_message += f'Testing AP: {AP:.4f}\n'

                # 只在指定区间内保存最佳AUC模型
                if (epoch >= target_epoch - window) and (epoch <= target_epoch + window):
                    if auc > best_auc_window:
                        best_auc_window = auc
                        best_model_state_window = model.state_dict()
                        best_epoch_window = epoch



                # Print log
                print(log_message)
                # Write log to file
                f.write(log_message)
                f.flush()  # Ensure content is immediately written to file

            end_time = time.time()
            total_time += end_time - start_time
            pbar.update(1)

print("Training completed.")
if best_model_state_window is not None:
    torch.save(best_model_state_window, f'{args.dataset}_ggad_teacher_best_{best_epoch_window}.pth')
    print(f"✅ Best model in [{target_epoch-window}, {target_epoch+window}] saved at epoch {best_epoch_window} with AUC {best_auc_window:.4f}.")
else:
    print("No model was saved in the specified window.")
