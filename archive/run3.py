import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch.nn as nn

from models.model3 import Model_ggad as Model
from utils.utils import *

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
                    default='questions')  # 'BlogCatalog'  'Flickr'  'ACM'  'cora'  'citeseer'  'pubmed', 'Amazon_no_isolate
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
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'Flickr', 'BlogCatalog', 'elliptic']:
        args.lr = 1e-3
    elif args.dataset == 'ACM':
        args.lr = 3e-3
    elif args.dataset in ['Amazon_no_isolate']:
        args.lr = 1e-3
    elif args.dataset in ['YelpChi_no_isolate', 'fb']:
        args.lr = 1e-3
    elif args.dataset in [ 'photo']:
        args.lr = 1e-4
    elif args.dataset in ['reddit', 'weibo', 'tf_finace']:
        args.lr = 1e-3
    elif args.dataset in ['tolokers_no_isolated']:
        args.lr = 1e-3
    elif args.dataset in ['questions']:
        args.lr = 1e-3
if args.num_epoch is None:
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'elliptic', 'photo']:
        args.num_epoch = 500
    elif args.dataset in ['BlogCatalog', 'Flickr', 'ACM',
                          'YelpChi_no_isolate']:
        args.num_epoch = 500
    elif args.dataset in ['reddit', 'weibo', 'fb']:
        args.num_epoch = 500
    elif args.dataset in ['tf_finace']:
        args.num_epoch = 1500
    elif args.dataset in ['Amazon_no_isolate']:
        args.num_epoch = 800
    elif args.dataset in ['tolokers_no_isolated', 'questions']:
        args.num_epoch = 800

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

if args.dataset in ['Amazon_no_isolate', 'YelpChi_no_isolate ', 'tf_finace', 'reddit',
                    'tolokers_no_isolated', 'questions', 'elliptic']:
    features, _ = preprocess_features(features)
else:
    features = features.todense()
dgl_graph = adj_to_dgl_graph(adj)

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
model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
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

# Train model
with tqdm(total=args.num_epoch) as pbar:
    pbar.set_description('Training')
    total_time = 0
    log_data = []
    for epoch in range(args.num_epoch):
        start_time = time.time()
        model.train()
        optimiser.zero_grad()

        # Train model
        train_flag = True

        emb, emb_combine, logits, emb_con, emb_abnormal, emb_orign = model(features, adj, abnormal_label_idx,
                                                                           normal_label_idx,
                                                                           train_flag)
        # if epoch % 10 == 0:
        #     # save data for tsne
        #     pass
        #
        #     # tsne_data_path = 'draw/tfinance_recon_total/tsne_data_{}.mat'.format(str(epoch))
        #     # io.savemat(tsne_data_path, {'emb': np.array(emb.cpu().detach()), 'ano_label': ano_label,
        #     #                             'abnormal_label_idx': np.array(abnormal_label_idx),
        #     #                             'normal_label_idx': np.array(normal_label_idx)})

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

        # emb_norm = emb / torch.norm(emb, dim=-1, keepdim=True)

        # emb_old = torch.squeeze(emb_old)
        # emb_old_norm = emb_old / torch.norm(emb_old, dim=-1, keepdim=True)

        sim_matrix = torch.mm(emb_norm, emb_norm.T)
        raw_adj = torch.squeeze(raw_adj)
        similar_matrix = sim_matrix * raw_adj

        # sim_matrix_old = torch.mm(emb_old_norm, emb_old_norm.T)
        # raw_adj = torch.squeeze(raw_adj)
        # similar_matrix_old = sim_matrix_old * raw_adj

        r_inv = torch.pow(torch.sum(raw_adj, 0), -1)
        r_inv[torch.isinf(r_inv)] = 0.
        affinity = torch.sum(similar_matrix, 0) * r_inv

        # affinity = torch.sum(similar_matrix, 0) / torch.sum(raw_adj, 0)
        # affinity_old = torch.sum(similar_matrix_old, 0) / torch.sum(raw_adj, 0)

        affinity_normal_mean = torch.mean(affinity[normal_label_idx])
        affinity_abnormal_mean = torch.mean(affinity[abnormal_label_idx])

        confidence_margin = 0.7
        loss_margin = (confidence_margin - (affinity_normal_mean - affinity_abnormal_mean)).clamp_min(min=0)

        # affinity_mean = torch.mean((affinity_old[abnormal_label_idx] - affinity[abnormal_label_idx]), 0)
        # affinity_mean = torch.mean((affinity_old[abnormal_label_idx] - torch.mean(affinity[normal_label_idx])), 0)
        # loss_margin_old = (confidence_margin - affinity_mean).clamp_min(min=0)

        # emb_con = emb_con / torch.norm(emb_con, dim=-1, keepdim=True)

        diff_attribute = torch.pow(emb_con - emb_abnormal, 2)
        loss_rec = torch.mean(torch.sqrt(torch.sum(diff_attribute, 1)))

        # diff_attribute_raw = torch.pow(torch.squeeze(features)[abnormal_label_idx, :] - emb_rec, 2)
        # loss_rec_raw = torch.mean(torch.sqrt(torch.sum(diff_attribute_raw, 1)))

        loss = 1 * loss_margin + 1 * loss_bce + 1 * loss_rec


        loss.backward()
        optimiser.step()
        end_time = time.time()
        total_time += end_time - start_time
        # print('Total time is', total_time)

        # Record the log
        log_data.append([np.array(loss_margin.detach()), np.array(loss_bce.detach()),
                         np.array(loss_rec.detach()), np.array(loss.detach())])

        if epoch % 2 == 0:
            logits = np.squeeze(logits.cpu().detach().numpy())
            lbl = np.squeeze(lbl.cpu().detach().numpy())
            auc = roc_auc_score(lbl, logits)
            # print('Traininig {} AUC:{:.4f}'.format(args.dataset, auc))
            # AP = average_precision_score(lbl, logits, average='macro', pos_label=1, sample_weight=None)
            # print('Traininig AP:', AP)

            print("Epoch:", '%04d' % (epoch), "train_loss_margin=", "{:.5f}".format(loss_margin.item()))
            # print("Epoch:", '%04d' % (epoch), "train_loss_margin_old=", "{:.5f}".format(loss_margin_old.item()))
            print("Epoch:", '%04d' % (epoch), "train_loss_bce=", "{:.5f}".format(loss_bce.item()))
            # print("Epoch:", '%04d' % (epoch), "rec_loss=", "{:.5f}".format(loss_rec_raw.item()))
            print("Epoch:", '%04d' % (epoch), "rec_loss=", "{:.5f}".format(loss_rec.item()))
            print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item()))
            print("=====================================================================")

        if epoch % 30 == 0:
            real_abnormal_label_idx = np.array(all_idx)[np.argwhere(ano_label == 1).squeeze()].tolist()
            real_normal_label_idx = np.array(all_idx)[np.argwhere(ano_label == 0).squeeze()].tolist()
            overlap = list(set(real_abnormal_label_idx) & set(real_normal_label_idx))
            # extend_label = torch.zeros(emb_combine.size(0), 1)
            # extend_label[abnormal_label_idx] = 1
            # extend_label[real_abnormal_label_idx] = 2

            # data_dict = dict([('embedding', emb_combine), ('Label', extend_label)])
            #
            # scio.savemat('embedding/{}_{}.mat'.format(args.dataset, epoch), data_dict)

            # real_affinity, index = torch.sort(affinity[real_abnormal_label_idx])
            # real_affinity = real_affinity[:600]
            # 
            # draw_pdf(np.array(affinity[real_normal_label_idx].detach().cpu()),
            #          np.array(affinity[abnormal_label_idx].detach().cpu()),
            #          np.array(real_affinity.detach().cpu()), args.dataset, epoch)

            # Save emb and emb_con
            data_dict = dict([('emb', emb_orign.detach().numpy()), ('feat', features.detach().numpy()),
                              ('emb_outlier', emb_con.detach().numpy()),
                              ('ano_label', ano_label), ('idx_test', idx_test), ('idx_train', idx_train),
                              ('normal_label_idx', normal_label_idx)])
            scio.savemat('emb_benefit/{}_{}.mat'.format(args.dataset, epoch), data_dict)

            # Save the log data
            # pd.DataFrame(log_data).to_csv('rebutall/util_rebutall_{}.csv'.format(args.dataset))

        if epoch % 10 == 0:
            model.eval()
            train_flag = False
            emb, emb_combine, logits, emb_con, emb_abnormal, emb_orign = model(features, adj, abnormal_label_idx, normal_label_idx,
                                                                    train_flag)
            # evaluation on the valid and test node
            logits = np.squeeze(logits[:, idx_test, :].cpu().detach().numpy())
            auc = roc_auc_score(ano_label[idx_test], logits)
            print('Testing {} AUC:{:.4f}'.format(args.dataset, auc))
            AP = average_precision_score(ano_label[idx_test], logits, average='macro', pos_label=1, sample_weight=None)
            print('Testing AP:', AP)
        if epoch == 30 :
            weight_save_path = "teacher/ggad_teacher_best_epoc.{}.pth".format(epoch)  # Define save path
            torch.save(model.state_dict(), weight_save_path)
            print(f"✅ New best model saved at epoch {epoch} with AUC {auc:.4f}.")

        if epoch == 80:
            weight_save_path = "teacher/ggad_teacher_best_epoc.{}.pth".format(epoch)  # Define save path
            torch.save(model.state_dict(), weight_save_path)
            print(f"✅ New best model saved at epoch {epoch} with AUC {auc:.4f}.")

