import logging
import numpy as np
from tqdm import tqdm
import torch

import sys
sys.path.append('pygod')
import importlib

import os

# --- 关键：添加自定义加载器所需的库 ---
import scipy.io as sio
import scipy.sparse as sp
import random
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from collections import Counter
# -----------------------------------------

from pygod.utils import load_data 
from pygod.metrics import eval_roc_auc,eval_average_precision,eval_ndcg,eval_precision_at_k,eval_recall_at_k
from pygod.models import ADANET
from torch_geometric.utils import to_dense_adj,add_remaining_self_loops,add_self_loops

from model.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from model.models import build_model
from model.god import god_evaluation

import logging

# --- 关键：添加你的自定义MAT加载器 ---
def load_custom_mat_as_pyg(dataset_path: str, 
                           train_rate: float = 0.3, 
                           val_rate: float = 0.1, 
                           normal_sample_rate: float = 0.5):
    """
    加载一个自定义的 .mat 文件, 将其转换为 PyGod 风格的 PyTorch Geometric Data 对象,
    并包含自定义的半监督拆分。
    """
    
    # 1. 加载 .mat 文件
    print(f"Loading custom .mat file from: {dataset_path}")
    data_mat = sio.loadmat(dataset_path)

    # 2. 灵活提取数据
    attr = data_mat['Attributes'] if ('Attributes' in data_mat) else data_mat['X']
    network = data_mat['Network'] if ('Network' in data_mat) else data_mat['A']
    
    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    # 3. 创建 PyGod 风格的 (0, 1, 2, 3) 标签
    if 'str_anomaly_label' in data_mat and 'attr_anomaly_label' in data_mat:
        print("Found attr/str labels. Creating PyGod-style y (0-3).")
        str_labels = np.squeeze(np.array(data_mat['str_anomaly_label']))
        attr_labels = np.squeeze(np.array(data_mat['attr_anomaly_label']))
        y = attr_labels + (str_labels * 2) 
    else:
        print("Warning: attr/str labels not found. Using binary 'Label' or 'gnd'.")
        label = data_mat['Label'] if ('Label' in data_mat) else data_mat['gnd']
        y = np.squeeze(np.array(label))

    
    # 4. 实现你的训练/验证/测试拆分
    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    
    all_idx = list(range(num_node))
    random.seed(42) # 为了可复现性，设置一个种子
    random.shuffle(all_idx)
    
    idx_train = all_idx[: num_train]
    idx_val = all_idx[num_train: num_train + num_val]
    idx_test = all_idx[num_train + num_val:]

    print('Training set label distribution:', Counter(y[idx_train]))
    print('Test set label distribution:', Counter(y[idx_test]))

    # 5. 实现半监督采样逻辑
    all_normal_label_idx = [i for i in idx_train if y[i] == 0]
    normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * normal_sample_rate)]
    print(f"Total train nodes: {len(idx_train)}. Known normal nodes: {len(normal_label_idx)}")

    # 6. 转换为 PyTorch Tensors
    edge_index, _ = from_scipy_sparse_matrix(adj)
    x_tensor = torch.FloatTensor(feat.toarray()) 
    y_tensor = torch.LongTensor(y)

    
    # 7. 创建并返回 PyG Data 对象 (包含自定义键)
    data_obj = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
    
    data_obj.train_idx = torch.LongTensor(idx_train)
    data_obj.val_idx = torch.LongTensor(idx_val)
    data_obj.test_idx = torch.LongTensor(idx_test) # <--- 属性在这里被添加
    data_obj.normal_label_idx = torch.LongTensor(normal_label_idx) # <--- 属性在这里被添加
    
    # 存储原始标签以供分析 (可选)
    if 'str_anomaly_label' in data_mat:
        data_obj.attr_labels = torch.LongTensor(attr_labels)
        data_obj.str_labels = torch.LongTensor(str_labels)

    return data_obj
# -----------------------------------------


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, model_name,aggr_f,lr_f, max_epoch_f, alpha_f,dropout_f,loss_f,loss_weight_f,T_f, num_hidden=16,logger=None,use_ssl=False,return_edge_score=False):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    # --- 修复： pretrain 也应该使用正确的 epoch 数 ---
    epoch_iter = range(max_epoch) # <--- 使用 max_epoch (Stage 1 的 epoch)

    outlier_score_list=[]
    edge_outlier_score_list=[]
    for epoch in epoch_iter:
        model.train()
        loss, loss_dict,final_edge_mask_rate = model(x, graph.edge_index)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

    # return best_model
    return model,np.array(outlier_score_list),np.array(edge_outlier_score_list)

def main(args):
    device = "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch # Stage 1 pretrain epochs

    num_hidden = args.num_hidden

    node_encoder_num_layers = args.node_encoder_num_layers
    edge_encoder_num_layers = args.edge_encoder_num_layers
    subgraph_encoder_num_layers = args.subgraph_encoder_num_layers

    attr_decoder_num_layers= args.attr_decoder_num_layers
    struct_decoder_num_layers= args.struct_decoder_num_layers

    attr_encoder_name = args.attr_encoder
    struct_encoder_name = args.struct_encoder
    topology_encoder_name=args.topology_encoder

    attr_decoder_name = args.attr_decoder
    struct_decoder_name= args.struct_decoder

    replace_rate = args.replace_rate
    weight_decay=args.weight_decay
    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr # Stage 1 pretrain LR

    model_name=args.model_name # <--- 修复了 .model -> .model_name
    aggr_f=args.aggr_f
    max_epoch_f = args.max_epoch_f # Stage 2 fit epochs
    lr_f = args.lr_f # Stage 2 fit LR
    alpha_f= args.alpha_f
    dropout_f=args.dropout_f
    loss_f=args.loss_f
    loss_weight_f=args.loss_weight_f
    T_f=args.T_f

    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    max_pu_epoch=args.max_pu_epoch
    each_pu_epoch=args.each_pu_epoch

    
    # --- 关键修改 1: 替换数据加载 ---
    # 使用你的自定义加载器
    graph = load_custom_mat_as_pyg(dataset_path=dataset_name, 
                                   train_rate=0.3,
                                   val_rate=0.1, 
                                   normal_sample_rate=0.5) # <-- 你的半监督逻辑
    
    num_features=graph.x.size()[1]
    num_classes=4 

    args.num_features = num_features

    # --- 关键修改 2: 创建 Stage 1 的“干净”预训练图 ---
    print("Creating pre-training graph from *only* normal_label_idx...")
    pretrain_nodes_idx = graph.normal_label_idx # 已知正常节点
    
    # 创建一个只包含这些正常节点的子图
    pretrain_graph = graph.subgraph(pretrain_nodes_idx)
    
    # 为这个子图添加自环
    pretrain_graph.edge_index, _ = add_remaining_self_loops(pretrain_graph.edge_index, num_nodes=pretrain_graph.num_nodes)
    print(f"Pre-training graph: {pretrain_graph.num_nodes} nodes, {pretrain_graph.num_edges} edges.")

    # --- 关键修改 3: 为 Stage 2 (评估) 准备 *完整* 图 ---
    print("Preparing full graph for Stage 2 (fit & evaluation)...")
    graph.edge_index, _ = add_remaining_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
    # -----------------------------------------------


    auc_score_list = []

    attr_mask,struct_mask=None,None
    pretrain_auc_score_list=[]
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        seed=int(seed)
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{node_encoder_num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}")
        else:
            logger = None

        attr_model,struct_model,topology_model = build_model(args)
        attr_model.to(device)
        struct_model.to(device)
        topology_model.to(device)

        if args.use_ssl:
            attr_remask=None
            struct_remask=None
            print('======== train attr encoder ========')
            if args.use_encoder_num>=1:

                optimizer = create_optimizer(optim_type, attr_model, lr, weight_decay)

                if use_scheduler:
                    logging.info("Use schedular")
                    scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
                else:
                    scheduler = None
                    
                # --- 关键修改 4: 在“干净”子图上预训练 ---
                x = pretrain_graph.x # <--- 使用子图的特征
                if not load_model:
                    # <--- 使用子图 pretrain_graph 和 Stage 1 的 epoch 数
                    attr_model,attr_outlier_list,_= pretrain(attr_model, pretrain_graph, x, optimizer, max_epoch, device, scheduler, model_name=model_name,aggr_f=aggr_f,lr_f=lr_f, max_epoch_f=max_epoch_f, alpha_f=alpha_f,dropout_f=dropout_f,loss_f=loss_f,loss_weight_f=loss_weight_f,T_f=T_f, num_hidden=args.num_hidden,logger=logger,use_ssl=args.use_ssl)
                    attr_model = attr_model.cpu()
                
                if load_model:
                    logging.info("Loading Model ... ")
                    attr_model.load_state_dict(torch.load("checkpoint.pt"))
                if save_model:
                    logging.info("Saveing Model ...")
                    torch.save(attr_model.state_dict(), "checkpoint.pt")
                
                attr_model = attr_model.to(device)
                attr_model.eval()

            print('======== train struct encoder ========')
            if args.use_encoder_num>=2:

                optimizer = create_optimizer(optim_type, struct_model, lr, weight_decay)

                if use_scheduler:
                    logging.info("Use schedular")
                    scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
                else:
                    scheduler = None
                    
                x = pretrain_graph.x
                if not load_model:
                    struct_model,struct_node_outlier_list,struct_outlier_list= pretrain(struct_model, pretrain_graph, x, optimizer, max_epoch, device, scheduler,  model_name=model_name,aggr_f=aggr_f,lr_f=lr_f, max_epoch_f=max_epoch_f, alpha_f=alpha_f,dropout_f=dropout_f,loss_f=loss_f,loss_weight_f=loss_weight_f,T_f=T_f, num_hidden=args.num_hidden,logger=logger,use_ssl=args.use_ssl)
                    struct_model = struct_model.cpu()

                if load_model:
                    logging.info("Loading Model ... ")
                    struct_model.load_state_dict(torch.load("checkpoint.pt"))
                if save_model:
                    logging.info("Saveing Model ...")
                    torch.save(struct_model.state_dict(), "checkpoint.pt")
                
                struct_model = struct_model.to(device)
                struct_model.eval()

            print('======== train topology encoder ========')
            if args.use_encoder_num>=3:

                optimizer = create_optimizer(optim_type, topology_model, lr, weight_decay)

                if use_scheduler:
                    logging.info("Use schedular")
                    scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
                else:
                    scheduler = None
                    
                x = pretrain_graph.x
                if not load_model:
                    topology_model,topology_node_outlier_list,topology_outlier_list= pretrain(topology_model, pretrain_graph, x, optimizer, max_epoch, device, scheduler,  model_name=model_name,aggr_f=aggr_f,lr_f=lr_f, max_epoch_f=max_epoch_f, alpha_f=alpha_f,dropout_f=dropout_f,loss_f=loss_f,loss_weight_f=loss_weight_f,T_f=T_f, num_hidden=args.num_hidden,logger=logger,use_ssl=args.use_ssl)
                    topology_model = topology_model.cpu()
                
                if load_model:
                    logging.info("Loading Model ... ")
                    topology_model.load_state_dict(torch.load("checkpoint.pt"))
                if save_model:
                    logging.info("Saveing Model ...")
                    torch.save(topology_model.state_dict(), "checkpoint.pt")
                
                topology_model = topology_model.to(device)
                topology_model.eval()

        print('finish one train!')
        
        # --- 关键修改 5: 修复 god_evaluation 的参数传递 ---
        auc_score,ap_score,ndcg_score,pk_score,rk_score,final_outlier,_= god_evaluation(
            dataset_name,model_name,attr_encoder_name,struct_encoder_name,topology_encoder_name,
            attr_decoder_name,struct_decoder_name,attr_model,struct_model,topology_model,
            graph, # 11. 传递 *完整* 图
            graph.x, # 12. 传递 *完整* 特征
            
            # --- 确保这里的变量与 god.py 中的定义匹配 ---
            aggr_f,       # 13.
            lr_f,         # 14.
            max_epoch_f,  # 15. <-- 修复：使用 max_epoch_f 而不是 max_epoch
            alpha_f,      # 16.
            dropout_f,    # 17.
            loss_f,       # 18.
            loss_weight_f,# 19.
            T_f,          # 20.
            args.num_hidden, # 21.
            
            node_encoder_num_layers,edge_encoder_num_layers,subgraph_encoder_num_layers,
            attr_decoder_num_layers,struct_decoder_num_layers,use_ssl=args.use_ssl,
            use_encoder_num=args.use_encoder_num,attention=args.attention,
            sparse_attention_weight=args.sparse_attention_weight,theta=args.theta,eta=args.eta,
            
            # vvv 传递你的半监督索引 vvv
            y_true=graph.y,
            idx_test=graph.test_idx,
            normal_label_idx=graph.normal_label_idx
            )
        # ----------------------------------------------------
            
        auc_score_list.append(auc_score)

        if logger is not None:
            logger.finish()

    final_auc, final_auc_std = np.mean(auc_score_list), np.std(auc_score_list)

    print(f"# final_auc: {final_auc*100:.2f}±{final_auc_std*100:.2f}")

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    import os
    
    args = build_args()
        
    if args.use_cfg:
        print(f"Original dataset path: {args.dataset}")
        
        dataset_key = os.path.basename(args.dataset)
        dataset_key = os.path.splitext(dataset_key)[0]
        
        original_dataset_path = args.dataset
        args.dataset = dataset_key
        print(f"Loading config for dataset key: '{dataset_key}'")

        args = load_best_configs(args, "config_ada-gad.yml")
        
        args.dataset = original_dataset_path

    if args.alpha_f=='None' or args.alpha_f is None: # 检查 None
        args.alpha_f=None

    if args.all_encoder_layers!=0:
        args.node_encoder_num_layers=args.all_encoder_layers
        args.edge_encoder_num_layers=args.all_encoder_layers
        args.subgraph_encoder_num_layers=args.all_encoder_layers

    print(args)
    
    main(args)