import copy
from tqdm import tqdm
import torch
import torch.nn as nn

# 假设 model.utils 在你的路径中
from model.utils import create_optimizer, accuracy

import sys

import importlib
from pygod.utils import load_data
from pygod.metrics import eval_roc_auc,eval_average_precision,eval_ndcg,eval_precision_at_k,eval_recall_at_k
from pygod.models import ADANET # 假设 ADANET 在你的 pygod 库中
from torch_geometric.utils import to_dense_adj
import numpy as np

anomaly_num_dict={'weibo':868,'reddit':366,'disney':6,'books':28,'enron':5,'inj_cora':138,'inj_amazon':694,'inj_flickr':4414}

# --- 关键修改 1: 修改函数签名 ---
# 添加 y_true, idx_test, 和 normal_label_idx 参数
def god_evaluation(data_name,model_name,attr_encoder_name,struct_encoder_name,topology_encoder_name,
attr_decoder_name,struct_decoder_name,attr_ssl_model,struct_ssl_model,topology_ssl_model,graph, x, 
aggr_f,lr_f, max_epoch_f, alpha_f,dropout_f,loss_f,loss_weight_f,T_f,num_hidden,node_encoder_num_layers,edge_encoder_num_layers,subgraph_encoder_num_layers,
attr_decoder_num_layers=1,struct_decoder_num_layers=1,use_ssl=False,use_encoder_num=1,attention=None,sparse_attention_weight=0.001,
theta=1.001,eta=1.001, y_true=None, idx_test=None, normal_label_idx=None):
    
    if use_encoder_num==1:
        attr_ssl_model.eval()
    if use_encoder_num==2:
        attr_ssl_model.eval()
        struct_ssl_model.eval()
    if use_encoder_num==3:
        attr_ssl_model.eval()
        struct_ssl_model.eval()
        topology_ssl_model.eval()
        
    
    model= eval(model_name)(epoch=max_epoch_f,aggr=aggr_f,hid_dim=num_hidden,alpha=alpha_f,dropout=dropout_f,\
        lr=lr_f,loss_name=loss_f,loss_weight=loss_weight_f,T=T_f,use_encoder_num=use_encoder_num,attention=attention,\
            attr_encoder_name=attr_encoder_name,struct_encoder_name=struct_encoder_name,topology_encoder_name=topology_encoder_name,attr_decoder_name=attr_decoder_name,struct_decoder_name=struct_decoder_name,\
            node_encoder_num_layers=node_encoder_num_layers,edge_encoder_num_layers=edge_encoder_num_layers,subgraph_encoder_num_layers=subgraph_encoder_num_layers,\
                attr_decoder_num_layers=attr_decoder_num_layers,\
                struct_decoder_num_layers=struct_decoder_num_layers,sparse_attention_weight=sparse_attention_weight,theta=theta,eta=eta)

    # --- 关键修改 2: 为 Stage 2 (fit) 创建半监督训练掩码 ---
    # 我们假设 ADANET.fit() 会自动查找 graph.train_mask
    if normal_label_idx is not None:
        print("Applying semi-supervised train_mask for Stage 2 (fit)...")
        # 确保掩码在正确的设备上
        train_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=x.device) 
        train_mask[normal_label_idx] = True
        graph.train_mask = train_mask
    # --------------------------------------------------------

    if use_ssl and use_encoder_num>0:
        if use_encoder_num==1:
            # model.fit 现在会使用 graph.train_mask
            model.fit(graph,pretrain_attr_encoder=attr_ssl_model.encoder,pretrain_struct_encoder=None,pretrain_topology_encoder=None)
        elif use_encoder_num==2:
            model.fit(graph,pretrain_attr_encoder=attr_ssl_model.encoder,pretrain_struct_encoder=struct_ssl_model.encoder,pretrain_topology_encoder=None) 
        elif use_encoder_num==3: 
            model.fit(graph,pretrain_attr_encoder=attr_ssl_model.encoder,pretrain_struct_encoder=struct_ssl_model.encoder,pretrain_topology_encoder=topology_ssl_model.encoder)    
        else:
            assert(f'wrong encoder num: {use_encoder_num}')
    else:
        model.fit(graph) # 同样会使用 graph.train_mask
        
    labels = model.predict(graph)

    outlier_scores = model.decision_function(graph)
    edge_outlier_scores = model.decision_struct_function(graph)

    # --- 关键修改 3: 仅在测试集 (idx_test) 上评估 ---
    if idx_test is not None and y_true is not None:
        print(f"Evaluating AUC on test set (size {len(idx_test)})...")
        test_idx_cpu = idx_test.cpu()
        y_true_cpu = y_true.bool().cpu()
        
        # 确保 outlier_scores 在 CPU 上
        outlier_scores_cpu = outlier_scores
        if isinstance(outlier_scores, torch.Tensor):
             outlier_scores_cpu = outlier_scores.cpu().numpy()
            
        auc_score = eval_roc_auc(y_true_cpu[test_idx_cpu].numpy(), outlier_scores_cpu[test_idx_cpu])
    else:
        # 如果没有提供索引，则退回到原始的全局评估
        print("Warning: idx_test not provided. Evaluating on full graph.")
        auc_score = eval_roc_auc(graph.y.bool().cpu().numpy(), outlier_scores)
    # ----------------------------------------------------

    print(f'auc_score: {auc_score:.4f}',)

    return auc_score,None,None,None,None,outlier_scores,edge_outlier_scores