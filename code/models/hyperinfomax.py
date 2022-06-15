import torch
import torch.nn as nn
from layers import HGNN, MLPReadout
from layers import INFOMAX, RestCross
from layers import Disc_INFOMIN as INFOMIN
from icecream import ic
from torch_geometric.nn import global_mean_pool
import numpy as np
import torch.autograd.profiler as profiler
from torch_scatter import scatter

class HyperInfomax(nn.Module):
    def __init__(self, args, n_features, device, writer):
        super(HyperInfomax, self).__init__()

        self.feature_emb = nn.Embedding(n_features, args.dim)

        self.feature_emb_edge = nn.Embedding(n_features, args.dim)


        self.hgnn = HGNN(args.dim, args.hid_units, device)
        self.edgePred = RestCross(args.dim, args.edge_num, eval(args.l0_para))
        self.infomax = INFOMAX(args.hid_units, args.n_neg_max,  device)
        self.infomin = INFOMIN(args.hid_units, args.n_neg_min,  device)
        self.readout = MLPReadout(args.hid_units, 1, nn.Sigmoid())
        self.args = args
        self.device = device


    def edge_stat(self, adj, batch):
        round_mat = scatter(torch.round(adj), batch, dim=0)
        grezero_mat = scatter((adj>0).float(), batch, dim=0)
        ones = torch.ones_like(round_mat)

        round_stat_list = []
        grezero_stat_list = []
        for i in range(self.args.num_features+1):
            round_stat = torch.sum((round_mat==i).float()).unsqueeze(0)
            grezero_stat = torch.sum((grezero_mat==i).float()).unsqueeze(0) 
            round_stat_list.append(round_stat)
            grezero_stat_list.append(grezero_stat)
        round_stat_list = torch.cat(round_stat_list).unsqueeze(1)
        grezero_stat_list = torch.cat(grezero_stat_list).unsqueeze(1)
        return round_stat_list, grezero_stat_list 

    def edge_pred(self, features, batch, is_training, record=False, epoch=-1):
        #Edge prediction graph-wise
        
        adj, l0_penaty, n_edge  = self.edgePred(features, batch, is_training)
        n_edge = n_edge.to(self.device)
        edge_batch = torch.LongTensor(range(batch.max()+1)).to(self.device).repeat_interleave(n_edge)
        round_stat, grezero_stat = None, None 


        return adj, l0_penaty, edge_batch, (round_stat, grezero_stat) 


    def pred(self, pred_logits):
        predictions = self.readout(pred_logits)
        return torch.squeeze(predictions)

    def l0_hirs(self, info_data, train, record=False, epoch=-1):
        node_index, edge_index, batch = info_data
        #node_index, batch = node_info_pair 
        nb_nodes = node_index.shape[0]
        features = self.feature_emb(node_index)
        features_edge = self.feature_emb_edge(node_index)
        adj, l0_penaty, edge_batch, edge_stats = self.edge_pred(features_edge, batch, train, record, epoch)
        c, h  = self.hgnn(features, adj, batch)
        edge_num = torch.sum(torch.ones_like(adj[adj>0]))

        return c, h, edge_batch, l0_penaty, edge_num, edge_stats

    def run_pred(self, data, train, record=False, epoch=-1):
        c, h, edge_batch, l0_penaty, edges, edge_stats = self.l0_hirs(data, train, record=record, epoch=epoch)
        pred = self.pred(c)

        if not train:
            return pred, (l0_penaty, edges) 
        else:
            return c, h, edge_batch, l0_penaty, edges, pred, edge_stats 


    def cal_similarity(self, c_p, c_n):
        cos = nn.CosineSimilarity(dim=0, eps=1e-6).to(self.device)
        p_mean = torch.mean(c_p, dim=0)
        n_mean = torch.mean(c_n, dim=0)
        return cos(p_mean, n_mean)


    def forward(self, pos_data, neg_data, train, record=False, epoch=-1):
        c_p, h_p, edge_batch_p, l0_penalty_p, edges_p, pred_p, edge_stats_p = self.run_pred(pos_data, train, record=record, epoch=epoch)
        c_n, h_n, edge_batch_n, l0_penalty_n, edges_n, pred_n, edge_stats_n = self.run_pred(neg_data, train, record=record, epoch=epoch)

        hc_loss = self.infomax(c_p, h_p, edge_batch_p, c_n, h_n, edge_batch_n)
        infomax_loss = hc_loss
        infomin_loss_p = self.infomin(h_p, edge_batch_p)
        infomin_loss_n = self.infomin(h_n, edge_batch_n)
        infomin_loss = (infomin_loss_p + infomin_loss_n)/2

        distance_c = self.cal_similarity(c_p, c_n)
        distance_h = self.cal_similarity(h_p, h_n)

        l0_penalty = l0_penalty_p + l0_penalty_n
        n_edges = edges_p + edges_n

        return pred_p, pred_n, infomax_loss, infomin_loss, (distance_c, distance_h), (l0_penalty, n_edges), (edge_stats_p, edge_stats_n)



