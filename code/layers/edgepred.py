import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from icecream import ic
from torch_scatter import scatter

class LinearTransform(nn.Module):
    def __init__(self, d, n_edge, l0_para):
        super(LinearTransform, self).__init__()
        self.fc = nn.Linear(d, n_edge)
        self.n_edge = n_edge
        self.l0_binary = L0_Hard_Concrete(*l0_para)

        for m in self.modules():
            self.weights_init(m)
            

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, batch, is_training):
        edge_logits = self.fc(x)
        edge_binary, l0_penaty = self.l0_binary(edge_logits, batch, is_training)
        n_edges = torch.LongTensor([self.n_edge for i in range(batch.max()+1)])
        return edge_binary, l0_penaty, n_edges



class RestCross(nn.Module):
    """
    The rest  cross function will sum all node embeddings in a graph as u_{total}.
    Then, a node u_i concat the u_{totoal} and input into a MLP.
    """
    def __init__(self, d, n_edge, l0_para):
        super(RestCross, self).__init__()
        #self.fc = nn.Linear(2*d, n_edge)
        self.n_edge = n_edge
        self.l0_binary = L0_Hard_Concrete(*l0_para)
        #self.f_k = nn.Bilinear(d, d, n_edge)
        self.f_k = nn.Linear(2*d, n_edge)
        #self.f_k = nn.Linear(d, n_edge)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    
    def forward(self, x, batch, is_training):

        x_t = scatter(x, batch, dim=0, reduce='sum')

        ones = torch.ones_like(batch) 
        index = scatter(ones, batch)
        x_t = torch.repeat_interleave(x_t, index, dim=0)
        x_rest = x_t - x

        edge_logits = self.f_k(torch.cat([x, x_rest], dim=1))
        edge_binary, l0_penaty = self.l0_binary(edge_logits, batch, is_training)
        n_edges = torch.LongTensor([self.n_edge for i in range(batch.max()+1)])
        return edge_binary, l0_penaty, n_edges 



class L0_Hard_Concrete(nn.Module):
    def __init__(self, temp, inter_min, inter_max):
        super(L0_Hard_Concrete, self).__init__()
        self.temp = temp
        self.inter_min = inter_min
        self.inter_max = inter_max
        self.hardtanh = nn.Hardtanh(0, 1)
        self.pdist = nn.PairwiseDistance(p=1)

    def perm_distance(self, s):
        index_tensor = torch.tensor(range(s.shape[0]))
        index_comb = torch.combinations(index_tensor)
        perm_s = s[index_comb]
        s_1 = (perm_s[:,0,:])
        s_2 = (perm_s[:,1,:])

        return self.pdist(s_1, s_2)
    
    def forward(self, loc, batch, is_training):

        if is_training:
            u = torch.rand_like(loc)
            logu = torch.log2(u)
            logmu = torch.log2(1-u)
            sum_log = loc + logu - logmu
            s = torch.sigmoid(sum_log/self.temp)
            s = s * (self.inter_max - self.inter_min) + self.inter_min
        else:
            s = torch.sigmoid(loc) * (self.inter_max - self.inter_min) + self.inter_min

        #s = torch.clamp(s, min=0, max=1)
        s = self.hardtanh(s)

        l0_matrix = torch.sigmoid(loc - self.temp * np.log2(-self.inter_min/self.inter_max))

        #original penalty
        l0_penaty = l0_matrix.mean()

        return s, l0_penaty

