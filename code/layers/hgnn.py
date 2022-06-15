import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from icecream import ic
from torch_scatter import scatter

class HGNN(nn.Module):
    def __init__(self, in_ch, out_ch, device):
        super(HGNN, self).__init__()
        self.device = device
        self.f = nn.Sequential(
                nn.Linear(in_ch,in_ch),
                nn.ReLU(),
                )

    def forward(self, x, H, batch):
        """
        x: the embedding of the vectors (V, d)
        H: the edge matrix (V, E) batch-wise diagnal matrix
        edge_num: the number of edges of each data samples (batch_size)
        """
        expand_x = x.unsqueeze(1).repeat([1,H.shape[-1], 1])
        expand_H = H.unsqueeze(2)
        edge_emb_tensor = expand_x * expand_H      #[V, n_edge, dim]
        edge_dim  = scatter(edge_emb_tensor, batch, dim=0, reduce='mean') #[batch_size, n_edge, dim]
        edge_dim_flatten = edge_dim.view(-1, edge_dim.shape[2])
        edge_dim_flatten = self.f(edge_dim_flatten) 
        edge_dim = edge_dim_flatten.view(edge_dim.shape)

        rep_index = scatter(torch.ones_like(batch), batch)
        repeat_edge_dim = torch.repeat_interleave(edge_dim, rep_index, dim=0) 
        node_emb_tensor = repeat_edge_dim * expand_H
        node_emb = node_emb_tensor.mean(dim=1)

        #convert edge_emb from [batch-size,n_edge, dim] to [batch_size*n_edge, dim]
        h_e = edge_dim.view(-1, edge_dim.shape[-1]) 
        c =  global_mean_pool(node_emb, batch)
        return c, h_e 


