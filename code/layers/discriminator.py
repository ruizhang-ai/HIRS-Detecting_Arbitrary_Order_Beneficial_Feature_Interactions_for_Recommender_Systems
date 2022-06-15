import torch
import torch.nn as nn
from icecream import ic
from torch_geometric.nn import global_add_pool 

class Discriminator_MLP(nn.Module):
    def __init__(self, h_d, c_d, device):
        super(Discriminator_MLP, self).__init__()
        self.f = nn.Linear(c_d, h_d)
        #self.f_1 = nn.Linear(h_d+c_d, 4*h_d)
        #self.f_2 = nn.Linear(4*h_d, 1)
        self.act = nn.ReLU()
        self.device = device

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h):
        """
        """
        out_c = self.f(c)
        logits = torch.sum(out_c*h, 1) 

        return self.act(logits)

class Discriminator_Bilinear(nn.Module):
    def __init__(self, n_h, n_c, device):
        super(Discriminator_Bilinear, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_c, 1)
        self.device = device

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h):
        """
        """
        logits = torch.squeeze(self.f_k(c, h))

        return logits

