import torch
import torch.nn as nn
from icecream import ic


class MLPReadout(nn.Module):
    def __init__(self, in_dim, out_dim, act):
        """
        out_dim: the final prediction dim, usually 1
        act: the final activation, if rating then None, if CTR then sigmoid
        """
        super(MLPReadout, self).__init__()
        self.layer1 = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU() 
        self.out_act = act 
    
    def forward(self, x):
        ret = self.layer1(x)
        return self.out_act(ret)

