import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GATLayer(nn.Module):
    def __init__(self,d_i,d_o,act=F.relu,eps=1e-6):
        super(GATLayer,self).__init__()
        self.f = nn.Linear(2*d_i,d_o)
        self.w = nn.Linear(2*d_i,1)
        self.act = act
        self._init_weights()
        self.eps = torch.tensor([eps])
        
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)
    
    def forward(self,x,adj):
        """
        features -> N,i node features
        src -> N,E adjacency matrix from source nodes to edges
        tgt -> N,E adjacency matrix from target nodes to edges
        """
        zero = torch.zeros_like(adj).unsqueeze(-1)
        hsrc = x.unsqueeze(0) + zero # 1,N,i
        htgt = x.unsqueeze(1) + zero # N,1,i
        h = torch.cat([hsrc,htgt],dim=2) # N,N,2i
        
        a = self.w(h) # N,N,1
        a_sqz = a.squeeze(2) # N,N
        a_zro = -1e16*torch.ones_like(a_sqz) # N,N
        a_msk = torch.where(adj>0,a_sqz,a_zro) # N,N
        a_att = F.softmax(a_msk,dim=1) # N,N
        
        y = self.act(self.f(h)) # N,N,o
        y_att = a_att.unsqueeze(-1)*y # N,N,o
        o = y_att.sum(dim=1).squeeze()
        
        return o

if __name__ == "__main__":
    g = GATLayer(3,10)
    x = torch.tensor([[0.,0,0],[1,1,1]])
    adj = torch.tensor([[0.,1],[1,0]])
    y = g(x,adj)
    print(y)
