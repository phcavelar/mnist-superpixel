import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GATLayerAdj(nn.Module):
    """
    More didatic (and slower) GAT layer
    """

    def __init__(self,d_i,d_o,act=F.relu,eps=1e-6):
        super(GATLayerAdj,self).__init__()
        self.f = nn.Linear(2*d_i,d_o)
        self.w = nn.Linear(2*d_i,1)
        self.act = act
        self._init_weights()
        self.eps = torch.tensor([eps])
        
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)
    
    def forward(self,x,adj,src,tgt,Msrc,Mtgt):
        """
        features -> N,i node features
        adj -> N,N adjacency matrix
        src -> E,i source index for edges
        tgt -> E,i target index for edges
        Mtgt -> N,E adjacency matrix from source nodes to edges
        Mtgt -> N,E adjacency matrix from target nodes to edges
        """
        N = x.size()[0]
        hsrc = x.unsqueeze(0).expand(N,-1,-1) # 1,N,i
        htgt = x.unsqueeze(1).expand(-1,N,-1) # N,1,i
        
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

class GATLayerEdgeAverage(nn.Module):
    def __init__(self,d_i,d_o,act=F.relu,eps=1e-6):
        super(GATLayerEdgeAverage,self).__init__()
        self.f = nn.Linear(2*d_i,d_o)
        self.w = nn.Linear(2*d_i,1)
        self.act = act
        self._init_weights()
        self.eps = torch.tensor([eps])
        
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)
    
    def forward(self,x,adj,src,tgt,Msrc,Mtgt):
        """
        features -> N,i node features
        adj -> N,N adjacency matrix
        src -> E,i source index for edges
        tgt -> E,i target index for edges
        Mtgt -> N,E adjacency matrix from source nodes to edges
        Mtgt -> N,E adjacency matrix from target nodes to edges
        """
        hsrc = x[src] # E,i
        htgt = x[tgt] # E,i
        h = torch.cat([hsrc,htgt],dim=1) # E,2i
        y = self.act(self.f(h)) # E,o
        a = self.w(h) # E,1
        assert not torch.isnan(a).any()
        a_sum = torch.mm(Mtgt,a) + self.eps # N,E x E,1 = N,1
        assert not torch.isnan(a_sum).any()
        o = torch.mm(Mtgt,y * a) / a_sum # N,1
        assert not torch.isnan(o).any()

        return o

class GATLayerEdgeSoftmax(nn.Module):
    def __init__(self,d_i,d_o,act=F.relu,eps=1e-6):
        super(GATLayerEdgeSoftmax,self).__init__()
        self.f = nn.Linear(2*d_i,d_o)
        self.w = nn.Linear(2*d_i,1)
        self.act = act
        self._init_weights()
        self.eps = torch.tensor([eps])
        
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)
    
    def forward(self,x,adj,src,tgt,Msrc,Mtgt):
        """
        features -> N,i node features
        adj -> N,N adjacency matrix
        src -> E,i source index for edges
        tgt -> E,i target index for edges
        Mtgt -> N,E adjacency matrix from source nodes to edges
        Mtgt -> N,E adjacency matrix from target nodes to edges
        """
        hsrc = x[src] # E,i
        htgt = x[tgt] # E,i
        h = torch.cat([hsrc,htgt],dim=1) # E,2i
        y = self.act(self.f(h)) # E,o
        # FIXME Manual softmax doesn't as expected numerically
        a = self.w(h) # E,1
        assert not torch.isnan(a).any()
        a_base, _ = torch.max(a,0,keepdim=True)#[0] + self.eps
        assert not torch.isnan(a_base).any()
        a_norm = a-a_base
        assert not torch.isnan(a_norm).any()
        a_exp = torch.exp(a_norm)
        assert not torch.isnan(a_exp).any()
        a_sum = torch.mm(Mtgt,a_exp) + self.eps # N,E x E,1 = N,1
        assert not torch.isnan(a_sum).any()
        o = torch.mm(Mtgt,y * a_exp) / a_sum # N,1
        assert not torch.isnan(o).any()

        return o

if __name__ == "__main__":
    g = GATLayer(3,10)
    x = torch.tensor([[0.,0,0],[1,1,1]])
    adj = torch.tensor([[0.,1],[1,0]])
    y = g(x,adj)
    print(y)
