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
    
    def forward(self,x,src,tgt):
        """
        features -> N,i node features
        src -> N,E adjacency matrix from source nodes to edges
        tgt -> N,E adjacency matrix from target nodes to edges
        """
        hsrc = torch.matmul(src.t(),x) # E,i
        htgt = torch.mm(tgt.t(),x) # E,i
        h = torch.cat([hsrc,htgt],dim=1) # E,2i
        y = self.act(self.f(h)) # E,o
        # FIXME Manual softmax doesn't as expected numerically
        a = self.w(h) # E,1
        a_max = torch.mean(a,0,keepdim=True)#[0] + self.eps
        a_norm = a-a_max
        a_exp = torch.exp(a_norm)
        a_sum = torch.mm(tgt,a_exp) + 0 # N,E x E,1 = N,1
        a_tgt = torch.mm(tgt.t(), a_sum) # E,N x N,1 = E,1
        a_att = a_exp/(a_tgt+self.eps) # E,1
        o = torch.mm(tgt,y * a_att) # N,1
        
        print("hsrc ~ ",   hsrc.size(),   hsrc)
        assert not torch.isnan(hsrc).any()
        print("htgt ~ ",   htgt.size(),   htgt)
        assert not torch.isnan(htgt).any()
        print("h ~ ",      h.size(),      h)
        assert not torch.isnan(h).any()
        print("y ~ ",      y.size(),      y)
        assert not torch.isnan(y).any()
        print("a ~ ",      a.size(),      a)
        assert not torch.isnan(a).any()
        print("a_max ~ ",  a_max.size(),  a_max)
        assert not torch.isnan(a_max).any()
        print("a_norm ~ ", a_norm.size(), a_norm)
        assert not torch.isnan(a_norm).any()
        print("a_exp ~ ",  a_exp.size(),  a_exp)
        assert not torch.isnan(a_exp).any()
        print("a_sum ~ ",  a_sum.size(),  a_sum)
        assert not torch.isnan(a_sum).any()
        print("a_tgt ~ ",  a_tgt.size(),  a_tgt)
        assert not torch.isnan(a_tgt).any()
        print("a_att ~ ",  a_att.size(),  a_att)
        assert not torch.isnan(a_att).any()
        print("o ~ ",      o.size(),      o)
        assert not torch.isnan(o).any()
        
        return o

if __name__ == "__main__":
    g = GATLayer(3,10)
    x = torch.tensor([[0.,0,0],[1,1,1]])
    src = torch.tensor([[0.],[1]])
    tgt = torch.tensor([[1.],[0]])
    y = g(x,src,tgt)
    print(y)
