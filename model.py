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
        #a = torch.exp(self.w(h)) # E,1
        #print("\n"*2)
        #print("AAAAAAAAAAAAAAAAA")
        #print("h",h.size(),h)
        #print("a",a.size(),a)
        #a_sum = torch.mm(tgt,a) # N,E x E,1 = N,1
        #print("a_sum",a_sum.size(),a_sum)
        #a_tgt = torch.mm(tgt.t(), a_sum) # E,N x N,1 = E,1
        #print("a_tgt",a_tgt.size(),a_tgt)
        #a_att = a/(a_tgt+self.eps) # E,1
        #print("a_att",a_att.size(),a_att)
        #y = torch.mm(tgt,y * a_att) # N,1
        y = torch.mm(tgt,y) # N,1
        #print("y",y.size(),y)
        return y

if __name__ == "__main__":
    g = GATLayer(3,10)
    x = torch.tensor([[0.,0,0],[1,1,1]])
    src = torch.tensor([[0.],[1]])
    tgt = torch.tensor([[1.],[0]])
    y = g(x,src,tgt)
    print(y)
