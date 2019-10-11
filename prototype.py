from tqdm import tqdm

import random
import copy
import numpy as np
import scipy as sp
from skimage.segmentation import slic
import networkx as nx
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST

from model import GATLayerEdgeSoftmax as GATLayer

NP_TORCH_FLOAT_DTYPE = np.float32
NP_TORCH_LONG_DTYPE = np.int64

def get_graph_from_image(image,desired_nodes=75): 
    # load the image and convert it to a floating point data type
    segments = slic(image, n_segments=desired_nodes, slic_zero = True)
    asegments = np.array(segments)

    num_nodes = np.max(asegments)
    nodes = {
        node: {
            "rgb_list": [],
            "pos_list": []
        } for node in range(num_nodes+1)
    }

    height = image.shape[0]
    width = image.shape[1]
    for y in range(height):
        for x in range(width):
            node = asegments[y,x]
            rgb = image[y,x,:]
            pos = np.array([float(x)/width,float(y)/height])
            nodes[node]["rgb_list"].append(rgb)
            nodes[node]["pos_list"].append(pos)
        #end for
    #end for
    
    G = nx.Graph()
    
    for node in nodes:
        nodes[node]["rgb_list"] = np.stack(nodes[node]["rgb_list"])
        nodes[node]["pos_list"] = np.stack(nodes[node]["pos_list"])
        # rgb
        rgb_mean = np.mean(nodes[node]["rgb_list"], axis=0)
        #rgb_std = np.std(nodes[node]["rgb_list"], axis=0)
        #rgb_gram = np.matmul( nodes[node]["rgb_list"].T, nodes[node]["rgb_list"] ) / nodes[node]["rgb_list"].shape[0]
        # Pos
        pos_mean = np.mean(nodes[node]["pos_list"], axis=0)
        #pos_std = np.std(nodes[node]["pos_list"], axis=0)
        #pos_gram = np.matmul( nodes[node]["pos_list"].T, nodes[node]["pos_list"] ) / nodes[node]["pos_list"].shape[0]
        # Debug
        
        features = np.concatenate(
          [
            np.reshape(rgb_mean, -1),
            #np.reshape(rgb_std, -1),
            #np.reshape(rgb_gram, -1),
            np.reshape(pos_mean, -1),
            #np.reshape(pos_std, -1),
            #np.reshape(pos_gram, -1)
          ]
        )
        G.add_node(node, features = list(features))
    #end
    
    # From https://stackoverflow.com/questions/26237580/skimage-slic-getting-neighbouring-segments
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
    vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    # Adjacency loops
    for i in range(bneighbors.shape[1]):
        if bneighbors[0,i] != bneighbors[1,i]:
            G.add_edge(bneighbors[0,i],bneighbors[1,i])
    
    # Self loops
    for node in nodes:
        G.add_edge(node,node)
    
    return G

def batch_graphs(gs):
    NUM_FEATURES = len(gs[0].nodes[0]["features"])
    G = len(gs)
    N = sum(len(g.nodes) for g in gs)
    M = 2*sum(len(g.edges) for g in gs)
    adj = np.zeros([N,N])
    src = np.zeros([M])
    tgt = np.zeros([M])
    Msrc = np.zeros([N,M])
    Mtgt = np.zeros([N,M])
    Mgraph = np.zeros([N,G])
    h = np.zeros([N,NUM_FEATURES])
    
    n_acc = 0
    m_acc = 0
    for g_idx, g in enumerate(gs):
        n = len(g.nodes)
        m = len(g.edges)
        
        for e,(s,t) in enumerate(g.edges):
            adj[n_acc+s,n_acc+t] = 1
            adj[n_acc+t,n_acc+s] = 1
            
            src[m_acc+e] = n_acc+s
            tgt[m_acc+e] = n_acc+t
            
            src[m_acc+m+e] = n_acc+t
            tgt[m_acc+m+e] = n_acc+s
            
            Msrc[n_acc+s,m_acc+e] = 1
            Mtgt[n_acc+t,m_acc+e] = 1
            
            Msrc[n_acc+t,m_acc+m+e] = 1
            Mtgt[n_acc+s,m_acc+m+e] = 1
            
        for i in g.nodes:
            h[n_acc+i,:] = g.nodes[i]["features"]
            Mgraph[n_acc+i,g_idx] = 1
        
        n_acc += n
        m_acc += 2*m
    return (
        h.astype(NP_TORCH_FLOAT_DTYPE),
        adj.astype(NP_TORCH_FLOAT_DTYPE),
        src.astype(NP_TORCH_LONG_DTYPE),
        tgt.astype(NP_TORCH_LONG_DTYPE),
        Msrc.astype(NP_TORCH_FLOAT_DTYPE),
        Mtgt.astype(NP_TORCH_FLOAT_DTYPE),
        Mgraph.astype(NP_TORCH_FLOAT_DTYPE)
    )
    
def to_cuda(x):
    return x.cuda()
    
def split_dataset(labels, valid_split=0.1):
    idx = np.random.permutation(len(labels))
    valid_idx = []
    train_idx = []
    label_count = [0 for _ in range(1+max(labels))]
    valid_count = [0 for _ in label_count]
    
    for i in idx:
        label_count[labels[i]] += 1
    
    
    for i in idx:
        l = labels[i]
        if valid_count[l] < label_count[l]*valid_split:
            valid_count[l] += 1
            valid_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, valid_idx
    
def test(images,labels,indexes,model, desc="Test "):
    test_accs = []
    for i in tqdm(range(len(indexes)), total=len(indexes), desc=desc):
        with torch.no_grad():
            idx = indexes[i]
        
            graphs = [get_graph_from_image(images[idx])]
            batch_labels = labels[idx:idx+1]
            pyt_labels = torch.tensor(batch_labels)
            
            h,adj,src,tgt,Msrc,Mtgt,Mgraph = batch_graphs(graphs)
            h,adj,src,tgt,Msrc,Mtgt,Mgraph = map(torch.tensor,(h,adj,src,tgt,Msrc,Mtgt,Mgraph))
            
            if USE_CUDA:
                h,adj,src,tgt,Msrc,Mtgt,Mgraph,pyt_labels = map(to_cuda,(h,adj,src,tgt,Msrc,Mtgt,Mgraph,pyt_labels))
            
            y = model(h,adj,src,tgt,Msrc,Mtgt,Mgraph)
            
            pred = torch.argmax(y,dim=1).detach().cpu().numpy()
            acc = np.sum((pred==batch_labels).astype(float)) / batch_labels.shape[0]
            
            test_accs.append(acc)
    return test_accs


class GAT_MNIST(nn.Module):
    
    def __init__(self,num_features,num_classes):
        super(GAT_MNIST,self).__init__()
        
        self.GAT_layer_sizes = [num_features,32,64,64]
        self.MLP_layer_sizes = [self.GAT_layer_sizes[-1],32,num_classes]
        self.MLP_acts = [F.relu,lambda x:x]
        
        self.GAT_layers = nn.ModuleList(
              [
                GATLayer(d_in,d_out)
                for d_in,d_out in zip(self.GAT_layer_sizes[:-1],self.GAT_layer_sizes[1:])
              ]
        )
        self.MLP_layers = nn.ModuleList(
              [
                nn.Linear(d_in,d_out)
                for d_in,d_out in zip(self.MLP_layer_sizes[:-1],self.MLP_layer_sizes[1:])
              ]
        )
    
    def forward(self,x,adj,src,tgt,Msrc,Mtgt,Mgraph):
        for l in self.GAT_layers:
            x = l(x,adj,src,tgt,Msrc,Mtgt)
        x = torch.mm(Mgraph.t(),x)
        for layer,act in zip(self.MLP_layers,self.MLP_acts):
            x = act(layer(x))
        return x

if __name__ == "__main__":
    USE_CUDA = True and torch.cuda.is_available()
    dset = MNIST("./mnist",download=True)
    imgs = dset.data.unsqueeze(-1).numpy().astype(np.float64)
    labels = dset.targets.numpy()
    
    train_idx, valid_idx = map(np.array,split_dataset(labels))
    
    epochs = 100
    batch_size = 32
    
    NUM_FEATURES = 3
    NUM_CLASSES = 10
    
    model = GAT_MNIST(num_features=NUM_FEATURES,num_classes=NUM_CLASSES)
    if USE_CUDA:
        model = model.cuda()
    
    opt = torch.optim.Adam(model.parameters())
    
    best_valid_acc = 0.
    best_model = copy.deepcopy(model)
    
    last_epoch_train_loss = 0.
    last_epoch_train_acc = 0.
    last_epoch_valid_acc = 0.
    
    interrupted = False
    for e in tqdm(range(epochs), total=epochs, desc="Epoch "):
        try:
            train_losses = []
            train_accs = []
            
            indexes = train_idx[np.random.permutation(len(train_idx))]
            
            for b in tqdm(range(0,len(indexes),batch_size), total=len(indexes)/batch_size, desc="Instances "):
            
                opt.zero_grad()
                
                batch_indexes = indexes[b:b+batch_size]
                
                with multiprocessing.Pool(16) as p:
                    graphs = p.map(get_graph_from_image, imgs[batch_indexes])
                    
                batch_labels = labels[batch_indexes]
                pyt_labels = torch.tensor(batch_labels)
                
                h,adj,src,tgt,Msrc,Mtgt,Mgraph = batch_graphs(graphs)
                h,adj,src,tgt,Msrc,Mtgt,Mgraph = map(torch.tensor,(h,adj,src,tgt,Msrc,Mtgt,Mgraph))
                
                if USE_CUDA:
                    h,adj,src,tgt,Msrc,Mtgt,Mgraph,pyt_labels = map(to_cuda,(h,adj,src,tgt,Msrc,Mtgt,Mgraph,pyt_labels))
                    
                y = model(h,adj,src,tgt,Msrc,Mtgt,Mgraph)
                loss = F.cross_entropy(input=y,target=pyt_labels)
                
                pred = torch.argmax(y,dim=1).detach().cpu().numpy()
                acc = np.sum((pred==batch_labels).astype(float)) / batch_labels.shape[0]
                mode = sp.stats.mode(pred)
                
                tqdm.write(
                      "{loss:.4f}\t{acc:.2f}%\t{mode} (x{modecount})\t\tLAST {eloss:.4f} {etacc:.2f}% {evacc:.2f}%".format(
                          loss=loss.item(),
                          acc=100*acc,
                          mode=mode[0][0],
                          modecount=mode[1][0],
                          eloss=last_epoch_train_loss,
                          etacc=last_epoch_train_acc,
                          evacc=last_epoch_valid_acc
                      )
                )
                
                loss.backward()
                opt.step()
                
                train_losses.append(loss.detach().cpu().item())
                train_accs.append(acc)
            
        except KeyboardInterrupt:
            print("Training interrupted!")
            interrupted = True
            
        valid_accs = test(imgs,labels,valid_idx,model,desc="Validation ")
                
        last_epoch_train_loss = np.mean(train_losses)
        last_epoch_train_acc = 100*np.mean(train_accs)
        last_epoch_valid_acc = 100*np.mean(valid_accs)
        
        if last_epoch_valid_acc>best_valid_acc:
            best_valid_acc = last_epoch_valid_acc
            best_model = copy.deepcopy(model)
        
        tqdm.write("EPOCH SUMMARY {loss:.4f} {t_acc:.2f}% {v_acc:.2f}%".format(loss=last_epoch_train_loss, t_acc=last_epoch_train_acc, v_acc=last_epoch_valid_acc))
        
        if interrupted:
            break
    
    test_dset = MNIST("./mnist",train=False,download=True)
    test_imgs = test_dset.data.unsqueeze(-1).numpy().astype(np.float64)
    test_labels = test_dset.targets.numpy()
    
    test_accs = test(test_imgs,test_labels,list(range(len(test_labels))),best_model,desc="Test ")
    last_epoch_acc = 100*np.mean(accs)
    print("TEST RESULTS: {acc:.2f}%".format(acc=last_epoch_acc))
