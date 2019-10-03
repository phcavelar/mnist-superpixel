from tqdm import tqdm

import numpy as np
import scipy as sp
from skimage.segmentation import slic
import networkx as nx
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST

import model

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
            #np.reshape(pos_mean, -1),
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

    for i in range(bneighbors.shape[1]):
        if bneighbors[0,i] != bneighbors[1,i]:
            G.add_edge(bneighbors[0,i],bneighbors[1,i])
        #end if
    return G

def batch_graphs(gs):
    NUM_FEATURES = len(gs[0].nodes[0]["features"])
    G = len(gs)
    N = sum(len(g.nodes) for g in gs)
    M = 2*sum(len(g.edges) for g in gs)
    adj = np.zeros([N,N])
    src = np.zeros([N,M])
    tgt = np.zeros([N,M])
    graph = np.zeros([N,G])
    h = np.zeros([N,NUM_FEATURES])
    
    n_acc = 0
    m_acc = 0
    for g_idx, g in enumerate(gs):
        n = len(g.nodes)
        m = 2*len(g.edges)
        
        for e,(s,t) in enumerate(g.edges):
            adj[n_acc+s,n_acc+t] = 1
            adj[n_acc+t,n_acc+s] = 1
            
            src[n_acc+s,m_acc+e*1] = 1
            tgt[n_acc+t,m_acc+e*1] = 1
            src[n_acc+s,m_acc+e*2] = 1
            tgt[n_acc+t,m_acc+e*2] = 1
            
        for i in g.nodes:
            h[n_acc+i,:] = g.nodes[i]["features"]
            graph[n_acc+i,g_idx] = 1
        
        n_acc += n
        m_acc += m
    return map(lambda x:x.astype(np.float32),(h,adj,src,tgt,graph))
    
def to_cuda(x):
    return x.cuda()

if __name__ == "__main__":
    USE_CUDA = True and torch.cuda.is_available()
    dset = MNIST("./mnist",download=True)
    imgs = dset.train_data.unsqueeze(-1).numpy().astype(np.float64)
    labels = dset.train_labels.numpy()
    
    epochs = 100
    batch_size = 10
    
    NUM_FEATURES = 1
    NUM_CLASSES = 10
    
    gat1 = model.GATLayer(NUM_FEATURES,32).cuda()
    gat2 = model.GATLayer(32,64).cuda()
    gat3 = model.GATLayer(64,NUM_CLASSES,act=lambda x:x).cuda()
    if USE_CUDA:
        gat1,gat2,gat3 = map(to_cuda,(gat1,gat2,gat3))
    
    opt = torch.optim.Adam([*gat1.parameters(),*gat2.parameters(),*gat3.parameters()])
    
    layers = [gat1,gat2,gat3]
    
    for e in tqdm(range(epochs), total=epochs, desc="Epoch "):
    
        losses = []
        accs = []
        
        indexes = np.random.permutation(len(dset))
        
        for b in tqdm(range(0,len(dset),batch_size), total=len(dset)/batch_size, desc="Instances "):
        
            opt.zero_grad()
            
            batch_indexes = indexes[b:b+batch_size]
            
            with multiprocessing.Pool(16) as p:
                graphs = p.map(get_graph_from_image, imgs[batch_indexes])
                
            batch_labels = labels[batch_indexes]
            pyt_labels = torch.tensor(batch_labels)
            
            h,adj,src,tgt,graph = batch_graphs(graphs)
            h,adj,graph = map(torch.tensor,(h,adj,graph))
            
            if USE_CUDA:
                h,adj,graph,pyt_labels = map(to_cuda,(h,adj,graph,pyt_labels))
            
            x = h
            for l in layers:
                x = l(x,adj)
                
            y = torch.mm(graph.t(),x)
            loss = F.cross_entropy(input=y,target=pyt_labels)
            
            pred = torch.argmax(y,dim=1).detach().cpu().numpy()
            acc = np.sum((pred==batch_labels).astype(float)) / batch_labels.shape[0]
            mode = sp.stats.mode(pred)
            
            tqdm.write("{loss:.4f}\t{acc:.2f}%\t{mode} (x{modecount})".format(loss=loss.item(), acc=100*acc, mode=mode[0][0], modecount=mode[1][0]))
            
            loss.backward()
            opt.step()
            
            losses.append(loss.detach().cpu().item())
            accs.append(acc)
        tqdm.write("EPOCH SUMMARY {loss:.4f} {acc:.2f}%".format(loss=np.mean(losses), acc=100*np.mean(accs)))
