from tqdm import tqdm
import pickle

import numpy as np
import scipy as sp
from skimage.segmentation import slic
import networkx as nx
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST

from model import GAT_MNIST

NP_TORCH_FLOAT_DTYPE = np.float32
NP_TORCH_LONG_DTYPE = np.int64

NUM_FEATURES = 3
NUM_CLASSES = 10

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
    
def save_model(fname, model):
    torch.save(model.state_dict(),"{fname}.pt".format(fname=fname))
    
def load_model(fname, model):
    model.load_state_dict(torch.load("{fname}.pt".format(fname=fname)))

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

def train(model, optimiser, images, labels, train_idx, use_cuda, batch_size=1, disable_tqdm=False):
    train_losses = []
    train_accs = []
    
    indexes = train_idx[np.random.permutation(len(train_idx))]
    
    for b in tqdm(range(0,len(indexes),batch_size), total=len(indexes)/batch_size, desc="Instances ", disable=disable_tqdm):
    
        optimiser.zero_grad()
        
        batch_indexes = indexes[b:b+batch_size]
        
        with multiprocessing.Pool(16) as p:
            graphs = p.map(get_graph_from_image, images[batch_indexes])
            
        batch_labels = labels[batch_indexes]
        pyt_labels = torch.tensor(batch_labels)
        
        h,adj,src,tgt,Msrc,Mtgt,Mgraph = batch_graphs(graphs)
        h,adj,src,tgt,Msrc,Mtgt,Mgraph = map(torch.tensor,(h,adj,src,tgt,Msrc,Mtgt,Mgraph))
        
        if use_cuda:
            h,adj,src,tgt,Msrc,Mtgt,Mgraph,pyt_labels = map(to_cuda,(h,adj,src,tgt,Msrc,Mtgt,Mgraph,pyt_labels))
            
        y = model(h,adj,src,tgt,Msrc,Mtgt,Mgraph)
        loss = F.cross_entropy(input=y,target=pyt_labels)
        
        pred = torch.argmax(y,dim=1).detach().cpu().numpy()
        acc = np.sum((pred==batch_labels).astype(float)) / batch_labels.shape[0]
        mode = sp.stats.mode(pred)
        
        tqdm.write(
              "{loss:.4f}\t{acc:.2f}%\t{mode} (x{modecount})".format(
                  loss=loss.item(),
                  acc=100*acc,
                  mode=mode[0][0],
                  modecount=mode[1][0],
              )
        )
        
        loss.backward()
        optimiser.step()
        
        train_losses.append(loss.detach().cpu().item())
        train_accs.append(acc)
        
    return train_losses, train_accs

def test(model, images, labels, indexes, use_cuda, desc="Test ", disable_tqdm=False):
    test_accs = []
    for i in tqdm(range(len(indexes)), total=len(indexes), desc=desc, disable=disable_tqdm):
        with torch.no_grad():
            idx = indexes[i]
        
            graphs = [get_graph_from_image(images[idx])]
            batch_labels = labels[idx:idx+1]
            pyt_labels = torch.tensor(batch_labels)
            
            h,adj,src,tgt,Msrc,Mtgt,Mgraph = batch_graphs(graphs)
            h,adj,src,tgt,Msrc,Mtgt,Mgraph = map(torch.tensor,(h,adj,src,tgt,Msrc,Mtgt,Mgraph))
            
            if use_cuda:
                h,adj,src,tgt,Msrc,Mtgt,Mgraph,pyt_labels = map(to_cuda,(h,adj,src,tgt,Msrc,Mtgt,Mgraph,pyt_labels))
            
            y = model(h,adj,src,tgt,Msrc,Mtgt,Mgraph)
            
            pred = torch.argmax(y,dim=1).detach().cpu().numpy()
            acc = np.sum((pred==batch_labels).astype(float)) / batch_labels.shape[0]
            
            test_accs.append(acc)
    return test_accs
