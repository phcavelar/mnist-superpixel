from tqdm import tqdm
import fire

import copy
import time

import numpy as np
import scipy as sp
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST

from model import GAT_MNIST
import util

to_cuda = util.to_cuda

def train_model(
        epochs,
        batch_size,
        use_cuda,
        dset_folder,
        disable_tqdm=False,
        ):
    print("Reading dataset")
    dset = MNIST(dset_folder,download=True)
    imgs = dset.data.unsqueeze(-1).numpy().astype(np.float64)
    labels = dset.targets.numpy()
    train_idx, valid_idx = map(np.array,util.split_dataset(labels))
    
    print("Processing images into graphs...", end="")
    ptime = time.time()
    with multiprocessing.Pool() as p:
        graphs = np.array(p.map(util.get_graph_from_image, imgs))
    del imgs
    ptime = time.time() - ptime
    print(" Took {ptime}s".format(ptime=ptime))

    
    
    
    
    model_args = []
    model_kwargs = {}
    model = GAT_MNIST(num_features=util.NUM_FEATURES, num_classes=util.NUM_CLASSES)
    if use_cuda:
        model = model.cuda()
    
    opt = torch.optim.Adam(model.parameters())
    
    best_valid_acc = 0.
    best_model = copy.deepcopy(model)
    
    last_epoch_train_loss = 0.
    last_epoch_train_acc = 0.
    last_epoch_valid_acc = 0.
    
    interrupted = False
    for e in tqdm(range(epochs), total=epochs, desc="Epoch ", disable=disable_tqdm,):
        try:
            train_losses, train_accs = util.train(model, opt, graphs, labels, train_idx, batch_size=batch_size, use_cuda=use_cuda, disable_tqdm=disable_tqdm,)
            
            last_epoch_train_loss = np.mean(train_losses)
            last_epoch_train_acc = 100*np.mean(train_accs)
        except KeyboardInterrupt:
            print("Training interrupted!")
            interrupted = True
        
        valid_accs = util.test(model,graphs,labels,valid_idx,use_cuda,desc="Validation ", disable_tqdm=disable_tqdm,)
                
        last_epoch_valid_acc = 100*np.mean(valid_accs)
        
        if last_epoch_valid_acc>best_valid_acc:
            best_valid_acc = last_epoch_valid_acc
            best_model = copy.deepcopy(model)
        
        tqdm.write("EPOCH SUMMARY {loss:.4f} {t_acc:.2f}% {v_acc:.2f}%".format(loss=last_epoch_train_loss, t_acc=last_epoch_train_acc, v_acc=last_epoch_valid_acc))
        
        if interrupted:
            break
    
    util.save_model("best",best_model)
    util.save_model("last",model)


def test_model(
        use_cuda,
        dset_folder,
        disable_tqdm=False,
        ):
    best_model = GAT_MNIST(num_features=util.NUM_FEATURES, num_classes=util.NUM_CLASSES)
    util.load_model("best",best_model)
    if use_cuda:
        best_model = best_model.cuda()
    
    test_dset = MNIST(dset_folder,train=False,download=True)
    test_imgs = test_dset.data.unsqueeze(-1).numpy().astype(np.float64)
    with multiprocessing.Pool() as p:
        test_graphs = np.array(p.map(util.get_graph_from_image, test_imgs))
    del test_imgs
    test_labels = test_dset.targets.numpy()
    
    test_accs = util.test(best_model, test_graphs, test_labels, list(range(len(test_labels))), use_cuda, desc="Test ", disable_tqdm=disable_tqdm,)
    test_acc = 100*np.mean(test_accs)
    print("TEST RESULTS: {acc:.2f}%".format(acc=test_acc))

def main(
        train:bool=False,
        test:bool=False,
        epochs:int=100,
        batch_size:int=32,
        use_cuda:bool=True,
        disable_tqdm:bool=False,
        dset_folder:str = "./mnist"
        ):
    use_cuda = use_cuda and torch.cuda.is_available()
    if train:
        train_model(
                epochs = epochs,
                batch_size = batch_size,
                use_cuda = use_cuda,
                dset_folder = dset_folder,
                disable_tqdm = disable_tqdm,
                )
    if test:
        test_model(
                use_cuda=use_cuda,
                dset_folder = dset_folder,
                disable_tqdm = disable_tqdm,
                )

if __name__ == "__main__":
    fire.Fire(main)
