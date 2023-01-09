import torch
import dgl
import os
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import numpy as np
from utils.RecDatasets import *

class Collator(object):
    def __init__(self):
        pass

    def collate(self, batch):
        batch_graphs, batch_labels = map(list, zip(*batch))
        batch_graphs = dgl.batch(batch_graphs)
        batch_labels = torch.stack(batch_labels)
        return batch_graphs, batch_labels

def load_data(dataset, batch_size, num_workers=8, path='../data', include_val=False):
    path = os.path.join(path, dataset, 'feateng_data')

    target_train = os.path.join(path, 'target_train.csv')
    train_knn_neighbors = os.path.join(path, 'search_res_col_train.txt')

    if include_val:
        target_val = os.path.join(path, 'target_val.csv')
        val_knn_neighbors = os.path.join(path, 'search_res_col_val.txt')

    target_test = os.path.join(path, 'target_test.csv')
    test_knn_neighbors = os.path.join(path, 'search_res_col_test.txt')

    print('Loading retrieve pool...')
    retrieve_pool = os.path.join(path, 'search_pool.csv')
    retrieve_pool = pd.read_csv(retrieve_pool, header= None, sep=',', encoding='latin-1').values
    
    if dataset == 'ml-1m':
        used_fields = np.arange(7)
    elif dataset == 'lastfm':
        used_fields = np.asarray([0, 2, 5, 6, 7, 8])
    else:
        raise NotImplementedError
        
    train_dataset = RecDataset(target_train, train_knn_neighbors, retrieve_pool, used_fields)
    if include_val:
        val_dataset = RecDataset(target_val, val_knn_neighbors, retrieve_pool, used_fields)
    test_dataset = RecDataset(target_test, test_knn_neighbors, retrieve_pool, used_fields)
    
    collator = Collator()
    train_loader= DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collator.collate)
    if include_val:
        val_loader= DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collator.collate)
    test_loader= DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collator.collate)
    
    if include_val:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, test_loader
    
