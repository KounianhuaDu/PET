import torch
import dgl
import os
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import numpy as np
from utils.SequentialDatasets import *

class Collator(object):
    def __init__(self):
        pass

    def collate(self, batch):
        batch_graphs, batch_labels = map(list, zip(*batch))
        batch_graphs = dgl.batch(batch_graphs)
        batch_labels = torch.stack(batch_labels)
        return batch_graphs, batch_labels

def load_data(dataset, batch_size, ret_size=10, num_workers=8, path='../data'):
    path = os.path.join(path, dataset, 'feateng_data')

    target_train = os.path.join(path, 'target_train.csv')
    target_test = os.path.join(path, 'target_test.csv')

    train_knn_neighbors = os.path.join(path, f'search_res_col_train_{ret_size}.txt')
    test_knn_neighbors = os.path.join(path, f'search_res_col_test_{ret_size}.txt')

    retrieval_pool = os.path.join(path, 'search_pool.csv')
    
    if dataset == 'tmall':
        train_dataset = TmallDataset(target_train, train_knn_neighbors, retrieval_pool)
        test_dataset = TmallDataset(target_test, test_knn_neighbors, retrieval_pool)
    elif dataset == 'taobao':
        train_dataset = TaobaoDataset(target_train, train_knn_neighbors, retrieval_pool)
        test_dataset = TaobaoDataset(target_test, test_knn_neighbors, retrieval_pool)
    elif dataset == 'alipay':
        train_dataset = AlipayDataset(target_train, train_knn_neighbors, retrieval_pool)
        test_dataset = AlipayDataset(target_test, test_knn_neighbors, retrieval_pool)
    else:
        raise NotImplementedError
    
    collator = Collator()
    train_loader= DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collator.collate)
    test_loader= DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collator.collate)
        
    return train_loader, test_loader

