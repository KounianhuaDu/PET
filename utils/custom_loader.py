import torch
import dgl
import os
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.nn.functional as F
import pickle as pkl
import random
import pandas as pd
from utils import config
from tqdm import tqdm, trange
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
#dynamic sampler

used_fields = np.arange(7)

class RecDataset(Dataset):
    def __init__(self, target_data, neighbor_idxs, retrieve_pool):
        print('Loading target data...')
        self.X, self.y = [], []
        with open(target_data, 'r') as f:
            for l in f:
                line = list(map(int, l.split(',')))
                self.X.append(line[:-1])
                self.y.append(line[-1])
        self.X, self.y = np.asarray(self.X, dtype=list), np.asarray(self.y, dtype=int)
        self.len = len(self.y)

        print('Loading KNN indices...')
        with open(neighbor_idxs, 'r') as f:
            self.knn_idxs = [list(map(int, l.split(','))) for l in tqdm(f.readlines())]

        self.X_ret, self.y_ret = retrieve_pool
        self.X_ret, self.y_ret = np.asarray(self.X_ret, dtype=list), np.asarray(self.y_ret, dtype=int)

        self.feat_num = 15226

    def construct_graph(self, index):
        neighbors = self.knn_idxs[index]
        
        num_target = 1
        X_tar = self.X[index]
        X_ret, y_ret = self.X_ret[neighbors], self.y_ret[neighbors]
        
        target_ids, repeats = [X_tar[0]], [len(X_tar)]
        for x in X_ret:
            target_ids.append(x[0])
            repeats.append(len(x))
        
        # For target, the label value is initialized as 2 (the padding idx in the label embedding).
        y_all = np.concatenate([[2], y_ret], axis=0)
        
        num_instances = y_all.shape[0]
        feats = np.concatenate([X_tar, np.concatenate(X_ret, axis=0)], axis=0) + num_instances
        instances = np.repeat(np.arange(num_instances), repeats)
        edge_src = np.concatenate([feats, instances], axis=0)
        edge_dst = np.concatenate([instances, feats], axis=0)

        #re-index
        org_nids = np.sort(np.unique(edge_src)).reshape(-1)
        nid2idx = pd.Series(np.arange(org_nids.shape[0]), index=org_nids)
        
        edge_src = nid2idx[edge_src].values
        edge_dst = nid2idx[edge_dst].values
        first_feature_ids = org_nids - num_instances
        
        g = dgl.graph((edge_src, edge_dst))
        
        # indicators of the target, retrieved instances, 1st order feature nodes, and 2nd order feature nodes
        org_nids = first_feature_ids
        
        labels = np.concatenate((y_all, [2]*(g.num_nodes()-num_instances)), axis=0)
        g.ndata['label'] = torch.tensor(labels).long()
        g.ndata['org_nid'] = torch.tensor(org_nids).long()
        g.ndata['target_id'] = torch.zeros((g.num_nodes(), ), dtype=torch.long)
        g.ndata['target_id'][:num_instances] = torch.tensor(target_ids)
        g.ndata['is_target'] = (g.nodes() < num_target)
        g.ndata['is_instance'] = (g.nodes() < num_instances)
        g.ndata['is_feat'] = (g.nodes() >= num_instances)
        return g


    def __getitem__(self, index):
        g = self.construct_graph(index)
        return g, torch.tensor(self.y[index])

    def __len__(self):
        return self.len 

class Collator(object):
    def __init__(self):
        pass

    def collate(self, batch):
        batch_graphs, batch_labels = map(list, zip(*batch))
        batch_graphs = dgl.batch(batch_graphs)
        batch_labels = torch.stack(batch_labels)
        return batch_graphs, batch_labels

def load_data(dataset, batch_size, num_workers=8, path='../data'):
    path = os.path.join(path, dataset)

    target_train = os.path.join(path, 'target_train.csv')
    train_knn_neighbors = os.path.join(path, 'search_res_col_train.txt')
    
    target_valid = os.path.join(path, 'target_valid.csv')
    valid_knn_neighbors = os.path.join(path, 'search_res_col_valid.txt')

    target_test = os.path.join(path, 'target_test.csv')
    test_knn_neighbors = os.path.join(path, 'search_res_col_test.txt')

    print('Loading retrieve pool...')
    retrieve_pool = os.path.join(path, 'target_train.csv')
    X_ret, y_ret = [], []
    with open(retrieve_pool, 'r') as f:
        for l in f:
            line = list(map(int, l.split(',')))
            X_ret.append(line[:-1])
            y_ret.append(line[-1])
    retrieve_pool = (X_ret, y_ret)
    
    train_dataset = RecDataset(target_train, train_knn_neighbors, retrieve_pool)
    valid_dataset = RecDataset(target_valid, valid_knn_neighbors, retrieve_pool)
    test_dataset = RecDataset(target_test, test_knn_neighbors, retrieve_pool)
    
    collator = Collator()
    train_loader= DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collator.collate)
    valid_loader= DataLoader(
        dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collator.collate)
    test_loader= DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collator.collate)
    
    return train_loader, valid_loader, test_loader
    