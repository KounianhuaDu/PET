import torch
import dgl
import os
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

class RecDataset(Dataset):
    def __init__(self, target_data, neighbor_idxs, retrieve_pool, used_fields):
        print('Loading target data...')
        self.lines = pd.read_csv(target_data, header= None, sep=',', encoding='latin-1').values
        self.X, self.y = self.lines[:, used_fields], self.lines[:, -1]
        self.len = len(self.y)

        print('Loading KNN indices...')
        with open(neighbor_idxs, 'r') as f:
            self.knn_idxs = [list(map(int, l.split(','))) for l in tqdm(f.readlines())]

        self.retrieve_pool = retrieve_pool
        self.X_ret, self.y_ret = self.retrieve_pool[:, used_fields], self.retrieve_pool[:, -1]


    def construct_graph(self, index):
        neighbors = self.knn_idxs[index]
        
        num_target = 1
        X_tar = self.X[index]
        X_ret, y_ret = self.X_ret[neighbors], self.y_ret[neighbors]
        X_all = np.concatenate([X_tar.reshape(1, -1), X_ret], axis=0)
        # For target, the label value is initialized as 2 (the padding idx in the label embedding).
        y_all = np.concatenate([[2], y_ret], axis=0)
        
        num_instances = y_all.shape[0]
        feats = X_all.reshape(-1) + num_instances
        instances = np.repeat(np.arange(num_instances), X_all.shape[1])
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
        g.ndata['target_id'][:num_instances] = torch.tensor(X_all[:, 1])
        g.ndata['is_target'] = (g.nodes() < num_target)
        g.ndata['is_instance'] = (g.nodes() < num_instances)
        g.ndata['is_feat'] = (g.nodes() >= num_instances)
        return g


    def __getitem__(self, index):
        g = self.construct_graph(index)
        return g, torch.tensor(self.y[index])

    def __len__(self):
        return self.len 


    