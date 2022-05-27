import torch
import dgl
import os
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import random
import pandas as pd
import numpy as np

class TmallDataset(Dataset):
    def __init__(self, target_data, neighbor_idxs, retrieve_pool, ret_size=None):
        
        self.ret_size = ret_size

        self.lines = pd.read_csv(target_data, header= None, sep=',', encoding='latin-1')
        self.feat1 = np.asarray(self.lines[0].astype(int).values)
        self.feat2 = np.asarray(self.lines[1].astype(int).values)
        self.feat3 = np.asarray(self.lines[2].astype(int).values)
        self.feat4 = np.asarray(self.lines[3].astype(int).values)
        self.feat5 = np.asarray(self.lines[4].astype(int).values)
        self.feat6 = np.asarray(self.lines[6].astype(int).values)
        self.feat7 = np.asarray(self.lines[7].astype(int).values)
        self.feat8 = np.asarray(self.lines[8].astype(int).values)
        self.label = np.asarray(self.lines[9].astype(int).values)
        self.len = len(self.lines)

        with open(neighbor_idxs, 'r') as f:
            self.knn_idxs = f.readlines()

        self.retrieval_pool = pd.read_csv(retrieve_pool, header= None, sep=',', encoding='latin-1')
        self.knn_feat1 = np.asarray(self.retrieval_pool[0].astype(int).values)
        self.knn_feat2 = np.asarray(self.retrieval_pool[1].astype(int).values)
        self.knn_feat3 = np.asarray(self.retrieval_pool[2].astype(int).values)
        self.knn_feat4 = np.asarray(self.retrieval_pool[3].astype(int).values)
        self.knn_feat5 = np.asarray(self.retrieval_pool[4].astype(int).values)
        self.knn_feat6 = np.asarray(self.retrieval_pool[6].astype(int).values)
        self.knn_feat7 = np.asarray(self.retrieval_pool[7].astype(int).values)
        self.knn_feat8 = np.asarray(self.retrieval_pool[8].astype(int).values)
        self.knn_label = np.asarray(self.retrieval_pool[9].astype(int).values)
        self.num_field = 8
        self.feat_num = 1529675

    def construct_graph(self, index):
        #idxs, neighbors, user, movie, genres, gender, age, occupation, label = items
        neighbor_idxs = self.knn_idxs[index]
        neighbors = list(map(int, neighbor_idxs.split(','))) 
        if self.ret_size != None:
            neighbors = neighbors[:self.ret_size]
        
        num_target = 1
        
        #for target, the label value is initialized as 2 (the padding idx in the label embedding).
        labels = [2]
        #obtain features for neighbors
        feat1 = np.concatenate(([self.feat1[index]], self.knn_feat1[neighbors]), axis=0)
        feat2 = np.concatenate(([self.feat2[index]], self.knn_feat2[neighbors]), axis=0)
        feat3 = np.concatenate(([self.feat3[index]], self.knn_feat3[neighbors]), axis=0)
        feat4 = np.concatenate(([self.feat4[index]], self.knn_feat4[neighbors]), axis=0)
        feat5 = np.concatenate(([self.feat5[index]], self.knn_feat5[neighbors]), axis=0)
        feat6 = np.concatenate(([self.feat6[index]], self.knn_feat6[neighbors]), axis=0)
        feat7 = np.concatenate(([self.feat7[index]], self.knn_feat7[neighbors]), axis=0)
        feat8 = np.concatenate(([self.feat8[index]], self.knn_feat8[neighbors]), axis=0)
        
        labels = np.concatenate((labels, self.knn_label[neighbors]), axis=0)
        #re-index the instances
        num_instances = feat1.shape[0]
        new_instances = np.arange(num_instances)
         
        #src&dst for the first order data-feature edges
        edge_src = np.concatenate(
            (
                feat1 + num_instances, feat2 + num_instances, feat3 + num_instances,  
                feat4 + num_instances, feat5 + num_instances, feat6 + num_instances,
                feat7 + num_instances, feat8 + num_instances, 
                new_instances, new_instances, new_instances, new_instances, 
                new_instances, new_instances, new_instances, new_instances 
                
            )
        )
        edge_dst = np.concatenate(
            (
                new_instances, new_instances, new_instances, new_instances, 
                new_instances, new_instances, new_instances, new_instances,
                feat1 + num_instances, feat2 + num_instances, feat3 + num_instances,  
                feat4 + num_instances, feat5 + num_instances, feat6 + num_instances,
                feat7 + num_instances, feat8 + num_instances
            )
        )
        #re-index
        org_nids = np.sort(np.unique(edge_src)).reshape(-1)
        max_nid = org_nids.shape[0]
        nid2idx = np.zeros((org_nids[-1]+1, ), dtype=int)
        
        nid2idx[org_nids] = np.arange(org_nids.shape[0])
        
        edge_src = nid2idx[edge_src]
        edge_dst = nid2idx[edge_dst]
        first_feature_ids = org_nids - num_instances
        
        g = dgl.graph((edge_src, edge_dst))
        
        # indicators of the target, retrieved instances, 1st order feature nodes, and 2nd order feature nodes
        org_nids = first_feature_ids
        
        labels = np.concatenate((labels, [2]*(g.num_nodes()-len(labels))), axis=0)
        g.ndata['label'] = torch.tensor(labels).long()
        g.ndata['org_nid'] = torch.tensor(org_nids).long()
        
        g.ndata['is_target'] = (g.nodes() < num_target)
        g.ndata['is_instance'] = (g.nodes() < num_instances)
        g.ndata['is_feat'] = (g.nodes() >= num_instances)
        return g

    def __getitem__(self, index):
        g = self.construct_graph(index)
        return g, torch.tensor(self.label[index])

    def __len__(self):
        return self.len 

class TaobaoDataset(Dataset):
    def __init__(self, target_data, neighbor_idxs, retrieve_pool, ret_size=None):

        self.ret_size = ret_size

        self.lines = pd.read_csv(target_data, header= None, sep=',', encoding='latin-1')
        self.feat1 = np.asarray(self.lines[0].astype(int).values)
        self.feat2 = np.asarray(self.lines[1].astype(int).values)
        self.feat3 = np.asarray(self.lines[2].astype(int).values)
        self.feat4 = np.asarray(self.lines[3].astype(int).values)
        
        self.label = np.asarray(self.lines[5].astype(int).values)
        self.len = len(self.lines)
        with open(neighbor_idxs, 'r') as f:
            self.knn_idxs = f.readlines()
        self.retrieval_pool = pd.read_csv(retrieve_pool, header= None, sep=',', encoding='latin-1')
        self.knn_feat1 = np.asarray(self.retrieval_pool[0].astype(int).values)
        self.knn_feat2 = np.asarray(self.retrieval_pool[1].astype(int).values)
        self.knn_feat3 = np.asarray(self.retrieval_pool[2].astype(int).values)
        self.knn_feat4 = np.asarray(self.retrieval_pool[3].astype(int).values)
        
        self.knn_label = np.asarray(self.retrieval_pool[5].astype(int).values)
        
    def construct_graph(self, index):
        #idxs, neighbors, user, movie, genres, gender, age, occupation, label = items
        neighbor_idxs = self.knn_idxs[index]
        neighbors = list(map(int, neighbor_idxs.split(','))) 
        if self.ret_size != None:
            neighbors = neighbors[:self.ret_size]
        
        num_target = 1
        
        #for target, the label value is initialized as 2 (the padding idx in the label embedding).
        labels = [2]
        #obtain features for neighbors
        feat1 = np.concatenate(([self.feat1[index]], self.knn_feat1[neighbors]), axis=0)
        feat2 = np.concatenate(([self.feat2[index]], self.knn_feat2[neighbors]), axis=0)
        feat3 = np.concatenate(([self.feat3[index]], self.knn_feat3[neighbors]), axis=0)
        feat4 = np.concatenate(([self.feat4[index]], self.knn_feat4[neighbors]), axis=0)
        
        labels = np.concatenate((labels, self.knn_label[neighbors]), axis=0)
        #re-index the instances
        num_instances = feat1.shape[0]
        new_instances = np.arange(num_instances)
         
        #src&dst for the first order data-feature edges
        edge_src = np.concatenate(
            (
                feat1 + num_instances, feat2 + num_instances, feat3 + num_instances,  
                feat4 + num_instances,  
                new_instances, new_instances, new_instances, new_instances
                
            )
        )
        edge_dst = np.concatenate(
            (
                new_instances, new_instances, new_instances, new_instances, 
                feat1 + num_instances, feat2 + num_instances, feat3 + num_instances,  
                feat4 + num_instances
            )
        )
        #re-index
        org_nids = np.sort(np.unique(edge_src)).reshape(-1)
        max_nid = org_nids.shape[0]
        nid2idx = np.zeros((org_nids[-1]+1, ), dtype=int)
        
        nid2idx[org_nids] = np.arange(org_nids.shape[0])
        
        edge_src = nid2idx[edge_src]
        edge_dst = nid2idx[edge_dst]
        first_feature_ids = org_nids - num_instances
        
        g = dgl.graph((edge_src, edge_dst))
        
        # indicators of the target, retrieved instances, 1st order feature nodes, and 2nd order feature nodes
        org_nids = first_feature_ids
        
        labels = np.concatenate((labels, [2]*(g.num_nodes()-len(labels))), axis=0)
        g.ndata['label'] = torch.tensor(labels).long()
        g.ndata['org_nid'] = torch.tensor(org_nids).long()
        
        g.ndata['is_target'] = (g.nodes() < num_target)
        g.ndata['is_instance'] = (g.nodes() < num_instances)
        g.ndata['is_feat'] = (g.nodes() >= num_instances)
        return g

    def __getitem__(self, index):
        g = self.construct_graph(index)
        return g, torch.tensor(self.label[index])

    def __len__(self):
        return self.len 

class AlipayDataset(Dataset):
    def __init__(self, target_data, neighbor_idxs, retrieve_pool, ret_size=None):

        self.ret_size = ret_size
        
        self.lines = pd.read_csv(target_data, header= None, sep=',', encoding='latin-1')
        self.feat1 = np.asarray(self.lines[0].astype(int).values)
        self.feat2 = np.asarray(self.lines[1].astype(int).values)
        self.feat3 = np.asarray(self.lines[2].astype(int).values)
        self.feat4 = np.asarray(self.lines[3].astype(int).values)
        self.feat5 = np.asarray(self.lines[4].astype(int).values)
        
        self.label = np.asarray(self.lines[6].astype(int).values)
        self.len = len(self.lines)

        with open(neighbor_idxs, 'r') as f:
            self.knn_idxs = f.readlines()

        self.retrieval_pool = pd.read_csv(retrieve_pool, header= None, sep=',', encoding='latin-1')
        self.knn_feat1 = np.asarray(self.retrieval_pool[0].astype(int).values)
        self.knn_feat2 = np.asarray(self.retrieval_pool[1].astype(int).values)
        self.knn_feat3 = np.asarray(self.retrieval_pool[2].astype(int).values)
        self.knn_feat4 = np.asarray(self.retrieval_pool[3].astype(int).values)
        self.knn_feat5 = np.asarray(self.retrieval_pool[4].astype(int).values)

        self.knn_label = np.asarray(self.retrieval_pool[6].astype(int).values)
        

    def construct_graph(self, index):
        #idxs, neighbors, user, movie, genres, gender, age, occupation, label = items
        neighbor_idxs = self.knn_idxs[index]
        neighbors = list(map(int, neighbor_idxs.split(','))) 
        if self.ret_size != None:
            neighbors = neighbors[:self.ret_size]
        
        num_target = 1
        
        #for target, the label value is initialized as 2 (the padding idx in the label embedding).
        labels = [2]

        #obtain features for neighbors
        feat1 = np.concatenate(([self.feat1[index]], self.knn_feat1[neighbors]), axis=0)
        feat2 = np.concatenate(([self.feat2[index]], self.knn_feat2[neighbors]), axis=0)
        feat3 = np.concatenate(([self.feat3[index]], self.knn_feat3[neighbors]), axis=0)
        feat4 = np.concatenate(([self.feat4[index]], self.knn_feat4[neighbors]), axis=0)
        feat5 = np.concatenate(([self.feat5[index]], self.knn_feat5[neighbors]), axis=0)
        
        labels = np.concatenate((labels, self.knn_label[neighbors]), axis=0)

        #re-index the instances
        num_instances = feat1.shape[0]
        new_instances = np.arange(num_instances)

        #src&dst for the first order data-feature edges
        edge_src = np.concatenate(
            (
                feat1 + num_instances, feat2 + num_instances, feat3 + num_instances,  
                feat4 + num_instances, feat5 + num_instances,  
                new_instances, new_instances, new_instances, new_instances, new_instances
                
            )
        )
        edge_dst = np.concatenate(
            (
                new_instances, new_instances, new_instances, new_instances, new_instances,
                feat1 + num_instances, feat2 + num_instances, feat3 + num_instances,  
                feat4 + num_instances, feat5 + num_instances
            )
        )

        #re-index
        org_nids = np.sort(np.unique(edge_src)).reshape(-1)
        nid2idx = np.zeros((org_nids[-1]+1, ), dtype=int)
        nid2idx[org_nids] = np.arange(org_nids.shape[0])
        
        edge_src = nid2idx[edge_src]
        edge_dst = nid2idx[edge_dst]
        first_feature_ids = org_nids - num_instances
        
        g = dgl.graph((edge_src, edge_dst))
        
        # indicators of the target, retrieved instances, 1st order feature nodes, and 2nd order feature nodes
        org_nids = first_feature_ids
        
        labels = np.concatenate((labels, [2]*(g.num_nodes()-len(labels))), axis=0)
        g.ndata['label'] = torch.tensor(labels).long()
        g.ndata['org_nid'] = torch.tensor(org_nids).long()
        
        g.ndata['is_target'] = (g.nodes() < num_target)
        g.ndata['is_instance'] = (g.nodes() < num_instances)
        g.ndata['is_feat'] = (g.nodes() >= num_instances)
        return g


    def __getitem__(self, index):
        g = self.construct_graph(index)
        return g, torch.tensor(self.label[index])

    def __len__(self):
        return self.len 

