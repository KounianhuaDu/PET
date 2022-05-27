import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import random


class PETLayer(nn.Module):
    def __init__(self, in_feat, hidden_feat):
        super(PETLayer, self).__init__()
        
        self.K = nn.Linear(2*in_feat, hidden_feat)
        self.V = nn.Linear(2*in_feat, hidden_feat)
        self.Q = nn.Linear(in_feat, hidden_feat)
        self.W = nn.Linear(in_feat + hidden_feat, hidden_feat)

        self.edge_W = nn.Linear(2*hidden_feat + in_feat, hidden_feat)


    def forward(self, g):

        edge_embds = g.edata['h']

        src, dst = g.edges()
        node_embds = g.ndata['h']
        src_messages = node_embds[src]*edge_embds
        src_messages = torch.cat((src_messages, node_embds[src]), 1)

        g.ndata['Q'] = self.Q(g.ndata['h'])
        g.edata['K'] = self.K(src_messages)
        g.edata['V'] = self.V(src_messages)

        g.apply_edges(fn.v_mul_e('Q', 'K', 'alpha'))
        g.edata['alpha'] = edge_softmax(g, g.edata['alpha'])
        g.edata['V'] = g.edata['alpha']*g.edata['V'] 

        g.update_all(fn.copy_e('V', 'h_n'), fn.sum('h_n', 'h_n'))
        g.ndata['h'] = self.W(torch.cat((g.ndata['h_n'], g.ndata['h']), 1))

        edge_embds = torch.cat((g.ndata['h'][src], g.ndata['h'][dst], g.edata['h']), 1)
        edge_embds = self.edge_W(edge_embds)
        g.edata['h'] = edge_embds

        return g

class PET(nn.Module):
    def __init__(self, num_layers, in_feat, hidden_feat, dropout=0.1):
        super(PET, self).__init__()
        self.num_layers = num_layers
        self.layer1 = PETLayer(in_feat=in_feat, hidden_feat=hidden_feat)
        self.layer2 = PETLayer(in_feat=hidden_feat, hidden_feat=hidden_feat)
        if num_layers == 3:
            self.layer3 = PETLayer(in_feat=hidden_feat, hidden_feat=hidden_feat)

        #layernorm before each propogation
        self.layernorm = nn.LayerNorm(hidden_feat)

        #dropout layer
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, g):
        g = self.layer1(g)
        g.ndata['h'] = F.relu_(g.ndata['h'])
        g.ndata['h'] = self.dropout(self.layernorm(g.ndata['h']))
        g.edata['h'] = F.relu_(g.edata['h'])
        g.edata['h'] = self.dropout(self.layernorm(g.edata['h']))
        
        g = self.layer2(g)
        g.ndata['h'] = F.relu_(g.ndata['h'])
        g.ndata['h'] = self.dropout(self.layernorm(g.ndata['h']))
        g.edata['h'] = F.relu_(g.edata['h'])
        g.edata['h'] = self.dropout(self.layernorm(g.edata['h']))
        
        if self.num_layers == 3:
            g = self.layer3(g)
            g.ndata['h'] = F.relu_(g.ndata['h'])
            g.ndata['h'] = self.dropout(self.layernorm(g.ndata['h']))
            g.edata['h'] = F.relu_(g.edata['h'])
            g.edata['h'] = self.dropout(self.layernorm(g.edata['h']))
        
        return g


class LinkPredict(nn.Module):
    def __init__(self, num_layers, num_feat, in_feat, hidden_feat, dropout):
        super(LinkPredict, self).__init__()
        
        self.in_feat = in_feat
        self.num_feat = num_feat

        self.feat_embds_l1 = nn.Embedding(self.num_feat, in_feat)
        self.feat_embds_l1.weight = nn.init.xavier_uniform_(self.feat_embds_l1.weight)

        self.label_embd = nn.Embedding(3, in_feat, padding_idx=2)
        self.label_embd.weight = nn.init.xavier_uniform_(self.label_embd.weight)
        
        self.edge_label_embd = nn.Embedding(3, 2*in_feat, padding_idx=2)
        self.edge_label_embd.weight = nn.init.xavier_uniform_(self.edge_label_embd.weight)

        self.gnn = PET(num_layers, in_feat, in_feat)
        
        # NOTE: Compat with different hidden_size settings
        if isinstance(hidden_feat, list):
            last_size = in_feat
            self.predict = []
            for h in hidden_feat:
                self.predict.append(
                    nn.Sequential(
                        nn.Linear(last_size, h),
                        nn.BatchNorm1d(h),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    )
                )
                last_size = h
            self.predict.append(nn.Linear(last_size, 1))
            self.predict = nn.Sequential(*self.predict)
        elif np.isscalar(hidden_feat):
            self.predict = nn.Sequential(
                nn.Linear(in_feat, hidden_feat),
                nn.BatchNorm1d(hidden_feat),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_feat, hidden_feat),
                nn.BatchNorm1d(hidden_feat),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_feat, 1)
            )
        else:
            raise NotImplementedError

    def forward(self, g):
        #label embedding initialization.
        #vec(0) for the targets, embedding(y) for the retrieved instances.
        label_idxs = g.ndata['label'][g.ndata['is_instance']]
        instance_embds = self.label_embd(label_idxs)

        #1st order feature ids & feature embedding initialization
        feat_ids = g.ndata['org_nid'][g.ndata['is_feat']] 
        feat_embds_l1 = self.feat_embds_l1(feat_ids)
        
        #apply on node embeddings
        g.ndata['h'] = torch.zeros((g.num_nodes(), self.in_feat)).to(g.device)
        g.ndata['h'][[g.ndata['is_instance']]] = instance_embds
        g.ndata['h'][[g.ndata['is_feat']]] = feat_embds_l1

        #apply on edge embeddings
        g.edata['h'] = torch.zeros((g.num_edges(), self.in_feat)).to(instance_embds.device)
        in_src, in_dst, in_eids = g.in_edges(g.nodes()[g.ndata['is_instance']], form='all')
        out_src, out_dst, out_eids = g.out_edges(g.nodes()[g.ndata['is_instance']], form='all')
        
        g.edata['h'][in_eids] = self.edge_label_embd(g.ndata['label'][in_dst])[:, :self.in_feat]
        g.edata['h'][out_eids] = self.edge_label_embd(g.ndata['label'][out_src])[:, self.in_feat:]

        #Message passing
        g = self.gnn(g)
        pred = self.predict(g.ndata['h'][g.ndata['is_target']])
        pred = torch.sigmoid(pred)
        return pred.squeeze(1)