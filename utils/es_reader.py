from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, MultiSearch
from torch.utils.data import Dataset
import time
import numpy as np
import pandas as pd
import pickle as pkl
import sys
import configparser

def select_pos_str(input_str, pos_list):
    return ','.join(np.array(input_str.split(','))[pos_list].tolist())

class ESReader(object):
    def __init__(self, 
                 index_name,
                 size,
                 host_url = 'localhost:9200'):
        
        self.es = Elasticsearch(host_url)
        self.index_name = index_name
        self.size = size

    # For UBR
    def query_ubr(self, queries, sync_ids):
        ms = MultiSearch(using=self.es, index=self.index_name)
        for i, q in enumerate(queries):
            s = Search().filter("terms", sync_id=[sync_ids[i]]).filter("terms", label=['1']).query("match", line=q)[1:self.size+1]
            ms = ms.add(s)
        responses = ms.execute()

        res_lineno_batch = []
        # res_line_batch = []
        for response in responses:
            res_lineno = []
            res_line = []
            for hit in response:
                res_lineno.append(str(hit.line_no))
                # res_line.append(list(map(int, hit.line.split(','))))
            if res_lineno == []:
                res_lineno.append('-1')
            res_lineno_batch.append(res_lineno)
            # res_line_batch.append(res_line)
        return res_lineno_batch#, res_line_batch

    # For RIM: not groupping by label, query_rim1 is for sequential data setting
    def query_rim1(self, queries, sync_ids):
        ms = MultiSearch(using=self.es, index=self.index_name)
        for q in queries:
            s = Search().query("match", line=q)[:self.size]
            ms = ms.add(s)
        responses = ms.execute()

        res_lineno_batch = []
        # res_line_batch = []
        label_batch = []
        for response in responses:
            # print("len of res:{}".format(len(response)))
            res_lineno = []
            # res_line = []
            # labels = []
            for hit in response:
                res_lineno.append(str(hit.line_no))
                # res_line.append(list(map(int, hit.line.split(','))))
                # labels.append(hit.label)
            if res_lineno == []:
                res_lineno.append('-1')
                # labels.append('0')
            res_lineno_batch.append(res_lineno)
            # res_line_batch.append(res_line)
            # label_batch.append(labels)
        return res_lineno_batch#, label_batch, res_line_batch

    # For RIM: avazu and criteo
    def query_rim_ac(self, queries, sync_ids):
        ms = MultiSearch(using=self.es, index=self.index_name)
        for i, q in enumerate(queries):
            s = Search().query("match", sync_id=sync_ids[i])[:self.size]
            ms = ms.add(s)
        responses = ms.execute()

        res_lineno_batch = []
        # res_line_batch = []
        # label_batch = []
        for response in responses:
            res_lineno = []
            # res_line = []
            # labels = []
            for hit in response:
                res_lineno.append(str(hit.line_no))
                # res_line.append(list(map(int, hit.line.split(','))))
                # labels.append(hit.label)
            if res_lineno == []:
                res_lineno.append('-1')
                # labels.append('0')
            res_lineno_batch.append(res_lineno)
            # res_line_batch.append(res_line)
            # label_batch.append(labels)
        return res_lineno_batch#, label_batch, res_line_batch

class queryGen(object):
    def __init__(self,
                 target_file,
                 batch_size,
                 sync_c_pos,
                 query_c_pos):
        
        self.batch_size = batch_size
        self.sync_c_pos = sync_c_pos
        self.query_c_pos = np.asarray(list(map(int, query_c_pos.split(','))))

        self.target_data = pd.read_csv(target_file, sep=',', index_col=False, header=None)
        # self.target_data = pd.DataFrame(np.load(target_file))
        self.query_data = self.target_data[self.query_c_pos].apply(lambda l: ','.join(map(str, l)), 1).values
        print(self.target_data.head(10))
        print(self.query_data[:10])
        exit(-1)
        self.sync_id_data = self.target_data[self.sync_c_pos].values.astype(str)
        self.dataset_size = len(self.target_data)
        
        self.query_fn = lambda x: ','.join(x)
        
        if self.dataset_size % self.batch_size == 0:
            self.total_step = int(self.dataset_size / self.batch_size)
        else:
            self.total_step = int(self.dataset_size / self.batch_size) + 1
        self.step = 0
        print('data loaded')
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.step == self.total_step:
            raise StopIteration
        
        if self.step != self.total_step - 1:
            q_batch = self.query_data[self.step * self.batch_size: (self.step + 1) * self.batch_size]
            sync_id_batch = self.sync_id_data[self.step * self.batch_size: (self.step + 1) * self.batch_size]
        else:
            q_batch = self.query_data[self.step * self.batch_size:]
            sync_id_batch = self.sync_id_data[self.step * self.batch_size:]
            
        # print(q_batch[:10])
        # print(sync_id_batch[:10])
        # exit(-1)

        self.step += 1

        return q_batch, sync_id_batch
    
    
class queryGenEfficient(Dataset):
    def __init__(self, target_range):
        self.target_min, self.target_max = target_range
        self.target_size = self.target_max - self.target_min
        
    def __len__(self):
        return self.target_size
    
    def __getitem__(self, index):
        return None, self.target_min + index


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('PLEASE INPUT [DATASET] [BATCH_SIZE] [RETRIEVE_SIZE]')
        sys.exit(0)
    dataset = sys.argv[1]
    batch_size = int(sys.argv[2])
    size = int(sys.argv[3])

    # read config file
    cnf = configparser.ConfigParser()
    cnf.read('config.ini')

    # query generator
    query_generator = queryGen(cnf.get(dataset, 'target_train_file'),
                               batch_size,
                               cnf.getint(dataset, 'sync_c_pos'),
                               cnf.get(dataset, 'query_c_pos'))
    es_reader = ESReader(dataset, size)

    t = time.time()
    for batch in query_generator:
        q_batch, sync_id_batch = batch

        res_lineno_batch, res_line_batch = es_reader.query_rim1(q_batch)
        # plist = zip(q_batch, res_line_batch, label_batch)
        # print(res_lineno_batch)
        # print(res_line_batch)

        print('time: %.4f seconds' % (time.time() - t))
        t = time.time()