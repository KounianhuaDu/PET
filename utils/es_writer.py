from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, MultiSearch
import elasticsearch.helpers
import time
import numpy as np
from tqdm import tqdm
import sys

class ESWriter(object):
    def __init__(self,
                 input_file,
                 index_name,
                 sync_c_pos,
                 invalid_range=(-1, -1),
                 host_url = 'localhost:9200'):

        self.input_file = input_file
        self.es = Elasticsearch(host_url)
        self.index_name = index_name
        self.sync_c_pos = sync_c_pos
        
        self.invalid_min, self.invalid_max = invalid_range
        
        # delete if there is existing index
        self.es.indices.delete(index=self.index_name, ignore=[400, 404])
        # create index
        self._create_index()

    def _create_index(self):
        self.create_index_body = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "my_analyzer": {
                        "tokenizer": "my_tokenizer"
                        }
                    },
                    "tokenizer": {
                        "my_tokenizer": {
                        "type": "pattern",
                        "pattern": ","
                        }
                    }
                },
            },
            "mappings": {
                "properties": {
                "line": {
                    "type": "text",
                    'analyzer': 'my_analyzer',
                    'search_analyzer': 'my_analyzer'
                    }
                }
            },
        }
        self.es.indices.create(index=self.index_name, body=self.create_index_body, ignore=400)
        print('index created')

    def write(self):
        t = time.time()
        with open(self.input_file, 'r') as f:
            docs = []
            batch_num = 0
            line_no = 0
            for line in tqdm(f):
                if (line_no) >= self.invalid_min and (line_no) < self.invalid_max:
                    line_no += 1
                    continue
                # print(f'started from {line_no}')
                # exit(-1)
                
                line_str = line[:-1]
                line_item = line[:-1].split(',')
                sync_id = line_item[self.sync_c_pos]
                label = line_item[-1]
                
                doc = {
                    'sync_id': sync_id,
                    'label': label,
                    'line': line_str,
                    'line_no': line_no
                }
                docs.append(doc)
                line_no += 1

                if len(docs) == 1000:
                    actions = [{
                        '_op_type': 'index',
                        '_index': self.index_name,  
                        '_source': d
                    }
                    for d in docs]
                    elasticsearch.helpers.bulk(self.es, actions)
                    batch_num += 1
                    docs = []
                    if batch_num % 1000 == 0:
                        print('\n{} data samples have been inserted'.format(batch_num * 1000))

            # the last bulk
            if docs != []:
                actions = [{
                    '_op_type': 'index',
                    '_index': self.index_name,  
                    '_source': d
                } 
                for d in docs]
                elasticsearch.helpers.bulk(self.es, actions)

        print('data insert time: %.2f seconds' % (time.time() - t))
        print('last line_no is {}'.format(line_no))

if __name__ == "__main__":
    pass