from es_reader import queryGen, ESReader

from tqdm import tqdm
import os
import csv
import argparse
import numpy as np
import pandas as pd

def pre_search_rim(query_generator, es_reader, search_res_col_file, sequential=False):
    query_fn = es_reader.query_rim1 if sequential else es_reader.query_rim_ac
    with open(search_res_col_file, 'w') as f:
        writer = csv.writer(f)
        for q_batch, sync_id_batch in tqdm(query_generator, total=query_generator.total_step, dynamic_ncols=True):
            res_lineno_batch = query_fn(q_batch, sync_id_batch)
            writer.writerows(res_lineno_batch)

"""
    TODO:
        Within this function, you need to merge all training and testing data separately,
        and save them as `target_train.txt` and `target_test.txt`,
        no label information required.
"""
def merge_files(base_folder, format, update_target=False):
    if update_target or not os.path.exists(os.path.join(base_folder, 'target_train' + format)):
        print('Writing train target file...')
        X = np.load(os.path.join(base_folder, 'train_input_all.npy'))
        pd.DataFrame(X, index=None).to_csv(
            os.path.join(base_folder, 'target_train' + format), sep=',', header=False, index=False)
    else:
        print('Skip train target file writing...')
    
    if update_target or not os.path.exists(os.path.join(base_folder, 'target_test' + format)):
        print('Writing test target file...')
        X = np.load(os.path.join(base_folder, 'test_input_part_0.npy'))
        pd.DataFrame(X, index=None).to_csv(
            os.path.join(base_folder, 'target_test' + format), sep=',', header=False, index=False)
    else:
        print('Skip test target file writing...')

available_datasets = ['avazu', 'criteo', 'taobao', 'tmall', 'alipay', 'custom']
sequential_datasets = ['taobao', 'tmall', 'alipay', 'custom']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=available_datasets)
    parser.add_argument('--query-columns', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--ret-size', type=int, default=10)
    parser.add_argument('--update-target', action='store_true', default=False)
    args = parser.parse_args()
    
    sync_c_pos = {
        'avazu': 9,
        'criteo': 32,
        'tmall': 0,
        'taobao': 0,
        'alipay': 0,
        'custom': 0,
    }
    
    query_c_pos = {
        'avazu': '8,9,10',
        'criteo': '13,22,32,34',
        'tmall': '0,1,2,3,4',
        'taobao': '0,1,2',
        'alipay': '0,1,2,3',
        'custom': args.query_columns,
    }
    
    format = {
        'avazu': '.txt',
        'criteo': '.txt',
        'tmall': '.csv',
        'taobao': '.csv',
        'alipay': '.csv',
        'custom': '.csv',
    }
    
    feateng_folder = {
        'avazu': 'processed',
        'criteo': 'processed',
        'tmall': 'feateng_data',
        'taobao': 'feateng_data',
        'alipay': 'feateng_data',
        'custom': '.',
    }
    
    base_folder = os.path.join('../data', args.dataset, feateng_folder[args.dataset])
    merge_files(base_folder, format[args.dataset], args.update_target)
    
    target_train_file = os.path.join(base_folder, 'target_train' + format[args.dataset])
    target_test_file = os.path.join(base_folder, 'target_test' + format[args.dataset])
    
    search_res_col_train_file = os.path.join(base_folder, f'search_res_col_train_{args.ret_size}.txt')
    search_res_col_test_file = os.path.join(base_folder, f'search_res_col_test_{args.ret_size}.txt')

    # query generator
    query_cols = query_c_pos[args.dataset]
    if query_cols == '':
        # get the number of columns
        with open(os.path.join(base_folder, 'target_train' + format[args.dataset])) as f:
            l = f.readline().strip().split(',')
            query_cols = ','.join([str(i) for i in range(len(l) - 1)])
    query_generator_train = queryGen(target_train_file,
                                     args.batch_size,
                                     sync_c_pos[args.dataset],
                                     query_cols)
    query_generator_test = queryGen(target_test_file,
                                    args.batch_size,
                                    sync_c_pos[args.dataset],
                                    query_cols)
    es_reader = ESReader(args.dataset, args.ret_size)

    print('target train pre searching...')
    pre_search_rim(query_generator_train,
                    es_reader,
                    search_res_col_train_file,
                    args.dataset in sequential_datasets)
    
    print('target test pre searching...')
    pre_search_rim(query_generator_test,
                    es_reader,
                    search_res_col_test_file,
                    args.dataset in sequential_datasets)
