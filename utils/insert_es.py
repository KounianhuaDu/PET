import os
import argparse
from es_writer import ESWriter
import numpy as np
import pandas as pd
# import logging
# logging.basicConfig(level=logging.INFO)

"""
    TODO:
        Within this function, you need to merge all your retrieve data,
        and concatenate feature & label into one file.
        
        You should have retrieved features and labels under `base_folder`,
        which corresponds to `X_ret` and `y_ret` in following codes.
"""
def merge_pool_file(base_folder, format, update_pool=False):
    if update_pool or (not os.path.exists(os.path.join(base_folder, 'search_pool' + format))):
        print('Generating search pool...')
        X_ret = np.load(os.path.join(base_folder, 'train_input_all.npy'))
        y_ret = np.load(os.path.join(base_folder, 'train_output_all.npy')).reshape(-1, 1)
        ret_pool = np.concatenate([X_ret, y_ret], axis=1)
        pd.DataFrame(ret_pool, index=None).to_csv(
            os.path.join(base_folder, 'search_pool' + format), sep=',', header=False, index=False)
    else:
        print('Skip pool generation...')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['avazu', 'criteo', 'alipay', 'taobao', 'tmall'])
    parser.add_argument('--update-pool', action='store_true', default=False)
    args = parser.parse_args()
    
    sync_c_pos = {
        'avazu': 9,
        'criteo': 32,
        'tmall': 0,
        'taobao': 0,
        'alipay': 0
    }
    
    format = {
        'avazu': '.txt',
        'criteo': '.txt',
        'tmall': '.csv',
        'taobao': '.csv',
        'alipay': '.csv'
    }
    
    feateng_folder = {
        'avazu': 'processed',
        'criteo': 'processed',
        'tmall': 'feateng_data',
        'taobao': 'feateng_data',
        'alipay': 'feateng_data',
    }
    
    base_folder = os.path.join('../data', args.dataset, feateng_folder[args.dataset])
    merge_pool_file(base_folder, format[args.dataset], args.update_pool)

    # ESWriter
    eswriter = ESWriter(
        os.path.join(base_folder, 'search_pool' + format[args.dataset]),
        args.dataset,
        sync_c_pos[args.dataset]
    )
    eswriter.write()