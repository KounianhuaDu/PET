# PET
This is the repository for PET.

## Dependencies

- dgl 0.7.0
- pytorch 1.8.0
- tensorboard 2.7.0
- sklearn 0.22.1
- torchmetrics 0.7.3 (for efficient model evaluation, especially when using multi GPU)
- elasticsearch-dsl 7.4.0 (requires local elasticsearch lib)
- elasticsearch


## Data Processing
Run following in a tmux session to establish an `elasticsearch` daemon

```bash
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.1-linux-x86_64.tar.gz
tar -xf elasticsearch-7.17.1-linux-x86_64.tar.gz
cd elasticsearch-7.17.1/bin/
./elasticsearch
```

**Note:** The produced retrieve pool has an order corresponding to the shuffled dataset. If you re-split or re-shuffle the dataset, you will need to run retrieve codes again.

Under the [`utils`](utils/) folder, to produce retrieve pool of *DATASET* with size *k*, run

```bash
python insert_es.py --dataset DATASET
```

```bash
python pre_search_new.py --dataset DATASET --ret-size k
```

For processed dataset, you can download via:
```bash
wget https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tmall-ret.zip
```

## Preparing your own datasets

To run with your own datasets, you need to prepare `search_pool.csv`, `target_train.csv` and `target_test.csv`
yourself in the folder `data/custom` similar to the link above.  We describe the concrete requirements as follows.

`search_pool.csv`, `target_train.csv` and `target_test.csv` should have the same columns, representing the
search pool, the training set and the test set respectively.  Moreover,

* Each file should not contain CSV headers.
* Each column should only contain categorical values encoded as integers.  The same integer in the same column
  in all three tables refer to the same categorical value.
* The last column represents the row label, which should be always binary (i.e. 0 or 1).

Once done, you can run

```bash
python insert_es.py --dataset custom
python pre_search_new.py --dataset custom
# or you can specify which subset of columns to compute the similarity metric like following
python pre_search_new.py --dataset custom --query-columns 0,1,2,3
```

## Run
Under the test folder:

For CTR task, run:
```bash
python run_PET_sequential.py --dataset tmall --in_size 16 --lr 5e-4 --wd 1e-4 --batch_size 100
```
```bash
python run_PET_sequential.py --dataset taobao --in_size 16 --lr 1e-4 --wd 5e-4 --batch_size 200
```
```bash
python run_PET_sequential.py --dataset alipay --in_size 32 --lr 5e-4 --wd 5e-4 --batch_size 100
```

For top-N recommendation task:
```bash
python run_PET_rec.py --dataset ml-1m --batch_size 100
```
```bash
python run_PET_rec.py --dataset lastfm --batch_size 500
```

If you prepared the custom datasets as above, supply `custom` in the `--dataset` option.
