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

For raw datasets, you can download from the links in the paper, or start from the joined tabular files:
```bash
wget https://s3.us-west-2.amazonaws.com/dgl-data/dataset/joined_tabulars.zip

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

