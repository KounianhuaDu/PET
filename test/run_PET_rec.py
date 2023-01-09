import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torchmetrics import AUROC
import dgl
import argparse
import os
import sys
import random
sys.path.append("..")
import numpy as np
from tqdm import tqdm
from utils import RecDataloader, config
from utils.ranking_utils import get_ranking_quality
from algo.PET import LinkPredict
from tensorboardX import SummaryWriter

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(args):
    torch.autograd.set_detect_anomaly(True)
    log_name = args.model + '_' + args.dataset + '_' + str(args.num) +'.txt'
    log_file = open(os.path.join(args.out_path, log_name), "w+")

    train_dir = os.path.join(args.out_path, args.model + '_' + args.dataset +'_' + str(args.num) + '_train')
    eval_dir = os.path.join(args.out_path, args.model + '_' + args.dataset +'_' + str(args.num) + '_eval')
    train_writer = SummaryWriter(log_dir=train_dir)
    eval_writer = SummaryWriter(log_dir=eval_dir)

    #step 1: Check device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    #step 2: Load data 
    if args.use_val:
        train_loader, val_loader, test_loader = RecDataloader.load_data(
            args.dataset, args.batch_size, args.num_workers, args.data_path, include_val=True)
    else:
        train_loader, test_loader = RecDataloader.load_data(
            args.dataset, args.batch_size, args.num_workers, args.data_path, include_val=False)
        val_loader = test_loader
    print('Data loaded.')
    log_file.write('Data loaded.\n')
    
    #step 3: Create model and training components
    model = LinkPredict(
        num_layers = args.N,
        num_feat=config.num_feat[args.dataset],
        in_feat=args.in_size,
        hidden_feat=args.hidden_size,
        dropout=args.dropout
    )
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print('Model created.')
    log_file.write('Model created.\n')

    #step 4: Training
    
    print('Start training.')
    log_file.write('Start training.\n')
    kill_cnt = 0
    
    def eval(loader):
        model.eval()
        with torch.no_grad():
            validate_loss, preds, target_ids, metric = [], [], [], AUROC()
            with tqdm(total=len(loader), dynamic_ncols=True) as t:
                for g, label in loader:
                    g, label = g.to(device), label.to(device)
                    logits = model(g)
                    # compute loss
                    val_loss = criterion(logits, label.float())
                    val_auc = metric(logits, label)
                    validate_loss.append(val_loss.item())
                    
                    preds += logits.cpu().numpy().tolist()
                    target_ids += g.ndata['target_id'][g.ndata['is_target']].cpu().numpy().tolist()
                    
                    t.update()
                    t.set_postfix({
                        'valid loss': f'{val_loss.item():.4f}',
                        'valid auc': f'{val_auc.item():.4f}'
                    })

            validate_loss = np.mean(validate_loss)
            validate_auc = metric.compute().item()
            ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr = get_ranking_quality(preds, target_ids)

            eval_writer.add_scalar('eval_loss', validate_loss, train_iter)
            eval_writer.add_scalar('eval_auc', validate_auc, train_iter)
            
            return validate_loss, validate_auc, hr_1, hr_5, hr_10, ndcg_5, ndcg_10, mrr
    
    train_iter, early_stop = 0, False
    loss, auc, hr_1, hr_5, hr_10, ndcg_5, ndcg_10, mrr = eval(val_loader)
                
    print("Val Loss: {:.5}, Val AUC: {:.5}, Val HR@1: {:.5}, Val HR@5: {:.5}, Val HR@10: {:.5}, Val NDCG@5: {:.5}, Val NDCG@10: {:.5}, Val MRR: {:.5}\n".\
            format(loss, auc, hr_1, hr_5, hr_10, ndcg_5, ndcg_10, mrr))
    
    test_losses, test_mrrs = [loss], [mrr]
    for epoch in range(args.epochs):
        # Training and validation 
        if args.load or early_stop:
            break
        
        train_loss, metric = [], AUROC()
        model.train()
        # with tqdm(total=len(train_loader), dynamic_ncols=True) as t:
        for step, batch in enumerate(train_loader):
            g, label = batch
            g, label = g.to(device), label.to(device)
            logits = model(g)
            
            # compute loss
            tr_loss = criterion(logits, label.float())
            tr_auc = metric(logits, label)
            train_loss.append(tr_loss.item())

            # backward
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()

            train_iter = epoch * len(train_loader) + step

            if step % 10 == 0:
                train_writer.add_scalar('train_loss', train_loss[-1], train_iter)
            if step % 100 == 0:
                print(f'Epoch: {epoch}, Step: {step}, Train loss: {train_loss[-1]:.4f}, Train AUC: {tr_auc.item():.4f}')
                log_file.write('Epoch {}, step {}/{}, train loss: {:.4f}\n'.format(epoch, step, len(train_loader), train_loss[-1]))
            
            if (step + 1) % (len(train_loader) // 2) == 0:
                loss, auc, hr_1, hr_5, hr_10, ndcg_5, ndcg_10, mrr = eval(val_loader)
    
                #validate
                test_mrrs.append(mrr)
                if test_mrrs[-1] > max(test_mrrs[:-1]):
                    print('Saving model...')
                    torch.save(model.state_dict(), os.path.join(args.out_path, args.model+'_'+args.dataset+'_'+str(args.num)))

                test_losses.append(loss)
                if len(test_losses) > 2 and epoch > 0:
                    if (test_losses[-1] > test_losses[-2] and test_losses[-2] > test_losses[-3]):
                        early_stop = True
                    if (test_losses[-2] - test_losses[-1]) <= 0.0001 and (test_losses[-3] - test_losses[-2]) <= 0.0001:
                        early_stop = True
                
                print("Val Loss: {:.5}, Val AUC: {:.5}, Val HR@1: {:.5}, Val HR@5: {:.5}, Val HR@10: {:.5}, Val NDCG@5: {:.5}, Val NDCG@10: {:.5}, Val MRR: {:.5}\n".\
                        format(loss, auc, hr_1, hr_5, hr_10, ndcg_5, ndcg_10, mrr))
                        
                if early_stop:
                    break
            
        train_loss = np.mean(train_loss)
        train_auc = metric.compute().item()

        print("In epoch {}, Train Loss: {:.5}, Train AUC: {:.5}, Val Loss: {:.5}, Val AUC: {:.5}, Val HR@1: {:.5}, Val HR@5: {:.5}, Val HR@10: {:.5}, Val NDCG@5: {:.5}, Val NDCG@10: {:.5}, Val MRR: {:.5}\n".\
            format(epoch, train_loss, train_auc, loss, auc, hr_1, hr_5, hr_10, ndcg_5, ndcg_10, mrr))
        log_file.write("In epoch {}, Train Loss: {:.5}, Train AUC: {:.5}, Val Loss: {:.5}, Val AUC: {:.5}, Val HR@1: {:.5}, Val HR@5: {:.5}, Val HR@10: {:.5}, Val NDCG@5: {:.5}, Val NDCG@10: {:.5}, Val MRR: {:.5}\n".\
            format(epoch, train_loss, train_auc, loss, auc, hr_1, hr_5, hr_10, ndcg_5, ndcg_10, mrr))
        
        if kill_cnt >= args.early_stop:
            break
    
    # test use the best model
    model.load_state_dict(torch.load(os.path.join(args.out_path, args.model+'_'+args.dataset+'_'+str(args.num))))
    test_loss, test_auc, hr_1, hr_5, hr_10, ndcg_5, ndcg_10, mrr = eval(test_loader)
    
    print("Test Loss: {:.5}\nTest AUC: {:.5}\nTest HR@1: {:.5}\nTest HR@5: {:.5}\nTest HR@10: {:.5}\nTest NDCG@5: {:.5}\nTest NDCG@10: {:.5}\nMRR: {:.5}\n".\
        format(test_loss, test_auc, hr_1, hr_5, hr_10, ndcg_5, ndcg_10, mrr))
    log_file.write("Test Loss: {:.5}\nTest AUC: {:.5}\nTest HR@1: {:.5}\nTest HR@5: {:.5}\nTest HR@10: {:.5}\nTest NDCG@5: {:.5}\nTest NDCG@10: {:.5}\nMRR: {:.5}\n".\
        format(test_loss, test_auc, hr_1, hr_5, hr_10, ndcg_5, ndcg_10, mrr))
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parser For Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset", default="ml-1m", help="Dataset to use, default: tmall")
    parser.add_argument("--data_path", default="../data", help="Path to save the data")
    parser.add_argument("--out_path", default="../output", help="Path to save the output")

    parser.add_argument("--model", default="PET", help="Model Name")
    parser.add_argument("--num", default="0", help="Model number")
    
    parser.add_argument('--load', action='store_true', default=False, help='Load trained model')
    
    parser.add_argument("--K", default=10, type=int, help="Retrieval size.")
    parser.add_argument("--N", default=2, type=int, help="Number of GNN layers.")
    parser.add_argument("--in_size", default=16, type=int, help="Initial dimension size for entities.")
    parser.add_argument('--hidden_size', type=int, default=[200, 80], nargs='*', help='Hidden dimension size for MLP.')

    parser.add_argument("--gpu", type=int, default="0", help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0")
    parser.add_argument("--epochs", type=int, default=500, help="Maximum number of epochs")

    parser.add_argument("--batch_size", type=int, default=100, help="Batch size.")
    parser.add_argument("--wd", type=float, default=1e-4, help="L2 Regularization for Optimizer")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--lr-decay", type=float, default=1, help="Exponential decay of learning rate")

    parser.add_argument("--num_workers", type=int, default=10, help="Number of processes to construct batches")
    parser.add_argument("--early_stop", default=3, type=int, help="Patience for early stop.")

    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout.")
    parser.add_argument("--use_val", action='store_true', help='Use validation set')

    args = parser.parse_args()

    print(args)

    main(args)
