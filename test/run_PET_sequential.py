import argparse
import os
import sys

import dgl
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("..")

import numpy as np
from tqdm import tqdm
from algo.PET import LinkPredict
from tensorboardX import SummaryWriter
from torchmetrics import AUROC
from utils import SequentialDataloader, config
from utils.utils import *


def main(args):
    torch.autograd.set_detect_anomaly(True)
    log_name = args.model + "_" + args.dataset + "_" + args.num + ".txt"
    log_file = open(os.path.join(args.out_path, log_name), "w+")
    np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    train_dir = os.path.join(args.out_path, args.model + "_" + args.dataset + "_" + args.num + "_train")
    eval_dir = os.path.join(args.out_path, args.model + "_" + args.dataset + "_" + args.num + "_eval")
    train_writer = SummaryWriter(log_dir=train_dir)
    eval_writer = SummaryWriter(log_dir=eval_dir)

    # step 1: Check device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    # step 2: Load data
    train_loader, test_loader = SequentialDataloader.load_data(
        args.dataset, args.batch_size, args.K, args.num_workers, args.data_path
    )
    print("Data loaded.")
    log_file.write("Data loaded.\n")

    # step 3: Create model and training components
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
    decay = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    print("Model created.")
    log_file.write("Model created.\n")

    # step 4: Training
    print("Start training.")
    log_file.write("Start training.\n")
    best_auc = 0.0
    kill_cnt = 0
    for epoch in range(args.epochs):
        # Training and validation
        model.train()
        train_loss, metric = [], AUROC()
        train_auc = []
        with tqdm(total=len(train_loader), dynamic_ncols=True) as t:
            for step, batch in enumerate(train_loader):
                g, label = batch
                g = g.to(device)
                label = label.to(device)
                logits = model(g)

                # compute loss
                tr_loss = criterion(logits, label.float())
                train_loss.append(tr_loss.item())
                tr_auc = metric(logits, label)
                train_auc.append(tr_auc)

                # backward
                optimizer.zero_grad()
                tr_loss.backward()
                optimizer.step()
                
                t.update()
                t.set_description(desc=f'Epoch: {epoch}/{args.epochs}')
                t.set_postfix({
                    'train loss': f'{tr_loss.item():.4f}',
                    'train auc': f'{tr_auc.item():.4f}'
                })

                train_iter = epoch * len(train_loader) + step
                if step % 10 == 0:
                    train_writer.add_scalar("train_loss", tr_loss.item(), train_iter)
                    train_writer.add_scalar("train auc", tr_auc.item(), train_iter)
                if step % 100 == 0:
                    log_file.write("Epoch {}, step {}/{}, train loss: {:.4f}\n".format(epoch, step, len(train_loader), tr_loss))

        train_loss = np.mean(train_loss)
        train_auc = metric.compute().item()
        log_file.write("Epoch {}, train loss: {}, train auc: {}\n".format(epoch, train_loss, train_auc))

        decay.step()

        model.eval()
        with torch.no_grad():
            validate_loss, metric = [], AUROC()
            with tqdm(total=len(test_loader), dynamic_ncols=True) as t:
                for step, batch in enumerate(test_loader):
                    g, label = batch
                    g = g.to(device)
                    label = label.to(device)
                    logits = model(g)
                    # compute loss
                    val_loss = criterion(logits, label.float())
                    val_auc = metric(logits, label)
                    validate_loss.append(val_loss.item())
                    
                    t.update()
                    t.set_description(desc=f'Epoch: {epoch}/{args.epochs}')
                    t.set_postfix({
                        'valid loss': f'{val_loss.item():.4f}',
                        'valid auc': f'{val_auc.item():.4f}'
                    })

            validate_loss = np.mean(validate_loss)
            validate_auc = metric.compute().item()

            eval_writer.add_scalar("eval_loss", validate_loss, train_iter)
            eval_writer.add_scalar("eval_auc", validate_auc, train_iter)

            # validate
            if validate_auc > best_auc:
                best_auc = validate_auc
                best_epoch = epoch
                torch.save(
                    model.state_dict(), os.path.join(args.out_path, args.model + "_" + args.dataset + "_" + args.num)
                )
                kill_cnt = 0
                print("saving model...")
                log_file.write("saving model...\n")
            else:
                kill_cnt += 1
                if kill_cnt > args.early_stop:
                    print("early stop.")
                    log_file.write("early stop.\n")
                    print("best epoch:{}".format(best_epoch))
                    log_file.write("best epoch:{}\n".format(best_epoch))
                    break

            print(
                "In epoch {}, Train Loss: {:.4f}, Train AUC: {:.4f}, Valid Loss: {:.5}, Valid AUC: {:.5}\n".format(
                    epoch, train_loss, train_auc, validate_loss, validate_auc
                )
            )
            log_file.write(
                "In epoch {}, Train Loss: {:.4f}, Train AUC: {:.4f}, Valid Loss: {:.5}, Valid AUC: {:.5}\n".format(
                    epoch, train_loss, train_auc, validate_loss, validate_auc
                )
            )

    # test use the best model
    model.eval()
    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(args.out_path, args.model + "_" + args.dataset + "_" + args.num)))
        test_loss = []
        test_auc = []
        test_logloss = []
        with tqdm(total=len(test_loader), dynamic_ncols=True) as t:
            for step, batch in enumerate(test_loader):
                g, label = batch
                g = g.to(device)
                label = label.to(device)
                logits = model(g)
                # compute loss
                loss = criterion(logits, label.float())
                auc = evaluate_auc(logits.detach().cpu().numpy(), label.detach().cpu().numpy())
                log_loss = evaluate_logloss(logits.detach().cpu().numpy(), label.detach().cpu().numpy())

                test_loss.append(loss.item())
                test_auc.append(auc)
                test_logloss.append(log_loss)

        test_loss = np.sum(test_loss)
        test_auc = np.mean(test_auc)
        test_logloss = np.mean(test_logloss)
        print(
            "Test Loss: {:.5}\n, AUC: {:.5}\n, Logloss: {:.5}\n".format(
                test_loss, test_auc, test_logloss
            )
        )
        log_file.write(
            "Test Loss: {:.5}\n, AUC: {:.5}\n, Logloss: {:.5}\n".format(
                test_loss, test_auc, test_logloss
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser For Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset", default="tmall", help="Dataset to use, default: tmall")
    parser.add_argument("--data_path", default="../data", help="Path to save the data")
    parser.add_argument("--out_path", default="../output", help="Path to save the output")

    parser.add_argument("--model", default="PET", help="Model Name")
    parser.add_argument("--num", default="0", help="Model number")

    parser.add_argument("--K", default=10, type=int, help="Retrieval size.")
    parser.add_argument("--N", default=3, type=int, help="Number of GNN layers.")
    parser.add_argument("--in_size", default=16, type=int, help="Initial dimension size for entities.")
    parser.add_argument('--hidden_size', type=int, default=[200, 80], nargs='*', help='Hidden dimension size for MLP.')

    parser.add_argument("--gpu", type=int, default="0", help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0")
    parser.add_argument("--epochs", type=int, default=500, help="Maximum number of epochs")

    parser.add_argument("--batch_size", type=int, default=100, help="Batch size.")
    parser.add_argument("--wd", type=float, default=5e-4, help="L2 Regularization for Optimizer")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--lr-decay", type=float, default=1, help="Exponential decay of learning rate")

    parser.add_argument("--num_workers", type=int, default=10, help="Number of processes to construct batches")
    parser.add_argument("--early_stop", default=3, type=int, help="Patience for early stop.")

    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout.")

    args = parser.parse_args()

    print(args)

    main(args)
