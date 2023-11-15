import csv
import sys
import time
import random
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch import nn
from torch import optim
from torch.utils import data
import torch.nn.init as init
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score

from utils.models_transformer import Net
from utils.tools import seed_everything
from utils.losses import ConLoss

SEED = 2352


def display(bi_acc, f1):
    print(f"Binary accuracy on test set is {bi_acc:.4f}")
    print(f"F1-score on test set is {f1:.4f}")


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth
    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def train(train_set, valid_set, test_set, input_dims, args, log):
    patience = args.patience
    num_epochs = args.num_epochs
    model_path = args.model_path
    output_path = args.output_path
    output_dim = args.output_dim
    seq_len = args.seq_len

    param_grid = {
        'embed_dim': [40],
        'dim_total_proj': [32],
        'num_layers': [2],
        'num_heads': [2],
        'batch_size': [16],
        'weight_decay': [0],
        'learning_rate': [0.001],
        'attn_dropout': [0.2],
        'embed_dropout': [0.2]
    }

    grid = ParameterGrid(param_grid)

    log.write(f'There are {len(grid)} hyper-parameter settings in total.\n\n')

    with open(args.output_path, 'w+') as out:
        writer = csv.writer(out)
        writer.writerow([
            'embed_dim',
            'dim_total_proj',
            'num_layers',
            'num_heads',
            'batch_size',
            'weight_decay',
            'learning_rate',
            'attn_dropout',
            'embed_dropout',
            'Min Validation Loss',
            'Happy accuracy', 'Happy f1_score',
            'Sad accuracy', 'Sad f1_score',
            'Angry accuracy', 'Angry f1_score',
            'Neutral accuracy', 'Neutral f1_score'
        ])

    min_mae = float('Inf')
    max_corr = 0
    max_multi_acc = 0
    max_bi_acc = 0
    max_f1 = 0
    param_num = 0

    for params in grid:
        param_num += 1

        seed_everything(SEED)

        embed_dim = params['embed_dim']
        dim_total_proj = params['dim_total_proj']
        num_layers = params['num_layers']
        num_heads = params['num_heads']
        batch_sz = params['batch_size']
        decay = params['weight_decay']
        lr = params['learning_rate']
        attn_dropout = params['attn_dropout']
        embed_dropout = params['embed_dropout']

        model = Net(text_input_size=input_dims[2], video_input_size=input_dims[1], audio_input_size=input_dims[0],
                    embed_dim=embed_dim, dim_total_proj=dim_total_proj,
                    num_layers=num_layers, num_heads=num_heads,
                    output_dim=output_dim, seq_len=seq_len,
                    attn_dropout=attn_dropout, embed_dropout=embed_dropout)

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args.cuda:
            print('now use GPU to accelerate...')
            model.to(DEVICE)

        print('=' * 40)
        print('Hyperparameter:' + '{}/{}'.format(param_num, len(grid)).rjust(25))
        print('-' * 40)
        total_parameter = sum([param.nelement() for param in model.parameters()])
        print(f'Now there are {total_parameter} parameters in total')
        print('-' * 40)
        print('embed_dim'.ljust(30) + '= ' + str(embed_dim))
        print('dim_total_proj'.ljust(30) + '= ' + str(dim_total_proj))
        print('num_layers'.ljust(30) + '= ' + str(num_layers))
        print('num_heads'.ljust(30) + '= ' + str(num_heads))
        print('batch_size'.ljust(30) + '= ' + str(batch_sz))
        print('weight_decay'.ljust(30) + '= ' + str(decay))
        print('learning_rate'.ljust(30) + '= ' + str(lr))
        print('attn_dropout'.ljust(30) + '= ' + str(attn_dropout))
        print('embed_dropout'.ljust(30) + '= ' + str(embed_dropout))
        print('=' * 40)

        CE_loss = nn.CrossEntropyLoss(reduction='sum')
        Contrastive_loss = ConLoss(reduction='sum')

        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
        # scheduler = ReduceLROnPlateau(opt, mode='min', patience=20, factor=0.1, verbose=True)

        # setup training
        complete = True
        min_valid_loss = float('Inf')

        train_iterator = data.DataLoader(
            train_set, batch_size=batch_sz, shuffle=True, num_workers=8)
        valid_iterator = data.DataLoader(
            valid_set, batch_size=batch_sz, shuffle=True, num_workers=8)
        test_iterator = data.DataLoader(
            test_set, batch_size=batch_sz, shuffle=True, num_workers=8)

        # train
        print('Start Training...')
        curr_patience = patience

        for e in range(num_epochs):
            model.train()
            avg_train_loss = 0.0
            avg_ce_loss = 0.0
            avg_cl_loss = 0.0

            train = tqdm(train_iterator, leave=True)

            for x_a, x_v, x_t, y in train:

                train.set_description(f'Epoch {e+1:02d}/{num_epochs}')

                x_a, x_v, x_t, y = x_a.float(), x_v.float(), x_t.float(), y.float()
                if args.cuda:
                    x_a, x_v, x_t, y = x_a.to(DEVICE), x_v.to(DEVICE), x_t.to(DEVICE), y.to(DEVICE)
                y = y.long()

                # train Total
                output, similarity_cube = model(x_t, x_v, x_a)

                cl_loss = Contrastive_loss(similarity_cube, y, args.signature)

                output = output.view(-1, 2)  # for CrossEntropy loss
                y = y.view(-1)

                ce_loss = CE_loss(output, y)
                loss = ce_loss + cl_loss

                # Adam
                model.zero_grad()
                loss.backward()
                opt.step()

                avg_train_loss += loss.item()
                avg_ce_loss += ce_loss.item()
                avg_cl_loss += cl_loss.item()

            avg_train_loss /= len(train_set)  # reduction - 'sum'
            avg_ce_loss /= len(train_set)  # reduction - 'sum'
            avg_cl_loss /= len(train_set)  # reduction - 'sum'

            # Terminate the training process if run into NaN
            if np.isnan(avg_train_loss):
                print("Training got into NaN values...\n\n")
                complete = False
                break

            model.eval()
            avg_valid_loss = 0.0
            with torch.no_grad():
                for x_a, x_v, x_t, y in valid_iterator:

                    x_a, x_v, x_t, y = x_a.float(), x_v.float(), x_t.float(), y.float()
                    if args.cuda:
                        x_a, x_v, x_t, y = x_a.to(DEVICE), x_v.to(DEVICE), x_t.to(DEVICE), y.to(DEVICE)
                    y = y.long()

                    output, _ = model(x_t, x_v, x_a)

                    output = output.view(-1, 2)
                    y = y.view(-1)

                    valid_loss = CE_loss(output, y)

                    avg_valid_loss += valid_loss.item()

            avg_valid_loss = avg_valid_loss / len(valid_set)  # reduction - 'sum'
            # scheduler.step(avg_valid_loss)

            if np.isnan(avg_valid_loss):
                print("Validating got into NaN values...\n\n")
                complete = False
                break

            print(f"Average Train loss: {avg_train_loss:.4f}\n\t[ "
                  f"Average CE loss: {avg_ce_loss:.4f} | "
                  f"Average CL loss: {avg_cl_loss:.4f} ]\n"
                  f"Average Validation loss: {avg_valid_loss:.4f}")

            if avg_valid_loss < min_valid_loss:
                curr_patience = patience
                min_valid_loss = avg_valid_loss
                torch.save(model, model_path)
                print("\nFound new best model, saving to disk...\n")
            else:
                curr_patience -= 1

            if curr_patience <= 0:
                break

        if complete:
            best_model = torch.load(model_path)

            if args.cuda:
                output_test = torch.Tensor().to(DEVICE)
                y = torch.Tensor().to(DEVICE)
            else:
                output_test = torch.Tensor()
                y = torch.Tensor()
            with torch.no_grad():
                for x_a, x_v, x_t, y_test in test_iterator:

                    x_a, x_v, x_t, y_test = x_a.float(), x_v.float(), x_t.float(), y_test.float()
                    if args.cuda:
                        x_a, x_v, x_t, y_test = x_a.to(DEVICE), x_v.to(DEVICE), x_t.to(DEVICE), y_test.to(DEVICE)

                    output, _ = best_model(x_t, x_v, x_a)

                    output_test_temp = output

                    y = torch.cat((y, y_test), dim=0)
                    output_test = torch.cat((output_test, output_test_temp), dim=0)

            ########################################################################
            # these are the needed metrics
            print('-' * 40)
            emos = ['Neutral', 'Happy', 'Sad', 'Angry']
            test_preds = output_test.view(-1, 4, 2).cpu().detach().numpy()
            test_truth = y.view(-1, 4).cpu().detach().numpy()

            f1 = {'Neutral': 0, 'Happy': 0, 'Sad': 0, 'Angry': 0}
            acc = {'Neutral': 0, 'Happy': 0, 'Sad': 0, 'Angry': 0}

            for emo_ind in range(4):
                print(f'{emos[emo_ind]}: ')
                test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
                test_truth_i = test_truth[:, emo_ind]
                f1[emos[emo_ind]] = f1_score(test_truth_i, test_preds_i, average='weighted')
                acc[emos[emo_ind]] = accuracy_score(test_truth_i, test_preds_i)

                display(acc[emos[emo_ind]], f1[emos[emo_ind]])
                if emo_ind == 3:
                    print('=' * 40)
                    print()
                else:
                    print('-' * 40)

            with open(output_path, 'a+') as out:
                writer = csv.writer(out)
                writer.writerow([
                    embed_dim,
                    dim_total_proj,
                    num_layers,
                    num_heads,
                    batch_sz,
                    decay,
                    lr,
                    attn_dropout,
                    embed_dropout,
                    min_valid_loss,
                    acc['Happy'], f1['Happy'],
                    acc['Sad'], f1['Sad'],
                    acc['Angry'], f1['Angry'],
                    acc['Neutral'], f1['Neutral']
                ])
