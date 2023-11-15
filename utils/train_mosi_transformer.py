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
from utils.losses import ARLoss, ConLoss

SEED = 2352


def display(mae, corr, multi_acc_7, multi_acc_5, bi_acc, f1):
    print('-' * 40)
    print(f"MAE on test set is {mae:.4f}")
    print(f"Correlation w.r.t human evaluation on test set is {corr:.4f}")
    print(f"Multiclass(7) accuracy on test set is {multi_acc_7:.4f}")
    print(f"Multiclass(5) accuracy on test set is {multi_acc_5:.4f}")
    print(f"Binary accuracy on test set is {bi_acc:.4f}")
    print(f"F1-score on test set is {f1:.4f}")
    print('=' * 40)
    print()


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
    use_bert = args.use_bert

    # Glove
    param_grid = {
        'embed_dim': [40],
        'dim_total_proj': [32],
        'num_layers': [5],
        'num_heads': [10],
        'batch_size': [128],
        'weight_decay': [0.002],
        'learning_rate': [0.001],
        'attn_dropout': [0.2],
        'embed_dropout': [0.3],
        'cl_t_weight': [0.333],
        'cl_v_weight': [0.333],
        'cl_a_weight': [0.333],
        'ar_weight': [1.0],
        'cl_weight': [1.0]
    }
    # param_grid = {
    #     'embed_dim': [40],
    #     'dim_total_proj': [32],
    #     'num_layers': [1, 2, 3, 4, 5],
    #     'num_heads': [1, 2, 4, 8, 10],
    #     'batch_size': [128],
    #     'weight_decay': [0],
    #     'learning_rate': [0.001],
    #     'attn_dropout': [0.0],
    #     'embed_dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
    #     'cl_t_weight': [0.333],
    #     'cl_v_weight': [0.333],
    #     'cl_a_weight': [0.333],
    #     'ar_weight': [1.0],
    #     'cl_weight': [1.0]
    # }

    # Bert
    # param_grid = {
    #     'embed_dim': [40],
    #     'dim_total_proj': [32],
    #     'num_layers': [1, 2, 3, 4, 5],
    #     'num_heads': [1, 2, 4, 8, 10],
    #     'batch_size': [64],
    #     'weight_decay': [0.002],
    #     'learning_rate': [0.001],
    #     'attn_dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
    #     'embed_dropout': [0.1, 0.2, 0.3, 0.4, 0.5]
    # }

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
            'cl_t_weight',
            'cl_v_weight',
            'cl_a_weight',
            'ar_weight',
            'cl_weight',
            'Min Validation Loss',
            'Test MAE', 'Test Corr',
            'Test multiclass(7) accuracy',
            'Test multiclass(5) accuracy',
            'Test binary accuracy', 'Test f1_score'
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
        cl_t_weight = params['cl_t_weight']
        cl_v_weight = params['cl_v_weight']
        cl_a_weight = params['cl_a_weight']
        ar_weight = params['ar_weight']
        cl_weight = params['cl_weight']

        model = Net(text_input_size=input_dims[2], video_input_size=input_dims[1], audio_input_size=input_dims[0],
                    embed_dim=embed_dim, dim_total_proj=dim_total_proj,
                    num_layers=args.num_layers, num_heads=args.num_heads,
                    output_dim=output_dim, seq_len=seq_len,
                    attn_dropout=args.attn_dropout, embed_dropout=args.embed_dropout, use_bert=use_bert,
                    attn_dropout_v=args.attn_dropout_v, attn_dropout_a=args.attn_dropout_a)

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
        print('cl_t_weight'.ljust(30) + '= ' + str(cl_t_weight))
        print('cl_v_weight'.ljust(30) + '= ' + str(cl_v_weight))
        print('cl_a_weight'.ljust(30) + '= ' + str(cl_a_weight))
        print('ar_weight'.ljust(30) + '= ' + str(ar_weight))
        print('cl_weight'.ljust(30) + '= ' + str(cl_weight))
        print('=' * 40)

        AR_loss = ARLoss(reduction='sum')
        L1_loss = nn.L1Loss(reduction='sum')
        Contrastive_loss = ConLoss(reduction='sum', weights=(args.cl_t_weight, args.cl_v_weight, args.cl_a_weight))

        if use_bert:
            bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            bert_params = list(model.text_model.named_parameters())
            bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
            bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
            model_params_other = [p for n, p in list(model.named_parameters()) if 'text_model' not in n]
            optimizer_grouped_parameters = [
                {'params': bert_params_decay, 'weight_decay': 0.001, 'lr': 5e-5},
                {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': 5e-5},
                {'params': model_params_other, 'weight_decay': 0.0, 'lr': lr}
            ]
            opt = optim.Adam(optimizer_grouped_parameters)
        else:
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

        # setup training
        complete = True
        min_valid_loss = float('Inf')

        train_iterator = data.DataLoader(
            train_set, batch_size=args.batch_sz, shuffle=True, num_workers=8)
        valid_iterator = data.DataLoader(
            valid_set, batch_size=args.batch_sz, shuffle=True, num_workers=8)
        test_iterator = data.DataLoader(
            test_set, batch_size=args.batch_sz, shuffle=True, num_workers=8)

        # train
        print('Start Training...')
        curr_patience = patience
        for e in range(num_epochs):
            model.train()
            avg_train_loss = 0.0
            avg_l1_loss = 0.0
            avg_ar_loss = 0.0
            avg_cl_loss = 0.0

            train = tqdm(train_iterator, leave=True)

            for x_a, x_v, x_t, y, meta in train:

                train.set_description(f'Epoch {e + 1:02d}/{num_epochs}')

                x_a, x_v, x_t, y = x_a.float(), x_v.float(), x_t.float(), y.float()
                if args.cuda:
                    x_a, x_v, x_t, y = x_a.to(DEVICE), x_v.to(DEVICE), x_t.to(DEVICE), y.to(DEVICE)

                # train Total
                output, similarity_cube, _, _ = model(x_t, x_v, x_a)

                l1_loss = L1_loss(output, y)
                ar_loss = AR_loss(output, y)
                cl_loss = Contrastive_loss(similarity_cube, y, args.signature)
                loss = l1_loss + args.ar_weight * ar_loss + args.cl_weight * cl_loss

                # Adam
                model.zero_grad()
                loss.backward()
                opt.step()

                avg_train_loss += loss.item()
                avg_l1_loss += l1_loss.item()
                avg_ar_loss += args.ar_weight * ar_loss.item()
                avg_cl_loss += args.cl_weight * cl_loss.item()

            avg_train_loss /= len(train_set)  # reduction - 'sum'
            avg_l1_loss /= len(train_set)  # reduction - 'sum'
            avg_ar_loss /= len(train_set)  # reduction - 'sum'
            avg_cl_loss /= len(train_set)  # reduction - 'sum'

            # Terminate the training process if run into NaN
            if np.isnan(avg_train_loss):
                print("Training got into NaN values...\n\n")
                complete = False
                break

            model.eval()
            avg_valid_loss = 0.0
            with torch.no_grad():
                for x_a, x_v, x_t, y, meta in valid_iterator:

                    x_a, x_v, x_t, y = x_a.float(), x_v.float(), x_t.float(), y.float()
                    if args.cuda:
                        x_a, x_v, x_t, y = x_a.to(DEVICE), x_v.to(DEVICE), x_t.to(DEVICE), y.to(DEVICE)

                    output, _, _, _ = model(x_t, x_v, x_a)

                    valid_loss = L1_loss(output, y)

                    avg_valid_loss += valid_loss.item()

            avg_valid_loss = avg_valid_loss / len(valid_set)  # reduction - 'sum'
            # scheduler.step(avg_valid_loss)

            if np.isnan(avg_valid_loss):
                print("Validating got into NaN values...\n\n")
                complete = False
                break

            print(f"Average Train loss: {avg_train_loss:.4f}\n\t[ "
                  f"Average L1 loss: {avg_l1_loss:.4f} | "
                  f"Average AR loss: {avg_ar_loss:.4f} | "
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
            # video_ids = []
            # t_v_attn = []
            # t_a_attn = []

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

                    output, _, t_v, t_a = best_model(x_t, x_v, x_a)

                    output_test_temp = output

                    # video_ids.append(list(meta))
                    # t_v_attn.append(t_v.cpu().detach().numpy())
                    # t_a_attn.append(t_a.cpu().detach().numpy())
                    y = torch.cat((y, y_test), dim=0)
                    output_test = torch.cat((output_test, output_test_temp), dim=0)

            ########################################################################
            # video_ids = np.concatenate(video_ids)
            # t_v_attn = np.concatenate(t_v_attn)
            # t_a_attn = np.concatenate(t_a_attn)
            # np.save('./[CA]mosi_video_ids.npy', video_ids)
            # np.save('./[CA]mosi_t_v_attn.npy', t_v_attn)
            # np.save('./[CA]mosi_t_a_attn.npy', t_a_attn)
            # these are the needed metrics
            test_preds = output_test.view(-1).cpu().detach().numpy()
            test_truth = y.view(-1).cpu().detach().numpy()

            exclude_zero = True
            non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

            test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
            test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
            test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
            test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

            mae = np.mean(np.absolute(test_preds - test_truth))
            corr = np.corrcoef(test_preds, test_truth)[0][1]
            mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
            mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
            f1 = f1_score((test_truth[non_zeros] > 0), (test_preds[non_zeros] > 0), average='weighted')

            binary_truth = (test_truth[non_zeros] > 0)
            binary_preds = (test_preds[non_zeros] > 0)

            bi_acc = accuracy_score(binary_truth, binary_preds)
            display(mae, corr, mult_a7, mult_a5, bi_acc, f1)

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
                    cl_t_weight,
                    cl_v_weight,
                    cl_a_weight,
                    ar_weight,
                    cl_weight,
                    min_valid_loss,
                    mae, corr,
                    mult_a7,
                    mult_a5,
                    bi_acc, f1
                ])

            return bi_acc
