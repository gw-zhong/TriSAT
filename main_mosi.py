import os
import argparse

import optuna
import torch
import numpy as np
from optuna.samplers import TPESampler

from utils.dataset import load_mosi, load_mosi_bert
from utils.tools import Logger, seed_everything
from utils.train_mosi_transformer import train

SEED = 2352

os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def init_envs(seed):
    log = Logger('.')

    # noinspection PyBroadException
    try:
        log.write(
            f'os[\'CUDA_VISIBLE_DEVICES\']     = {os.environ["CUDA_VISIBLE_DEVICES"]}\n'
        )
    except Exception:
        log.write(f'os[\'CUDA_VISIBLE_DEVICES\']     = None\n')

    log.write(f'torch.__version__              = {torch.__version__}\n')
    log.write(f'torch.version.cuda             = {torch.version.cuda}\n')
    log.write(
        f'torch.backends.cudnn.version() = {torch.backends.cudnn.version()}\n')
    log.write(
        f'torch.cuda.device_count()      = {torch.cuda.device_count()}\n')

    log.write('\n')

    return log


def main(args):
    log = init_envs(args)
    seed_everything(SEED)

    # parse the input args
    run_id = args.run_id
    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir
    signature = args.signature

    log.write(f'Training initializing... Setup ID is: {run_id}\n')

    # prepare the paths for storing models and outputs
    args.model_path = os.path.join(model_dir, 'model_{}_{}.pt'.format(
        signature, run_id))

    args.output_path = os.path.join(
        output_dir, 'results_{}_{}.csv'.format(signature, run_id))

    log.write(f'Temp location for models: {args.model_path}\n'
              f'Grid search results are in: {args.output_path}\n\n')

    if args.use_bert:
        train_set, valid_set, test_set, input_dims = load_mosi_bert(data_dir)
    else:
        train_set, valid_set, test_set, input_dims = load_mosi(data_dir)
    print(f"input_dims is {input_dims}")

    ans = train(train_set, valid_set, test_set, input_dims, args, log)
    return ans


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="mosi")

    arg_parser.add_argument(
        '--run_id', dest='run_id', type=int, default=100)

    arg_parser.add_argument(
        '--num_epochs', dest='num_epochs', type=int, default=100)
    arg_parser.add_argument(
        '--patience', dest='patience', type=int, default=10)

    arg_parser.add_argument(
        '--signature', dest='signature', type=str, default='mosi')
    arg_parser.add_argument(
        '--cuda', dest='cuda', type=bool, default=True)
    arg_parser.add_argument(
        '--use_bert', dest='use_bert', type=bool, default=False)

    arg_parser.add_argument(
        '--data_dir', dest='data_dir', type=str, default='input')
    arg_parser.add_argument(
        '--model_dir', dest='model_dir', type=str, default='models/mosi')
    arg_parser.add_argument(
        '--output_dir', dest='output_dir', type=str, default='results/mosi')

    arg_parser.add_argument(
        '--output_dim', dest='output_dim', type=int, default=1)
    arg_parser.add_argument(
        '--seq_len', dest='seq_len', type=int, default=50)

    # Hyperparameters
    arg_parser.add_argument(
        '--embed_dim', type=int, default=40)
    arg_parser.add_argument(
        '--dim_total_proj', type=int, default=32)
    arg_parser.add_argument(
        '--num_layers', type=int, default=5)
    arg_parser.add_argument(
        '--num_heads', type=int, default=10)
    arg_parser.add_argument(
        '--batch_sz', type=int, default=128)
    arg_parser.add_argument(
        '--decay', type=float, default=0.002)
    arg_parser.add_argument(
        '--lr', type=float, default=0.001)
    arg_parser.add_argument(
        '--attn_dropout', type=float, default=0.2)
    arg_parser.add_argument(
        '--attn_dropout_v', type=float, default=0.0)
    arg_parser.add_argument(
        '--attn_dropout_a', type=float, default=0.0)
    arg_parser.add_argument(
        '--embed_dropout', type=float, default=0.3)
    arg_parser.add_argument(
        '--cl_t_weight', type=float, default=0.333)
    arg_parser.add_argument(
        '--cl_v_weight', type=float, default=0.333)
    arg_parser.add_argument(
        '--cl_a_weight', type=float, default=0.333)
    arg_parser.add_argument(
        '--ar_weight', type=float, default=1.0)
    arg_parser.add_argument(
        '--cl_weight', type=float, default=1.0)

    args = arg_parser.parse_args()

    sampler = TPESampler(seed=SEED)
    url = f'sqlite:///trisat.db'

    def objective(trial):
        # args.num_layers = trial.suggest_int('num_layers', 1, 5, step=1)
        # args.num_heads = trial.suggest_categorical('num_heads', choices=[1, 2, 4, 8, 10])
        # args.attn_dropout = trial.suggest_float('attn_dropout', 0.0, 0.5)
        # args.attn_dropout_v = trial.suggest_float('attn_dropout_v', 0.0, 0.5)
        # args.attn_dropout_a = trial.suggest_float('attn_dropout_a', 0.0, 0.5)
        # args.embed_dropout = trial.suggest_float('embed_dropout', 0.0, 0.5)
        # args.cl_t_weight = trial.suggest_float('cl_t_weight', 0.0, 1.0)
        # args.cl_v_weight = trial.suggest_float('cl_v_weight', 0.0, 1.0)
        # args.cl_a_weight = trial.suggest_float('cl_a_weight', 0.0, 1.0)
        # args.ar_weight = trial.suggest_float('ar_weight', 0.0, 1.0)
        # args.cl_weight = trial.suggest_float('cl_weight', 0.0, 1.0)

        ans = main(args)

        return ans

    n_trials = 100
    print('Start Hyperparameter Tuning....')
    print(f'Total Trials Are {n_trials}....')
    print(f'Your Database Url is {url}....')
    study = optuna.create_study(
        direction='maximize', sampler=sampler,
        study_name='mosi[acc2]', storage=url, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)
    best_trial = study.best_trial
    print('-' * 50)
    print(f'Best Accuracy is: {best_trial.value:.4f}')
    print(f'Best Hyperparameters are:\n{best_trial.params}')

    df = study.trials_dataframe().to_csv(
        'results/mosi/[optuna][acc2]results_mosi_{}.csv'.format(args.run_id))


