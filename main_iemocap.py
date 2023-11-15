import os
import argparse
import torch
import numpy as np

from utils.dataset import load_iemocap
from utils.tools import Logger, seed_everything
from utils.train_iemocap_reproduce import train

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

    train_set, valid_set, test_set, input_dims = load_iemocap(data_dir)
    print(f"input_dims is {input_dims}")

    train(train_set, valid_set, test_set, input_dims, args, log)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="iemocap")

    arg_parser.add_argument(
        '--run_id', dest='run_id', type=int, default=1)

    arg_parser.add_argument(
        '--num_epochs', dest='num_epochs', type=int, default=200)
    arg_parser.add_argument(
        '--patience', dest='patience', type=int, default=10)

    arg_parser.add_argument(
        '--signature', dest='signature', type=str, default='iemocap')
    arg_parser.add_argument(
        '--cuda', dest='cuda', type=bool, default=True)

    arg_parser.add_argument(
        '--data_dir', dest='data_dir', type=str, default='input')
    arg_parser.add_argument(
        '--model_dir', dest='model_dir', type=str, default='models/iemocap')
    arg_parser.add_argument(
        '--output_dir', dest='output_dir', type=str, default='results/iemocap')

    arg_parser.add_argument(
        '--output_dim', dest='output_dim', type=int, default=8)
    arg_parser.add_argument(
        '--seq_len', dest='seq_len', type=int, default=20)

    args = arg_parser.parse_args()

    main(args)
