import os
import sys
import random
import numpy as np
# import matplotlib.pyplot as plt

import torch
# from torchvision import utils


def total(params):
    '''
    count the total number of hyperparameter settings
    '''
    settings = 1
    for k, v in params.items():
        settings *= len(v)
    return settings


# always call this before training for deterministic results
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


# torch cuda, save load model -----------
def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x

    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x.data[0]  # Variable(x)
        return x[0]

    return x.item()


def save_state(model, step, epoch, model_path):
    torch.save({
        "model": model.state_dict(),
        "step": step,
        "epoch": epoch,
    }, str(model_path))


def load_state(model, model_path):
    if os.path.exists(model_path):
        state = torch.load(model_path)
        start_step = state["step"]
        start_epoch = state["epoch"]
        model.load_state_dict(state["model"])
        # print(f"Restore model, step: {start_step} epoch: {start_epoch:.2f}")
        return model, start_step, start_epoch
    else:
        # print(f"Model path {model_path} is not exist.")
        return model, 0, 0


# -----------


class Logger(object):
    def __init__(self, out_dir):
        self.terminal = sys.stdout
        self.out_dir = out_dir
        file_path = os.path.join(out_dir, 'log1.log')

        self.ensure_dir(out_dir)
        self.open(file_path, mode='a')

    def ensure_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            # time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

# -----------


def np_float32_to_uint8(x, scale=255):
    return (x * scale).astype(np.uint8)


def np_uint8_to_float32(x, scale=255):
    return (x / scale).astype(np.float32)


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)
    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)
    else:
        raise NotImplementedError
