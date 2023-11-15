# CMU Multimodal SDK, CMU Multimodal Model SDK

# Multimodal Language Analysis with Recurrent Multistage Fusion, Paul Pu Liang, Ziyin Liu, Amir Zadeh, Louis-Philippe Morency - https://arxiv.org/abs/1808.03920

# in_dimensions: the list of dimensionalities of each modality

# cell_size: lstm cell size

# in_modalities: is a list of inputs from each modality - the first dimension of all the modality inputs must be the same, it will be the batch size. The second dimension is the feature dimension. There are a total of n modalities.

# steps: number of iterations for the recurrent fusion

import torch
import time
from torch import nn
import torch.nn.functional as F
from six.moves import reduce

from utils.baselines.submodels import TextSubNet
from utils.models_transformer import BertTextEncoder


class RecurrentFusion(nn.Module):

    def __init__(self, in_dimensions, cell_size):
        super(RecurrentFusion, self).__init__()
        self.in_dimensions = in_dimensions
        self.cell_size = cell_size
        self.model = nn.LSTM(sum(in_dimensions), cell_size)

        self.text_model = BertTextEncoder()  # BERT version

        self.t_subnet = TextSubNet(768, 768, 768)
        self.v_subnet = TextSubNet(35, 35, 35)
        self.a_subnet = TextSubNet(74, 74, 74)

        self.out = nn.Linear(cell_size, 1)

    def __call__(self, in_modalities, steps=1):
        return self.fusion(in_modalities, steps)

    def fusion(self, in_modalities, steps=1):

        in_modalities[0] = self.text_model(in_modalities[0])  # BERT version

        in_modalities[0] = self.t_subnet(in_modalities[0])
        in_modalities[1] = self.v_subnet(in_modalities[1])
        in_modalities[2] = self.a_subnet(in_modalities[2])

        bs = in_modalities[0].shape[0]
        model_input = torch.cat(in_modalities, dim=1).view(1, bs, -1).repeat([steps, 1, 1])
        hidden, cell = (torch.zeros(1, bs, self.cell_size), torch.zeros(1, bs, self.cell_size))
        hidden = hidden.cuda()
        cell = cell.cuda()
        # print('model_input.device = ', model_input.device)
        # print('hidden.device = ', hidden.device)
        # print('cell.device = ', cell.device)
        for i in range(steps):
            outputs, last_states = self.model(model_input, [hidden, cell])
        outputs = self.out(outputs.squeeze(0))
        return outputs, last_states[0], last_states[1]

    def forward(self, x):
        print("Not yet implemented for nn.Sequential")
        exit(-1)


if __name__ == "__main__":
    print("This is a module and hence cannot be called directly ...")
    print("A toy sample will now run ...")

    from torch.autograd import Variable
    import torch.nn.functional as F
    import numpy

    inputx = Variable(torch.Tensor(numpy.zeros([128, 50, 300])), requires_grad=True)
    inputy = Variable(torch.Tensor(numpy.array(numpy.zeros([128, 50, 35]))), requires_grad=True)
    inputz = Variable(torch.Tensor(numpy.array(numpy.zeros([128, 50, 74]))), requires_grad=True)

    modalities = [inputx, inputy, inputz]

    fmodel = RecurrentFusion([300, 35, 74], 8)

    out = fmodel(modalities, steps=1)

    output = out[0].squeeze(0)
    print(output.shape)  # (batch, 1)
