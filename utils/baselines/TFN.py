# CMU Multimodal SDK, CMU Multimodal Model SDK

# Tensor Fusion Network for Multimodal Sentiment Analysis, Amir Zadeh, Minghai Chen, Soujanya Poria, Erik Cambria, Louis-Philippe Morency - https://arxiv.org/pdf/1707.07250.pdf

# in_modalities: is a list of inputs from each modality - the first dimension of all the modality inputs must be the same, it will be the batch size. The second dimension is the feature dimension. There are a total of n modalities.

# out_dimension: the output of the tensor fusion

import torch
import time
from torch import nn
import torch.nn.functional as F
from six.moves import reduce

from utils.baselines.submodels import TextSubNet
from utils.models_transformer import BertTextEncoder


class TensorFusion(nn.Module):

    def __init__(self, in_dimensions, out_dimension):
        super(TensorFusion, self).__init__()
        self.tensor_size = reduce(lambda x, y: x * y, in_dimensions)
        self.linear_layer = nn.Linear(self.tensor_size, out_dimension)
        self.in_dimensions = in_dimensions
        self.out_dimension = out_dimension

        self.text_model = BertTextEncoder()  # BERT version

        self.t_subnet = TextSubNet(768, 768, 768)
        self.v_subnet = TextSubNet(35, 35, 35)
        self.a_subnet = TextSubNet(74, 74, 74)

        # self.t_subnet = TextSubNet(300, 300, 300)
        # self.v_subnet = TextSubNet(35, 35, 35)
        # self.a_subnet = TextSubNet(74, 74, 74)

    def __call__(self, in_modalities):
        return self.fusion(in_modalities)

    def fusion(self, in_modalities):

        in_modalities[0] = self.text_model(in_modalities[0])  # BERT version

        in_modalities[0] = self.t_subnet(in_modalities[0])
        in_modalities[1] = self.v_subnet(in_modalities[1])
        in_modalities[2] = self.a_subnet(in_modalities[2])

        bs = in_modalities[0].shape[0]
        tensor_product = in_modalities[0]

        # calculating the tensor product

        for in_modality in in_modalities[1:]:
            tensor_product = torch.bmm(tensor_product.unsqueeze(2), in_modality.unsqueeze(1))
            tensor_product = tensor_product.view(bs, -1)

        return self.linear_layer(tensor_product)

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
    inputy = Variable(torch.Tensor(numpy.array(numpy.zeros([128, 50, 20]))), requires_grad=True)
    inputz = Variable(torch.Tensor(numpy.array(numpy.zeros([128, 50, 5]))), requires_grad=True)

    fmodel = TensorFusion([300, 20, 5], 1)

    modalities = [inputx, inputy, inputz]

    out = fmodel(modalities)
    print(out.shape)  # (batch, 1)

