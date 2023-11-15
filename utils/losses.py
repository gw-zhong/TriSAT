import torch
import torch.nn as nn
import torch.nn.functional as F


class ARLoss(nn.Module):
    """
    suitable for mosi/mosei, because they're regression tasks,
    but not suitable for iemocap, because it's a classification task.
    """
    def __init__(self, reduction='mean'):
        super(ARLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        batch, dim = input.shape
        eps = 1e-06

        classes = torch.round(target)
        zero = torch.tensor(0., device=target.device)
        loss = torch.Tensor().to(target.device)

        for i in range(batch):
            delta = input[i] - classes[i]  # delta -> (output_dim)
            batch_loss = torch.Tensor().to(target.device)
            for j in range(dim):
                if delta[j] >= 0:
                    z = torch.ceil(input[i][j]).detach()
                    temp_loss = torch.max(
                        zero, torch.abs(delta[j]) - torch.abs(input[i][j] - z) + eps
                    )
                else:
                    z = torch.floor(input[i][j]).detach()
                    temp_loss = torch.max(
                        zero, torch.abs(delta[j]) - torch.abs(input[i][j] - z)
                    )
                batch_loss = torch.cat((batch_loss, temp_loss.reshape(1, 1)), dim=1)
            loss = torch.cat((loss, batch_loss), dim=0)

        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class ConLoss(nn.Module):
    """
    a simple implementation for self-supervised contrastive loss

    batch, _ = target.shape
    label = torch.arange(batch, device=target.device)

    logits_t = similarity_cube[label, :, label]
    logits_v = similarity_cube[label, label, :]
    logits_a = similarity_cube.transpose(0, 1)[label, :, label]

    loss_t = F.cross_entropy(logits_t, label, reduction=self.reduction)
    loss_v = F.cross_entropy(logits_v, label, reduction=self.reduction)
    loss_a = F.cross_entropy(logits_a, label, reduction=self.reduction)

    loss = (loss_t + loss_v + loss_a) / 3
    """
    def __init__(self, reduction='mean', weights=(1.0, 1.0, 1.0)):
        super(ConLoss, self).__init__()
        self.reduction = reduction
        self.weights = weights

    def forward(self, similarity_cube, target, dataset):

        assert dataset == 'mosi' or dataset == 'mosei' or dataset == 'iemocap', \
            'dataset must be mosi/mosei/iemocap'

        batch, dim = target.shape
        eps = torch.tensor([1e-6], device=target.device)
        eps = eps.repeat(batch ** 2)

        if dataset == 'mosi' or dataset == 'mosei':

            classes = torch.round(target)
            classes[classes >= 0] = 3  # 2-class fine-grained
            classes[classes < 0] = -3  # 2-class fine-grained
            mask = get_mask_cube(classes, batch)

            loss = get_contrastive_loss(similarity_cube, mask, eps, batch, weights=self.weights)

            if self.reduction == 'sum':
                loss = loss.sum() / batch
            if self.reduction == 'mean':
                loss = loss.mean()

            return loss

        elif dataset == 'iemocap':

            loss = torch.zeros(batch**2, device=target.device)
            for idx in range(dim):
                classes = target[:, [idx]]
                classes[classes == 0] = 10  # prevent 0 and False
                mask = get_mask_cube(classes, batch)

                tmp_loss = get_contrastive_loss(similarity_cube, mask, eps, batch, weights=self.weights)
                loss += tmp_loss

            loss /= dim

            if self.reduction == 'sum':
                loss = loss.sum() / batch
            if self.reduction == 'mean':
                loss = loss.mean()

            return loss


def get_mask_cube(classes, batch):
    """
    :param classes: need to be (batch, 1)
    :param batch: the batch_size
    :return: the mask cube - (batch, batch, batch)
    """
    matrix_t = classes.repeat(1, batch)
    matrix_v = classes.transpose(0, 1).repeat(batch, 1)
    matrix_a = classes.unsqueeze(-1).repeat(1, batch, batch)
    mask_flat = torch.eq(matrix_t, matrix_v)
    intersection = matrix_t * mask_flat  # use matrix_v * mask_flat is also right
    intersection = intersection.unsqueeze(0).repeat(batch, 1, 1)
    mask = torch.eq(intersection, matrix_a).float()

    return mask


def get_contrastive_loss(similarity_cube, mask, eps, batch, weights):
    """
    :param similarity_cube: (batch, batch, batch)
    :param mask: (batch, batch, batch)
    :param eps: default=1e-6, in order to avoid divide by 0
    :param batch: the batch_size
    :param weights: the weights of loss_t, loss_v, and loss_a
    :return: the contrastive loss - (batch**2)
    """
    logit_t = similarity_cube.transpose(1, 2).contiguous().view(-1, batch)
    mask_t = mask.transpose(1, 2).contiguous().view(-1, batch)

    logit_v = similarity_cube.view(-1, batch)
    mask_v = mask.view(-1, batch)

    logit_a = similarity_cube.permute(1, 2, 0).view(-1, batch)
    mask_a = mask.permute(1, 2, 0).view(-1, batch)

    loss_t = -torch.sum(F.log_softmax(logit_t, dim=-1) * mask_t, dim=-1) / (
        torch.max(torch.sum(mask_t, dim=-1), eps))
    loss_v = -torch.sum(F.log_softmax(logit_v, dim=-1) * mask_v, dim=-1) / (
        torch.max(torch.sum(mask_v, dim=-1), eps))
    loss_a = -torch.sum(F.log_softmax(logit_a, dim=-1) * mask_a, dim=-1) / (
        torch.max(torch.sum(mask_a, dim=-1), eps))

    # loss = (loss_t + loss_v + loss_a) / 3
    loss = weights[0] * loss_t + weights[1] * loss_v + weights[2] * loss_a

    return loss
