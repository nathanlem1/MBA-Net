"""
This implements label smoothing which prevents over-fitting and over-confidence for a classification task i.e. it is a
regularization technique. You can refer for more information on label smoothing here:
https://arxiv.org/pdf/1906.02629.pdf

"""
import torch
import torch.nn.functional as F
import torch.nn as nn


def reduce_loss(loss, reduction='mean'):
    """ Reduce loss
    Args:
        loss: The output (loss here).
        reduction: The reduction to apply to the output (loss here) such as 'mean', 'sum' or 'none'.
    return:
        reduced output (loss here).
    """
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """ Label smoothing cross entropy loss
    Args:
        epsilon: A small constant (smoothing value) to encourage the model to be less confident on the training set.
        reduction: The reduction to apply to the output (loss here) such as 'mean', 'sum' or 'none'.
        preds: Predictions
        target: Labels
    """
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]  # n is the number of classes.
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)  # The negative log likelihood loss

        ls_ce = (loss / n) * self.epsilon + (1 - self.epsilon) * nll

        return ls_ce


# # Usage
# criterion = LabelSmoothingCrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)   # Predictions
# target = torch.empty(3, dtype=torch.long).random_(5)  # Label
# loss = criterion(input.cuda(), target.cuda())
# print(loss)
# # loss.backward()
# # optimizer.step()

