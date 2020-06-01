import torch

import ignite
from ignite.metrics import Recall, Metric
from ignite.exceptions import NotComputableError
from ignite._utils import to_onehot


class Sensitivity(Recall):
    pass


class Specificity(Metric):
    def __init__(self, average=False, output_transform=lambda x: x):
        super(Specificity, self).__init__(output_transform)
        self._average = average

    def reset(self):
        self._n = None
        self._true_negatives = None

    def update(self, output):
        y_pred, y = output
        dtype = y_pred.type()

        if not (y.ndimension() == y_pred.ndimension() or y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError("y must have shape of (batch_size, ...) and y_pred "
                             "must have shape of (batch_size, num_classes, ...) or (batch_size, ...).")

        if y.ndimension() > 1 and y.shape[1] == 1:
            y = y.squeeze(dim=1)

        if y_pred.ndimension() > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=1)

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if not (y_shape == y_pred_shape):
            raise ValueError("y and y_pred must have compatible shapes.")

        if y_pred.ndimension() == y.ndimension():
            # Maps Binary Case to Categorical Case with 2 classes
            y_pred = y_pred.unsqueeze(dim=1)
            y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)

        y = to_onehot(y.view(-1), num_classes=y_pred.size(1))
        indices = torch.max(y_pred, dim=1)[1].view(-1)
        y_pred = to_onehot(indices, num_classes=y_pred.size(1))

        y_pred = y_pred.type(dtype)
        y = y.type(dtype)

        correct = y * y_pred
        wrong = (1-y) * y_pred

        true_negatives = correct.sum(dim=0)
        ridx = torch.arange(
            true_negatives.size(0)-1, -1, -1).to(true_negatives.device)
        true_negatives = true_negatives.index_select(0, ridx)

        false_positives = wrong.sum(dim=0)
        n = true_negatives + false_positives

        if self._n is None:
            self._n = n
            self._true_negatives = true_negatives
        else:
            self._n += n
            self._true_negatives += true_negatives

    def compute(self):
        if self._n is None:
            raise NotComputableError(
                'Specificity must have at least one example before it can be computed')
        result = self._true_negatives / self._n
        result[result != result] = 0.0
        if self._average:
            return result.mean().item()
        else:
            return result
