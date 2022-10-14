from collections import defaultdict
from typing import Dict

from clad.utils.meta import SODA_CATEGORIES
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
from torch.utils.data import DataLoader


class AMCAtester:
    """
    Class that calculates the AMCA for the given testset. Every time the model is evaluated it will update
    the results. The AMCA, when calling summarize, is calculated as the average class accuracy over all evaluations
    points. Class accuracy is the mean accuracy of each class.
    """

    def __init__(self, test_loader: torch.utils.data.DataLoader, model: nn.Module, device: str = 'cuda'):

        self.loader = test_loader
        self.model = model
        self.device = device

        self.accs = defaultdict(list)
        self.loss = defaultdict(list)

    def evaluate(self):
        """
        Evaluate the model on the testset.
        """
        accs, loss = test_cladc(self.model, self.loader, self.device)
        for key, value in loss.items():
            self.loss[key].append(value)
        for key, value in accs.items():
            self.accs[key].append(value)

    def summarize(self, print_results=True):
        """
        Returns the accuracy and loss dictionaries and AMCA.
        :param print_results: if True, print a formatted version of the results.
        """
        avg_accuracies = {k: np.mean(v) for k, v in self.accs.items()}
        avg_losses = {k: np.mean(v) for k, v in self.loss.items()}

        amca = np.mean(list(avg_accuracies.values()))

        if print_results:
            print('Current class accuracies:')
            for k, v in sorted(self.accs.items()):
                print(f'{SODA_CATEGORIES[k]:20s}: {v[-1] * 100:.2f}%')
            print(f'AMCA after {len(avg_accuracies)} test points: \n'
                  f'{"AMCA":20s}: {amca * 100:.2f}% \n')

        return avg_accuracies, avg_losses, amca


def test_cladc(model: nn.Module, test_loader: DataLoader, device: str = 'cuda') -> [Dict, Dict]:
    """
    Tests a given model on a given dataloader, returns accuracies and losses per class.
    :param model: the model to test.
    :param test_loader: A DataLoader of the testset
    :param device: cuda/cpu device
    :return: Two dictionaries with the accuracies and losses for each class.
    """
    losses, length, correct = defaultdict(float), defaultdict(float), defaultdict(float)

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target, reduction='none')
        pred = output.argmax(dim=1)

        for lo, pr, ta in zip(loss, pred, target):
            ta = ta.item()
            losses[ta] += lo.item()
            length[ta] += 1
            if pr.item() == ta:
                correct[ta] += 1

    return {label: correct[label] / length[label] for label in length}, \
           {label: losses[label] / length[label] for label in length}
