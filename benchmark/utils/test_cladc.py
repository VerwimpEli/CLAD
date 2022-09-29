from collections import defaultdict
from meta import SODA_CATEGORIES
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
from torch.utils.data import DataLoader


class AMCAtester:

    def __init__(self, test_loader: torch.utils.data.DataLoader, model: nn.Module, device: str = 'cuda'):

        self.loader = test_loader
        self.model = model
        self.device = device

        self.accs = defaultdict(list)
        self.loss = defaultdict(list)

    def evaluate(self):
        accs, loss = test_cladc(self.model, self.loader, self.device)
        for key, value in loss.items():
            self.loss[key].append(value)
        for key, value in accs.items():
            self.accs[key].append(value)

    def summarize(self, print_results=True):
        avg_accuracies = {k: np.mean(v) for k, v in self.accs.items()}
        avg_losses = {k: np.mean(v) for k, v in self.accs.items()}

        amca = np.mean(list(avg_accuracies.values()))

        if print_results:
            for k, v in avg_accuracies.items():
                print(f'{SODA_CATEGORIES[k]:20s}: {v*100:.2f}%')
            print(f'{"AMCA":20s}: {amca*100:.2f}%')

        return avg_accuracies, avg_losses


def test_cladc(model: nn.Module, test_loader: DataLoader, device: str = 'cuda'):
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

    return {label: losses[label] / length[label] for label in losses.keys()}, \
           {label: correct[label] / length[label] for label in losses.keys()}

