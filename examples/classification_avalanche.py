from clad import *

import torch
import torchvision.models
from torch.nn import Linear

import argparse

from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, amca_metrics
from avalanche.logging import TextLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import Naive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='result',
                        help='Name of the result files')
    parser.add_argument('--root', default="../../data",
                        help='Root folder where the data is stored')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Num workers to use for dataloading')
    parser.add_argument('--store', action='store_true',
                        help="If set the prediciton files required for submission will be created")
    parser.add_argument('--no_cuda', action='store_true',
                        help='If set, training will be on the CPU')
    parser.add_argument('--store_model', action='store_true',
                        help="Stores model if specified. Has no effect is store is not set")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    model = torchvision.models.resnet18(pretrained=True)
    model.fc = Linear(model.fc.in_features, 7, bias=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    batch_size = 10

    cladc = cladc_avalanche(args.root)

    text_logger = TextLogger(open(f"./{args.name}.log", 'w'))
    interactive_logger = InteractiveLogger()
    plugins = []

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(stream=True), loss_metrics(stream=True), amca_metrics(),
        loggers=[text_logger, interactive_logger])

    strategy = Naive(
        model, optimizer, criterion, train_mb_size=batch_size, train_epochs=1, eval_mb_size=256, device=device,
        evaluator=eval_plugin, plugins=plugins)

    for i, experience in enumerate(cladc.train_stream):

        strategy.train(experience, eval_streams=[], shuffle=False, num_workers=args.num_workers)
        results = strategy.eval(cladc.test_stream, num_workers=args.num_workers)

        print(results)

    if args.store_model:
        torch.save(model.state_dict(), f'./{args.name}.pt')


if __name__ == '__main__':
    main()
