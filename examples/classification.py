from clad import *
import torch
import torchvision.models
from torch.nn import Linear
from torch.utils.data import DataLoader
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='result',
                        help='Name of the result files')
    parser.add_argument('--root', default="../../data",
                        help='Root folder where the data is stored')
    parser.add_argument('--store', action='store_true',
                        help="If set the prediciton files required for submission will be created")
    parser.add_argument('--test', action='store_true',
                        help='If set model will be evaluated on test set, else on validation set')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If set, training will be on the CPU')
    parser.add_argument('--store_model', action='store_true',
                        help="Stores model if specified. Has no effect is store is not set")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    model = torchvision.models.resnet18(weights=False)
    model.fc = Linear(model.fc.in_features, 7, bias=True)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    batch_size = 10

    train_sets = get_cladc_train(args.root)
    val_set = get_cladc_val(args.root)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4)
    tester = AMCAtester(val_loader, model, device)

    for t, ts in enumerate(train_sets):
        print(f'Training task {t}')
        loader = DataLoader(ts, batch_size=batch_size, num_workers=4, shuffle=False)
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print('testing....')
        tester.evaluate()
        tester.summarize(print_results=True)

    if args.store_model:
        torch.save(model.state_dict(), f'./{args.name}.pt')


if __name__ == '__main__':
    main()
