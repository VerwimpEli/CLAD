import re
import argparse
import torchvision.models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_multi_dataset_generic_benchmark
from avalanche.logging import TextLogger
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin

from detection_util import *
from detection_strategy import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='result',
                        help='Name of the result files')
    parser.add_argument('--root', default="../data",
                        help='Root folder where data is stored.')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Num workers to use for dataloading.')
    parser.add_argument('--test', action='store_true',
                        help='If set model will be evaluated on test set, else on validation set')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--store', action='store_true',
                        help='If set the predicition files required for submission will be created')
    parser.add_argument('--store_model', action='store_true',
                        help="Stores model if specified. Has no effect is store is not set")
    parser.add_argument('--load_model', type=str, default=None,
                        help='Loads model with given name. Model should be stored in current folder')
    parser.add_argument('--test_only', action='store_true')
    args = parser.parse_args()

    ######################################
    #                                    #
    #  Editing below this line allowed   #
    #                                    #
    ######################################

    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    data_root = args.root = f"{args.root}/SSLAD-2D/labeled"

    epochs = 10
    batch_size = 1

    # Setup model, optimizer and dummy criterion
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 7)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    lr_scheduler_plugin = LRSchedulerPlugin(lr_scheduler)

    plugins = [DetectionStrategyPlugin(), lr_scheduler_plugin]

    ######################################
    #                                    #
    # No editing below this line allowed #
    #                                    #
    ######################################

    if args.load_model is not None:
        model.load_state_dict(torch.load(f"./{args.load_model}"))

    # Setup Benchmark
    train_datasets, val_datasets = create_train_val_set(data_root, validation_proportion=0.1)

    if args.test:
        eval_datasets, _ = create_test_set_from_json(data_root)
    else:
        eval_datasets = val_datasets

    benchmark = create_multi_dataset_generic_benchmark(train_datasets=train_datasets, test_datasets=eval_datasets)

    # Setup evaluation and logging
    test_split = "test" if args.test else "val"

    result_file = open(f"./{args.name}_{test_split}.txt", "w")
    logger = TextLogger(result_file)
    gt_path = f"{args.root}/annotations/instance_{test_split}.json"
    store = None if not args.store else f"{args.name}_{test_split}"
    eval_plugin = EvaluationPlugin(detection_metrics(gt_path, experience=True, store=store, pred_only=args.test),
                                   loggers=logger)

    # Create strategy.
    criterion = empty
    strategy = DetectionBaseStrategy(
        model, optimizer, criterion, train_mb_size=batch_size, train_epochs=epochs,
        eval_mb_size=batch_size, device=device, evaluator=eval_plugin, plugins=plugins)

    if args.test_only:
        results = strategy.eval(benchmark.test_stream, num_workers=args.num_workers)
        task_mean_map = sum(float(re.findall(r'\d+.\d+', rv)[-1]) for rv in results.values()) / len(results)
        result_file.writelines([f"Task mean MAP: {task_mean_map:.3f} \n"])

    else:
        for train_exp in benchmark.train_stream:
            strategy.train(train_exp, num_workers=args.num_workers)

        # Only evaluate at the end of training
        results = strategy.eval(benchmark.test_stream, num_workers=args.num_workers)
        task_mean_map = sum(float(re.findall(r'\d+.\d+', rv)[-1]) for rv in results.values()) / len(results)
        result_file.writelines([f"Task mean MAP: {task_mean_map:.3f} \n"])

    if args.store_model:
        torch.save(model.state_dict(), f'./{args.name}.pt')

    result_file.close()


if __name__ == '__main__':
    main()
