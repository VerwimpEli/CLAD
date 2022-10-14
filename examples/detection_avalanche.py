import clad

import logging
import argparse
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from avalanche.training.supervised.naive_object_detection import ObjectDetectionTemplate
from avalanche.evaluation.metrics import timing_metrics, loss_metrics
from avalanche.evaluation.metrics.detection import DetectionMetrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import LRSchedulerPlugin, EvaluationPlugin


# This sets the root logger to write to stdout (your console).
# Your script/app needs to call this somewhere at least once.
logging.basicConfig(level=logging.NOTSET)


def train_cladd_avalanche():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../../data', help='root of SODA10M dataset')
    parser.add_argument('--no_cuda', action='store_true', help='disable cuda training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    torch.random.manual_seed(1997)
    benchmark = clad.get_cladd_avalanche(args.root)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    num_classes = benchmark.n_classes + 1  # N classes + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    train_mb_size = 5
    warmup_factor = 1.0 / 1000
    warmup_iters = min(1000, len(benchmark.train_stream[0].dataset) // train_mb_size - 1)

    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)
    
    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = ObjectDetectionTemplate(
        model=model,
        optimizer=optimizer,
        train_mb_size=train_mb_size,
        train_epochs=1,
        eval_mb_size=train_mb_size,
        device=device,
        plugins=[
            LRSchedulerPlugin(
                lr_scheduler,
                step_granularity="iteration",
                first_exp_only=True,
                first_epoch_only=True,
            )
        ],
        evaluator=EvaluationPlugin(
            timing_metrics(epoch=True),
            loss_metrics(epoch_running=True),
            DetectionMetrics(default_to_coco=True),
            loggers=[InteractiveLogger()],
        ),
    )

    # TRAINING LOOP
    print("Starting experiment...")
    for i, experience in enumerate(benchmark.train_stream):
        print("Start of experience: ", experience.current_experience)
        print("Train dataset contains", len(experience.dataset), "instances")

        cl_strategy.train(experience, num_workers=4)
        print("Training completed")

        cl_strategy.eval(benchmark.test_stream, num_workers=4)
        print("Evaluation completed")


if __name__ == "__main__":
    train_cladd_avalanche()
