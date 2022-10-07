import logging
import os
import argparse
from datetime import datetime
import clad

from detectron2.model_zoo import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.logger import setup_logger


def train_cladd_detectron():

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="yaml config file")
    parser.add_argument('--root', default='../../data', help='root of SODA10M dataset')
    parser.add_argument('--outdir', default="detectron_results", help="Base output dir")
    parser.add_argument('--no_cuda', action='store_true', help="Run on cpu")
    parser.add_argument('--resume', action='store_true', help="Resume training")
    parser.add_argument('--test_only', action='store_true', help="Don't train, only test")
    args = parser.parse_args()

    # This registers the CLAD-D datasets in Detectron2. They're accessible with the names cladd_T[i]_[split], with
    # i the task-ID and [split] one of train/val/test.
    clad.register_cladd_detectron(args.root)

    cfg = setup(args)

    if not args.test_only:
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        trainer.train()

    setup_detectron_logger(args)
    predictor = DefaultPredictor(cfg)

    for test_dataset in cfg.DATASETS.TEST:
        evaluator = COCOEvaluator(test_dataset, output_dir=f"{cfg.OUTPUT_DIR}/{test_dataset}")
        val_loader = build_detection_test_loader(cfg, test_dataset)
        print(inference_on_dataset(predictor.model, val_loader, evaluator))


def setup(args):
    """
    Sets up config file.
    """

    cfg = get_cfg()

    # Loads basic config file and then merges with args.config
    cfg.merge_from_file(model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_C4.yaml"))
    cfg.merge_from_file(args.config)
    cfg.MODEL.DEVICE = "cpu" if args.no_cuda else cfg.MODEL.DEVICE

    # Makes a new sub-folder in the current output dir to store results of this run, unless resume is set
    if args.outdir is not None and not args.resume:
        time_stamp = datetime.now().strftime('%H%M%S')
        cfg.OUTPUT_DIR = f"{args.outdir}/output_{time_stamp}"
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    else:
        cfg.OUTPUT_DIR = args.outdir
    return cfg


def setup_detectron_logger(args):
    # If there's no training, set up logger.
    if args.test_only:
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()


if __name__ == '__main__':
    train_cladd_detectron()
