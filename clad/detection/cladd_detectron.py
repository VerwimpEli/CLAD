"""
All helper code to prepare the detectron versions of the CLAD-D benchmark.
"""
from clad.utils.meta import CLADD_TRAIN_VAL_DOMAINS, CLADD_TEST_DOMAINS, SODA_CATEGORIES
from clad.detection.cladd_utils import create_match_dict_fn_img

import json
import os

from typing import Dict
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from collections import defaultdict


def register_cladd_detectron(root: str):
    """
    This method will register the CLAD-D datasets in Detectron2. They will be registered as cladd_T[i]_[split], with
    i the task-ID and [split] one of train/val/test.
    :param root: the root directory
    """

    # These refer to the original SODA10M splits, not those of CLAD-D
    cladd_trainval_orig_splits = ['train', 'val', 'val', 'val']

    for t, (task_dict, orig_split) in enumerate(zip(CLADD_TRAIN_VAL_DOMAINS, cladd_trainval_orig_splits)):
        for split in ['train', 'val']:
            dataset_name = f'cladd_T{t+1}_{split}'
            DatasetCatalog.register(dataset_name,
                                    lambda r=root, o=orig_split, s=split,
                                    td=task_dict: _get_cladd_detectron_domain_set(r, o, s, td))
            MetadataCatalog.get(dataset_name).set(thing_classes=list(SODA_CATEGORIES.values()))

    for t, task_dict in enumerate(CLADD_TEST_DOMAINS):
        dataset_name = f'cladd_T{t+1}_test'
        DatasetCatalog.register(dataset_name,
                                lambda r=root, td=task_dict: _get_cladd_detectron_domain_set(r, 'test', 'test', td))
        MetadataCatalog.get(dataset_name).set(thing_classes=list(SODA_CATEGORIES.values()))


def _get_cladd_detectron_domain_set(root_dir: str, soda_split: str, cladd_split: str, task_dict: Dict[str, str]):
    """
    This shouldn't be called directly, but it filters out the images and annotations that don't match the task
    dictionary. The origin split is the original split of the haitain dataset, while split is whether the
    training or validation data is asked for by the caller of this method.
    """
    full_set = get_soda10m_detectron_dict(root_dir, soda_split)
    match_fn = create_match_dict_fn_img(task_dict)

    matched_images = [img for img in full_set if match_fn(0, {0: img})]
    validation_proportion = 0.1

    cut_off = int((1.0 - validation_proportion) * len(matched_images))
    if cladd_split == "train":
        matched_images = matched_images[:cut_off]
    elif cladd_split == "val":
        matched_images = matched_images[cut_off:]

    return matched_images


def get_soda10m_detectron_dict(root: str, soda_split: str):
    """
    This method loads the original SODA10M split annotations in detectron format, which is slightly different from
    the default ones.
    :param root: the root dir of the SODA10M dataset
    :param soda_split: train/val/test split of SODA10M to get the correct file/images
    :return: Dictionary with the annotations.
    """
    dict_path = os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations', f'instance_{soda_split}.json')
    with open(dict_path, 'r') as f:
        instances = json.load(f)

    img_annots = defaultdict(list)
    for obj in instances['annotations']:
        obj['category_id'] -= 1
        obj['bbox_mode'] = BoxMode.XYWH_ABS
        img_annots[obj['image_id']].append(obj)

    dataset_dicts = []
    for img in instances['images']:
        img['file_name'] = os.path.join(root, 'SSLAD-2D', 'labeled', soda_split, img['file_name'])
        img['annotations'] = img_annots[img['id']]
        dataset_dicts.append(img)

    return dataset_dicts
