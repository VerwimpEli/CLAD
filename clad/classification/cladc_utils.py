from clad.utils.utils import *

import datetime
import re
import os
import numpy as np
import torch.utils.data
import torchvision

from typing import Optional, Callable, Any, Sequence, Dict, List
from functools import lru_cache
from PIL import Image
from collections import defaultdict


class CladClassification(torch.utils.data.Dataset):
    """
    A class that creates a Clad-C dataset (train, val or test), by cutting out the labelled objects in the
    SODA10M dataset. This dataset is meant to be trained in chronological order, so far training
    self.chronological_sort() should always be called before training, and the dataloader shouldn't be shuffling data.
    This doens't matter for the validation and/or testset (but it is more efficient to keep the default order).

    :param root: root directory of the datasets
    :param ids: a sequence of the object ids that are part of the dataset
    :param annot_file: the .json file that contains the object and image annotations for this dataset
    :param img_size: the width and height of the cut-out objects
    :param transform and optional callabale that transforms the cut-out objects
    :param meta: an optional user-defined str for the current set, usefull if objects are all from the same domain.
    """

    _default_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.3252, 0.3283, 0.3407), (0.0265, 0.0241, 0.0252))
    ])

    def __init__(
            self,
            root: str,
            ids: Sequence[int],
            annot_file: str,
            img_size: int = 64,
            transform: Optional[Callable] = None,
            meta: str = None
    ):
        super(CladClassification).__init__()

        split = annot_file.split('_')[-1].split('.')[0]

        self.img_folder = os.path.join(root, 'SSLAD-2D', 'labeled', split)
        self.ids = ids
        self.obj_annotations, self.img_annotations = load_obj_img_dic(annot_file)
        self.img_size = img_size
        self.transform = transform if transform is not None else CladClassification._default_transform

        self._prev_loaded = defaultdict(int)
        self._sorted = False

        self.meta = meta

    @property
    def targets(self):
        return [self._load_target(obj_id) for obj_id in self.ids]

    def _load_image(self, obj_id: int) -> Image.Image:

        file_name = self.img_annotations[self.obj_annotations[obj_id]['image_id']]['file_name']
        img = _load_image(os.path.join(self.img_folder, file_name))

        yb, ye, xb, xe = squarify_bbox(self.obj_annotations[obj_id]['bbox'])
        img = img.crop((xb, yb, xe, ye))
        img = img.resize((self.img_size, self.img_size))

        return img

    def _load_target(self, obj_id: int) -> int:
        return self.obj_annotations[obj_id]['category_id']

    def _check_order(self, item):
        """
        Checks that the data is loaded chronologically if the dataset was sorted. This isn't completely fool-proof,
        if chronological_sort is not called this doens't mean anything since then the ids won't be ordered and only
        the indexes are checked.
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        if item < self._prev_loaded[worker_id]:
            print("[WARNING] Accessing non-sequential elements in CLAD-C. This can happen for rehearsal based solutions"
                  " etc. Make sure you're training sequentially and not repeating samples though.")
            self._sorted = False  # Issued warning, no reason to check further now.
        else:
            self._prev_loaded[worker_id] = item

    def __getitem__(self, item):

        if self._sorted:
            self._check_order(item)

        obj_id = self.ids[item]
        image = self._load_image(obj_id)
        target = self._load_target(obj_id)

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.ids)

    def chronological_sort(self):
        time_stamps, video_ids, sequence_ids = [], [], []

        for sample in self.ids:
            sample_image_annot = self.img_annotations[self.obj_annotations[sample]["image_id"]]

            try:
                time_stamp = datetime.datetime.strptime(sample_image_annot["date"].strip() +
                                                        sample_image_annot["time"].strip(), '%Y%m%d%H%M%S')
            except ValueError:
                try:
                    time_stamp = datetime.datetime.strptime(sample_image_annot["date"].strip() +
                                                            sample_image_annot["time"].strip(), '%Y-%m-%d%H:%M:%S')
                except ValueError:
                    print(f"Error with image id: {sample_image_annot['id']}, time: {sample_image_annot['time']} \t"
                          f"date: {sample_image_annot['date']}, period: {sample_image_annot['period']}")
                    raise ValueError

            video_id, seq_id = re.findall(r'\d+', sample_image_annot['file_name_old'].split('/')[-1])
            video_ids.append(video_id)
            sequence_ids.append(seq_id)
            time_stamps.append(time_stamp)

        # Sort by time, then video id, then seq id
        order = np.lexsort((sequence_ids, video_ids, time_stamps))
        self.ids = [self.ids[i] for i in order]
        self._sorted = True


def get_matching_classification_set(root: str, annot_file: str, match_fn: Callable, img_size=64, transform=None,
                                    meta: str = None) -> CladClassification:
    """
    Creates CladClassification set from a match_fn

    :param root: root path of where to look for pickled object files
    :param annot_file: annotation file from root
    :param match_fn: A function that takes a sample, the obj and img dicts and return T/F if a sample should be
                     in the dataset
    :param img_size: Size of the rescaled, cut out object
    :param transform: Transformation to apply to images. If None, _default_transform will be applied.
    :param meta: optional meta information in a str that will be stored with the dataset.
    :return: CladClassification object
    """

    obj_dic, img_dic = load_obj_img_dic(annot_file)
    object_ids = [obj_id for obj_id in _load_object_ids(obj_dic) if match_fn(obj_id, img_dic, obj_dic)]

    return CladClassification(root, object_ids, annot_file, img_size, transform, meta)


def get_cladc_domain_sets(root: str, annot_file: str, domains: Sequence[str], img_size=64, transform=None,
                          match_fn: Callable = None) -> Sequence[CladClassification]:
    """
    Creates a sequence of sets of the specified domains, with all samples matching the possibly specified match_fn
    """
    if match_fn is None:
        def match_fn(*args): return True

    domain_dicts = create_domain_dicts(domains)
    domain_set = []
    for domain_dict in domain_dicts:
        domain_match_fn = create_match_dict_fn(domain_dict)
        ds = get_matching_classification_set(root, annot_file, lambda *args: domain_match_fn(*args) and match_fn(*args),
                                             img_size, transform, meta='-'.join(domain_dict.values()))
        domain_set.append(ds)
    return domain_set


def create_match_dict_fn(match_dict: Dict[Any, Any]) -> Callable:
    """
    Creates a method that returns true if the object specified by the obj_id
    is in the specified domain of the given match_dict.
    :param match_dict: dictionary that should match the objects
    :return: a function that evaluates to true if the object is from the given date
    """

    def match_fn(obj_id, img_dic, obj_dic):
        img_annot = img_dic[obj_dic[obj_id]['image_id']]
        for key, value in match_dict.items():
            if isinstance(value, List):
                if img_annot[key] not in value:
                    return False
            else:
                if img_annot[key] != value:
                    return False
        else:
            return True

    return match_fn


def _load_object_ids(obj_dic, min_area=1024, remove_occluded=True):
    """
    This prepares a set of obj_ids with possible objects, with small and occluded ones removed.
    """
    objects = []
    for obj_id, obj in obj_dic.items():
        if obj["area"] < min_area:
            continue
        if remove_occluded and obj["occluded"] == 0:
            continue
        objects.append(obj_id)
    return objects


@lru_cache(16)
def _load_image(path) -> Image.Image:
    return Image.open(path).convert('RGB')
