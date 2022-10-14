from clad.utils.utils import *

import torchvision
import torch
import random

from PIL import Image
from collections import defaultdict
from torch.utils.data import DataLoader
from typing import Optional, List, Sequence, Callable, Dict, Any


class CladDetection(torch.utils.data.Dataset):
    """
    A class that creates a Clad-D dataset, which will covers a given domain. Class incremental style datasets
    isn't supported in this dataset.

    :param root: root path of the folder where files are stored
    :param ids: Ids of the images that should be in the dataset
    :param transform: Transform to be applied to images before returning
    :param meta: Any string with usefull meta information.
    """

    def __init__(self, root: str,
                 ids: Sequence[int],
                 annot_file: str,
                 transform: Optional[Callable] = None,
                 meta: str = None,
                 ):
        super(CladDetection).__init__()

        split = annot_file.split('_')[-1].split('.')[0]

        self.img_folder = os.path.join(root, 'SSLAD-2D', 'labeled', split)
        self.ids = ids
        self.transform = transform if transform is not None else get_transform(split == 'train')
        self.meta = meta

        self.obj_annotations, self.img_annotations = load_obj_img_dic(annot_file)
        self._remove_empty_images()
        self.img_anns = self._create_index()

    def _remove_empty_images(self):
        """
        Required because torchvision models can't handle empty lists for bbox in targets
        """
        non_empty_images = set()
        for obj in self.obj_annotations.values():
            non_empty_images.add(obj["image_id"])
        self.ids = [img_id for img_id in self.ids if img_id in non_empty_images]

    def _create_index(self):
        img_anns = defaultdict(list)
        for ann in self.obj_annotations.values():
            img_anns[ann['image_id']].append(ann)
        return img_anns

    @property
    def targets(self):
        """
        Get a list of all category ids, required for Avalanche.
        """
        targets = []
        for img_id in self.img_anns:
            targets.extend(obj['category_id'] for obj in self.img_anns[img_id])
        return torch.tensor(targets)

    def _load_target(self, index: str):
        img_id = self.ids[index]
        img_objects = self.img_anns[img_id]

        boxes = []
        for obj in img_objects:
            bbox = obj["bbox"]
            # Convert from x, y, h, w to x0, y0, x1, y1
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor([obj["category_id"] for obj in img_objects], dtype=torch.int64)
        area = torch.as_tensor([obj["area"] for obj in img_objects])
        iscrowd = torch.as_tensor([obj["iscrowd"] for obj in img_objects], dtype=torch.int64)

        # Targets should all be tensors
        target = {"boxes": boxes, "labels": labels, "image_id": torch.as_tensor(img_id, dtype=torch.int64),
                  "area": area, "iscrowd": iscrowd}

        return target

    def _load_image(self, index):
        file_name = self.img_annotations[self.ids[index]]['file_name']
        return Image.open(os.path.join(self.img_folder, file_name)).convert('RGB')

    def __getitem__(self, index):
        image = self._load_image(index)
        target = self._load_target(index)

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.ids)


def get_matching_detection_set(root: str, annot_file: str, match_fn: Callable, transform=None,
                               meta: str = None) -> CladDetection:
    """
    Creates CladDetection set from a match_fn

    :param root: root path of where to look for pickled object files
    :param annot_file: annotation file from root
    :param match_fn: A function that takes a sample, the obj and img dicts and return T/F if a sample should be
                     in the dataset
    :param transform: Transformation to apply to images. If None, _default_transform will be applied.
    :param meta: optional meta information in a str that will be stored with the dataset.
    :return: CladDetection object
    """

    _, img_dic = load_obj_img_dic(annot_file)
    img_ids = [image for image in img_dic if match_fn(image, img_dic)]

    return CladDetection(root, img_ids, annot_file, transform, meta)


def get_cladd_domain_sets(root: str, annot_file: str, domains: Sequence[str], transform: Callable = None,
                          match_fn: Callable = None) -> Sequence[CladDetection]:
    """
    :param root: Root directory of the dataset
    :param annot_file: the annotation file for the dataset
    :param domains: the domains that should be included in the domains sets (e.g. ['period', 'city'])
    :param match_fn: a method that returns True if a given sample should be in the dataset.
    :param transform
    """

    if match_fn is None:
        def match_fn(*args): return True

    domain_dicts = create_domain_dicts(domains)
    domain_set = []
    for domain_dict in domain_dicts:
        domain_match_fn = create_match_dict_fn_img(domain_dict)
        ds = get_matching_detection_set(root, annot_file, lambda *args: domain_match_fn(*args) and match_fn(*args),
                                        transform, meta='-'.join(domain_dict.values()))
        domain_set.append(ds)
    return domain_set


def create_match_dict_fn_img(match_dict: Dict[Any, Any]) -> Callable:
    """
    Creates a method that returns true if the image specified by the img_id
    is in the specified domain of the given match_dict.
    :param match_dict: dictionary that should match the objects
    :return: a function that evaluates to true if the object is from the given date
    """

    def match_fn(img_id: int, img_dic: Dict[str, Dict]) -> bool:
        img_annot = img_dic[img_id]
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


def create_val_from_trainset(trainset: CladDetection, root, val_transform, split, proportion=0.1):
    cut_off = int((1.0 - proportion) * len(trainset))
    all_imgs = trainset.ids
    trainset.ids = all_imgs[:cut_off]
    val_ids = all_imgs[cut_off:]
    return CladDetection(root, val_ids, os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations',
                                                     f'instance_{split}.json'), val_transform, trainset.meta)


# Below adapted from pytorch vision example on detection, but removed unnecessary code.

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = torchvision.transforms.functional.to_tensor(image)
        return image, target


def get_transform(train):
    transform_arr = [ToTensor()]
    if train:
        transform_arr.append(RandomHorizontalFlip(0.5))
    return Compose(transform_arr)
