import random

import torch
import torch.utils.data
from typing import Callable, List, Any, Dict, Tuple, Union, Optional
from collections import defaultdict
from functools import lru_cache

import torchvision
from PIL import Image
import os
import json

haitain_domains = {
    "period": ["Daytime", "Night"],
    "weather": ["Clear", "Overcast", "Rainy"],
    "city": ["Guangzhou", "Shenzhen", "Shanghai"],
    "location": ["Citystreet", "Countryroad", "Highway"]
}


class HaitainDetectionSet(torch.utils.data.Dataset):

    def __init__(self, root: str,
                 samples: List[str],
                 annotation_file: str = None,
                 transform: Any = None,
                 meta: Any = None,
                 test: bool = False,
                 ):
        """
        :param root: root path of the folder where files are stored
        :param samples: File names of samples
        :param test: If set no labels are loaded.
        :param transform: Transform to be applied to images before returning
        :param meta: Anything to describe data. (Usefull during testing).
        """
        self.root = root
        self.samples = samples
        self.transform = transform
        self.meta = meta
        self.test = test

        if not test:
            self.annotation_file = annotation_file
            self.img_annotations, self.obj_annotations = _load_annotations(annotation_file)
            self._remove_empty_images()
        else:
            self.annotation_file, self.img_annotations, self.obj_annotations = None, None, None

        self.categories = {1: "Pedestrian", 2: "Cyclist", 3: "Car", 4: "Truck", 5: "Tram", 6: "Tricycle"}

    def _remove_empty_images(self):
        """
        Required because torchvision models can't handle empty lists for bbox in targets
        """
        non_empty_images = set()
        for obj in self.obj_annotations.values():
            non_empty_images.add(obj["image_id"])
        self.samples = [sample for sample in self.samples if sample in non_empty_images]

    def _get_objects_in_image(self, img_id: int):
        return [obj for obj in self.obj_annotations.values() if obj['image_id'] == img_id]

    @property
    def targets(self):
        """
        make dict with {img_id: [targets]}
        :return:
        """
        if self.test:
            return [0 for _ in range(len(self.samples))]

        img_objects = defaultdict(list)
        for obj_values in self.obj_annotations.values():
            img_objects[obj_values['image_id']].append(obj_values['category_id'])

        targets = []
        for img_id in self.samples:
            targets.extend(img_objects[img_id])
        return torch.tensor(targets)
        # return torch.cat([self._load_target(i)['labels'] for i in range(len(self))])

    def _load_target(self, index: str):
        img_id = self.img_annotations[self.samples[index]]['id']
        img_objects = self._get_objects_in_image(img_id)

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

    def __getitem__(self, index):
        if self.test:
            file_name = self.samples[index]
        else:
            file_name = self.img_annotations[self.samples[index]]['file_name']
        img = Image.open(os.path.join(self.root, file_name)).convert('RGB')

        if self.test:
            return self.transform(img, {'image_id': file_name})
        else:
            target = self._load_target(index)

            if self.transform is not None:
                img, target = self.transform(img, target)

            return img, target

    def __len__(self):
        return len(self.samples)


def get_full_set(root: str, split: str, transform: Callable = None, meta: str = None) -> HaitainDetectionSet:
    return get_matching_set(root, split, match_fn=None, transform=transform, meta=meta)


def get_matching_set(root: str, split: str,
                     match_fn: Callable[[str, Any], bool] = None,
                     transform: Optional[Callable] = None,
                     meta: str = None) -> HaitainDetectionSet:
    """
    :param root:
    :param split:
    :param match_fn:
    :param meta
    :param transform
    :return: A HaitainDetectionSet where all samples satisify the match_fn
    """

    annotation_path = os.path.join(root, 'annotations', f'instance_{split}.json')
    image_annotations, _ = _load_annotations(annotation_path)

    if match_fn is not None:
        samples = [image for image in image_annotations if match_fn(image, image_annotations)]
    else:
        samples = list(image_annotations.keys())

    root = os.path.join(root, split)
    return HaitainDetectionSet(root, samples, annotation_path, transform, meta)


def get_domain_sets(root: str, split: str, keys: List[str] = None,
                    match_fn: Callable[[Any], bool] = None,
                    transform: Callable = None) \
        -> Union[List[HaitainDetectionSet], HaitainDetectionSet]:
    """
    :param root: Root directory of the dataset
    :param split: Which split of the directory to load
    :param keys: Which keys should be used in the split. Should be one or more of ['period', 'city', 'weather',
                'location', 'categories'.]
    :param match_fn: A function that returns true if the sample should be included and false otherwise. This is used
                     before applying the keys split.
    :param transform
    :return:
    """
    annotation_path = os.path.join(root, 'annotations', f'instance_{split}.json')
    image_annotations, obj_annotations = _load_annotations(annotation_path)

    if match_fn is None:
        samples = get_full_set(root, split).samples
    else:
        samples = get_matching_set(root, split, match_fn).samples

    root = os.path.join(root, split)

    if keys is None:
        return HaitainDetectionSet(root, samples, annotation_path, transform)
    else:
        domain_keys = _create_domain_keys(keys)
        domain_images = {key: [] for key in domain_keys}

        for image in samples:
            image_annot = image_annotations[image]
            obj_key = _get_domain_key_from_image_annot(image_annot, keys)
            domain_images[obj_key].append(image)

        # TODO: would be easier if meta is transformed back to dict.
        return [HaitainDetectionSet(root, domain_images[key], annotation_path, transform,
                                    meta=key) for key in domain_keys if len(domain_images[key]) > 0]


def create_match_fn_from_dict(match_dict: Dict[str, Union[str, List[str]]]) -> \
        Callable[[str, Any], bool]:
    def match_fn(img_name: str, img_annotations: Dict[str, Dict]) -> bool:
        img_annot = img_annotations[img_name]
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


def _create_domain_keys(domains: List[str]) -> List[str]:
    """
    :return: A list of unique keys by the domains specified
    """
    domain_keys = []
    for domain in domains:
        new_keys = []
        values = haitain_domains[domain]
        if len(domain_keys) == 0:
            new_keys = values
        else:
            for key in domain_keys:
                for value in values:
                    new_keys.append(f"{key}{value}")
        domain_keys = new_keys
    return domain_keys


def _get_domain_key_from_image_annot(img_annot: Dict, domains: List[str]) -> str:
    """
    :param img_annot: The annotation dictionary for the image involved.
    :param domains: list of domains for which to contstruct the key
    :return: A string that serves as a unique key for the domain
    """
    domain_key = ""
    for domain in domains:
        domain_key += img_annot[domain]
    return domain_key


@lru_cache(128)
def _load_annotations(file_name: str) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:

    with open(file_name, 'r') as f:
        annot_json = json.load(f)

    obj_annotations = {obj['id']: obj for obj in annot_json["annotations"]}
    img_annotations = {img['id']: img for img in annot_json["images"]}

    # _correct_data(img_annotations, obj_annotations)

    return img_annotations, obj_annotations


def _correct_data(img_dic, obj_dic):
    """
    Corrects known mistakes in the annotations files.
    :param img_dic:
    :param obj_dic:
    :return:
    """
    for key in img_dic:
        if img_dic[key]["time"] == '145960':
            img_dic[key]["time"] = '150000'
        elif img_dic[key]["time"] == '102360':
            img_dic[key]["time"] = '102400'
        elif img_dic[key]["time"] == '1221831':
            img_dic[key]["time"] = '122131'
        elif img_dic[key]["time"] == '1420214':
            img_dic[key]["time"] = '142021'
        elif img_dic[key]["time"] == '10:00':
            img_dic[key]["time"] = '100000'
        elif img_dic[key]["time"] == '13:00':
            img_dic[key]["time"] = '130000'
        elif img_dic[key]["time"] == ' 13:00':
            img_dic[key]["time"] = '130000'
        elif img_dic[key]["time"] == '12:00':
            img_dic[key]["time"] = '120000'

        if img_dic[key]["date"] == "201810152":
            img_dic[key]["date"] = "20181015"
        elif img_dic[key]["date"] == "50181015":
            img_dic[key]["date"] = "20181015"

        img_dic[key]['time'] = img_dic[key]['time'].strip()
        img_dic[key]['date'] = img_dic[key]['date'].strip()
        if img_dic[key]["time"] == '1111523':
            img_dic[key]["time"] = '111152'


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
