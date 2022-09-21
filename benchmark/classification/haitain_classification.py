import datetime
import re
from typing import Iterator, List

import numpy as np
import pandas as pd
import torch.utils.data
import torchvision
import json
import pickle
import os
from PIL import Image
from itertools import product

from torch.utils.data.sampler import T_co

haitain_domains = {
    "period": ["Daytime", "Night"],
    "weather": ["Clear", "Overcast", "Rainy"],
    "city": ["Guangzhou", "Shenzhen", "Shanghai"],
    "location": ["Citystreet", "Countryroad", "Highway"]
}

categories = {
    1: "Pedestrain",
    2: "Cyclist",
    3: "Car",
    4: "Truck",
    5: "Tram (Bus)",
    6: "Tricycle"
}

START_NIGHT = "195035"  # As long as times don't contain non-numerical chars, str comparison is fine.

_default_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.3252, 0.3283, 0.3407), (0.0265, 0.0241, 0.0252))
])


class HaitainObject:
    """
    A class that contains the image, label and obj id.
    """
    def __init__(self, x, y, obj_id):
        self.x = x
        self.y = y
        self.id = obj_id


class HaitainObjectSet(torch.utils.data.Dataset):
    """
    Dataset wrapper for the haitain data.
    """
    def __init__(self, samples: List[HaitainObject], transform=None, meta=None, image_annot=None, obj_annot=None,
                 img_size=None, root=None, split=None):
        """
        :param samples: An iterable of haitain objects
        :param transform: a transformation to be applied to every object before __getitem__
        :param meta: Some string, to define the dataset
        :param image_annot: The annotations of the images, required if meta data shouldn't be lost
        :param obj_annot: The annotations of the objects, mostly to map obj_id to image_id.
        """
        super(HaitainObjectSet, self).__init__()
        self.samples = samples
        self.transform = transform
        self.meta = meta
        self.image_annot = image_annot
        self.obj_annot = obj_annot
        self.img_size = img_size
        self.root = root
        self.split = split

    @property
    def targets(self):
        """
        :return: List of all labels in the set
        """
        return [int(s.y) for s in self.samples]

    def get_image_annotation(self, item):
        """
        :param item: Index of required sample in self.samples
        :return: Annotation of the image to which the object belongs
        """
        return self.image_annot[self.obj_annot[self.samples[item].id]["image_id"]]

    def lazy_get(self, item):
        obj_id = self.samples[item].id
        file_name = self.image_annot[self.obj_annot[obj_id]['image_id']]['file_name']
        # img = torchvision.io.read_image(os.path.join(self.root, self.split, file_name))
        img = Image.open(os.path.join(self.root, self.split, file_name)).convert('RGB')
        yb, ye, xb, xe = _rescale_bbox(self.obj_annot[obj_id]['bbox'])
        img = img.crop((xb, yb, xe, ye))
        x = img.resize((self.img_size, self.img_size))
        if self.transform is not None:
            x = self.transform(x)
        y = self.obj_annot[obj_id]['category_id']
        return x, y

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.lazy_get(item)

    def chronological_sort(self):
        time_stamps, video_ids, sequence_ids = [], [], []
        for sample in self.samples:
            sample_image_annot = self.image_annot[self.obj_annot[sample.id]["image_id"]]

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
        self.samples = [self.samples[i] for i in order]


class HaitainObjectTestSet(torch.utils.data.Dataset):

    def __init__(self, object_set: HaitainObjectSet, samples=None):
        if samples is None:
            self.samples = [sample[0] for sample in object_set]
        else:
            self.samples = samples
        self.targets = [0 for _ in range(len(self.samples))] # Dummy targets

    def __getitem__(self, item):
        return self.samples[item].contiguous(), self.targets[item]

    def __len__(self):
        return len(self.samples)


def _rescale_bbox(bbox):
    x, y, w, h = bbox
    padding = 5

    size = w if w > h else h
    size = size // 2
    size += padding
    if size > 539:
        size = 539  # Size of images is 1920 * 1080: size can be max 539

    cx, cy = x + w // 2, y + h // 2
    xb, xe, yb, ye = cx - size, cx + size, cy - size, cy + size

    if xb < 0:
        xe += -1 * xb
        xb = 0
    if xe >= 1920:
        xb -= (xe - 1919)
        xe = 1919
    if yb < 0:
        ye += -1 * yb
        yb = 0
    if ye >= 1080:
        yb -= (ye - 1079)
        ye = 1079
    return yb, ye, xb, xe


def _create_domain_keys(keys):
    """
    :return: Dict{key: value ... }
    """
    domain_keys = []
    for key in keys:
        new_keys = []
        values = haitain_domains[key]
        if len(domain_keys) == 0:
            new_keys = values
        else:
            for domain in domain_keys:
                for value in values:
                    new_keys.append(f"{domain}{value}")
        domain_keys = new_keys
    return domain_keys


def _get_domain_key_from_image(img_annot, keys):
    domain_key = ""
    for key in keys:
        domain_key += img_annot[key]
    return domain_key


def _correct_data(img_dic, obj_dic, split):
    if split == "train":
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

    elif split == "val":
        for key in img_dic:
            img_dic[key]['time'] = img_dic[key]['time'].strip()
            img_dic[key]['date'] = img_dic[key]['date'].strip()
            if img_dic[key]["time"] == '1111523':
                img_dic[key]["time"] = '111152'


def _load_obj_img_dic(root: str, split: str):
    if split not in ["train", "test", "val", "train_224", "test_224", "val_224"]:
        raise ValueError(f"Split {split} not recognized")

    split = split.split("_")[0]
    with open(f"{root}/annotations/instance_{split}.json", "r") as f:
        annot_json = json.load(f)

    obj_dic = {obj["id"]: obj for obj in annot_json["annotations"]}
    img_dic = {img["id"]: img for img in annot_json["images"]}

    _correct_data(img_dic, obj_dic, split)

    return obj_dic, img_dic


def _load_all_objects(root, split):
    objects = []
    files = os.listdir(f"{root}/processed/{split}")

    for file in files:
        with open(f"{root}/processed/{split}/{file}", "rb") as f:
            data_file: dict = pickle.load(f)

        for obj_id in data_file:
            objects.append(HaitainObject(data_file[obj_id][0], data_file[obj_id][1], obj_id))
    return objects


def _load_all_objects_ids(obj_dic, min_area=1024, remove_occluded=True):
    objects = []
    for obj_id, obj in obj_dic.items():
        if obj["area"] < min_area:
            continue
        if remove_occluded and obj["occluded"] == 0:
            continue
        objects.append(HaitainObject(None, obj['category_id'], obj_id))
    return objects


def get_matching_set(root, split, match_fn, img_size=None, transform=None) -> HaitainObjectSet:
    """
    :param root: root path of where to look for pickled object files
    :param split: train, val or test split
    :param match_fn: A function that takes a sample, the obj and img dicts and return T/F if a sample should be
                     in the dataset
    :param img_size: If set, the objects will be lazely loaded and cutted from the images rather than loading them
                     in memory. This is much slower, but required for larger images.
    :param transform: Transofrmation to apply to images. If None, _default_transform will be applied.
    :return: an HaitainObjectSet containing all samples that evaluate to true in match_fn
    """

    obj_dic, img_dic = _load_obj_img_dic(root, split)
    if img_size is None:
        all_objects = _load_all_objects(root, split)
    else:
        all_objects = _load_all_objects_ids(obj_dic)

    if transform is None:
        transform = _default_transform

    matching = [obj for obj in all_objects if match_fn(obj, img_dic, obj_dic)]
    return HaitainObjectSet(matching, transform, image_annot=img_dic, obj_annot=obj_dic,
                            root=root, split=split, img_size=img_size)


def create_match_date(date):
    """
    :param date: date in yyyymmdd format, as a string
    :return: a function that evaluates to true if the object is from the given date
    """
    def match_fn(obj, img_dic, obj_dic):
        img_annot = img_dic[obj_dic[obj.id]['image_id']]
        return img_annot["Date"] == date
    return match_fn


def create_match_dict_fn(match_dict):
    """
    :param match_dict: dictionary that should match the objects
    :return: a function that evaluates to true if the object is from the given date
    """
    def match_fn(obj, img_dic, obj_dic):
        img_annot = img_dic[obj_dic[obj.id]['image_id']]
        for key, value in match_dict.items():
            if img_annot[key] != value:
                return False
        else:
            return True
    return match_fn


def get_haitain_domain_sets(root, split, keys=None, match_fn=None, img_size=None):
    """
    :param root: root file of dataset
    :param split: training, validation or test
    :param keys: object sets will be split for all combinations of the given values of these keys.
    :return: List[HaitainObjectSet]
    """
    obj_dic, img_dic = _load_obj_img_dic(root, split)
    if match_fn is None and img_size is None:
        objects = _load_all_objects(root, split)
    elif match_fn is None and img_size is not None:
        objects = _load_all_objects_ids(obj_dic)
    else:
        objects = get_matching_set(root, split, match_fn, img_size).samples

    if keys is None:
        return HaitainObjectSet(objects, _default_transform, image_annot=img_dic, obj_annot=obj_dic,
                                root=root, split=split, img_size=img_size)
    else:
        domain_keys = _create_domain_keys(keys)
        domain_objects = {key: [] for key in domain_keys}

        for obj in objects:
            image_annot = img_dic[obj_dic[obj.id]["image_id"]]
            obj_key = _get_domain_key_from_image(image_annot, keys)
            domain_objects[obj_key].append(obj)

        return [HaitainObjectSet(domain_objects[key], _default_transform, key, img_dic, obj_dic,
                                 root=root, split=split, img_size=img_size) for key in domain_keys]


class ChronologicalSampler(torch.utils.data.Sampler):
    """
    A chronological sampler for the HaitainObjectSet. Order is based on timestamp - video_id - seq_id
    """

    def __init__(self, dataset: HaitainObjectSet):
        super(ChronologicalSampler, self).__init__(dataset)
        self.dataset = dataset
        time_stamps, video_ids, sequence_ids = [], [], []
        for sample in dataset.samples:
            sample_image_annot = self.dataset.image_annot[self.dataset.obj_annot[sample.id]["image_id"]]

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
        self.indices = np.lexsort((sequence_ids, video_ids, time_stamps))

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def get_item_time(self, item):
        """
        Not optimal here, but access to current time is nice during training. But dataloader
        does only return (x, y).
        """
        obj_idx = self.indices[item]
        image_annot = self.dataset.get_image_annotation(obj_idx)
        return image_annot["date"].strip(), image_annot["time"].strip()


class Logger:
    def __init__(self, type='class'):
        if type == 'class':
            self.keys = ['time', 'period', 'weather', 'city', 'location',
                         'acc ped', 'acc cyc', 'acc car', 'acc tru', 'acc tra', 'acc tri',
                         'loss ped', 'loss cyc', 'loss car', 'loss tru', 'loss tra', 'loss tri',
                         'count ped', 'count cyc', 'count car', 'count tru', 'count tra', 'count tri']
        else:
            raise ValueError
        self.data = []
        self.df = pd.DataFrame()

    def append(self, result, test_id, run_id):
        result = {k: v for k, v in zip(self.keys, result)}
        result['test_id'] = test_id
        result['run_id'] = run_id
        self.data.append(result)

    def build_df(self):
        new_df = pd.DataFrame(self.data)
        self.df.replace([-1.0], np.NaN, inplace=True)  # print(logger)
        self.df = pd.concat([self.df, new_df])

    def average_by_class(self):
        self.df['acc'] = self.df[['acc ped', 'acc cyc', 'acc car', 'acc tru', 'acc tra', 'acc tri']].mean(1)

    def plot_stats(self, ax, domains, average=False, metric='acc', class_keys=None, ylim=(0, 1), color=None):
        if average:
            self.average_by_class()

        domain_values = [haitain_domains[domain] for domain in domains]
        if average:
            value_keys = [metric]
        else:
            if class_keys is None:
                class_keys = ["ped", "cyc", "car", "tru", "tra", "tri"]
            value_keys = [f'{metric} {ck}' for ck in class_keys]

        grouped_by_run = self.df.groupby([*domains, 'test_id', 'run_id'])[value_keys].mean()
        grouped = grouped_by_run.groupby([*domains, 'test_id'])
        accuracies = grouped.mean()
        std = grouped.std()

        x_labels = self.df.groupby(['test_id'])['time'].min()
        x_labels = pd.to_datetime(x_labels)
        x = range(len(x_labels))

        for comb in product(*domain_values):
            try:
                for vk in value_keys:
                    y = accuracies.loc[comb][vk]
                    dev = std.loc[comb][vk]
                    label = _beautify_label(vk, comb)
                    line = ax.plot(x, y, '-o', label=label, lw=2, color=color, zorder=2)
                    ax.fill_between(x, y-dev, y+dev, color=line[0].get_color(), alpha=0.25, zorder=1)
            except KeyError:
                print(comb)
                continue

        day_night_idx = _get_night_switches(x_labels)
        for i in range(0, len(day_night_idx), 2):
            ax.axvspan(day_night_idx[i], day_night_idx[i+1], color='gainsboro', alpha=0.75, zorder=0)

        ax.set_ylim(ylim)
        x_labels = [xl.strftime('%d/%m - %H:%M') for xl in x_labels]
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=75)
        # ax.legend(bbox_to_anchor=(1.2, 1.0))
        ax.legend()

    def save(self, path):
        self.df.to_pickle(path)

    def load(self, name):
        self.df = pd.read_pickle(f'./results/stats_{name}.pkl')

    def load_path(self, path):
        self.df = pd.read_pickle(path)

    def table_print(self, timestamp, avg=False):
        table_data = self.df[self.df['time'] == timestamp]
        columns = ['acc'] if avg else self.keys[5:11]
        grouped = table_data.groupby(['city', 'weather', 'period', 'location'])[columns].mean()

        for city in haitain_domains['city']:
            for weather in haitain_domains['weather']:
                for period in haitain_domains['period']:
                    for location in haitain_domains['location']:
                        try:
                            to_print = grouped.loc[city, weather, period, location].values
                            for value in to_print:
                                if np.isnan(value):
                                    print('/, ', end="")
                                else:
                                    print(f"{value}, ", end="")
                        except KeyError:
                            to_print = '/, ' * len(columns)
                            print(to_print, end="")
                print()

    def __repr__(self):
        return self.df.__repr__()

    def __getitem__(self, item):
        return self.df.__getitem__(item)


def _beautify_label(vk, comb):
    if len(comb) == 0:
        return vk
    else:
        result = f"{vk}: "
        for c in comb:
            result += f"{c}, "
    return result[:-2]


def _get_night_switches(times):
    switch = []
    night = times.iloc[0].strftime('%H%M%S') >= START_NIGHT
    for i, time in enumerate(times):
        time = time.strftime('%H%M%S')
        time_is_night = time >= START_NIGHT
        if night != time_is_night:
            switch.append(i-1)
            night = time_is_night
    if len(switch) % 2 != 0:
        switch.append(-1)
    return switch


def key_to_domains(key):
    domains = [a for a in re.split(r'([A-Z][a-z]*)', key) if a]
    return domains


def log_avalanche_results(results, domain_keys, logger, test_id, run_id=0):

    for i, dk in enumerate(domain_keys):
        # Don't have access to timestamps here.
        log_result = [datetime.datetime(1970, 1, 1, 0, 0, 0), *key_to_domains(dk)]
        accs = []
        for key, value in results.items():
            try:
                exp_id = int(re.findall(r'\d+', key.split('/')[-1])[0])
            except IndexError:
                continue

            if exp_id == i:
                metric_name = key.split('/')[0]
                if metric_name == "Top1_ClassAcc_Exp":
                    for t in range(1, 7):
                        try:
                            accs.append(value[t])
                        except KeyError:
                            accs.append(-1)

        log_result.extend(accs)
        logger.append(log_result, test_id=test_id, run_id=run_id)
