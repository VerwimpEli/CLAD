import os.path
import pickle

from avalanche.benchmarks.utils import AvalancheDataset

import benchmark.classification.haitain_classification as hc
from avalanche.evaluation import Metric, GenericPluginMetric


def create_val_set(root, img_size, avalanche=True):
    def val_match_fn_1(obj, img_dic, obj_dic):
        img_annot = img_dic[obj_dic[obj.id]['image_id']]
        date = img_annot["date"]
        return not (date == "20191120" or date == "20191117" or date == "20191111" or
                    (date == "20191121" and img_annot['period'] == "Night"))

    def val_match_fn_2(obj, img_dic, obj_dic):
        img_annot = img_dic[obj_dic[obj.id]['image_id']]
        time = img_annot['time']
        date = img_annot["date"]
        return obj.y == 6 or (obj.y == 2 and date == "20181015" and (time == '152030'
                                                                     or time == '160000'))

    val_set_1 = hc.get_matching_set(root, 'val', val_match_fn_1, img_size=img_size)
    val_set_2 = hc.get_matching_set(root, 'train', val_match_fn_2, img_size=img_size)

    if avalanche:
        return [AvalancheDataset(val_set_1), AvalancheDataset(val_set_2)]
    else:
        return [val_set_1, val_set_2]


def create_test_set_from_pkl(root, avalanche=True):
    file_names = os.listdir(os.path.join(root, 'test_track3A'))
    file_names = sorted(file_names, key=lambda x: int(x[0:2]))
    test_sets = []
    for file in file_names:
        with open(os.path.join(root, 'test_track3A', file), 'rb') as f:
            ds = pickle.load(f)
            ds = hc.HaitainObjectTestSet(None, ds.samples)
            test_sets.append(ds)
    if avalanche:
        test_sets = [AvalancheDataset(ts) for ts in test_sets]

    return test_sets, None


def create_test_set(root, img_size, avalanche=True):
    test_sets = hc.get_haitain_domain_sets(root, 'test', ["period", "weather", "city", "location"],
                                           img_size=img_size)
    test_sets_keys = [ds.meta for ds in test_sets if len(ds) > 0]

    if avalanche:
        return [AvalancheDataset(test_set) for test_set in test_sets if len(test_set) > 0], test_sets_keys
    else:
        return [ts for ts in test_sets if len(ts) > 0], test_sets_keys


def create_train_set(root, img_size, avalanche=True, transform=None):
    task_dicts = [{'date': '20191111', 'period': 'Daytime'},
                  {'date': '20191111', 'period': 'Night'},
                  {'date': '20191117', 'period': 'Daytime'},
                  {'date': '20191117', 'period': 'Night'},
                  {'date': '20191120', 'period': 'Daytime'},
                  {'date': '20191121', 'period': 'Night'}, ]

    match_fn = (hc.create_match_dict_fn(td) for td in task_dicts)

    train_sets = [hc.get_matching_set(root, 'val', mf, img_size=img_size, transform=transform) for mf in match_fn]
    for ts in train_sets:
        ts.chronological_sort()

    if avalanche:
        return [AvalancheDataset(train_set) for train_set in train_sets]
    else:
        return train_sets


class ClassEvaluationPlugin(GenericPluginMetric):

    def __init__(self, reset_at, emit_at, mode, store=None):
        self._class_results = ClassificationOutputMetric(store)
        super(ClassEvaluationPlugin, self).__init__(
            self._class_results, reset_at=reset_at, emit_at=emit_at, mode=mode)

    def update(self, strategy):
        self._class_results.update(strategy.mb_output)


class ClassificationOutputMetric(Metric):
    """
    This is a hack to get the outputs in a seperate file. Rather than in the logging file.
    In the future Avalanche should support which metrics are logged to which file.
    """
    def __init__(self, store: str=None):
        self.store = store
        if self.store is not None:
            index = 0
            while os.path.isfile(f"./{store}_{index}_output.txt"):
                index += 1
            self.store = f"{self.store}_{index}"
        self._output = ""

    def update(self, prediction):
        for line in prediction:
            for elem in line:
                self._output += f"{elem}, "
            self._output += "\n"

    def result(self):
        if self.store is not None and len(self._output) > 0:
            file_name = f"{self.store}_output.txt"
            with open(file_name, 'a') as f:
                f.write(self._output)
        return ""

    def reset(self) -> None:
        self._output = ""
