from clad.detection.cladd_utils import *
from clad.utils.meta import CLADD_TRAIN_VAL_DOMAINS, CLADD_TEST_DOMAINS
import os


def get_cladd_trainval(root: str, train_transform: Callable = None, val_transform: Callable = None, avalanche=False)\
        -> [Sequence[CladDetection], Sequence[CladDetection]]:
    """
    Creates the CLAD-D benchmarks train and validation sets, as in the ICCV '21 challenge. The validation set is
    10% of the training sets of each task. This isn't attached to any framework and only depends on PyTorch itself.
    The dataset objects are in COCO format.

    :param root: root path to the dataset
    :param train_transform: transformation for the train set. If none is given, the default one is used. See
                           `get_tranform`.
    :param val_transform: transformation for the validation set. If none is given, the default is used.
    :param avalanche: If true, Avalanche-type datasets will be returned.
    """
    if train_transform is None:
        train_transform = get_transform(train=True)
    if val_transform is None:
        val_transform = get_transform(train=False)

    splits = ['train', 'val', 'val', 'val']
    match_fns = [create_match_dict_fn_img(td) for td in CLADD_TRAIN_VAL_DOMAINS]
    train_sets = [get_matching_detection_set(root,
                                             os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations',
                                                          f'instance_{split}.json'),
                                             match_fn, train_transform) for
                  match_fn, split in zip(match_fns, splits)]

    validation_proportion = 0.1
    val_sets = [create_val_from_trainset(ts, root, val_transform, split, validation_proportion) for ts, split in
                zip(train_sets, splits)]

    if avalanche:
        from avalanche.benchmarks.utils import AvalancheDataset
        return [AvalancheDataset(train_set) for train_set in train_sets], \
               [AvalancheDataset(val_set) for val_set in val_sets]
    else:
        return train_sets, val_sets


def get_cladd_test(root: str, transform: Callable = None, avalanche=False) -> Sequence[CladDetection]:
    """
     Creates the CLAD-D benchmarks train and validation sets, as in the ICCV '21 challenge. This isn't attached to any
     framework and only depends on PyTorch itself. The dataset objects are in COCO format.

    :param root: root path to the dataset
    :param transform: transformation for the test set. If none is given, the default one is used. See
                           `get_tranform`.
    :param avalanche: If true, Avalanche-type datasets will be returned.
    """

    if transform is None:
        transform = get_transform(train=True)

    match_fns = [create_match_dict_fn_img(td) for td in CLADD_TEST_DOMAINS]
    test_sets = [get_matching_detection_set(root,
                                            os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations',
                                                         f'instance_test.json'),
                                            match_fn, transform) for match_fn in match_fns]

    if avalanche:
        from avalanche.benchmarks.utils import AvalancheDataset
        return [AvalancheDataset(test_set) for test_set in test_sets]
    else:
        return test_sets


def get_cladd_avalanche(root: str):
    from avalanche.benchmarks.scenarios.detection_scenario import DetectionCLScenario

    train_sets, val_sets = get_cladd_trainval(root, avalanche=True)
    return DetectionCLScenario({"train": (train_sets, ), "test": (val_sets, )}, n_classes=6)


def collate_fn_cladd(batch):
    return list(zip(*batch))
