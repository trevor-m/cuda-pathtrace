from load_data import load_exr_data, get_patches
import torch
from torch.utils.data import TensorDataset

def get_dataset(patches=True):
    train_inputs = []
    train_targets = []

    for i in range(33):
        if patches:
            train_patches, gt_patches = get_patches("../data/{}_train.exr".format(i), "../data/{}_gt.exr".format(i), patch_size=256, num_patches=10, preprocess=True)
            train_inputs.extend([torch.Tensor(x) for x in train_patches])
            train_targets.extend([torch.Tensor(x) for x in gt_patches])
        else:
            train_inputs.extend(torch.Tensor(
                load_exr_data("../data/{}_train.exr".format(i), preprocess=True, concat=True))
            )
            train_targets.extend(torch.Tensor(
                load_exr_data("../data/{}_gt.exr".format(i), preprocess=True, concat=True, target=True))
            )

    train_inputs = torch.stack(train_inputs, 0)
    train_targets = torch.stack(train_targets, 0)
    train_dataset = TensorDataset(train_inputs, train_targets)

    test_inputs = torch.stack([torch.Tensor(load_exr_data("../data/0_train.exr", preprocess=True, concat=True))], 0)
    test_targets = torch.stack([torch.Tensor(load_exr_data("../data/0_gt.exr", preprocess=True, concat=True, target=True))], 0)
    test_dataset = TensorDataset(test_inputs, test_targets)
    return train_dataset, test_dataset