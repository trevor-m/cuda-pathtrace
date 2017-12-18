from load_data import load_exr_data
import torch
from torch.utils.data import TensorDataset

def get_dataset():
    train_inputs = []
    train_targets = []

    for i in range(16):
        train_inputs.append(torch.Tensor(
            load_exr_data("training/{}_train.exr".format(i), preprocess=True, concat=True))
        )
        train_targets.append(torch.Tensor(
            load_exr_data("training/{}_gt.exr".format(i), preprocess=True, concat=True, target=True))
        )

    train_inputs = torch.stack(train_inputs, 0)
    train_targets = torch.stack(train_targets, 0)

    train_dataset = TensorDataset(train_inputs, train_targets)
    test_dataset = TensorDataset(train_inputs, train_targets)
    return train_dataset, test_dataset