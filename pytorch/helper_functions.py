from timeit import default_timer as timer

import torch
from torch import nn

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start: float,
                     end: float,
                     device: torch.device):
    """ Prints difference between start and end time."""

    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

