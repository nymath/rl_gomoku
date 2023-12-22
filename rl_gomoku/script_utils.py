import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import PolicyValueNet
from torch.optim import Optimizer
from typing import TypedDict


class ModelArgs(TypedDict):
    in_channels: int
    drop_out: int
    learning_rate: float
    weight_decay: float
    device: float
    use_attention: bool



class TrainingArgs(TypedDict):
    n_iterations: int
    buffer_size: int
    n_playout: int
    board_size: int
    update_times_per_episode: int
    log_freq: int
    batch_size: int


def default_args():
    model_args: ModelArgs = {
        "in_channels": 7,
        "drop_out": 0.0001,
        "device": "cuda:0",
        "learning_rate": 0.0002,
        "weight_decay": 0.00001,
        "use_attention": True,
    }
    
    training_args: TrainingArgs = {
        "n_iterations": 5000,
        "buffer_size": 8192,
        "n_playout": 400,
        "board_size": 15,
        "update_times_per_episode": 5,
        "log_freq": 200,
        "batch_size": 256,
    }
    return model_args, training_args


def create_model_from_args(args: ModelArgs):
    model = PolicyValueNet(
        in_channels=args['in_channels'],
        drop_out=args['drop_out'],
        device=args['device'],
        lr=args['learning_rate'],
        weight_decay=args['weight_decay'],
        use_attention=args['use_attention']
    )
    return model

    
    



