from scipy.sparse.linalg import spsolve

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from argparse import ArgumentParser

from src.model import PINN
from src.train import Trainer
from src.utils import plot_timestep


def main(args):
    """
    Entry point for training or evaluating a Physics-Informed Neural Network
    for the 2D heat equation.

    Depending on the input arguments, the function either trains the model
    or loads a pretrained checkpoint and visualizes the solution at different
    time instants.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments specifying the training method and mode.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = PINN(input_size=3)

    if args.train:
        trainer = Trainer(model=model, method=args.method)
        trainer.train()
    else:
        model.load_state_dict(
            torch.load(
                "./models/model_simple_initial_condition.pt",
                map_location=torch.device(device)
            )
        )
        model.to(device)

    for timestep in np.linspace(0, 600, 10, endpoint=True):
        plot_timestep(
            model,
            timestep / 600,
            name=args.method,
            device=device,
            dtype=torch.float
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--method", choices=["normal", "fourier"], default="normal")
    parser.add_argument("--train", action="store_true", help="Train the model")

    args = parser.parse_args()
    main(args=args)
