import torch
import numpy as np
import os

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam

from .normal.loss import LossNormal
from .fourier.fourier_loss import FourierLoss
from .fourier.fourier import Fourier


class Trainer:
    """
    Trainer class for Physics-Informed Neural Networks.

    This class handles the optimization loop, learning rate scheduling,
    loss computation and model checkpointing.
    """

    def __init__(self, model, method, epochs= 1000, lr= 1e-3, num_points_sampled= 20000, num_physical_sampled= 10000, device= None):
        """
        Initialize the training procedure.

        Parameters
        ----------
        model : torch.nn.Module
            PINN model to be trained.
        method : str
            Training method ('normal' or 'fourier').
        epochs : int, optional
            Number of training epochs.
        lr : float, optional
            Initial learning rate.
        num_points_sampled : int, optional
            Number of samples used for boundary and initial conditions.
        num_physical_sampled : int, optional
            Number of collocation points for the PDE residual.
        device : torch.device, optional
            Device used for training.
        """
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.device = device if device else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=self.epochs // 3, gamma=0.1)

        if method == 'normal':
            self.loss = LossNormal(model, num_points_sampled, num_physical_sampled, device=self.device)
            self.save_dir = "normal"
            os.makedirs(f"./models/{self.save_dir}", exist_ok=True)

        if method == 'fourier':
            self.loss = FourierLoss(model, num_points_sampled, num_physical_sampled, device= self.device)
            self.save_dir = "fourier"
            os.makedirs(f"./models/{self.save_dir}", exist_ok=True)

            fourier = Fourier()
            self.solution = fourier.solution(order= 18, sampling= 200000, nb_evaluations= 10, device= self.device, dtype= torch.float)

    def train(self):
        """
        Run the training loop and save the trained model parameters.
        """
        epoch_bar = tqdm(range(self.epochs), desc="Loss: 0.000000")

        for epoch in epoch_bar:
            self.optimizer.zero_grad()

            loss = self.loss.compute_loss()
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            epoch_bar.set_description(f"Loss: {loss.item():.6f}")
            lr_current = self.scheduler.get_last_lr()[0]
            epoch_bar.set_postfix(lr=f"{lr_current:.1e}")

        torch.save(
            self.model.state_dict(),
            f"./models/{self.save_dir}/best_model.pt"
        )
        print(f"Training finished: saved weights in ./models/{self.save_dir}/best_model.pt")
