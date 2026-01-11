import torch.nn as nn

class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for solving PDEs.

    The network takes spatial-temporal coordinates as input and outputs
    a scalar field value constrained in [0, 1] through a sigmoid activation.
    """

    def __init__(self, input_size, hidden_dim=500):
        """
        Initialize the PINN architecture.

        Args:
            input_size (int): Number of input features (e.g. x, y, t).
            hidden_dim (int, optional): Number of neurons in each hidden layer.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (..., input_size).

        Returns:
            torch.Tensor: Network output.
        """
        return self.layers(x)
