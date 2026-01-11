import torch
import numpy as np
import math

class Fourier:
    """
    Fourier-based analytical solver for the heat equation.
    """

    def initial_condition(self, x):
        """
        Rectangular initial condition.

        Args:
            x (torch.Tensor): Spatial coordinates.

        Returns:
            torch.Tensor: Initial values.
        """
        out = torch.zeros_like(x[..., 0])
        out[x[..., 0] < 1] = 1.0
        return out

    def initial_condition_circle(self, x):
        """
        Circular initial condition.

        Args:
            x (torch.Tensor): Spatial coordinates.

        Returns:
            torch.Tensor: Initial values.
        """
        dist = (x[..., 0] - 1.)**2 + (x[..., 1] - 0.5)**2
        out = torch.zeros_like(dist)
        out[dist < 0.25] = 1.0
        return out

    def solution(self, order, sampling, nb_evaluations=1, device=None, dtype=None):
        """
        Computes a Fourier-series solution for the heat equation.

        Args:
            order (int): Truncation order of the Fourier series.
            sampling (int): Monte Carlo samples for coefficient estimation.
            nb_evaluations (int, optional): Number of averaging iterations.
            device (str, optional): Computation device.
            dtype (torch.dtype, optional): Tensor type.

        Returns:
            callable: Function mapping (x, y, t) to solution value.
        """
        grid = torch.tensor(
            np.stack(np.meshgrid(range(order), range(order)), axis=-1),
            device=device, dtype=dtype
        )

        coeffs = torch.zeros(order, order, device=device, dtype=dtype)

        for _ in range(nb_evaluations):
            values = torch.rand(sampling, order, order, 2, device=device, dtype=dtype)
            values *= torch.tensor([2., 1.], device=device, dtype=dtype)
            evals = self.initial_condition_circle(values)

            coefs = 4 * torch.mean(
                evals *
                torch.cos(math.pi * values[..., 0] * grid[..., 0] / 2) *
                torch.cos(math.pi * values[..., 1] * grid[..., 1]),
                dim=0
            )

            coefs[0, :] /= 2
            coefs[:, 0] /= 2
            coeffs += coefs / nb_evaluations

        def sol(input):
            basis = (
                torch.cos(math.pi * input[..., 0].unsqueeze(-1).unsqueeze(-1) * grid[..., 0] / 2) *
                torch.cos(math.pi * input[..., 1].unsqueeze(-1).unsqueeze(-1) * grid[..., 1])
            )
            decay = torch.exp(
                -input[..., 2].unsqueeze(-1).unsqueeze(-1) *
                (grid[..., 0]**2 * math.pi**2 / 4 + grid[..., 1]**2 * math.pi**2) * 0.001
            )
            return torch.sum(coeffs * basis * decay, dim=(-1, -2))

        return sol
