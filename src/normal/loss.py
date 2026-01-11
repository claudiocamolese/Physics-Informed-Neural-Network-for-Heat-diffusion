import torch
import numpy as np
import torch.autograd as autograd

class LossNormal:
    """
    Loss module for a PINN solving the heat equation with
    initial, boundary, and physical constraints.
    """

    def __init__(self, model, num_points_sampled, num_physical_sampled, device=None):
        """
        Initialize the loss object.

        Args:
            model (nn.Module): PINN model.
            num_points_sampled (int): Number of points for boundary and initial conditions.
            num_physical_sampled (int): Number of points for PDE residual.
            device (str, optional): Computation device.
        """
        self.model = model
        self.num_points_sampled = num_points_sampled
        self.num_physical_sampled = num_physical_sampled
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def initial_loss(self):
        """
        Enforces the initial condition at t = 0.

        Returns:
            torch.Tensor: Mean squared error of the initial condition.
        """
        values_init = torch.rand(self.num_points_sampled, 3, device=self.device)
        values_init[:, 0] *= 2.0
        values_init[:, 1] *= 1.0
        values_init[:, 2] = 0.0

        label_init = torch.zeros(self.num_points_sampled, 1, device=self.device)
        label_init[values_init[:, 0] < 1.0] = 1.0

        u_pred = self.model(values_init)
        return torch.mean((u_pred - label_init) ** 2)

    def end_loss(self):
        """
        Enforces the final condition at t = 1.

        Returns:
            torch.Tensor: Mean squared error of the final condition.
        """
        values_end = torch.rand(self.num_points_sampled, 3, device=self.device)
        values_end[:, 0] *= 2.0
        values_end[:, 1] *= 1.0
        values_end[:, 2] = 1.0

        label_end = 0.5 * torch.ones(self.num_points_sampled, 1, device=self.device)
        u_pred = self.model(values_end)
        return torch.mean((u_pred - label_end) ** 2)

    def boundary_loss(self):
        """
        Enforces Neumann boundary conditions on the domain boundary.

        Returns:
            torch.Tensor: Boundary condition loss.
        """
        perimeter = 6.0
        values_limit = torch.rand(self.num_points_sampled, device=self.device) * perimeter
        values_border = torch.zeros(self.num_points_sampled, 3, device=self.device)

        mask1 = values_limit < 2.0
        mask2 = (values_limit >= 2.0) & (values_limit < 3.0)
        mask3 = (values_limit >= 3.0) & (values_limit < 5.0)
        mask4 = values_limit >= 5.0

        values_border[mask1, 0] = values_limit[mask1]
        values_border[mask1, 1] = 0.0

        values_border[mask2, 0] = 2.0
        values_border[mask2, 1] = values_limit[mask2] - 2.0

        values_border[mask3, 0] = values_limit[mask3] - 3.0
        values_border[mask3, 1] = 1.0

        values_border[mask4, 0] = 0.0
        values_border[mask4, 1] = values_limit[mask4] - 5.0

        values_border[:, 2] = torch.rand(self.num_points_sampled, device=self.device)
        values_border.requires_grad_(True)

        u = self.model(values_border)
        grad_u = autograd.grad(
            u, values_border,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        grads_border = (
            torch.sum(grad_u[mask1, 1] ** 2) +
            torch.sum(grad_u[mask2, 0] ** 2) +
            torch.sum(grad_u[mask3, 1] ** 2) +
            torch.sum(grad_u[mask4, 0] ** 2)
        ) / N

        return grads_border

    def physics_loss(self):
        """
        Enforces the heat equation residual.

        Returns:
            torch.Tensor: Physics loss.
        """
        space = torch.rand(self.num_physical_sampled, 2, device=self.device)
        space[:, 0] *= 2.0
        space[:, 1] *= 1.0
        space.requires_grad_(True)

        time = torch.rand(self.num_physical_sampled, 1, device=self.device) * 600.0
        time.requires_grad_(True)

        inputs = torch.cat((space, time / 600.0), dim=1)
        u = self.model(inputs)

        ut = autograd.grad(u, time, torch.ones_like(u), create_graph=True)[0]
        ux = autograd.grad(u, space, torch.ones_like(u), create_graph=True)[0]
        uxx = autograd.grad(ux, space, torch.ones_like(ux), create_graph=True)[0]

        D = 0.001
        residual = ut[:, 0] - D * (uxx[:, 0] + uxx[:, 1])
        return torch.sum(residual ** 2)

    def compute_loss(self):
        """
        Computes the total loss.

        Returns:
            torch.Tensor: Total loss.
        """
        return (self.initial_loss() + self.end_loss() + self.boundary_loss() + self.physics_loss())
