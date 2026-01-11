import torch
import math
import torch.autograd as autograd


class FourierLoss:
    def __init__(
        self,
        model,
        num_points_sampled,
        num_physical_sampled,
        device=None,
        dtype=torch.float,
        fourier_oracle=None,  # opzionale: soluzione Fourier per guida supervisionata
    ):
        self.model = model
        self.num_points_sampled = num_points_sampled
        self.num_physical_sampled = num_physical_sampled
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.fourier_oracle = fourier_oracle
        self.model.to(self.device)

    # ----------------------------
    # Initial condition (disk)
    # ----------------------------
    def initial_loss(self):
        N = self.num_points_sampled
        values = torch.rand(N, 3, device=self.device, dtype=self.dtype)
        values[:, 0] *= 2.0
        values[:, 1] *= 1.0
        values[:, 2] = 0.0

        label = torch.zeros(N, 1, device=self.device, dtype=self.dtype)
        mask = (values[:, 0] - 1.0) ** 2 + (values[:, 1] - 0.5) ** 2 < 0.25
        label[mask] = 1.0

        u_pred = self.model(values)
        return torch.mean((u_pred - label) ** 2)

    # ----------------------------
    # Final condition (t = 1)
    # ----------------------------
    def end_loss(self):
        N = self.num_points_sampled
        values = torch.rand(N, 3, device=self.device, dtype=self.dtype)
        values[:, 0] *= 2.0
        values[:, 1] *= 1.0
        values[:, 2] = 1.0

        label = torch.full(
            (N, 1),
            math.pi * 0.25 / 2.0,
            device=self.device,
            dtype=self.dtype
        )

        u_pred = self.model(values)
        return torch.mean((u_pred - label) ** 2)

    # ----------------------------
    # Boundary loss (Neumann BC)
    # ----------------------------
    def boundary_loss(self):
        N = self.num_points_sampled
        perimeter = 6.0

        values_limit = torch.rand(N, device=self.device) * perimeter
        values_border = torch.zeros(N, 3, device=self.device, dtype=self.dtype)

        mask1 = values_limit < 2.0
        mask2 = (values_limit >= 2.0) & (values_limit < 3.0)
        mask3 = (values_limit >= 3.0) & (values_limit < 5.0)
        mask4 = values_limit >= 5.0

        # Bottom
        values_border[mask1, 0] = values_limit[mask1]
        values_border[mask1, 1] = 0.0
        # Right
        values_border[mask2, 0] = 2.0
        values_border[mask2, 1] = values_limit[mask2] - 2.0
        # Top
        values_border[mask3, 0] = values_limit[mask3] - 3.0
        values_border[mask3, 1] = 1.0
        # Left
        values_border[mask4, 0] = 0.0
        values_border[mask4, 1] = values_limit[mask4] - 5.0

        values_border[:, 2] = torch.rand(N, device=self.device)
        values_border.requires_grad_(True)

        u = self.model(values_border)
        grad_u = autograd.grad(
            outputs=u,
            inputs=values_border,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        loss = (
            torch.sum(grad_u[mask1, 1] ** 2) +
            torch.sum(grad_u[mask2, 0] ** 2) +
            torch.sum(grad_u[mask3, 1] ** 2) +
            torch.sum(grad_u[mask4, 0] ** 2)
        )

        return loss / N

    # ----------------------------
    # Physics loss (heat equation)
    # ----------------------------
    def physics_loss(self):
        N = self.num_physical_sampled

        space = torch.rand(N, 2, device=self.device, dtype=self.dtype)
        space[:, 0] *= 2.0
        space[:, 1] *= 1.0
        space.requires_grad_(True)

        time = torch.rand(N, 1, device=self.device, dtype=self.dtype) * 600.0
        time.requires_grad_(True)

        t_norm = time / 600.0
        inputs = torch.cat((space, t_norm), dim=1)

        u = self.model(inputs)

        ut = autograd.grad(outputs=u, inputs=time, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        ux = autograd.grad(outputs=u, inputs=space, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        uxx = autograd.grad(outputs=ux, inputs=space, grad_outputs=torch.ones_like(ux), create_graph=True)[0]

        D = 0.001
        residual = ut[:, 0] - D * (uxx[:, 0] + uxx[:, 1])
        return torch.mean(residual ** 2)

    # ----------------------------
    # Fourier-informed loss
    # ----------------------------
    def fourier_guided_loss(self):
        if self.fourier_oracle is None:
            return 0.0

        N = self.num_points_sampled
        # Sampling space-time points
        values = torch.rand(N, 3, device=self.device, dtype=self.dtype)
        values[:, 0] *= 2.0
        values[:, 1] *= 1.0
        values[:, 2] *= 1.0  # time in [0,1]

        labels = self.fourier_oracle(values)
        preds = self.model(values)
        return torch.mean((preds - labels) ** 2)

    # ----------------------------
    # Total loss
    # ----------------------------
    def compute_loss(self, use_fourier=False):
        loss_total = self.initial_loss() + self.end_loss() + self.boundary_loss() + self.physics_loss()
        if use_fourier and self.fourier_oracle is not None:
            loss_total += self.fourier_guided_loss()
        return loss_total
