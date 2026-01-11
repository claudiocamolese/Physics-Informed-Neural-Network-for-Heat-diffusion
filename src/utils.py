import numpy as np
import matplotlib.pyplot as plt
import torch
import os


def plot_timestep(model, t, name, device=None, dtype=None):
    """
    Visualize the temperature field predicted by the PINN at a fixed time.

    The function evaluates the model on a dense spatial grid and produces
    a contour plot of the predicted temperature field, saving the result
    as an image.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PINN model approximating u(x, y, t).
    t : float
        Normalized time in the interval [0, 1].
    name : str
        Identifier used to organize output figures.
    device : torch.device, optional
        Device used for model evaluation.
    dtype : torch.dtype, optional
        Data type used for tensors.
    """
    os.makedirs(f"./figures/{name}", exist_ok=True)

    xs, ys, ts = np.meshgrid(
        np.arange(0, 2, 0.002),
        np.arange(0, 1, 0.002),
        [t]
    )

    with torch.no_grad():
        X_grid = torch.tensor(np.stack([xs, ys, ts], axis=-1)).to(device=device, dtype=dtype)

        Z_pt = torch.clamp(model(X_grid).squeeze(), 0, 1)
        Z_mesh = Z_pt.cpu().numpy()

    fig, ax = plt.subplots()
    cs = ax.contourf(
        xs.squeeze(),
        ys.squeeze(),
        Z_mesh,
        levels=np.linspace(0., 1., 100),
        vmin=0,
        vmax=1
    )
    fig.colorbar(cs)

    plt.savefig(f"./figures/{name}/{int(t * 1000):04d}.png")
    plt.close(fig)
