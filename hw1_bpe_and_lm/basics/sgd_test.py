from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration
        return loss

def test_sgd(lr: float = 1):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)

    losses = []
    for t in range(10):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        losses.append(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer

    return losses

def main():
    candidates = [1e1, 1e2, 1e3]
    all_losses = [[] for _ in candidates]
    for i, lr in enumerate(candidates):
        for t in range(10):
            losses = test_sgd(lr)
            all_losses[i].append(losses)
    import matplotlib.pyplot as plt

    all_losses_tensor = torch.tensor(all_losses)
    mean_losses = all_losses_tensor.mean(dim=1).cpu().numpy()
    iterations = range(mean_losses.shape[1])

    # Separate plots (1 per learning rate)
    fig, axes = plt.subplots(1, len(candidates), figsize=(14, 4), sharey=False)
    if len(candidates) == 1:
        axes = [axes]
    for ax, lr, losses_per_lr_tensor, mean_loss in zip(
        axes, candidates, all_losses_tensor, mean_losses
    ):
        losses_per_lr = losses_per_lr_tensor.cpu().numpy()
        for run_losses in losses_per_lr:
            ax.plot(iterations, run_losses, color="C0", alpha=0.3)
        ax.plot(iterations, mean_loss, color="C1", linewidth=2, label="mean")
        min_loss = losses_per_lr.min()
        max_loss = losses_per_lr.max()
        span = max(max_loss - min_loss, 1e-6)
        padding = 0.05 * span
        ax.set_ylim(min_loss - padding, max_loss + padding)
        ax.set_title(f"lr={lr:g}")
        ax.set_xlabel("Iteration")
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)
        ax.legend()
    axes[0].set_ylabel("Loss")
    fig.suptitle("SGD losses per learning rate (separate)")
    fig.tight_layout()
    fig.savefig("sgd_loss_curves_separate.png")

    # Joint plot (all means together)
    plt.figure(figsize=(8, 5))
    for lr, mean_loss in zip(candidates, mean_losses):
        plt.plot(iterations, mean_loss, label=f"lr={lr:g}")

    plt.xlabel("Iteration")
    plt.ylabel("Mean loss")
    plt.title("SGD learning rate comparison (joint means)")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)
    plt.tight_layout()
    plt.savefig("sgd_loss_curves_joint.png")
    plt.show()

if __name__ == "__main__":
    main()