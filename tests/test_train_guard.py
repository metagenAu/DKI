"""Regression test: a non-finite batch must not poison the training run.

A stiff replicator field (common with the nonlinear SiLU fitness) can make a
single batch's ODE solve return a non-finite state, hence a non-finite
loss/gradient. Before the guard, stepping on that wrote NaN into every weight
via Adam and every subsequent epoch read NaN forever. The guard skips the
optimiser step on a non-finite loss/gradient so the run continues from the last
good weights.
"""

from __future__ import annotations

import torch

import dki.train as train_mod
from dki.data import DKIData
from dki.train import TrainConfig, train


def _toy_data(N: int = 4, n: int = 8) -> DKIData:
    torch.manual_seed(0)
    p = torch.rand(n, N)
    p = p / p.sum(dim=-1, keepdim=True)
    z = (p > 0).float()
    z = z / z.sum(dim=-1, keepdim=True)
    return DKIData(
        p_train=p, z_train=z, p_val=p, z_val=z, p_all=p, z_all=z,
        p_test=None, z_test=None, n_species=N,
    )


def test_nonfinite_batch_does_not_corrupt_weights(monkeypatch):
    data = _toy_data()
    calls = {"grad": 0}
    real_softmax = torch.softmax

    def fake_equilibrium(func, z0, **kwargs):
        # Differentiable, finite surrogate for the equilibrium solve so normal
        # epochs produce real gradients and step the optimiser.
        out = real_softmax(func.fitness(z0), dim=-1)
        # Inject a non-finite result on the 2nd training (grad-enabled) solve.
        if torch.is_grad_enabled():
            calls["grad"] += 1
            if calls["grad"] == 2:
                out = out * float("inf")
        return out

    monkeypatch.setattr(train_mod, "equilibrium", fake_equilibrium)

    cfg = TrainConfig(
        epochs=5, batch_size=8, val_fraction=0.5, seed=0,
        save_predictions=False, nonlinear=True, out_dir="/tmp/dki-test",
    )
    model, result, _ = train(cfg, data=data)

    assert result.skipped_steps >= 1
    # The bad batch was skipped, so no weight is NaN/Inf.
    for p in model.parameters():
        assert torch.isfinite(p).all()
    # Training kept making progress: at least one finite val score recorded.
    assert any(torch.isfinite(torch.tensor(v)) for v in result.val_loss)
