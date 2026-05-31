"""The batched legacy path must reproduce the original per-sample dynamics.

These tests guard the equivalence the ``--batched`` flag relies on: the
batch-safe ``ODEFuncBatched`` computes the same replicator dynamics as the
original ``ODEFunc`` for a single sample, and integrating a whole batch in one
``odeint`` call matches looping ``odeint`` per sample.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_SPEC = importlib.util.spec_from_file_location(
    "baseline_runner",
    Path(__file__).resolve().parent.parent / "legacy" / "baseline_runner.py",
)
baseline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(baseline)


def _copy_weights(src, dst):
    dst.fcc1.load_state_dict(src.fcc1.state_dict())
    dst.fcc2.load_state_dict(src.fcc2.state_dict())


def test_single_sample_dynamics_match():
    """ODEFunc and ODEFuncBatched agree on a (1, N) state."""
    torch.manual_seed(0)
    N = 8
    orig = baseline.ODEFunc(N)
    batched = baseline.ODEFuncBatched(N)
    _copy_weights(orig, batched)

    y = torch.rand(1, N)
    y = y / y.sum()
    f_orig = orig(torch.tensor(0.0), y)
    f_batched = batched(torch.tensor(0.0), y)
    assert torch.allclose(f_orig, f_batched, atol=1e-6)


def test_batched_forward_is_per_row():
    """Each row of a batched forward equals that row solved alone."""
    torch.manual_seed(1)
    N, B = 6, 4
    batched = baseline.ODEFuncBatched(N)

    ys = torch.rand(B, N)
    ys = ys / ys.sum(dim=-1, keepdim=True)
    f_batch = batched(torch.tensor(0.0), ys)
    for i in range(B):
        f_row = batched(torch.tensor(0.0), ys[i].unsqueeze(0))
        assert torch.allclose(f_batch[i], f_row.squeeze(0), atol=1e-6)


def test_batched_integration_matches_loop():
    """integrate_set(batched=True) == integrate_set(batched=False) per row."""
    torch.manual_seed(2)
    N, B = 5, 3
    orig = baseline.ODEFunc(N)
    batched = baseline.ODEFuncBatched(N)
    _copy_weights(orig, batched)

    z = torch.rand(B, N)
    z = z / z.sum(dim=-1, keepdim=True)
    t = torch.tensor([0.0, 1.0])

    q_loop = baseline.integrate_set(orig, z, t, batched=False)
    q_batch = baseline.integrate_set(batched, z, t, batched=True)
    assert q_loop.shape == q_batch.shape == (B, N)
    assert torch.allclose(q_loop, q_batch, atol=1e-4)
