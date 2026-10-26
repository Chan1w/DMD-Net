import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.demand_gating import eps_from_x0, gate_x0_correction


def test_x0_gate_endpoints_and_gradient():
    torch.manual_seed(0)
    z_t = torch.randn(2, 4, 8, 8)
    eps_full = torch.randn_like(z_t)
    z_r = torch.randn_like(z_t)
    t = torch.tensor([100, 700])
    alphas = torch.linspace(0.999, 0.001, 1000)

    zeros = torch.zeros(2, 1, 8, 8, requires_grad=True)
    out_zero = gate_x0_correction(
        z_t, eps_full, z_r, zeros, t, alphas
    )
    assert torch.allclose(out_zero.x0, z_r, atol=1e-6)
    assert torch.allclose(out_zero.x0_reference, z_r, atol=1e-6)
    expected_zero_eps = eps_from_x0(z_t, z_r, t, alphas)
    assert torch.allclose(out_zero.eps, expected_zero_eps, atol=1e-5)

    ones = torch.ones(2, 1, 8, 8)
    out_one = gate_x0_correction(
        z_t, eps_full, z_r, ones, t, alphas
    )
    assert torch.allclose(out_one.x0, out_one.x0_full, atol=1e-6)
    assert torch.allclose(out_one.eps, eps_full, atol=1e-5)

    out_warmup = gate_x0_correction(
        z_t, eps_full, z_r, zeros, t, alphas, curriculum_mix=0.0
    )
    assert torch.allclose(out_warmup.x0, out_warmup.x0_full, atol=1e-6)

    demand = torch.full((2, 1, 8, 8), 0.5, requires_grad=True)
    output = gate_x0_correction(
        z_t, eps_full, z_r, demand, t, alphas
    )
    expected = z_r + 0.5 * (output.x0_full - z_r)
    assert torch.allclose(output.x0, expected, atol=1e-6)
    output.x0.square().mean().backward()
    assert demand.grad is not None and demand.grad.abs().sum() > 0


if __name__ == "__main__":
    test_x0_gate_endpoints_and_gradient()
    print("ALL PASS")
