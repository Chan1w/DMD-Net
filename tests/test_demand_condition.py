"""Demand-gating invariants for DemandConditionModel (spec 7.3 / 10.5).

Requires the configured SD 2.1 UNet (GPU recommended).  Verifies:

1. zero-control identity: with zero-conv weights at zero, injecting residuals
   must reproduce the base UNet output exactly (any D).
2. no per-skip adapter contains a second ``z_r`` projection.
3. attaching Stage 3 initially gives exactly the Stage-2 residual.
4. learned D projection changes the outgoing residual after its branch trains.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.builder import load_unet
from models.demand_condition_model import DemandConditionModel


def _set_zero_conv_nonzero(model: DemandConditionModel, value: float = 1.0):
    """Give every zero-conv a non-zero identity-ish weight + bias so gating can
    be observed (simulates a trained ConditionModel)."""
    for adapter in model.adapters:
        with torch.no_grad():
            adapter.zero_conv.weight.fill_(value)
            adapter.zero_conv.bias.zero_()


def _residual_norm(ctrl) -> float:
    return sum(r.abs().sum().item() for r in ctrl.down_residuals) + \
        ctrl.mid_residual.abs().sum().item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = load_unet(device, dtype=torch.float32)
    b, c, h, w = 1, 4, 48, 48
    z_t = torch.randn(b, c, h, w, device=device)
    t = torch.tensor([500], device=device, dtype=torch.long)
    enc = torch.zeros(b, 77, 768, device=device)
    z_r = torch.randn(b, 4, h, w, device=device)
    csi = torch.randn(b, 1, device=device)

    model = DemandConditionModel(unet).to(device)
    model.eval()

    with torch.no_grad():
        # --- 1. zero-control identity (zero-convs at zero) --------------------
        base = unet(z_t, t, encoder_hidden_states=enc).sample
        D_any = torch.rand(b, 1, h, w, device=device)
        ctrl = model(z_t, t, encoder_hidden_states=enc, rx_latents=z_r,
                     demand_map=D_any, csi=csi)
        injected = unet(z_t, t, encoder_hidden_states=enc,
                        down_block_additional_residuals=ctrl.down_residuals,
                        mid_block_additional_residual=ctrl.mid_residual).sample
        diff = (base - injected).abs().max().item()
        print(f"[zero-identity] max |base - injected| = {diff:.6e}")
        assert diff < 1e-5, "zero-conv injection changed the UNet output"

        # --- 2. z_r is not injected again inside a skip adapter ---------------
        assert all(
            not any("zr_proj" in name for name, _ in adapter.named_parameters())
            for adapter in model.adapters
        )

        # --- 3. new D branches initially produce exactly zero perturbation ----
        _set_zero_conv_nonzero(model, value=1.0)
        control_stage2 = model.forward_control(
            z_t, t, encoder_hidden_states=enc, rx_latents=z_r, csi=csi
        )
        control_initial_d = model(
            z_t, t, encoder_hidden_states=enc, rx_latents=z_r,
            demand_map=D_any, csi=csi,
        )
        for stage2_residual, stage3_residual in zip(
            tuple(control_stage2.down_residuals) + (control_stage2.mid_residual,),
            tuple(control_initial_d.down_residuals) + (
                control_initial_d.mid_residual,
            ),
        ):
            assert torch.equal(stage2_residual, stage3_residual)

        # --- 4. trained D branch can change outgoing skip residuals -----------
        for adapter in model.adapters:
            adapter.gate.gate_head.weight.normal_(0.0, 0.02)
            adapter.gate.modulation_head.weight.normal_(0.0, 0.02)
            adapter.gate.demand_zero_conv.weight.normal_(0.0, 0.02)

        D0 = torch.zeros(b, 1, h, w, device=device)
        ctrl0 = model(z_t, t, encoder_hidden_states=enc, rx_latents=z_r,
                      demand_map=D0, csi=csi)
        n0 = _residual_norm(ctrl0)
        print(f"[gate] D=0 total |residual| = {n0:.6e}")

        D1 = torch.ones(b, 1, h, w, device=device)
        ctrl1 = model(z_t, t, encoder_hidden_states=enc, rx_latents=z_r,
                      demand_map=D1, csi=csi)
        n1 = _residual_norm(ctrl1)
        print(f"[gate] D=1 total |residual| = {n1:.6e}")
        assert n1 > 0, "learned D path must pass a non-zero control residual"
        assert any(
            not torch.allclose(residual0, residual1)
            for residual0, residual1 in zip(
                tuple(ctrl0.down_residuals) + (ctrl0.mid_residual,),
                tuple(ctrl1.down_residuals) + (ctrl1.mid_residual,),
            )
        ), "D projection did not affect any outgoing Control residual"

    print("PASS")


if __name__ == "__main__":
    main()
