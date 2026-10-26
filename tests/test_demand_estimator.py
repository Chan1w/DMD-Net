"""DemandEstimator tensor contracts and receiver-input tests."""

import inspect
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.demand_estimator import DemandEstimator


def test_demand_estimator_shapes_and_range():
    estimator = DemandEstimator(latent_channels=4)
    z_r = torch.randn(2, 4, 48, 48)
    snr = torch.tensor([[-5.0], [20.0]])
    output = estimator(z_r, snr)
    assert output.logits.shape == (2, 1, 48, 48)
    assert output.demand.shape == (2, 1, 48, 48)
    assert (output.demand >= 0).all() and (output.demand <= 1).all()
    assert torch.isfinite(output.demand).all()


def test_demand_estimator_preserves_odd_spatial_size():
    estimator = DemandEstimator(latent_channels=4, base_channels=32,
                                mid_channels=48, context_channels=64,
                                attention_heads=4, local_blocks=1,
                                context_blocks=1)
    output = estimator(torch.randn(1, 4, 47, 45), torch.tensor([10.0]))
    assert output.demand.shape == (1, 1, 47, 45)


def test_demand_estimator_uses_only_receiver_feature_and_snr():
    signature = inspect.signature(DemandEstimator.forward)
    params = set(signature.parameters) - {"self"}
    assert params == {"z_r", "csi"}


def test_demand_estimator_backward_reaches_conv_attention_and_snr():
    estimator = DemandEstimator(latent_channels=4, base_channels=32,
                                mid_channels=48, context_channels=64,
                                attention_heads=4, local_blocks=1,
                                context_blocks=1)
    output = estimator(torch.randn(2, 4, 16, 16),
                       torch.tensor([[-3.0], [12.0]]))
    output.demand.mean().backward()
    assert estimator.stem[0].weight.grad is not None
    assert estimator.context[1].attention.in_proj_weight.grad is not None
    assert estimator.encoder_high.film.affine.weight.grad is not None


if __name__ == "__main__":
    test_demand_estimator_shapes_and_range()
    test_demand_estimator_preserves_odd_spatial_size()
    test_demand_estimator_uses_only_receiver_feature_and_snr()
    test_demand_estimator_backward_reaches_conv_attention_and_snr()
    print("ALL PASS")
