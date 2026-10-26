# DMD-Net

Official research code for **Spatially Adaptive Diffusion Restoration via Learned Demand Maps for Deep JSCC-Based Image Transmission**.

## Final three-stage pipeline

1. `scripts/stage1_train_latent_djscc.py`: train the SD2.1 latent DJSCC codec.
2. `scripts/stage2_train_control_sd21.py`: train the demand-free Condition model and original zero-convs.
3. `scripts/stage3_train_demand_estimator.py`: freeze Stage 2 and train the DemandEstimator plus zero-initialized Demand-Gate branches.

Receiver prompts are generated from the current DJSCC reconstruction `x_base`. The default `--prompt-source` is `receiver`; `original` is retained only for the Oracle-Prompt ablation.

## Test entry points

- `scripts/stage2_test_control_sd21.py`
- `scripts/stage3_test_demand_estimator.py`

## Main checkpoints

- `scripts/checkpoints/DJSCC_SD21/best.pth`
- `scripts/checkpoints/stage2_control_SD21_xbase/control_best.pth`
- `scripts/checkpoints/stage3_demand_control_SD21_xbase/best.pth`

Checkpoints and datasets are not bundled with this source migration. See `doc/final_paper_framework_and_method.md` for the complete method and experiment contract.

## Environment

Install a CUDA-compatible PyTorch build first, then install `requirements.txt`. SD2.1 and BLIP are loaded with `local_files_only=True`, so their Hugging Face snapshots must already exist on the execution host.
