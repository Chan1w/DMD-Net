# DMD-Net 最终代码迁移清单

> 源项目：`D:\python_project\ControlNet`  
> 目标项目：`D:\python_project\DMD-Net`  
> 目标：只保留论文最终确认的 Stage 1/2/3 方法、训练测试入口和必要依赖，避免把早期 RFRE、teacher/student、离线 D 标签和旧 joint 方案混入新项目。

---

## 1. 最终入口脚本

当前实际存在并应作为最终入口的文件如下：

```text
scripts/stage1_train_latent_djscc.py
scripts/stage2_train_control_sd21.py
scripts/stage2_test_control_sd21.py
scripts/stage3_train_demand_estimator.py
scripts/stage3_test_demand_estimator.py
```

需要注意：当前 Stage 1 文件名是：

```text
stage1_train_latent_djscc.py
```

不是旧名 `train_latent_djscc.py`，也不需要再创建第二份同功能脚本。

五个入口分别对应：

| 阶段 | 文件 | 功能 |
|---|---|---|
| Stage 1 | `stage1_train_latent_djscc.py` | 使用 SD2.1 VAE raw latent 训练 DJSCC |
| Stage 2 Train | `stage2_train_control_sd21.py` | 不引入 D，只训练 Condition Model 和原始 zero-conv |
| Stage 2 Test | `stage2_test_control_sd21.py` | 测试 DJSCC、SD2.1 base 和 Stage 2 Full Control |
| Stage 3 Train | `stage3_train_demand_estimator.py` | 冻结 Stage 2，训练 DemandEstimator 和 D skip branches |
| Stage 3 Test | `stage3_test_demand_estimator.py` | 测试 Demand Map、Demand Gate 和 x0 gate 完整方案 |

四个 Stage 2/3 脚本当前均以 `--prompt-source receiver` 为默认设置，即由接收端 `x_base` 生成 Prompt。

---

## 2. 必须迁移的核心代码

### 2.1 根目录公共文件

```text
config.py
metrics.py
utils.py
utils_distributed.py
requirements_rest.txt
```

作用：

- `config.py`：SD2.1/BLIP 模型 ID、DJSCC 和 DemandEstimator 默认参数；
- `metrics.py`：PSNR、SSIM、LPIPS；
- `utils.py`：随机种子、AverageMeter 和公共保存工具；
- `utils_distributed.py`：单卡/双卡 DDP 启动、同步和模型解包；
- `requirements_rest.txt`：当前环境依赖记录。

`config.py` 中仍有旧 RFRE、离线 D 标签、SD1.5 和旧 ControlNet 配置。若直接复制不会影响最终入口运行，但推荐在 DMD-Net 中删除这些旧配置，只保留：

```text
SD21_MODEL_ID
BLIP_MODEL_ID
DJSCC_BANDWIDTH
DJSCC_MID_CH
DJSCC_N_BLOCKS
DJSCC_EMB_DIM
VAL_SNRS
DDM_DEFAULTS
```

以及确实仍由新脚本使用的数据路径常量。

### 2.2 数据模块

```text
data/__init__.py
data/paired_channel_dataset.py
```

`paired_channel_dataset.py` 提供：

- `VOCImageDataset`；
- `ImageFolderDataset`；
- `[0,1] -> [-1,1]` 的 `to_vae_input`；
- `[-1,1] -> [0,1]` 的 `from_vae_output`。

以下旧数据模块不属于最终三阶段方案：

```text
data/demand_label_dataset.py
data/demand_label_schema.py
data/dense_quality.py
```

它们服务于已经放弃的离线 D 标签/teacher 路线，不建议迁入 DMD-Net。

### 2.3 最终模型模块

```text
models/__init__.py
models/channel.py
models/djscc.py
models/djscc_adapter.py
models/vae_adapter.py
models/cond_encoder.py
models/demand_estimator.py
models/demand_condition_model.py
models/demand_gating.py
models/loss.py
models/restorer.py
models/builder.py
```

依赖关系如下：

```text
stage1_train_latent_djscc.py
  -> models/djscc.py
       -> models/channel.py
  -> models/vae_adapter.py
  -> data/paired_channel_dataset.py

stage2_train/test
  -> models/builder.py
       -> models/djscc_adapter.py
            -> models/djscc.py
            -> models/vae_adapter.py
       -> models/demand_condition_model.py
            -> models/cond_encoder.py
  -> models/restorer.py
       -> models/demand_gating.py
            -> models/loss.py

stage3_train/test
  -> all Stage 2 dependencies
  -> models/demand_estimator.py
  -> models/demand_gating.py
```

各文件在最终方法中的功能：

| 文件 | 最终用途 |
|---|---|
| `channel.py` | 实值 AWGN 和逐样本功率归一化 |
| `djscc.py` | SD2.1 raw latent DJSCC Encoder/Decoder |
| `djscc_adapter.py` | 封装冻结 DJSCC，输出 `x_base`、`z_hat_raw`、`z_r` 所需观测 |
| `vae_adapter.py` | SD2.1 VAE raw/scaled latent 转换和带梯度 Decoder |
| `cond_encoder.py` | `z_r` 条件编码与 CSI embedding |
| `demand_estimator.py` | 输入 `z_r+SNR`，输出 Demand Map |
| `demand_condition_model.py` | Stage 2 zero-conv Condition 和 Stage 3 Demand Gate |
| `demand_gating.py` | 每一步 `x0=z_r+D*(x0_full-z_r)` |
| `loss.py` | `x0_from_eps` 等扩散数学函数 |
| `restorer.py` | Stage 2/3 DDIM 迭代恢复流程 |
| `builder.py` | SD2.1、BLIP、DJSCC 和最终模型构建函数 |

---

## 3. 当前存在的两处历史依赖

### 3.1 `models/builder.py` 的旧模型导入

当前 `builder.py` 顶层还包含：

```python
from .dual_controlnet import DualReliabilityControlNet
from .rfre import ReliabilityEstimator
```

因此，如果完全不修改代码而直接复制，为了让 Python import 成功，还必须附带：

```text
models/dual_controlnet.py
models/rfre.py
```

这两个文件不属于最终 DMD-Net 方法，只是被旧 builder 顶层导入。推荐在迁移时从 `builder.py` 删除上述导入，并删除只服务于它们的：

```text
build_rfre(...)
build_dual_controlnet(...)
```

完成精简后，DMD-Net 不需要复制 `dual_controlnet.py` 和 `rfre.py`。

### 3.2 Stage 3 测试对旧 joint 脚本的工具依赖

当前：

```text
scripts/stage3_test_demand_estimator.py
```

从旧脚本：

```text
scripts/test_joint_demand_control_sd21.py
```

导入了以下通用评测函数：

```text
METHODS
parse_snr_values
build_dataset
atomic_json
save_csv
seed_diffusion
run_restorer
image_metrics
aggregate
```

因此原样复制时必须附带 `test_joint_demand_control_sd21.py`。但该文件名称和内容属于旧 joint teacher/student 阶段，容易误导后续开发。

推荐在 DMD-Net 中新建：

```text
scripts/evaluation_utils.py
```

将上述纯评测函数迁入其中，再让 `stage3_test_demand_estimator.py` 从 `evaluation_utils.py` 导入。这样即可删除旧 `test_joint_demand_control_sd21.py` 依赖。

---

## 4. 两种可选迁移方案

### 4.1 方案 A：原样复制，最快可运行

需要复制的完整闭包：

```text
config.py
metrics.py
utils.py
utils_distributed.py
requirements_rest.txt

data/__init__.py
data/paired_channel_dataset.py

models/__init__.py
models/builder.py
models/channel.py
models/cond_encoder.py
models/demand_condition_model.py
models/demand_estimator.py
models/demand_gating.py
models/djscc.py
models/djscc_adapter.py
models/loss.py
models/restorer.py
models/vae_adapter.py

# 仅因 builder 旧 import 被迫包含
models/dual_controlnet.py
models/rfre.py

scripts/stage1_train_latent_djscc.py
scripts/stage2_train_control_sd21.py
scripts/stage2_test_control_sd21.py
scripts/stage3_train_demand_estimator.py
scripts/stage3_test_demand_estimator.py

# 仅因 Stage 3 test 借用评测函数被迫包含
scripts/test_joint_demand_control_sd21.py
```

优点：改动最少。  
缺点：DMD-Net 中仍混有旧 RFRE、DualControlNet 和旧 joint 命名。

### 4.2 方案 B：推荐的最终论文代码仓库

推荐目录：

```text
DMD-Net/
├─ config.py
├─ metrics.py
├─ utils.py
├─ utils_distributed.py
├─ requirements.txt
├─ README.md
├─ data/
│  ├─ __init__.py
│  └─ paired_channel_dataset.py
├─ models/
│  ├─ __init__.py
│  ├─ builder.py
│  ├─ channel.py
│  ├─ cond_encoder.py
│  ├─ demand_condition_model.py
│  ├─ demand_estimator.py
│  ├─ demand_gating.py
│  ├─ djscc.py
│  ├─ djscc_adapter.py
│  ├─ loss.py
│  ├─ restorer.py
│  └─ vae_adapter.py
├─ scripts/
│  ├─ evaluation_utils.py
│  ├─ stage1_train_latent_djscc.py
│  ├─ stage2_train_control_sd21.py
│  ├─ stage2_test_control_sd21.py
│  ├─ stage3_train_demand_estimator.py
│  └─ stage3_test_demand_estimator.py
├─ tests/
│  ├─ __init__.py
│  ├─ test_demand_estimator.py
│  ├─ test_demand_condition.py
│  ├─ test_demand_x0_gating.py
│  └─ test_stage1_djscc.py
└─ doc/
   ├─ final_paper_framework_and_method.md
   └─ dmd_net_code_migration_manifest.md
```

方案 B 需要两项小型重构：

1. 精简 `models/builder.py`，移除 RFRE 和 DualControlNet；
2. 新建 `scripts/evaluation_utils.py`，解除 Stage 3 test 对旧 joint 脚本的依赖。

这是最适合作为论文最终代码仓库的方案。

---

## 5. 建议迁移的测试

可以直接迁移：

```text
tests/__init__.py
tests/test_demand_estimator.py
tests/test_demand_condition.py
tests/test_demand_x0_gating.py
tests/test_scheduler_steps.py
```

需要重新整理后迁移：

- `test_online_demand_teacher.py`：其中 Demand Gate 零初始化测试仍有价值，但 teacher 测试已废弃。应提取有效部分并改名为 `test_demand_gate_zero_init.py`；
- 当前 `test_djscc_cbr.py` 测试的是 `models/deep_jscc.py`，不是最终 Stage 1 使用的 `models/djscc.py`。不应原样迁移，应该为 `models/djscc.py::calculate_cbr` 新建测试；
- `test_no_gt_at_inference.py` 当前主要检查旧 RFRE，应重写为检查 `DemandEstimator.forward(z_r, csi)`。

不建议迁移：

```text
tests/test_reliability_limits.py
tests/test_tensor_contracts.py
tests/test_freeze_gradients.py
tests/test_zero_control_identity.py
tests/test_demand_label_schema.py
tests/test_demand_losses.py
tests/test_online_demand_teacher.py  # 不应原样复制
```

这些测试主要对应旧 RFRE、旧 DualControlNet、离线标签或 teacher 方案。

---

## 6. 不应迁移到最终 DMD-Net 的旧代码

### 6.1 旧训练与标签路线

```text
scripts/build_demand_labels.py
scripts/train_demand_estimator.py
scripts/train_demand_estimator_sd21.py
scripts/train_demand_condition.py
scripts/train_joint_demand_control_sd21.py
scripts/test_demand_estimator_sd21.py
scripts/test_joint_demand_control_sd21.py  # 提取评测工具后删除
models/online_demand_teacher.py
models/demand_losses.py
```

### 6.2 旧 Reliability/RFRE 路线

```text
models/rfre.py
models/dual_controlnet.py
models/policy.py
scripts/train_rfre.py
scripts/train_policy.py
scripts/train_controlnet.py
scripts/infer_demand.py
```

### 6.3 与最终论文无关的实验代码

```text
models/deep_jscc.py
models/neural_text_jscc.py
scripts/train_djscc.py
scripts/train_neural_text_jscc.py
scripts/test_neural_text_jscc.py
scripts/text_transmission_ldpc.py
scripts/ablate_guidance.py
scripts/explore_strength_step.py
scripts/explore_strength_step_prompt.py
scripts/precompute_pairs.py
scripts/smoke_test_ddm.py
scripts/stress_test_ddm.py
```

这些文件可以保留在原 ControlNet 项目作为历史实验记录，但不应出现在最终 DMD-Net 主分支中。

---

## 7. 文档与绘图文件

必须迁移：

```text
doc/final_paper_framework_and_method.md
```

可选迁移：

```text
figure/DDM_模型架构与论文插图绘制说明.md
figure/DDM_论文插图图注.md
scripts/make_demand_estimator_diagram.py
```

旧的 Stage 说明、RFRE 迁移说明和标签构建说明可能包含已经废弃的设定。如果迁移，应放进 `doc/archive/`，不能与最终方法文档处于同一优先级。

---

## 8. Checkpoint、数据和模型缓存

以下不是源代码，但运行最终项目时必须提供：

```text
scripts/checkpoints/DJSCC_SD21/best.pth
scripts/checkpoints/stage2_control_SD21_xbase/control_best.pth
scripts/checkpoints/stage3_demand_control_SD21_xbase/best.pth
```

若需要继续训练，还应保留对应 `last.pth` 和 optimizer 状态。

外部数据：

```text
/home/wangch/projects/Dataset/VOC2012
/home/wangch/projects/Dataset/Kodak24
/home/wangch/projects/Dataset/picture   # 当前部分测试默认目录
```

Hugging Face 本地缓存至少应包含：

```text
LanguageMachines/stable-diffusion-2-1-base
Salesforce/blip-image-captioning-base
```

代码使用 `local_files_only=True`，如果服务器缓存中没有这些模型，将不能自动从网络加载。

---

## 9. 迁移后必须修改的硬编码路径

以下文件仍包含服务器端 `/home/wangch/projects/ControlNet`：

```text
scripts/stage2_train_control_sd21.py
scripts/stage2_test_control_sd21.py
scripts/stage3_train_demand_estimator.py
scripts/stage3_test_demand_estimator.py
```

如果服务器上的新项目路径改为：

```text
/home/wangch/projects/DMD-Net
```

则应同步更新默认 checkpoint 和 output 路径。更推荐的做法是：

- 项目内 checkpoint/output 使用相对路径；
- 数据集路径继续通过命令行参数传入；
- 不在代码中固定 Windows 本地路径。

例如：

```text
scripts/checkpoints/DJSCC_SD21/best.pth
scripts/checkpoints/stage2_control_SD21_xbase/control_best.pth
scripts/checkpoints/stage3_demand_control_SD21_xbase/best.pth
outputs/stage2_test
outputs/stage3_test
```

---

## 10. 最小第三方依赖

最终 Stage 1/2/3 主流程至少需要：

```text
torch
torchvision
numpy
pillow
tqdm
diffusers
transformers
huggingface_hub
safetensors
lpips
matplotlib
```

当前 `requirements_rest.txt` 是完整环境快照，包含大量 CUDA wheel 和非必要包。迁移时可以先保留以复现实验环境，随后再生成精简的 `requirements.txt`。

---

## 11. 推荐迁移顺序

1. 在 DMD-Net 建立推荐目录结构；
2. 复制五个最终入口脚本和核心模型依赖；
3. 精简 `config.py` 和 `models/builder.py`；
4. 抽取 `scripts/evaluation_utils.py`；
5. 将所有项目路径改为相对路径或 DMD-Net 路径；
6. 迁移三个阶段的 checkpoint；
7. 运行 Python 语法检查；
8. 运行 DemandEstimator、Demand Gate、x0 gate 和 DJSCC 单元测试；
9. 使用 `--max-samples 1 --max-steps 1` 分别进行 Stage 2/3 smoke test；
10. 对比迁移前后同一 seed、同一 observation、同一 checkpoint 的输出，确保数值一致。

---

## 12. 最终结论

如果目标只是立即运行，使用“方案 A 原样复制”。

如果目标是建立论文最终仓库，应采用“方案 B”：保留五个 Stage 入口、12 个核心模型/公共文件、必要测试和最终方法文档，同时移除 RFRE、teacher、离线 D 标签、旧 joint 和神经文本传输代码。

已按照方案 B 将最终代码迁入 `D:\python_project\DMD-Net`，并完成 builder、评测工具、配置和旧接口的精简。

