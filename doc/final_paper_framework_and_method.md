# 基于需求图与需求门控扩散修复的图像语义通信框架

> 文档状态：当前论文方案基线（2026-10-26）  
> 用途：论文写作、实验设计、方法绘图，以及向其他 Agent 交接项目上下文。  
> 说明：本文记录目前已经敲定的主方案。若旧文档、旧脚本注释或早期实验设定与本文冲突，应以本文和当前 Stage 1/2/3 代码为准。

---

## 1. 研究问题与论文核心思路

本项目研究低信噪比和受限信道资源下的图像传输与生成式恢复。发送端通过基于 SD2.1 VAE 潜空间的 DJSCC 传输图像特征，接收端首先获得基础重建结果，再利用冻结的 Stable Diffusion 2.1 生成先验提升视觉质量。

本论文针对的核心问题是：**现有生成式语义通信恢复通常对整幅图像的所有区域施加近似相同的生成式修复强度，无法区分“已经被 DJSCC 正确恢复的区域”和“仍需要生成先验补偿的区域”。** 统一修复容易使扩散模型改写本来可靠的内容，引入幻觉或像素偏移；同时，对严重退化区域又缺少明确的空间介入指示。

DiffJSCC 已经采用“从接收端初始重建图像中恢复文本描述，再作为扩散模型 Prompt”的处理方式。本项目沿用这一既有设定：发送端不单独传输文本，接收端从 \(x_{base}\) 生成 Prompt。该部分用于保持与 DiffJSCC 的系统设定和信道资源口径一致，**不是本文提出的创新点，也不是本文试图解决的主要问题**。

目前已经明确的创新点有两个：

1. **Diffusion Demand Map（D）概念与学习机制**：DemandEstimator 根据当前图像的接收潜变量 \(z_r\) 和信道 SNR，自适应学习空间需求图 \(D\)。D 值越大，表示该区域越需要生成模型介入；D 值越小，表示该区域越应保留 DJSCC 已恢复的信息。该机制使不同图像、不同信道条件和不同空间区域能够采用不同程度的生成式修复。
2. **基于原始 zero-conv 的 Demand Gate**：在传统 UNet+Condition Model 架构中，保留 Condition Model 到 UNet 的原始 zero-conv 路径，并在其基础上为每个多尺度 residual 增加由 D 控制的可学习 Demand Gate 分支。该分支通过 D projection、feature projection、门控和特征调制，有选择地改变 Condition Model 传给 UNet 的信息，而不是对所有位置进行统一缩放。

在当前实现中，D 具体作用于两个位置：

- 在 Condition Model 到冻结 UNet 的 13 个多尺度 residual/skip 上，通过可学习的 D 投影、门控和调制分支改变 Control 信息；
- 在每一步扩散的 clean latent 预测上，通过

  \[
  \hat x_0^D=z_r+D\odot(\hat x_0^{full}-z_r)
  \]

  控制生成结果偏离接收潜变量 \(z_r\) 的程度。

其中，多尺度 Demand Gate 是当前已经确认的第二项创新；每一步 \(x_0\) 修正是完整方法中的重要控制机制，但是否在论文中单独概括为第三项创新，需要结合后续实验和论文其他部分再决定，当前文档不提前作此主张。

整个方法采用三阶段训练，避免 D 的学习破坏基础通信能力和 Condition Model 的去噪控制能力。

---

## 2. 已经确定且不应随意改动的设计

以下内容是当前论文主方案的固定设定：

1. 基础生成模型使用 **Stable Diffusion 2.1 Base**，VAE、UNet、tokenizer 和 text encoder 均与 SD2.1 配套。
2. DJSCC 在 SD2.1 VAE 的 **raw latent** 上训练和传输。
3. 文本不经过独立信道；主方案 Prompt 由接收端的 \(x_{base}\) 生成。
4. 原图生成的 Prompt 只作为 **Oracle Prompt 上限消融**，不能作为主方法输入。
5. 接收端从 \(x_{base}\) 恢复 Prompt 的做法沿用 DiffJSCC，不列为本文创新点。
6. DemandEstimator 的唯一输入是 \(z_r+\mathrm{SNR}\)，不使用 \(x_{base}\)、原图、Prompt 或人工标签。
7. Stage 3 不使用 teacher/student，不构造 teacher D，不使用离线真实 D 标签。
8. Stage 2 完全不包含 D 相关结构，只训练普通 Condition Model 和原始 zero-conv。
9. Stage 3 冻结 Stage 2 的全部已有参数，只训练 DemandEstimator 和新接入 skip 的 D 分支。
10. D 不进入 Control encoder；D 只影响输出到 UNet 的多尺度 residual，以及每一步的 \(x_0\) 修正。
11. `DemandPathAdapter` 不再次输入或投影 \(z_r\)。\(z_r\) 已经通过 Condition encoder 进入 Control 主干。
12. Stage 3 的每条 D skip 分支以 zero-conv 结束，并从严格零扰动开始，接入瞬间保持 Stage 2 网络输出不变。
13. 最终图像质量训练目标使用像素 MSE、LPIPS 和标准扩散噪声 MSE，不使用 PSNR 作为直接损失；PSNR 通过像素 MSE 间接优化。

---

## 3. 整体端到端数据流

### 3.1 发送端与信道

对输入图像 \(x\)：

\[
x\xrightarrow{\text{Frozen SD2.1 VAE Encoder}}z^{raw}
\xrightarrow{\text{DJSCC Encoder}}s
\xrightarrow{\text{Power Normalization}}\bar s
\xrightarrow{\text{AWGN}}y.
\]

发送端只发送连续信道符号 \(\bar s\)，没有独立文本比特流。

### 3.2 接收端基础重建

\[
y\xrightarrow{\text{DJSCC Decoder}}\hat z^{raw}.
\]

由同一个接收 latent 得到两种表示：

\[
z_r=s_{vae}\hat z^{raw},
\]

\[
x_{base}=\operatorname{VAE}_{dec}(\hat z^{raw}).
\]

其中 \(z_r\) 是送入 SD2.1 UNet、Condition Model 和 DemandEstimator 的 scaled latent；\(x_{base}\) 是接收端可见的 DJSCC 像素域基础重建。

### 3.3 接收端 Prompt

主方案固定采用：

\[
\hat p=\operatorname{BLIP}(x_{base}),
\]

\[
e_p=\operatorname{SD2.1TextEncoder}(\hat p).
\]

BLIP、tokenizer 和 SD2.1 text encoder 均冻结。SD2.1 Prompt embedding 使用 OpenCLIP ViT-H/14 对应的 token hidden states，形状通常为 \([B,77,1024]\)。当前代码使用倒数第二层 hidden state，再经过 text encoder 的 final LayerNorm。

### 3.4 Demand 与扩散恢复

\[
D=E_D(z_r,\gamma),\qquad D\in[0,1]^{B\times1\times H_z\times W_z},
\]

其中 \(\gamma\) 是接收端已知的 SNR/CSI。

扩散恢复以 \(z_r\) 为 img2img 初始 latent。Condition Model 根据 \(z_t,t,z_r,\gamma,e_p\) 生成多尺度 residual；D 对这些 residual 进行可学习的局部调制，并在每一步进一步控制 \(x_0\) 向 \(z_r\) 对齐的程度。最终 latent 经冻结的 SD2.1 VAE Decoder 得到图像 \(\hat x\)。

### 3.5 总体流程图文本版

```text
Transmitter
  x_gt
    -> Frozen SD2.1 VAE Encoder
    -> z_gt_raw
    -> DJSCC Encoder(z_gt_raw, SNR)
    -> power-normalized channel symbols s

Channel
  s -> AWGN(SNR) -> y

Receiver base reconstruction
  y -> DJSCC Decoder(y, SNR) -> z_hat_raw
       ├─ × VAE scaling factor -> z_r
       └─ Frozen SD2.1 VAE Decoder -> x_base

Receiver text path
  x_base -> Frozen BLIP -> receiver prompt p_hat
         -> Frozen SD2.1 tokenizer/text encoder -> prompt embeddings e_p

Demand path
  (z_r, SNR) -> DemandEstimator -> D in [0,1]

Diffusion restoration
  z_r -> add noise at selected start timestep -> z_t
  (z_t, t, z_r, SNR, e_p) -> frozen Stage-2 Condition backbone
  (Control features H_l, resized D_l) -> learned D skip branches
  -> 12 down residuals + 1 mid residual
  -> Frozen SD2.1 UNet -> epsilon_full
  -> x0_full
  -> x0_D = z_r + D * (x0_full - z_r)
  -> convert x0_D back to epsilon_D
  -> DDIM scheduler step
  -> repeat for all denoising steps
  -> Frozen SD2.1 VAE Decoder -> final reconstruction x_hat
```

---

## 4. 符号和典型张量尺寸

以输入分辨率 \(384\times384\) 为例：

| 符号 | 含义 | 典型形状 |
|---|---|---|
| \(x_{gt}\) | 真值图像，VAE 输入范围为 \([-1,1]\) | \([B,3,384,384]\) |
| \(z_{gt}^{raw}\) | SD2.1 VAE 原始 latent | \([B,4,48,48]\) |
| \(z_{gt}\) | 乘 VAE scaling factor 后的真值 latent | \([B,4,48,48]\) |
| \(s\) | DJSCC 发送符号，默认 bandwidth=8 | \([B,8,48,48]\) |
| \(\hat z^{raw}\) | DJSCC 接收 raw latent | \([B,4,48,48]\) |
| \(z_r\) | 接收端 scaled latent | \([B,4,48,48]\) |
| \(x_{base}\) | DJSCC 像素域基础重建 | \([B,3,384,384]\) |
| \(e_p\) | SD2.1 Prompt embeddings | \([B,77,1024]\) |
| \(D\) | Demand map | \([B,1,48,48]\) |
| \(z_t\) | 第 t 步 noisy latent | \([B,4,48,48]\) |
| \(H_l\) | 第 l 个 Condition 多尺度特征 | \([B,C_l,H_l,W_l]\) |
| \(R_l\) | 注入 UNet 的 Control residual | 与对应 UNet skip 同形状 |
| \(\hat\epsilon\) | UNet 噪声预测 | \([B,4,48,48]\) |

必须区分 raw latent 和 scaled latent：

\[
z_{scaled}=s_{vae}z_{raw}.
\]

Stage 1 的 DJSCC 监督位于 raw latent 空间；Stage 2/3 的扩散 UNet 使用 scaled latent。

---

## 5. 沿用 DiffJSCC 的接收端 Prompt 设定

### 5.1 采用的实验协议

发送端不传输文本，接收端由当前信道实现下的 \(x_{base}\) 生成 Prompt：

\[
\hat p_\gamma=C(x_{base,\gamma}).
\]

不同 SNR 和不同噪声实现可能产生不同的 \(x_{base}\)，因此也可能产生不同 Prompt。训练时不能只按 `sample_id` 缓存 receiver Prompt；测试时必须针对每个 SNR 分别生成。

这一处理方式与 DiffJSCC 的接收端文本恢复思路一致。本项目采用它是为了复现并保持已有生成式 JSCC 系统的条件构建方式，而不是将其作为新的文本传输方案。

### 5.2 该设定在本文中的作用与边界

- 不引入额外文本信道和额外信道使用次数；
- 与 DJSCC 图像传输的带宽预算保持一致；
- Prompt 错误是接收端恢复链路的一部分，能够真实反映低 SNR 下的误差传播；
- 与“先基础重建、再提取语义、再生成增强”的接收端逻辑一致。

以上内容属于系统设定和公平比较协议。论文相关工作与方法部分应明确说明其来源于 DiffJSCC 类方法，不能将“从 \(x_{base}\) 生成 Prompt”写成本文贡献。

### 5.3 Prompt 消融

| 名称 | Prompt 来源 | 论文定位 |
|---|---|---|
| Receiver Prompt | \(\operatorname{BLIP}(x_{base})\) | 主实验协议，沿用 DiffJSCC，非创新点 |
| Oracle Prompt | \(\operatorname{BLIP}(x_{gt})\) 或真实标注 | 上限消融，不得作为主结果 |
| Empty Prompt | 空字符串 | 补充分析，用于观察文本条件影响 |

四个 Stage 2/3 训练测试脚本均支持：

```bash
--prompt-source receiver
--prompt-source original
```

当前主方案默认值已经统一为 `receiver`。为了实验日志可审计，正式实验命令仍建议显式写出 `--prompt-source receiver`。

---

## 6. Stage 1：训练 SD2.1 潜空间 DJSCC

### 6.1 目标

建立固定的基础通信链路，使发送 latent 经过功率约束和 AWGN 信道后，尽可能恢复原始 SD2.1 raw latent。

对应脚本：

```text
scripts/train_latent_djscc.py
```

核心模型：

```text
models/djscc.py
models/channel.py
models/vae_adapter.py
```

### 6.2 前向过程

\[
z^{raw}=\operatorname{Mode}(q_{VAE}(z|x_{gt})),
\]

\[
s=E_{JSCC}(z^{raw},\gamma),
\]

\[
\bar s=s\frac{\sqrt P}{\sqrt{\operatorname{mean}(s^2)+\varepsilon}},
\]

\[
y=\bar s+n,\qquad n\sim\mathcal N(0,\sigma^2),
\]

\[
\sigma^2=\frac{P}{10^{\gamma/10}},
\]

\[
\hat z^{raw}=G_{JSCC}(y,\gamma).
\]

### 6.3 DJSCC 结构

默认配置：latent channel=4、bandwidth channel=8、middle channel=128、SNR embedding=128、Encoder/Decoder 各 6 个 FiLM residual blocks。

Encoder：

```text
z_raw (4ch)
 -> Conv 3x3, 4 -> 128
 -> 6 x ResBlockFiLM(128)
 -> Conv 3x3, 128 -> 8
 -> per-sample power normalization
```

Decoder：

```text
received symbols (8ch)
 -> Conv 3x3, 8 -> 128
 -> 6 x ResBlockFiLM(128)
 -> Conv 3x3, 128 -> 4
 -> z_hat_raw
```

### 6.4 损失函数

\[
\mathcal L_{S1}=\operatorname{MSE}(\hat z^{raw},z^{raw}).
\]

best checkpoint 根据多个验证 SNR 上 latent MSE 的平均值选择。

### 6.5 训练与冻结模块

| 模块 | 状态 |
|---|---|
| SD2.1 VAE Encoder | 冻结，仅用于构建 latent cache |
| DJSCC Encoder/Decoder/SNR-FiLM | 训练 |
| AWGN | 无参数 |
| SD2.1 UNet、文本模块、Condition、Demand | 不参与 |

Stage 1 输出：

```text
scripts/checkpoints/DJSCC_SD21/best.pth
scripts/checkpoints/DJSCC_SD21/last.pth
```

### 6.6 信道资源口径

当前信道是实值 AWGN，一个发送张量元素计为一次 real channel use：

\[
\mathrm{CBR}=\frac{C_sH_sW_s}{3H_xW_x}.
\]

384×384 输入、8×48×48 符号时：

\[
\mathrm{CBR}=\frac{8\times48\times48}{3\times384\times384}=\frac1{24}.
\]

论文中必须明确这是 real channel use 口径；不可直接把连续 DJSCC 符号写成离散比特数。

---

## 7. Stage 2：不含 D 的 Condition Model 训练

### 7.1 目标

Stage 2 先学习普通 Control/Condition 功能。D 不存在，所有多尺度输出只经过原始 zero-conv 注入冻结 SD2.1 UNet。

对应脚本：

```text
scripts/stage2_train_control_sd21.py
scripts/stage2_test_control_sd21.py
```

架构版本：

```text
control_condition_zr_zero_conv_v3
```

### 7.2 输入

Condition Model 接收：

1. noisy latent \(z_t\)；
2. diffusion timestep \(t\)；
3. receiver Prompt embedding \(e_p\)；
4. DJSCC 接收 latent \(z_r\)；
5. SNR/CSI \(\gamma\)。

D 不参与 Stage 2。

### 7.3 Condition 主干

Condition 主干从 SD2.1 UNet encoder 初始化，包括 `conv_in`、time embedding、down blocks 和 mid block。

空间条件仅为 \(z_r\)：

\[
H_0=\operatorname{ConvIn}(z_t)+\operatorname{CondEncoder}(z_r).
\]

时间嵌入与 CSI 嵌入相加：

\[
e=e_t+e_{CSI}.
\]

Prompt embedding 进入带 cross-attention 的 Condition blocks，同时进入冻结 UNet。

Condition 主干产生 12 个 down features 和 1 个 mid feature，共 13 个输出位置。Stage 2 对每个特征只执行：

\[
R_l^{S2}=Z_l(H_l),
\]

其中 \(Z_l\) 是原始 zero-initialized 1×1 convolution。Stage 2 不包含 D projection、D gate、FiLM modulation 或额外 skip adapter。

### 7.4 扩散训练目标

\[
z_t=\sqrt{\bar\alpha_t}z_{gt}+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\]

Condition residual 注入冻结 UNet 后得到：

\[
\hat\epsilon_\theta=operatorname{UNet}_{frozen}
(z_t,t,e_p,R_1,\ldots,R_{13}).
\]

SD2.1 当前采用 epsilon prediction，损失为：

\[
\mathcal L_{S2}=\operatorname{MSE}(\hat\epsilon_\theta,\epsilon).
\]

### 7.5 训练与冻结模块

| 模块 | 状态 |
|---|---|
| Stage 1 DJSCC | 冻结 |
| SD2.1 VAE | 冻结 |
| BLIP、tokenizer、text encoder | 冻结 |
| SD2.1 UNet | 冻结，但允许梯度穿过 residual 输入 |
| Condition Model 主干 | 训练 |
| 13 个原始 zero-conv | 训练 |
| DemandEstimator/D skip 分支 | 不存在 |

`--freeze-control-encoder` 是低显存变体，会冻结从 UNet 复制的 encoder，仅训练条件编码和原始 zero-conv；论文主设置应明确是否启用，不能把两种配置混在同一结果表中。

Stage 2 每个 epoch 更新：

```text
control_best.pth
control_last.pth
```

收敛曲线每 200 个 batch 更新一次。

---

## 8. Stage 3：DemandEstimator 与 D 门控训练

### 8.1 目标

Stage 3 在已经训练好的 Stage 2 Condition Model 上增加 D 感知能力。Stage 2 主干保持完全冻结，避免 Demand 学习破坏已经建立的 Control 功能。

对应脚本：

```text
scripts/stage3_train_demand_estimator.py
scripts/stage3_test_demand_estimator.py
```

架构版本：

```text
DemandEstimator: demand_estimator_zr_snr_v3
Condition: demand_condition_zr_zero_init_skip_delta_v5
```

### 8.2 DemandEstimator 输入与语义

\[
D=\sigma(F_\phi(z_r,\gamma)).
\]

- 输入只包含 \(z_r\) 和 SNR；
- 输出是与 latent 同空间分辨率的单通道图；
- \(D\to0\)：倾向保留接收端 latent；
- \(D\to1\)：允许更强的生成式修正。

D 不是可靠性图。若与旧 RFRE/Reliability Map 对比，需要注意两者语义相反：旧可靠性高表示少修复，当前 Demand 高表示多介入。

### 8.3 DemandEstimator 架构

默认通道：64→96→128，SNR embedding 维度 64。

```text
z_r [B,4,48,48]
 -> Stem: Conv3x3 4->64 + GroupNorm + SiLU
 -> Encoder High: SNR-FiLM + 2 x LocalImageBlock
 -> Down: Conv3x3 stride2, 64->96                    [24,24]
 -> Encoder Mid: SNR-FiLM + 2 x LocalImageBlock
 -> Down: Conv3x3 stride2, 96->128                   [12,12]
 -> Context SNR-FiLM
 -> LocalImageBlock
 -> 2 x GlobalUnderstandingBlock / 8-head attention
 -> LocalImageBlock
 -> Decoder Mid: bilinear upsample + skip fuse + SNR-FiLM + local blocks
 -> Decoder High: bilinear upsample + skip fuse + SNR-FiLM + local blocks
 -> Refine LocalImageBlock
 -> GroupNorm + SiLU + Conv3x3 64->32 + SiLU + Conv1x1 32->1
 -> Sigmoid
 -> D [B,1,48,48]
```

`LocalImageBlock` 同时使用普通 3×3 depthwise convolution 和 dilation=2 的 3×3 depthwise convolution，再进行 1×1 融合、通道注意力和卷积 FFN。低分辨率 context 使用多头自注意力建模远距离区域关系。

SNR 通过零初始化 FiLM 注入各尺度：

\[
\operatorname{FiLM}(h,e_\gamma)
=h\odot(1+s(e_\gamma))+b(e_\gamma).
\]

### 8.4 D 在 Control skip 上的作用

对第 \(l\) 个冻结 Stage 2 Control feature \(H_l\)，先将 D 双线性缩放到对应空间尺寸：

\[
D_l=\operatorname{Resize}(D,H_l,W_l).
\]

Stage 2 基础 residual 保持：

\[
R_l^{base}=Z_l^{S2}(H_l).
\]

新增 D 分支执行：

\[
F_l=P_l^H(H_l),\qquad Q_l=P_l^D(D_l),
\]

\[
U_l=\operatorname{Fuse}_l([F_l,Q_l]),
\]

\[
G_l=\sigma(P_l^G(U_l)),
\]

\[
(S_l,B_l)=P_l^M(U_l),
\]

\[
\tilde H_l=(1+\tanh S_l)\odot H_l+B_l,
\]

\[
\Delta R_l^D=Z_l^D(G_l\odot\tilde H_l),
\]

\[
R_l^{S3}=R_l^{base}+\Delta R_l^D.
\]

其中：

- 每个 skip 拥有独立的 D projection、feature projection、fusion、gate、modulation 和 D zero-conv；
- gate 是逐样本、逐通道、逐空间位置的可学习张量，不是简单的 \(\rho+(1-\rho)D\)；
- `DemandPathAdapter` 输入只有 \(H_l,D_l\)，不输入 \(z_r\)；
- \(Z_l^D\) 权重和偏置严格初始化为零，所以刚接入 Stage 3 时

  \[
  \Delta R_l^D=0,
  \qquad R_l^{S3}=R_l^{base}.
  \]

这保证 Stage 3 从 Stage 2 的已训练功能出发，而不是突然扰动主网络。

### 8.5 D 在每一步 x0 上的作用

冻结 UNet 在 D residual 条件下预测完整噪声 \(\hat\epsilon^{full}\)，先反推：

\[
\hat x_0^{full}
=\frac{z_t-\sqrt{1-\bar\alpha_t}\hat\epsilon^{full}}
{\sqrt{\bar\alpha_t}}.
\]

然后以接收 latent \(z_r\) 为参考执行空间门控：

\[
\hat x_0^D=z_r+D\odot(\hat x_0^{full}-z_r).
\]

边界语义：

\[
D=0\Rightarrow\hat x_0^D=z_r,
\]

\[
D=1\Rightarrow\hat x_0^D=\hat x_0^{full}.
\]

再将 \(\hat x_0^D\) 转换回与 scheduler 兼容的噪声预测：

\[
\hat\epsilon^D
=\frac{z_t-\sqrt{\bar\alpha_t}\hat x_0^D}
{\sqrt{1-\bar\alpha_t}}.
\]

推理时 scheduler 使用 \(\hat\epsilon^D\) 完成当前步更新。相同 D 在所有扩散步骤使用，但会针对每一时刻的 \(\hat x_0^{full}\) 重新进行门控。

### 8.6 Stage 3 损失函数

训练时随机采样一个时间步，得到 Condition+UNet 的噪声预测和门控后的 \(\hat x_0^D\)。将 \(\hat x_0^D\) 通过冻结 VAE Decoder 解码为 \(\hat x\)。VAE 参数冻结，但保留从图像损失到 latent 输入的梯度。

像素损失：

\[
\mathcal L_{pixel}=\operatorname{MSE}(\hat x,x_{gt}).
\]

感知损失：

\[
\mathcal L_{LPIPS}=\operatorname{LPIPS}_{Alex}(\hat x,x_{gt}).
\]

加权图像损失：

\[
\mathcal L_{image}
=\frac{\lambda_p\mathcal L_{pixel}
+\lambda_l\mathcal L_{LPIPS}}
{\lambda_p+\lambda_l}.
\]

标准扩散损失：

\[
\mathcal L_{diff}
=\operatorname{MSE}(\hat\epsilon^{full},\epsilon).
\]

总损失：

\[
\mathcal L_{S3}
=\mathcal L_{image}+\lambda_d\mathcal L_{diff}.
\]

当前默认权重：

```text
lambda_pixel = 0.5
lambda_LPIPS = 0.5
lambda_diff  = 1.0
```

说明：\(\mathcal L_{diff}\) 使用 x0 最终门控之前的 `conditioned_eps`。因此它通过 D skip residual 约束 DemandEstimator 和 skip D 分支保持扩散去噪能力；图像损失同时监督 skip D 分支和 x0 D 门控。

PSNR 不直接作为损失。对于归一化图像，最小化 MSE 等价于提高 PSNR；训练日志中的 `psnr_proxy` 由像素 MSE 计算：

\[
\mathrm{PSNR}_{proxy}
=-10\log_{10}(\mathcal L_{pixel}/4),
\]

其中除以 4 是因为训练图像范围为 \([-1,1]\)，而标准 PSNR 按 \([0,1]\) 计算。

### 8.7 Stage 3 训练与冻结模块

| 模块 | 状态 |
|---|---|
| DJSCC | 冻结 |
| SD2.1 VAE Encoder/Decoder | 参数冻结，Decoder 保留输入梯度 |
| BLIP、tokenizer、text encoder | 冻结 |
| SD2.1 UNet | 参数冻结，保留对 Control residual 的梯度 |
| Stage 2 Condition 主干 | 完全冻结 |
| Stage 2 原始 zero-conv | 完全冻结 |
| DemandEstimator | 训练 |
| 13 个 D skip 分支 | 训练 |
| x0 gate | 无独立参数，梯度回传到 D |

Stage 3 输出：

```text
scripts/checkpoints/stage3_demand_control_SD21_xbase/best.pth
scripts/checkpoints/stage3_demand_control_SD21_xbase/last.pth
```

每个 epoch 更新 best/last，收敛曲线每 200 batch 更新一次。

---

## 9. 三阶段权重传递关系

```text
Stage 1
  train_latent_djscc.py
  -> DJSCC_SD21/best.pth
       |
       v
Stage 2
  frozen Stage-1 DJSCC + frozen SD2.1
  train Stage-2 Condition backbone + original zero-convs
  -> stage2_control_SD21_xbase/control_best.pth
       |
       v
Stage 3
  load and freeze Stage-2 Condition parameters
  attach zero-initialized D correction branch at every skip
  train DemandEstimator + D branches
  -> stage3_demand_control_SD21_xbase/best.pth
```

Stage 2 checkpoint的架构必须是：

```text
control_condition_zr_zero_conv_v3
```

Stage 3 runtime 架构必须是：

```text
demand_condition_zr_zero_init_skip_delta_v5
```

若出现 checkpoint architecture mismatch，通常表示把旧版含 full residual/旧 adapter 的权重加载到当前 zero-conv-only Stage 2/Stage 3 架构，需要使用当前 Stage 2 重新训练或使用匹配的 checkpoint。

---

## 10. 推理算法

对每个图像和 SNR：

1. 用冻结 DJSCC 产生 \(\hat z^{raw}\)、\(z_r\) 和 \(x_{base}\)；
2. 用 BLIP 从 \(x_{base}\) 生成 receiver Prompt；
3. 用 SD2.1 text encoder 得到 \(e_p\)；
4. 用 DemandEstimator 一次性预测 \(D=E_D(z_r,\gamma)\)；
5. 从 \(z_r\) 加噪到指定 img2img 起始时间步；
6. 对每个 DDIM 去噪时间步：
   - Stage 2 冻结 Condition 主干计算 \(H_l\)；
   - 每个 D branch 计算 \(R_l^{S3}\)；
   - 冻结 UNet 输出 \(\hat\epsilon^{full}\)；
   - 计算 \(\hat x_0^{full}\)；
   - 计算 \(\hat x_0^D=z_r+D(\hat x_0^{full}-z_r)\)；
   - 转为 \(\hat\epsilon^D\)，交给 scheduler 更新 \(z_t\)；
7. 用冻结 SD2.1 VAE Decoder 解码最终 latent。

Stage 2 推理没有 D skip branch 和 x0 gate；它直接使用训练后的 full-strength Control residual。

---

## 11. 主实验与评价设计

### 11.1 主比较路径

当前测试脚本输出以下方法：

| 方法 | 含义 |
|---|---|
| DJSCC | 直接使用 \(x_{base}\) |
| SD2.1 base | 无 Condition Model 的 SD2.1 img2img |
| Full Control | Stage 2 Condition Model，不使用 D |
| Joint/Stage3 gated | DemandEstimator + D skip + x0 gate |

Stage 2 测试比较前三者中的 DJSCC、SD2.1 base 和 Trained Control；Stage 3 测试同时比较 DJSCC、SD2.1 base、Full Control 和 Joint gated。

### 11.2 评价指标

- PSNR：像素保真度，越高越好；
- SSIM：结构保真度，越高越好；
- LPIPS：感知距离，越低越好；
- 可选增加语义指标，但不能替代上述主指标；
- 同时记录 D mean/min/max 和各尺度 learned gate mean，用于解释模型行为。

### 11.3 SNR 与随机性

- 在多个 SNR 上分别报告，不只报告跨 SNR 平均值；
- 不同恢复路径必须共享相同的 DJSCC observation 和 diffusion seed；
- 每个 SNR 的 receiver Prompt 来自该 SNR 对应的 \(x_{base}\)；
- 正式实验应固定数据划分、随机种子、DDIM steps、start strength、guidance scale 和 guidance rescale。

### 11.4 必做消融

1. 无 D 的 Stage 2 Full Control vs 完整 Stage 3，验证区域差异化生成修复的总体收益；
2. 固定或均匀 D vs 学习得到的空间 Demand Map，验证第一项创新；
3. 保留原始 zero-conv、移除 Demand Gate vs 加入 Demand Gate，验证第二项创新；
4. 仅 skip D gate、仅 x0 D gate、两者同时使用，分析两个 D 作用位置的贡献；
5. 不同图像损失权重 \(\lambda_p,\lambda_l\)；
6. 不同 \(\lambda_d\)，验证 diffusion loss 是否防止基础去噪能力退化；
7. DemandEstimator 输入消融可作为附加实验：\(z_r+SNR\) 为主方案，\(x_{base}+SNR\) 和 \(z_r+x_{base}+SNR\) 只作为对比，不改变主架构。

其中第 2 至第 4 项可能需要增加专门的测试开关；如果当前脚本尚未提供，不应在结果表中假装已经完成。

Receiver Prompt、Oracle Prompt 和 Empty Prompt 可以作为补充协议分析，用于排除文本质量带来的影响，但这组实验不用于证明本文创新，因为 receiver Prompt 本身沿用 DiffJSCC。

### 11.5 Prompt 可视化

当前 Stage 2/3 测试图会同时显示：

```text
Original prompt: ...
x_base prompt: ...
```

并用 `[USED]` 标记实际进入扩散模型的文本。JSON/CSV 同时保存 `prompt`、`prompt_source`、`original_prompt` 和 `receiver_prompt`，方便分析低 SNR 下的语义错误。

---

## 12. 当前已经确认的论文创新点

目前只明确主张以下两项创新。后续是否增加新的贡献点，应由实验结果和论文其余部分支持，不能仅根据已有模块数量拆分贡献。

1. **面向区域差异化生成修复的 Diffusion Demand Map**：提出 Demand Map 来描述各空间区域对生成式修复的需求强度。DemandEstimator 只使用接收端可获得的 \(z_r\) 和 SNR，在不同图像与不同信道条件下学习不同的空间需求分布，使可靠区域尽量保留 DJSCC 结果，使退化区域获得更强的生成先验。该 D 不依赖人工标签或 teacher/student，而是由最终图像质量目标和扩散去噪目标端到端诱导。
2. **在 Condition Model 原始 zero-conv 基础上引入 Demand Gate**：在传统 UNet+Condition Model 的多尺度 residual 注入路径中，保留已经训练好的原始 zero-conv，并并联加入由 D 驱动的可学习 Demand Gate 修正分支。每个尺度分别完成 Demand projection、Control feature projection、联合融合、空间/通道门控和特征调制，从而有选择地控制 Condition Model 向 UNet 传递的信息。D 分支以 zero-initialized convolution 结束，保证接入初始时对已训练 Stage 2 网络严格零扰动。

论文中不要把本方法描述成“准确预测真实 D 标签”，因为当前没有人工或 oracle D 标签。更准确的表述是：D 是由最终任务目标诱导出的可学习空间控制变量。

以下内容目前属于系统设定或方法实现，不单独主张为已经确认的创新：

- 从 \(x_{base}\) 恢复 Prompt：沿用 DiffJSCC；
- 三阶段训练：用于稳定优化和隔离各模块训练目标；
- 每一步的 x0 Demand 修正：属于当前完整方法的重要组成部分，是否独立概括为贡献需等待消融实验支持。

---

## 13. 绘图 Agent 必须体现的细节

### 13.1 总图布局

建议分为四个区域：

1. Transmitter；
2. AWGN Channel；
3. Receiver Base Reconstruction + Prompt/Demand；
4. Demand-aware Diffusion Restoration。

关键箭头：

- \(x_{gt}\to z^{raw}\to s\to y\to\hat z^{raw}\)；
- \(\hat z^{raw}\to x_{base}\to BLIP\to Prompt\to Text Encoder\)；
- \(\hat z^{raw}\to z_r\to DemandEstimator\)；
- \(z_r\) 同时进入扩散初始化、Condition encoder 和 x0 reference；
- D 分叉进入 13 个 skip D branches 和 x0 gate；
- Prompt embedding 同时进入 Condition cross-attention 和 UNet cross-attention。

### 13.2 Stage 2 图

- 不画 DemandEstimator；
- 不画 D；
- 每个 \(H_l\) 只连接一个原始 zero-conv；
- 画出 12 down residual + 1 mid residual 注入冻结 UNet；
- 标明只训练 Condition Model，UNet/VAE/DJSCC/文本模块冻结。

### 13.3 Stage 3 skip 放大图

必须画成并联结构：

```text
H_l -> Frozen Stage-2 ZeroConv ------------------------+
                                                        + -> R_l
H_l -> FeatureProj --+                                  |
                     +-> Fuse -> Gate + Modulation -> D-ZeroConv
D_l -> DemandProj ---+
```

不要把 D 画进 Control encoder，也不要把 \(z_r\) 画进 `DemandPathAdapter`。

### 13.4 x0 gate 放大图

```text
epsilon_full + z_t + alpha_t -> x0_full
z_r ------------------------------+
D --------------------------------+-> x0_D = z_r + D*(x0_full-z_r)
x0_D + z_t + alpha_t -> epsilon_D -> scheduler step
```

### 13.5 推荐颜色

- 蓝色：可训练模块；
- 灰色加雪花：冻结模块；
- 紫色：D 和 D 分支；
- 绿色：接收端观测 \(z_r,x_{base},SNR\)；
- 橙色：AWGN、加噪、插值和公式算子；
- 红色虚线：损失反向传播；
- 黑色虚线：Stage 间 checkpoint 初始化。

---

## 14. 当前代码入口与模块映射

| 功能 | 文件 |
|---|---|
| Stage 1 训练 | `scripts/train_latent_djscc.py` |
| Stage 2 训练 | `scripts/stage2_train_control_sd21.py` |
| Stage 2 测试 | `scripts/stage2_test_control_sd21.py` |
| Stage 3 训练 | `scripts/stage3_train_demand_estimator.py` |
| Stage 3 测试 | `scripts/stage3_test_demand_estimator.py` |
| 潜空间 DJSCC | `models/djscc.py` |
| AWGN 与功率归一化 | `models/channel.py` |
| DJSCC 接收观测封装 | `models/djscc_adapter.py` |
| SD2.1 VAE 适配 | `models/vae_adapter.py` |
| DemandEstimator | `models/demand_estimator.py` |
| Stage 2/3 Condition | `models/demand_condition_model.py` |
| x0 D 门控 | `models/demand_gating.py` |
| Stage 2/3 迭代恢复 | `models/restorer.py` |
| 模型加载、Prompt 编码、BLIP | `models/builder.py` |
| 主要超参数 | `config.py` |

服务器项目路径与主要 checkpoint：

```text
/home/wangch/projects/DMD-Net
/home/wangch/projects/DMD-Net/scripts/checkpoints/DJSCC_SD21/best.pth
/home/wangch/projects/DMD-Net/scripts/checkpoints/stage2_control_SD21_xbase/control_best.pth
/home/wangch/projects/DMD-Net/scripts/checkpoints/stage3_demand_control_SD21_xbase/best.pth
```

---

## 15. 推荐正式实验命令形式

Stage 2 主方案训练：

```bash
python scripts/stage2_train_control_sd21.py \
  --prompt-source receiver \
  --gpu-ids 0,1
```

Stage 2 主方案测试：

```bash
python scripts/stage2_test_control_sd21.py \
  --prompt-source receiver
```

Stage 3 主方案训练：

```bash
python scripts/stage3_train_demand_estimator.py \
  --prompt-source receiver \
  --gpu-ids 0,1
```

Stage 3 主方案测试：

```bash
python scripts/stage3_test_demand_estimator.py \
  --prompt-source receiver
```

原图 Prompt 上限实验只将参数改为：

```bash
--prompt-source original
```

若使用 `--prompt-mode empty`，则 Prompt source 不再影响推理输入。

---

## 16. 交接给其他 Agent 时的核对清单

其他 Agent 在修改代码、写论文或画图前，应先确认：

- [ ] 主 Prompt 来自当前 SNR 下的 \(x_{base}\)，不是原图；
- [ ] 文本不经过信道，不计额外无线带宽；
- [ ] Stage 1 只训练 latent DJSCC；
- [ ] Stage 2 不存在 D，只训练 Condition Model 和原始 zero-conv；
- [ ] Stage 3 冻结 Stage 2 全部已有参数；
- [ ] DemandEstimator 输入只有 \(z_r+SNR\)；
- [ ] D 不进入 Control encoder；
- [ ] skip D 分支使用 feature/D 投影、学习门控、FiLM 和 zero-initialized correction；
- [ ] x0 规则是 \(z_r+D(x_0^{full}-z_r)\)；
- [ ] Stage 3 没有 teacher/student 和 D 标签；
- [ ] Stage 3 损失是 Pixel MSE + LPIPS + diffusion MSE；
- [ ] 主测试报告 PSNR、SSIM、LPIPS，并按 SNR 分开统计；
- [ ] 原图 Prompt 只作为 Oracle 上限消融；
- [ ] 不把 DJSCC 连续信道符号直接等同于离散比特数；
- [ ] 所有 SD 组件均使用 SD2.1 配套权重和 latent scaling。

只要这些约束保持不变，后续可以继续调整学习率、损失权重、扩散步数、起始强度、数据集和实验规模，而不会改变论文方法的核心定义。

