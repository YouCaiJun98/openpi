# 四足机器人 Pi-0 微调管线

本目录包含 M20 四足轮足机器人使用的 Pi-0 数据转换、训练、serve 和 query 脚本。

Isaac Sim 侧的数据采集脚本与详细采样说明已归档到
[`isaac_sim/`](isaac_sim/) 目录。继续采集时，应运行 Isaac Sim 仓库中的原始脚本；
这里的副本用于记录训练数据的来源与格式。

## 数据流概览

完整数据流如下：

```text
Isaac Sim 原始轨迹
  -> IsaacTrajectoryDataset 严格校验
  -> 转换为本地 LeRobot dataset
  -> openpi 标准 LeRobot loader 生成 32 步动作窗口
  -> repack、归一化、图像缩放、文本 tokenization、维度 padding
  -> QuadrupedPi0 训练
```

当前已经转换并验证的数据集包含：

```text
轨迹数量：10
总帧数：  4113
采样频率：50 Hz
总时长：  约 82.26 秒
训练样本：4113 条滑动窗口样本
```

每条轨迹对应一条导航指令，也对应一个 LeRobot episode。每一帧都会作为一个训练样本的
起点，因此一条长度为 `N` 的轨迹会生成 `N` 条滑动窗口样本。

## Isaac Sim 原始轨迹

Docker 容器中的原始数据目录为：

```text
/workspace/openpi/data/vln_sample
```

目录中的每个子目录是一条完整导航轨迹，例如：

```text
data/vln_sample/
  20260601_131356_367343/
    metadata.json
    prompt.txt
    trajectory.npz
    action_chunks.npy
    action_chunk_valid_mask.npy
    front/
      000000.png
      ...
    bottom_undistorted/
      000000.png
      ...
    bottom_fisheye_raw/
      000000.png
      ...
```

训练只使用 `front/` 和 `bottom_undistorted/` 中的图像。`bottom_fisheye_raw/` 保留作
采集质量检查，不送入模型。

每帧本体状态为 `58` 维向量：

| 字段 | 维度 | 含义 |
| --- | ---: | --- |
| `base_ang_vel` | 3 | 机身坐标系下的 base 角速度 |
| `projected_gravity` | 3 | 投影到机身坐标系下的重力方向 |
| `normal_force` | 4 | 四个轮足与地面的法向接触力 |
| `joint_pos` | 16 | 12 个腿部关节与 4 个轮子关节的位置 |
| `joint_vel` | 16 | 12 个腿部关节与 4 个轮子关节的速度 |
| `last_action` | 16 | 上一帧 PPO policy 的原始输出；首帧填零 |

每帧监督标签 `action` 为 `16` 维 PPO policy 原始输出。它位于 action manager 的缩放、
offset 和 actuator 处理之前，因此微调后的 Pi-0 可以直接替换原 PPO policy，并复用后续
控制管线。

## 原始轨迹校验

[`isaac_trajectory_dataset.py`](isaac_trajectory_dataset.py) 提供单轨迹 raw dataloader。
转换前会逐帧执行以下检查：

- `state_dim=58`、`action_dim=16`、`action_chunk_size=32`。
- metadata 中的 `fps=50` 与 `dt=0.02` 相互匹配。
- `simulation_step` 连续，`simulation_time_s` 每帧递增 `0.02s`。
- 状态、动作和动作窗口中不包含 `NaN` 或无穷值。
- `last_action[t] == action[t - 1]`，首帧 `last_action` 为零。
- `state` 的最后 `16` 维与 `last_action` 一致。
- 每帧预采集的 `32x16` action chunk 与逐帧 PPO 输出严格对齐。
- 轨迹尾部不足 `32` 帧的 action chunk 使用零填充，valid mask 正确标记有效部分。
- 两路训练图像逐帧存在，尺寸一致。

多轨迹转换时，会先完成所有轨迹的校验，再创建 LeRobot dataset。这样可以避免将不完整
轨迹写入训练缓存。采集仍在运行时，只应将已经结束写入的轨迹复制到 `data/vln_sample`。

## 转换为 LeRobot Dataset

批量转换全部完整轨迹：

```bash
uv run examples/quadruped/convert_isaac_trajectory_to_lerobot.py \
  --trajectories-dir /workspace/openpi/data/vln_sample \
  --overwrite
```

输出写入本地 LeRobot cache：

```text
~/.cache/huggingface/lerobot/openpi/m20_quadruped_isaac
```

转换器将每条 Isaac Sim 轨迹保存为一个 LeRobot episode，并保留采集时的 `640x480` 原始
图像。图像只在训练 batch 进入模型前缩放为 `224x224`。

调试单条轨迹时，可以显式指定轨迹目录并使用独立 repo id：

```bash
uv run examples/quadruped/convert_isaac_trajectory_to_lerobot.py \
  --trajectory-dir /workspace/openpi/data/vln_sample/20260601_131356_367343 \
  --repo-id openpi/m20_quadruped_isaac_single_trajectory \
  --overwrite
```

## Openpi Dataloader

转换后的 LeRobot dataset 只保存逐帧 `action`，不直接将预采集的 `action_chunks.npy` 写入
LeRobot。训练时，[`src/openpi/training/data_loader.py`](../../src/openpi/training/data_loader.py)
根据 dataset 的 `fps` 和模型的 `action_horizon` 自动构造动作序列：

```python
delta_timestamps = [t / fps for t in range(action_horizon)]
```

当前配置为：

```text
fps            = 50
action_horizon = 32
```

因此每条训练样本包含：

```text
输入：
  当前时刻的前视图像
  当前时刻的底部去畸变图像
  当前时刻的 58 维本体状态
  当前轨迹的导航文本

监督标签：
  从当前时刻开始的 32 个 PPO action
  shape = [32, 16]
  覆盖时长 = 32 * 0.02s = 0.64s
```

轨迹尾部不足 `32` 帧的窗口由 LeRobot loader 按 episode 边界补齐，不会跨越到下一条轨迹。

[`LeRobotQuadrupedDataConfig`](../../src/openpi/training/misc/quadruped_config.py) 随后执行：

1. 将 LeRobot 字段 repack 为 `observation/front_image`、`observation/bottom_image`、
   `observation/state`、`actions` 和 `prompt`。
2. 将前视图像映射到 Pi-0 的 `base_0_rgb`。
3. 将底部去畸变图像映射到 Pi-0 的 `left_wrist_0_rgb`。
4. 为兼容原始 Pi-0 接口，创建全零占位图像 `right_wrist_0_rgb`；普通 Pi-0 模式下 mask
   为 `False`。
5. 使用 `compute_norm_stats.py` 生成的统计量归一化 state 与 action。
6. 将图像缩放为 `224x224`，对导航文本执行 tokenization。
7. 将 `16` 维 action 使用零填充扩展到 Pi-0 内部使用的 `32` 维。`58` 维 state 保持不变，
   在 `QuadrupedPi0` 内部由 `state_adapter` 投影到 `32` 维。

训练日志中可看到最终 batch 形状：

```text
images['base_0_rgb']:       [batch, 224, 224, 3]
images['left_wrist_0_rgb']: [batch, 224, 224, 3]
state:                      [batch, 58]
actions:                    [batch, 32, 32]
```

`actions` 的最后一维为 `32` 是 Pi-0 内部 padding 后的维度；真实监督信号仍然位于前
`16` 维。推理输出经过 `QuadrupedOutputs` 后也只保留前 `16` 维，再送入机器人原有的
action manager 后处理管线。

## 模型配置

四足模型配置位于
[`src/openpi/models/quadruped_pi0_config.py`](../../src/openpi/models/quadruped_pi0_config.py)：

| 参数 | 当前值 | 含义 |
| --- | ---: | --- |
| `state_dim` | 58 | 四足本体状态维度 |
| `adapter_hidden_dim` | 128 | state adapter 隐藏层维度 |
| `action_horizon` | 32 | 每次预测的动作数量 |
| `action_dim` | 32 | Pi-0 内部连续动作维度；真实控制动作占前 16 维 |

正式微调时冻结 backbone，仅训练四足适配相关参数：

```text
state_adapter
state_proj
action_in_proj
action_out_proj
action_time_mlp_in
action_time_mlp_out
```

## 训练配置

训练配置位于
[`src/openpi/training/misc/quadruped_config.py`](../../src/openpi/training/misc/quadruped_config.py)。

| 配置名 | 数据集 | 模型 | steps | batch size | 用途 |
| --- | --- | --- | ---: | ---: | --- |
| `pi0_quadruped_synthetic` | 随机代理数据 | dummy | 10 | 2 | 快速检查合成数据 pipeline |
| `pi0_quadruped_synthetic_base` | 随机代理数据 | Pi-0 base | 1 | 2 | 检查 base checkpoint 与 adapter |
| `pi0_quadruped_isaac_single` | 单条真实轨迹 | dummy | 10 | 2 | 最小化排查真实数据 |
| `pi0_quadruped_isaac_single_base` | 单条真实轨迹 | Pi-0 base | 1 | 2 | 单轨迹 base checkpoint 检查 |
| `pi0_quadruped_isaac` | 全部真实轨迹 | dummy | 10 | 2 | 多轨迹 dataloader smoke test |
| `pi0_quadruped_isaac_base` | 全部真实轨迹 | Pi-0 base | 1 | 2 | 多轨迹 base checkpoint 检查 |
| `pi0_quadruped_isaac_finetune` | 全部真实轨迹 | Pi-0 base | 20000 | 8 | 推荐：冻结 backbone 的正式 adapter 微调 |
| `pi0_quadruped_isaac_full_finetune` | 全部真实轨迹 | Pi-0 base | 20000 | 8 | 高显存：解除冻结的全参数微调 |
| `pi0_quadruped` | 正式数据集 | Pi-0 base | 20000 | 32 | 正式微调模板 |

`Pi-0 base` 配置使用：

```text
gs://openpi-assets/checkpoints/pi0_base/params
```

需要多 GPU 时，配置中的 `fsdp_devices="auto"` 会根据当前环境自动选择设备，不写死双卡
或八卡数量。

当前真实数据包含 `4113` 条滑动窗口样本。以 `batch_size=32` 训练时，每轮约包含：

```text
4113 / 32 = 128.5
```

即约 `129` 个 optimizer steps。正式模板的 `20000` steps 大致相当于重复遍历当前数据
`155` 次，因此当前数据适合打通 pipeline，还不足以支撑正式训练。

## 计算归一化统计量

每次重新转换数据集后，都应重新计算 normalization stats：

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_quadruped_isaac
```

当前 10 条轨迹已经验证通过，统计量已成功写入：

```text
assets/pi0_quadruped_isaac/openpi/m20_quadruped_isaac/norm_stats.json
```

## 运行训练

先运行轻量多轨迹 smoke test：

```bash
uv run scripts/train.py pi0_quadruped_isaac
```

再使用真实 Pi-0 base checkpoint 执行单步检查：

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_quadruped_isaac_base
uv run scripts/train.py pi0_quadruped_isaac_base
```

## 使用全部真实数据进行正式微调

“使用全部数据训练”和“训练全部参数”是两个不同概念：

| 模式 | 使用 10 条真实轨迹 | 更新 adapter 与 action/state 投影 | 更新 Pi-0 backbone |
| --- | --- | --- | --- |
| 推荐 adapter 微调 | 是 | 是 | 否 |
| 全参数微调 | 是 | 是 | 是 |

当前只有约 `82.26` 秒数据。建议优先运行 adapter 微调，降低过拟合风险和显存开销。需要
进行对照实验，或者后续数据规模明显增加后，再尝试全参数微调。

### 推荐：使用全部轨迹进行 adapter 微调

每次重新转换数据后，先计算当前正式配置的 normalization stats：

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi0_quadruped_isaac_finetune
```

开始训练：

```bash
uv run scripts/train.py pi0_quadruped_isaac_finetune \
  --exp-name m20_isaac_finetune_v1 \
  --fsdp-devices auto \
  --no-overwrite \
  --no-wandb-enabled
```

默认设置为：

```text
数据集：          openpi/m20_quadruped_isaac
初始权重：        gs://openpi-assets/checkpoints/pi0_base/params
训练参数：        state_adapter 与 action/state 投影相关参数
num_train_steps： 20000
global batch size：8
num_workers：     2
save_interval：   1000
FSDP：            自动使用全部可见 GPU
```

### 全参数微调

需要更新 Pi-0 backbone 时，使用独立配置：

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi0_quadruped_isaac_full_finetune

uv run scripts/train.py pi0_quadruped_isaac_full_finetune \
  --exp-name m20_isaac_full_finetune_v1 \
  --fsdp-devices auto \
  --no-overwrite \
  --no-wandb-enabled
```

该配置没有设置 `freeze_filter`，因此 Pi-0 backbone、action expert、视觉与语言相关参数、
state adapter 和 action/state 投影层都会更新。它比 adapter 微调需要更多显存，也更容易
在当前小数据集上过拟合。

### 根据 GPU 数量调整 batch size

`batch_size` 是 global batch size，必须能够被当前可见 GPU 数量整除。正式配置默认使用
`8`，适合 1、2、4 或 8 张可见 GPU。显存不足时可以覆盖该参数：

```bash
uv run scripts/train.py pi0_quadruped_isaac_finetune \
  --exp-name m20_isaac_finetune_bs2 \
  --batch-size 2 \
  --fsdp-devices auto \
  --no-overwrite \
  --no-wandb-enabled
```

双卡环境可尝试 `--batch-size 2`；八卡环境至少应使用 `--batch-size 8`。启动训练前，应停止
Isaac Sim 或其他占用 GPU 的进程。

### 恢复训练

checkpoint 默认保存在：

```text
checkpoints/<config-name>/<exp-name>/
```

例如：

```text
checkpoints/pi0_quadruped_isaac_finetune/m20_isaac_finetune_v1/
```

从已有 checkpoint 继续训练时，使用相同的配置名和实验名，并传入 `--resume`。不要同时
传入 `--overwrite`：

```bash
uv run scripts/train.py pi0_quadruped_isaac_finetune \
  --exp-name m20_isaac_finetune_v1 \
  --num-train-steps 30000 \
  --fsdp-devices auto \
  --resume \
  --no-overwrite \
  --no-wandb-enabled
```

## Serve 与 Query

推荐 adapter 微调完成后，serve 最新 checkpoint：

```bash
uv run examples/quadruped/serve_checkpoint.py \
  --config-name pi0_quadruped_isaac_finetune \
  --exp-name m20_isaac_finetune_v1
```

全参数微调完成后，改用对应配置：

```bash
uv run examples/quadruped/serve_checkpoint.py \
  --config-name pi0_quadruped_isaac_full_finetune \
  --exp-name m20_isaac_full_finetune_v1
```

在另一个终端发送 query。这个脚本是 standalone 的，不依赖 `openpi_client` 或仓库内的
Python 环境：

```bash
uv run examples/quadruped/query_policy.py
```

如果要把脚本拷贝到一个更简单的环境里运行，只需要安装轻量客户端依赖：

```bash
python -m pip install numpy msgpack websockets
python query_policy.py --host 127.0.0.1 --port 8000
```

## 合成数据回归测试

需要脱离 Isaac Sim 数据执行最小回归测试时，生成本地合成 LeRobot dataset：

```bash
uv run examples/quadruped/generate_synthetic_lerobot.py --overwrite
```

然后运行：

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_quadruped_synthetic
uv run scripts/train.py pi0_quadruped_synthetic
```
