# VLN 轨迹数据采集说明

本文档说明 `vln_sample.py` 中实现的轨迹采集逻辑。采集结果用于微调
Pi-0，使其替换现有 PPO policy，同时保留 Isaac Lab 中既有的 action
manager、关节目标换算和 articulation 控制流程。

## 1. 控制链路与学习目标

采集数据时，机器人仍然按照原有链路运行：

```text
用户导航指令
  -> VLN 模型
  -> 文本运动命令或局部运动增量
  -> 命令解析器
  -> 机身目标速度 [vx, vy, wz]
  -> PPO policy
  -> PPO 原始输出 [16]
  -> Isaac Lab action manager
  -> 腿部关节位置目标 [12] + 轮子关节速度目标 [4]
  -> robot articulation 和 actuator
```

监督标签是 **PPO policy 原始输出的 16 维 action**。记录 action 时，尚未
经过 action manager 的缩放和偏移处理。

因此，微调后的 Pi-0 只替换 PPO policy：

```text
图像 + 文本指令 + 本体状态
  -> 微调后的 Pi-0
  -> 与 PPO 输出兼容的原始 action [16]
  -> 现有 Isaac Lab action manager
  -> 现有 articulation 控制流程
```

采集器不会把最终写入 articulation 的 target 当成监督标签。现有 action
manager 仍然负责：

- 缩放 12 个腿部 action。
- 给腿部 action 加上默认站立姿态偏移。
- 把 4 个轮子 action 缩放成轮子速度目标。
- 把处理后的 target 写入 articulation。

当前 M20 配置中的缩放系数为：

```text
腿部 hip-x 关节 action scale:       0.125
其他腿部关节 action scale:          0.25
轮子速度 action scale:              5.0
```

这些缩放不会在采集器中重复执行。

## 2. 轨迹的开始与结束

只有命令行中以字面量 `goal` 开头的交互命令会创建一条轨迹：

```text
goal 前往自动售货机旁边
```

`goal` 后面的文字会被保存为该轨迹的文本指令。

以下命令不会创建轨迹：

```text
vln ...
move ...
back ...
left ...
right ...
turn ...
wait ...
script ...
photo ...
```

`vln` 仍然是可用的导航命令别名，但它只启动 VLN 导航，不触发数据采集。

主仿真循环消费到排队中的 `goal` 请求时，轨迹正式开始。出现以下任意条件
时，轨迹进入停止尾段：

- 用户手动输入 `stop`。
- VLN 返回 `stop`。
- VLN 交互轮数达到 `--vln_max_iterations`。
- VLN 交互过程抛出异常。

如果仿真关闭时仍在录制，当前轨迹会保存为：

```text
stop_reason = "simulation_closed"
```

如果上一条轨迹仍在录制时输入新的 `goal`，上一条轨迹会先保存为：

```text
stop_reason = "interrupted_by_new_goal"
```

然后再开始新轨迹。

## 3. 采样频率

M20 的 Isaac Lab 环境配置为：

```text
物理仿真步长:          0.005 s
控制 decimation:       4 个物理步
PPO policy 周期:       0.005 * 4 = 0.02 s
PPO policy 频率:       1 / 0.02 = 50 Hz
```

采集器严格按照 PPO policy 周期运行：每次 PPO 推理记录一帧，即 `50 Hz`。
采集器不使用真实墙钟时间决定是否采样。即使渲染或磁盘写入导致运行变慢，
相邻数据仍然对应相邻的 policy step。

两个机载相机采用相同的更新周期：

```text
camera update period = environment policy timestep = 0.02 s
```

接触力传感器每个物理步更新一次，即 `0.005 s`。采集器每个 policy step
读取一次其当前 buffer。

## 4. 单帧数据如何对齐

主循环中的关键顺序为：

```python
actions = policy(obs)
recorder.record_frame(env, actions, simulation_step)
obs, _, _, _ = env.step(actions)
```

因此，第 `t` 帧的语义是：

```text
observation[t] -> PPO 原始 action[t]
```

详细流程如下：

1. PPO policy 接收当前环境 observation。
2. PPO policy 计算原始输出 `action[t]`。
3. 在调用 `env.step(action[t])` 之前，采集器同步读取：
   - 前向机载相机 RGB。
   - 底部机载鱼眼相机 RGB。
   - 由同一张底部鱼眼图生成的去畸变 RGB。
   - 当前 robot state buffer。
   - 当前接触力 sensor buffer。
   - 采集器中保存的上一帧 action `action[t - 1]`。
   - 本帧刚计算出的 PPO 原始 action `action[t]`。
4. 环境接收并执行 `action[t]`。

因此，输入模态和监督 action 在同一个 policy step 上对齐。历史动作也不是
从之后的环境状态倒推得到的。

## 5. 机载相机

两个虚拟相机都挂载在：

```text
{ENV_REGEX_NS}/Robot/base_link
```

### 5.1 前向相机

前向相机为针孔 RGB 相机：

```text
名称:                  front_camera
默认分辨率:            640 x 480
默认偏移:              (-0.40, 0.0, 0.08) m，位于 base_link 坐标系
裁剪距离:              (0.05, 30.0) m
焦距:                  24.0
水平孔径:              20.955
```

### 5.2 底部鱼眼相机

底部相机为 polynomial fisheye RGB 相机：

```text
名称:                  bottom_fisheye_camera
默认分辨率:            640 x 480
默认偏移:              (0.0, 0.0, -0.085) m，位于 base_link 坐标系
鱼眼最大视场角:        200 度
去畸变输出视场角:      120 度
```

每个 policy step 都会保存：

- 原始鱼眼图。
- 从同一张鱼眼图生成的 rectilinear 去畸变图。

保留原始鱼眼图是为了以后可以重新调整去畸变参数，而不必重新采集轨迹。

相机分辨率和位姿可通过以下参数调整：

```text
--front_sensor_width
--front_sensor_height
--bottom_sensor_width
--bottom_sensor_height
--front_cam_x
--front_cam_y
--front_cam_z
--bottom_cam_x
--bottom_cam_y
--bottom_cam_z
--bottom_undistorted_fov
```

采集器保存高分辨率图像。训练时在 dataloader 中统一 resize 或 crop 到
`224 x 224`，而不是在仿真采集阶段提前缩小。

## 6. 本体状态向量

每帧保存一个原始的 `58` 维 state：

```text
state = concat(
    base_ang_vel,       # 3
    projected_gravity,  # 3
    normal_force,       # 4
    joint_pos,          # 16
    joint_vel,          # 16
    last_action,        # 16
)
```

具体布局如下：

| 切片 | 名称 | Shape | 语义 |
| --- | --- | --- | --- |
| `[0:3]` | `base_ang_vel` | `(3,)` | 机身坐标系下的 base 角速度，来自 `robot.data.root_ang_vel_b`。 |
| `[3:6]` | `projected_gravity` | `(3,)` | 投影到机身坐标系的重力方向，来自 `robot.data.projected_gravity_b`。 |
| `[6:10]` | `normal_force` | `(4,)` | 四个轮足接触力在世界坐标系 Z 轴上的正向分量，计算方式为 `maximum(contact_force_w[:, 2], 0)`。 |
| `[10:26]` | `joint_pos` | `(16,)` | articulation 中原始的绝对关节位置。不是 PPO observation 使用的 relative-to-default 关节位置。 |
| `[26:42]` | `joint_vel` | `(16,)` | articulation 中原始的关节速度。 |
| `[42:58]` | `last_action` | `(16,)` | 上一帧 PPO policy 的原始输出。每条轨迹的第一帧填零。 |

采集器不会应用 observation normalization、PPO observation scale、
action-manager scale 或默认关节位置偏移。

关节顺序固定为：

```text
0   fl_hipx_joint
1   fl_hipy_joint
2   fl_knee_joint
3   fr_hipx_joint
4   fr_hipy_joint
5   fr_knee_joint
6   hl_hipx_joint
7   hl_hipy_joint
8   hl_knee_joint
9   hr_hipx_joint
10  hr_hipy_joint
11  hr_knee_joint
12  fl_wheel_joint
13  fr_wheel_joint
14  hl_wheel_joint
15  hr_wheel_joint
```

## 7. 接触力

state 中的四个法向力来自：

```text
contact_sensor.data.net_forces_w
```

选取的四个 body 为：

```text
fl_wheel
fr_wheel
hl_wheel
hr_wheel
```

为了便于排查问题，`trajectory.npz` 同时保存：

```text
normal_force     shape [T, 4]
contact_force_w  shape [T, 4, 3]
```

`contact_force_w` 是世界坐标系下完整的 XYZ 接触力向量。
`normal_force` 仅保留其非负的世界坐标系 Z 轴分量，并写入 Pi-0 state。

## 8. PPO 原始 action

每帧保存：

```text
action       shape [16]
last_action  shape [16]
```

`action[t]` 是根据 `observation[t]` 计算出的 PPO policy 原始输出。

`last_action[t]` 是上一条已经记录的 PPO 原始输出。每条轨迹第一帧：

```text
last_action[0] = zeros(16)
```

action 维度遵循 action manager 的输入顺序：

```text
0:12   缩放和默认偏移处理之前的腿部位置控制 action
12:16  缩放之前的轮子速度控制 action
```

采集器不会把处理后的 articulation target 作为 Pi-0 监督标签。这是为了
在用 Pi-0 替换 PPO 后，保留原有后处理 pipeline。

## 9. 停止后的尾段

收到停止条件后，轨迹不会立刻关闭。

默认参数为：

```text
--stop_tail_seconds 0.75
```

在 `50 Hz` 下，采集器计算：

```text
round(0.75 / 0.02) = 38 帧
```

停止后的流程为：

1. controller 清除排队中的导航动作。
2. 机身速度命令变为零。
3. PPO policy 继续正常运行。
4. 采集器记录约 `0.75 s` 的站立过程。
5. 轨迹落盘。

尾段 action 是 PPO 在站立命令下产生的真实输出，不是人为填充的零向量。

## 10. 输出目录结构

默认顶级输出目录为：

```text
data/
```

可通过以下参数修改：

```text
--data_dir <path>
```

每条 `goal` 指令会创建一个按时间戳命名的子目录：

```text
data/
  20260601_201500_123456/
    metadata.json
    prompt.txt
    trajectory.npz
    action_chunks.npy
    action_chunk_valid_mask.npy
    front/
      000000.png
      000001.png
      ...
    bottom_undistorted/
      000000.png
      000001.png
      ...
    bottom_fisheye_raw/
      000000.png
      000001.png
      ...
```

三个图像目录使用相同的六位帧编号。例如：

```text
front/000127.png
bottom_undistorted/000127.png
bottom_fisheye_raw/000127.png
trajectory.npz 中的第 127 行
action_chunks.npy 中的第 127 行
```

都对应同一个 policy step。

## 11. trajectory.npz

`trajectory.npz` 保存逐帧数组：

| Key | Shape | Dtype | 含义 |
| --- | --- | --- | --- |
| `simulation_step` | `[T]` | integer | 从脚本进入主循环开始计数的全局 policy-step 编号。 |
| `simulation_time_s` | `[T]` | float | `simulation_step * dt`。 |
| `wall_time_ns` | `[T]` | integer | 由 `time.time_ns()` 生成的主机墙钟时间戳，仅用于诊断。 |
| `state` | `[T, 58]` | `float32` | 拼接后的原始 Pi-0 本体状态。 |
| `base_ang_vel` | `[T, 3]` | `float32` | 原始机身角速度。 |
| `projected_gravity` | `[T, 3]` | `float32` | 原始 projected gravity。 |
| `normal_force` | `[T, 4]` | `float32` | 世界坐标系 Z 轴方向的非负接触力。 |
| `contact_force_w` | `[T, 4, 3]` | `float32` | 世界坐标系下完整的接触力向量。 |
| `joint_pos` | `[T, 16]` | `float32` | articulation 中原始的绝对关节位置。 |
| `joint_vel` | `[T, 16]` | `float32` | articulation 中原始的关节速度。 |
| `last_action` | `[T, 16]` | `float32` | 上一帧 PPO 原始输出。 |
| `action` | `[T, 16]` | `float32` | 作为监督标签的当前 PPO 原始输出。 |

`T` 表示该轨迹记录的 policy frame 数量。

## 12. 滑动 action 窗口

Pi-0 输出 action chunk，而不是单个 action。采集器会根据逐帧 PPO action
预先生成滑动窗口。

默认 chunk 长度为：

```text
--action_chunk_size 32
```

在 `50 Hz` 下，一个完整 chunk 覆盖：

```text
32 * 0.02 = 0.64 s
```

对于每个帧编号 `t`：

```text
action_chunks[t] = action[t : t + 32]
```

保存文件为：

```text
action_chunks.npy
action_chunk_valid_mask.npy
```

shape 为：

```text
action_chunks.npy             [T, 32, 16] float32
action_chunk_valid_mask.npy   [T, 32]     bool
```

轨迹尾部可能不足 `32` 个未来 action。不可用的尾部会填零：

```text
action_chunks[t, valid_count:] = 0
```

并标为无效：

```text
action_chunk_valid_mask[t, valid_count:] = False
```

有效前缀标为 `True`。训练 dataloader 应使用 mask，或者直接丢弃不完整的
末尾窗口。

## 13. metadata.json

`metadata.json` 保存轨迹级 schema 和运行时信息：

```text
instruction
stop_reason
fps
dt
num_frames
state_dim
state_layout
action_dim
action_semantics
action_chunk_size
joint_names
wheel_body_names
images
```

其中，`action_semantics` 明确记录：

```text
raw PPO policy output before action-manager scaling and offsets
```

## 14. prompt.txt

`prompt.txt` 保存用户输入的导航指令，不包含开头的 `goal`。

例如：

```text
goal 前往自动售货机旁边
```

文件内容为：

```text
前往自动售货机旁边
```

## 15. 第一条轨迹的检查清单

采集第一条轨迹后，建议检查：

1. 输入 `goal ...` 后，脚本打印 `[DATA] Started trajectory: ...`。
2. 停止后，脚本打印 `[DATA] Recording 38 standing tail frames ...`。
3. 落盘后，脚本打印 `[DATA] Saved trajectory with ... frames: ...`。
4. 三个图像目录中的文件数量完全相同。
5. `trajectory.npz["state"].shape == (T, 58)`。
6. `trajectory.npz["action"].shape == (T, 16)`。
7. `action_chunks.npy.shape == (T, 32, 16)`。
8. `action_chunk_valid_mask.npy.shape == (T, 32)`。
9. 相邻 `simulation_step` 的差值严格为 `1`。
10. `last_action[0]` 全部为零。
11. 当 `t > 0` 时，`last_action[t]` 与 `action[t - 1]` 一致。
12. 相同帧编号的前向图、底部鱼眼原图和底部去畸变图应显示一致的机器人
    位姿和仿真时刻。

## 16. 当前适用范围

当前采集器用于单环境交互式采集：

```text
env_cfg.scene.num_envs = 1
```

它只记录第一个环境，并假定使用当前 M20 的关节名称和轮子 body 名称。
它不是通用的多机器人、多环境采集器。

每个 policy step 中，图像编码和磁盘写入会同步执行。当前阶段优先保证数据
对齐和可排查性。如果后续吞吐量成为瓶颈，可以再将图像编码和磁盘写入移动
到有界后台队列中，同时严格保留帧编号顺序。
