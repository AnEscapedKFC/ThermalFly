# GRaD-Nav 动作驱动的 3DGS 飞行模拟器

此目录是独立副本，不导入或修改 `gs_flight_simulator`、`grad_nav`。
输入标准 3DGS PLY，通过 GRaD-Nav 四维动作推进无人机，返回同一步的状态与第一视角。
会议室场景也支持 OpenFly-Agent RGB 指令导航，以及从已有示范拟合室内航点输出头。`pointcloud.py` 采用旋转长方体扫掠检测及首次接触前停止。

ThermalFly 仓库提供源码、配置和文档。运行已有场景需要在本地准备 3DGS 模型、碰撞几何和场景输入，复现示范还需要相应的轨迹或航点头权重。文档中指向录像、专家回合和运行报告的链接对应本地实验文件。场景准备方法见[会议室说明](scenes/room65_selected_anchored/README.md)和[走廊说明](scenes/corridor_native/README.md)。

[无人机模拟器原理与状态字段](SIMULATOR_PRINCIPLES_AND_STATE.md) 结合会议室巡航和沙发接触记录，说明每一步如何更新机体、渲染双模态画面，以及如何读取状态、动作和航点。

[走廊双模态飞行与 OpenFly RGB 导航](outputs/corridor_native_flights/README.md) 提供走廊长距离飞行、墙面碰撞和红色展板导航的第一视角录像、逐帧位姿、航点文件及复现命令。该场景使用 `corridor_full_native_20261121_1645/base_30000` 的 RGB 和红外模型。

## 用 OpenFly-Agent 在会议室进行 RGB 指令导航

从示范拟合、三帧视觉推理到航点执行和任务评分的完整过程，见[会议室 OpenFly RGB 导航流程](OPENFLY_RGB_WORKFLOW.md)。

默认指令是沿会议桌向门侧前进、绕过桌端、转向挂壁空调并悬停。策略读取英文指令与最近三次决策的 RGB 第一视角，首帧重复补齐历史。场景使用 `scenes/room65_selected_anchored/models/rgb.ply`、房间标定和原有碰撞表面，仅加载 RGB 高斯模型。

**执行已示范的绕桌指令**。室内航点头读取冻结的 OpenFly 视觉语言特征，预测机头坐标下的前移、左移、升降、偏航和停止。低层控制器使用当前机体状态跟踪航点。策略侧每次只能看到指令和第一视角，终点坐标、示范位姿、回合时间均不进入模型。

```bash
cd /home/zr_fan/3DGS_env/gs_gradnav_simulator
conda activate thermal_gaussian_5070
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -B -m openfly_rgb.navigate \
  --room-head outputs/openfly_room65_fit/room65_rgb.npz --resolution .5 --device cuda:0
```

**运行公开权重的原版动作头**。使用以下命令可以在相同的模拟器里评估原版 OpenFly 的离散决策。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -B -m openfly_rgb.navigate --resolution .5 --device cuda:0
```

模型独立运行在 `~/.cache/openfly-room65/venv` 中。首次在新机器上使用时，先在已配置好的模拟器环境下运行 `bash scripts/setup_openfly.sh`，下载固定版本的源码和约 15.1 GB 的官方权重。可用 `OPENFLY_CACHE_DIR` 更改安装和推理使用的目录。默认使用已完成闭环验证的 CUDA 四位 NF4 推理。场景始终在 GPU 渲染。可选 `--device cpu` 指定 CPU 模型推理，或用 `--device auto` 在 NF4 模式下按 7 GiB 空闲显存选择 CUDA 或 CPU。完整 7B 的 CPU 飞行尚未验证。[官方权重](https://huggingface.co/IPEC-COMMUNITY/openfly-agent-7b) · [上游推理实现](https://github.com/SHAILAB-IPEC/OpenFly-Platform/blob/c075075497a7122bad82f5b76b9be926ad5a81b3/train/eval.py)

要重建室内航点头，可运行下面的命令并选用新的输出名。拟合程序在当前 RGB 模型上重渲染原绕桌示范及其扰动视角，从实际位姿生成未来 1 秒的相对航点监督，再用径向基函数拟合轻量输出头。在线执行每个连续航点用 1.5 秒跟踪、0.5 秒稳定姿态。这个输出头针对一个房间的一条示范指令；任意新指令可通过原版动作头单独测评。在这台机器上请按顺序执行拟合与导航，两者分别需要加载一份 7B 模型。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -B -m openfly_rgb.fit_room \
  --output outputs/openfly_room65_fit/retrained.npz --resolution .5 --device cuda:0
```

自定义指令可通过命令行传入，例如从默认起点向左旋转约 45° 后悬停。下面的目标坐标、朝向只用于飞行结束后的评分。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -B -m openfly_rgb.navigate \
  --instruction "Turn left about 45 degrees in place, then hover." \
  --goal -1.55 1.10 1.41 --goal-yaw 29 --max-decisions 12
```

原版动作头每次推理输出一个动作。室内适配器将三个前进档映射为 0.15、0.30、0.45 m，转向映射为 ±15°。位置控制器以 20 Hz 跟踪目标，`STOP` 触发制动和位置保持。默认复用上游评测的 `vlnv1` 反归一化统计，该配置下升降和侧移通道恒为零。模型输出八维向量但无法匹配动作表时，回合以 `invalid_model_action` 结束。[动作统计](https://huggingface.co/IPEC-COMMUNITY/openfly-agent-7b/resolve/main/config.json)

结果写入新的 `outputs/openfly_rgb_<时间>/` 目录。打开 `rgb_fpv.mp4` 查看飞行，`decisions.jsonl` 记录模型原始输出和取帧编号，`frames.jsonl` 逐步保存实际控制及位姿。包含至少一个控制步的记录也可导入下方的飞行回放台。

`summary.json` 给出结束原因与终点评分。模型主动停止、距目标 ≤0.30 m、速度 ≤0.10 m/s，并在指定朝向时满足误差 ≤20°，即记为终点成功。中间路段的指令完成情况可通过视频和位姿记录核查。对于自定义指令，省略 `--goal` 时仅记录飞行，成功字段为 `null`。可用 `--spawn X Y Z --yaw DEG` 指定初始位姿，`--max-decisions` 设置决策上限。

绕桌指令另有 `route_check`。它沿参考线和实际飞行轨迹按最多 5 cm 间隔采样，用 0.35 m 半径分别计算参考路线被覆盖的比例、实飞路径贴近参考线的比例，并按顺序核对桌端的两个检查点。下列两次室内航点头测试均经过两个检查点并主动停止，两次碰撞数均为零。偏移测试从 `(-1.40, 1.22, 1.41) m`、偏航 `-6°` 起飞。

| 同一绕桌指令 | 决策数 | 终点距离误差 | 终点朝向误差 | 飞行录像 |
|---|---:|---:|---:|---|
| 原版动作头，示范起点 | 40 | 2.816 m，原地转向直至达到步数上限 | 10.02° | [RGB 第一视角](outputs/openfly_room65_rgb_baseline/rgb_fpv.mp4) |
| 室内航点头，示范起点 | 21 | 0.207 m | 3.73° | [RGB 第一视角](outputs/openfly_room65_rgb_nominal/rgb_fpv.mp4) |
| 室内航点头，偏移起点 | 22 | 0.258 m | 5.63° | [RGB 第一视角](outputs/openfly_room65_rgb_offset/rgb_fpv.mp4) |

三次运行均使用 320×240 RGB 输入和相同的碰撞表面。[俯视轨迹对比](outputs/openfly_room65_fit/trajectory_comparison.png)

## 启动会议室 RGB-T 无人机第一视角查看器

room65 第一视角使用 **PlanarGS RGB 30,000 步**和**锚定红外 24,000 步**，按原双相机支架同步渲染。碰撞检测继续使用锚定 RGB 24,000 步生成的 `collision/rgb_surface_m.ply`，巡航空域检查使用配套的 `collision/free_volume.npz`。场景来源及备份位置见[会议室场景说明](scenes/room65_selected_anchored/README.md)。

服务未启动时，可在本机终端运行以下命令。

```bash
cd /home/zr_fan/3DGS_env/gs_gradnav_simulator
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/zr_fan/miniconda3/envs/thermal_gaussian_5070/bin/python -B scenes/room65_selected_anchored/replay_viewer.py --port 7890
```

在「场景内置飞行」中选择原记录并点击「执行动作」，即可在当前场景重新飞行。也可导入原 `frames.jsonl` 执行动作，或导入 `waypoints.csv` 在「空间轨迹」中标出离散航点。点击画中画可切换 RGB 与热红外主画面。[打开会议室飞行回放台](http://127.0.0.1:7890/)

如需手动移动相机、并排观察同一位置的 RGB 与红外，请打开 7891 端口的双模态查看器。[双模态查看器](http://127.0.0.1:7891/) · [启动命令](../../ThermalGaussian_Surface/README.md#room65-双模态场景查看器)

以上命令以前台模式运行，按 `Ctrl+C` 停止服务。

## 自动飞行与视频（gate_mid 示例）

```bash
conda activate thermal_gaussian_5070
cd /home/zr_fan/3DGS_env/gs_gradnav_simulator
bash scripts/record_demo.sh \
  --scene ~/3DGS_env/no_train_outputs/export_gate_mid/gate_mid.ply
```

默认从场景生成普通障碍点云，启用碰撞；起点 `[-6,0,1.25]`、航向 0 度。
`configs/demo_actions.json` 预定义 **240 步、12 秒**动作：悬停、俯仰加速、制动、左右转向。
所有动作经过同一适配器，没有隐藏位置控制器或航点规划。
这是一段针对所选场景验证的演示序列，不保证任意场景或随机参数下都安全通过。

默认输出到本目录 `outputs/demo_<UTC时间>/`，拒绝覆盖非空目录：

| 文件 | 内容 |
|---|---|
| `fpv.mp4` | H.264，480×270、20 FPS；包含初始帧，完整序列为 241 帧、12.05 秒 |
| `states.jsonl` | 每一视频帧对应的动作、实际控制、状态、观测与碰撞回执 |
| `trajectory.csv` | 位置、姿态、速度、控制量、净空和事件 |
| `summary.json` | 执行结果、参数、帧数及完整解码验证 |
| `profile.json` / `sequence.json` | 实际运行配置与动作序列 |
| `obstacles.ply` / `obstacles.json` | 使用的普通点云及存在的导出元数据 |

碰撞或状态异常会提前结束并保存已录制部分。视频采用仿真时间轴，运行耗时可以不同。
使用 `imageio-ffmpeg` 自带 FFmpeg，不需要系统 FFmpeg。

复用已保存的点云或启用训练参数：

```bash
bash scripts/record_demo.sh \
  --point-cloud outputs/gate_mid_nominal/obstacles.ply \
  --output-dir outputs/another_flight
bash scripts/record_demo.sh --profile training --seed 7
```

实际参数会写入回执。相同种子可复现本模拟器的参数、控制和观测噪声。
`reset` 不传新种子时重放当前种子；需要不同参数抽样时显式传入新种子。
动作文件第四维的 `"hover"` 按本回合质量/最大推力换算为固定动作，不是高度反馈控制。

## 动作与训练配置

| `format` | 约定 |
|---|---|
| `gradnav_normalized` | tanh 后 `[roll,pitch,yaw,thrust]`，每维 `[-1,1]` |
| `gradnav_logits` | tanh 前输出，由本模块仅执行一次 tanh |
| `body_rates_thrust` | 已处理的 `[wx,wy,wz,c]`：rad/s、总推力比例 `[0,1]`；跳过 LPF/缩放/延迟 |

前两种默认经过 LPF(alpha=0.5)、角速度缩放 `0.5*a[:3]`、推力映射 `0.25*(aT+1)`，
再执行角速度延迟 0.8 和推力延迟 0.7。
零归一化动作的稳态推力为 0.25，不是悬停；名义参数悬停第四维约为 `0.660153846`。

- `configs/nominal.json`：开启缩放/LPF/延迟，固定物理参数，无噪声，用于演示与基准。
- `configs/training.json`：控制噪声标准差 0.01、观测噪声幅度 0.1；质量 `[1.0,1.2] kg`、
  最大推力 `[24,28] N`、惯量 ±20% 的按回合抽样。
- `--profile /路径/config.json` 可覆盖配置；也能读取原训练 YAML 的 `params.config/env_hyper`，
  采用 `player.LPF_eval/LPF_val`、`domain_randomization`、`obs_rms`。
  YAML 没有定义的物理范围来自本目录记录的原环境代码设置。

这里匹配训练代码的参数，未改用论文表 III 中不同的范围。
`delay.history=legacy_two_step` 明确保留原环境的两拍历史读法；`previous_step` 是通常的一阶延迟，
但会改变旧策略执行行为。动力学保留标准 wxyz 旋转公式，不复现原实现中不一致的旋转矩阵。

## 按步执行 API

```bash
bash run.sh --scene ~/3DGS_env/no_train_outputs/export_gate_mid/gate_mid.ply
```

打开 `http://127.0.0.1:8766` 可监看 FPV、执行单步、复位。
客户端无需浏览器。每步推进 `0.05 s`，含 10 个 `0.005 s` 子步。
等待推理、读取状态或画面时仿真时间不前进；下一步继承当前速度与姿态。

```bash
curl -X POST http://127.0.0.1:8766/api/step \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"action-0001","action":[0,0,0,0.660153846],"format":"gradnav_normalized","repeat":1}'
```

| 接口 | 用途 |
|---|---|
| `GET /api/state` | 最近一次完成的回执 |
| `POST /api/step` | 执行动作；repeat=1–200，逐控制周期更新滤波、历史、观测 |
| `POST /api/reset` | 复位，可传 seed、spawn、yaw |
| `GET /api/frame/fpv?frame_id=N` | 编号 N 的 JPEG 监看图像 |
| `GET /api/frame/fpv?frame_id=N&format=npy` | 编号 N 的原始 float32 RGB，H×W×3、范围 `[0,1]` |

POST 必须带唯一字符串 `request_id`，网络重试沿用同一 ID，不会重复执行动作。
相同 ID 换载荷、或 `expected_frame_id` 过期返回 409。
保存最近 128 个成功请求回执、32 个图像快照；过期图像返回 404。
CUDA 和状态更新由同一个工作线程串行执行；执行后渲染/编码器失败会停用动作服务，避免重试重复推进。

回执包含 episode/step/frame ID、仿真时间与实际推进时间、原动作与实际控制、完整状态、
碰撞与净空、completed/terminated/truncated/reason，以及与图像匹配的观测。
`completed` 表示动作时长执行完毕，不表示已经悬停或到达目标。
碰撞可能缩短最后一个子步；必须检查 `collision.event`，不能只看最终 `in_collision`。
速度、角速度、倾角、有限值及可选世界边界均检查；max_steps 为时间截断。
`min_sampled_safety_clearance` 是子步采样最小净空，不是完整连续表面的精确最小距离。

## 已有策略的观测

57 维布局按原仓库默认配置：

```text
[z,vz,quaternion_xyzw(4),applied_control(4),previous_control(4),
 world_velocity(3),visual_features(16),context_latent(24)]
```

状态回执四元数为 wxyz，策略观测转换为 xyzw。保留原代码的 5 帧 CENet 历史时序，
附有历史帧编号与有效性标记。RGB 按 AdaptiveAvgPool 缩到 224×224，不另加 ImageNet 归一化。

本次没有提供视觉/CENet/归一化权重，因此默认 `policy_ready=false`、完整策略向量为 null；
仍返回 17 维基础观测、真值状态和对应 RGB。预定义动作录制不需要这些权重。
不使用随机 CNN 或零潜变量冒充已有策略可用的学习特征。

外部导航进程可以使用原始观测构造输入；也可配置真实权重导出的 TorchScript 编码器和 JSON 归一化器：

```json
{"visual_encoder":"visual_encoder.ts","context_encoder":"context_encoder.ts","obs_rms":"obs_rms.json"}
```

```bash
bash run.sh --observation-assets /路径/assets.json
```

路径相对 manifest。视觉输入 `[1,3,224,224]`、输出 `[1,16]`；上下文输入 `[1,285]`、输出 `[1,24]`，
导出确定性均值或实际训练所需的采样包装。归一化 JSON 为 mean/var 两个 57 维数组及 epsilon=1e-5。
不自动执行普通 Python pickle 检查点，不自动下载或随机初始化学习网络。

## 文件与验证

`engine.py`：按步执行与状态检测；`app.py`：串行 API；`action_adapter.py`：动作；
`observations.py`：观测；`dynamics.py`：可配置动力学；`scene.py/cameras.py`：渲染；
`pointcloud.py/export_pointcloud.py`：原点云/OBB 路径；`record_sequence.py`：自动飞行与视频。

```bash
python -m unittest discover -s tests -q
```

依赖见 requirements.txt。脚本使用当前 Python，也支持 GS_FLIGHT_PYTHON。
复制整个目录与 PLY 数据即可搬迁，无需原项目源码。保留 LICENSE-third-party.md。
碰撞定义见 COLLISION.md。
