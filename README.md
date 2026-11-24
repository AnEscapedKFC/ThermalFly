# GRaD-Nav 动作驱动的 3DGS 飞行模拟器

此目录是独立副本，不导入或修改 `gs_flight_simulator`、`grad_nav`。
输入标准 3DGS PLY，通过 GRaD-Nav 四维动作推进无人机，返回同一步的状态与第一视角。
会议室场景也支持 OpenFly-Agent RGB 指令导航，以及从已有示范拟合室内航点输出头。`pointcloud.py` 采用旋转长方体扫掠检测及首次接触前停止。

ThermalFly 仓库提供源码、配置和文档。运行已有场景需要在本地准备 3DGS 模型、碰撞几何和场景输入，复现示范还需要相应的轨迹或航点头权重。文档中指向录像、专家回合和运行报告的链接对应本地实验文件。场景准备方法见[会议室说明](scenes/room65_selected_anchored/README.md)和[走廊说明](scenes/corridor_native/README.md)。

[无人机模拟器原理与状态字段](SIMULATOR_PRINCIPLES_AND_STATE.md) 结合会议室巡航和沙发接触记录，说明每一步如何更新机体、渲染双模态画面，以及如何读取状态、动作和航点。

[走廊双模态飞行与 OpenFly RGB 导航](outputs/corridor_native_flights/README.md) 提供走廊长距离飞行、墙面碰撞和红色展板导航的第一视角录像、逐帧位姿、航点文件及复现命令。该场景使用 `corridor_full_native_20261121_1645/base_30000` 的 RGB 和红外模型。

## 训练会议室与走廊共用的 RGB 航点策略

第一版采用行为克隆。冻结的 OpenFly-Agent 7B 读取英文指令、当前 RGB 和前两次决策的 RGB。共享 PyTorch 航点头还读取当前机体系线速度、角速度、重力方向和云台角。训练标签来自每个专家决策的三个连续航段。推理时只执行首段，刷新图像和机体状态后重新规划。

### 准备数据和提取特征

在仓库根目录运行以下命令。模拟器和训练脚本使用装有本仓库依赖的环境，OpenFly 使用 `scripts/setup_openfly.sh` 配置的独立推理环境。准备好本地专家数据集及 `splits.json`，每个回合须有 `episode.json`、`frames.jsonl`、`decisions.jsonl` 和决策引用的 RGB PNG。[逐字段数据格式](EXPERT_EPISODE_FORMAT.md)

```bash
conda activate thermal_gaussian_5070
# 首次安装 OpenFly 推理环境与官方权重时运行
bash scripts/setup_openfly.sh

DATASET=outputs/expert_rgbt_20261123
RUN=outputs/thermalfly_rgb_v1
python -m openfly_rgb.waypoint_data --dataset "$DATASET" --output "$RUN/cache"
python -m openfly_rgb.train_waypoint --cache "$RUN/cache" --output "$RUN/train"
```

缓存按回合写入，每回合清空图像历史。提取中断后可用相同命令复用已校验的回合。如有人工审核导出，在提取命令加 `--review <路径>` 以排除退回回合。`room_orbit_vary` 的英文指令在输入时校正为前半程经过高位、后半程经过低位，原始轨迹保持原样。重复训练时为 `--output` 选择新目录。

### 训练、选模与闭环复现

每个训练子集单独拟合降维和归一化。OpenFly 的 4,096 维视觉语言特征与 1,536 维当前图像空间特征各投影为 32 维，与 10 维状态拼接。训练比较线性头和单隐藏层 MLP，在固定验证集上选择配置并校准 STOP 阈值。最终策略保存在 `train/policy.pt`，离线指标和子集、种子、验证选择过程保存在 `train/report.json`。

默认按路线族、回合平衡采样，使用 AdamW，学习率和权重衰减均为 0.001，批大小 32，最多训练 100 轮，早停耐心为 10 轮。航点回归使用 Smooth L1，STOP 的二元交叉熵正类权重为 3。MLP 的隐藏层为 32 维，Dropout 为 0.1。验证集首槽 STOP 的 F1 用于选择阈值。

每个航段预测机头水平坐标系中的前、左、上、偏航增量，以及 STOP、时长、末速和云台俯仰。连续量仅用 MOVE 监督，走廊固定云台，缺失的云台标签和填充槽通过掩码处理。闭环沿当前实际速度衔接相邻航段，STOP 触发 2.5 秒主动悬停，随后检查速度与漂移。

```bash
# 专家目标和模型预测通过同一执行器。每条命令使用新的输出目录。
python -m openfly_rgb.waypoint_rollout --dataset "$DATASET" --episode room_inspect_ac \
  --oracle --output "$RUN/oracle_ac"
python -m openfly_rgb.waypoint_rollout --dataset "$DATASET" --episode room_inspect_ac \
  --head "$RUN/train/policy.pt" --output "$RUN/learned_ac"
python -m openfly_rgb.score_waypoint --dataset "$DATASET" "$RUN/oracle_ac" "$RUN/learned_ac"

# 同一起点、朝向、进场速度和云台角，分别要求观察笔记本与沙发
python -m openfly_rgb.waypoint_rollout --dataset "$DATASET" \
  --head "$RUN/train/policy.pt" --episode room_inspect_laptop room_inspect_sofa \
  --spawn-from room_inspect_ac --output "$RUN/paired_targets_replay"
```

闭环需要对应的本地场景模型、碰撞几何和相机标定。查看每次飞行的 `summary.json`、`decisions.jsonl`、`frames.jsonl` 与第一视角视频。事后评分使用真实相机位姿计算地标中心是否入画、接近距离和子目标观察顺序，结果写入 `geometry.json`。几何投影不能判断遮挡，物体辨认可结合视频复核。

### 2026-11-24 首次训练结果

会议室和走廊共 34 个回合。训练、验证、测试依次为 26/6/2 个回合和 555/110/53 个决策。两个物理场景均在训练集中。采集元数据的人工审核状态为 `pending`，本次使用通过结构检查的全部回合。训练仅使用 RGB。冻结 OpenFly 在 RTX 5070 Ti 上以 4 bit 提取特征，小型 PyTorch 航点头在 CPU 上训练。

| 训练回合 / 决策数 | 线性头验证损失 | MLP 验证损失 |
|---|---:|---:|
| 8 / 147 | 1.169 ± 0.033 | 0.825 ± 0.014 |
| 16 / 318 | **0.816 ± 0.004** | 0.822 ± 0.016 |
| 26 / 555 | 1.005 ± 0.021 | 0.904 ± 0.006 |

表中为按回合平均的验证损失，再对三个随机种子取均值与标准差。按组均值选出 16 回合线性头，选组内验证损失居中的种子 47 部署，STOP 阈值 0.7031。下表的误差对回合等权，STOP 单独列出实际决策数。

| 离线评估 | 首段位置误差 | 首段偏航误差 | 三段终点误差 | STOP 命中 / 误报 / 漏判 |
|---|---:|---:|---:|---:|
| 验证，6 回合 | 0.263 m | 4.32° | 0.580 m | 4 / 0 / 2 |
| 测试，2 回合 | 0.191 m | 7.04° | 0.417 m | 1 / 2 / 1 |

**在同一起点、同一任务、同一执行器下，专家命令完成 8/8 个留出回合，学习策略完成 0/6 个验证回合、0/2 个测试回合。** 成功要求发出 STOP、通过悬停稳定检查，并满足该回合的位置、航向和速度容差。8 次专家回放的终点误差均低于 0.05 m，四条带物体观察的回合都满足有序观察的几何判据。额外的三条训练路线控制检查也全部到点。对学习策略，8 个留出起点在线渲染所得的首段位置预测、偏航预测和 STOP 概率均与离线缓存重算一致。

| 学习策略的测试任务 | 实际飞行结果 |
|---|---|
| 走廊升高并加速 | 飞行 5.82 m 后触发已知自由空间边界，距目标 10.88 m |
| 搜索笔记本再观察沙发 | 飞行 1.60 m 后提前悬停，距目标 2.84 m。两处地标中心在本次飞行中均未进入 RGB 画面 |

另从 `room_inspect_ac` 的同一初始状态分别下达观察笔记本、观察沙发的指令。两次的首段位移预测相差 0.063 m，最后分别距各自参考终点 1.55 m、2.13 m，均未在目标的参考观察距离内将其置于画面中央。这组对照显示指令改变了预测，但当前策略尚未完成目标区分与抵近。

本地 `outputs/thermalfly_rgb_v1/` 下的 `oracle_heldout/`、`heldout/` 分别保存八条留出任务的专家回放与学习策略飞行记录，`same_start/` 保存换指令对照，`oracle/` 保存三条训练路线的执行器检查。八次留出策略飞行中有四次在错误位置停稳，两次发出 STOP 后悬停未稳，两次触发已知空间边界。下一轮可以优先补采远离终点却外观相似的继续飞行样本，以及目标仍在视野外时的转向搜索和偏航恢复样本。

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
