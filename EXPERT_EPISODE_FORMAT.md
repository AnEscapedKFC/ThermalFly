# 室内 RGB-T 导航专家轨迹数据格式与采集说明

一条回合（episode）是无人机从一次场景重置到终点悬停的完整飞行。一条训练样本对应其中一次导航决策。要回答“给模型看什么、要它预测什么”，先在 `decisions.jsonl` 选一行，通过 `observed_frame_id` 和 `history_frame_ids` 到 `frames.jsonl` 找图像和机体状态，从 `episode.json` 取语言指令，从该决策的 `teacher.chunk` 取航点和停止标签。本说明以 `rgbt_episode_v1` 已采集的会议室和走廊实飞轨迹为准，末尾保留策略驾驶和 DAgger 的扩展约定。[本批任务、视频和指标](outputs/expert_rgbt_20261123/README.md)

[文件布局](#files) · [航线与回合元数据](#route) · [观测与决策字段](#frames) · [真实训练样本](#example) · [专家采集流程](#workflow) · [场景、划分与审核文件](#shared) · [其他回合类型](#extensions)

<a id="files"></a>

## 文件布局与读取顺序

以下路径从本批数据集根目录 `outputs/expert_rgbt_20261123/` 起算。JSON 文件保存一个对象（个别报告保存数组）。JSONL 每行是一个独立 JSON 对象。帧号和决策号各自在回合内从 0 编号，跨回合关联时要同时使用 `episode_id`。

| 文件或目录 | 写入时机与作用 |
|---|---|
| `routes/<episode_id>.json` | 起飞前编辑的专家路线，包含指令、带时刻的世界系节点，可附云台和任务阶段。是采集脚本的输入。 |
| `episodes/<episode_id>/route.json` | 采集时把当次使用的路线复制到回合内。回放、重建教师航点时以这份快照为准。 |
| `episodes/<episode_id>/episode.json` | 一回合一份。给出指令、场景、来源、坐标和时间约定、停止条件以及实飞汇总。 |
| `episodes/<episode_id>/frames.jsonl` | 每个控制时刻一行。保存实际位姿、速度、入帧控制量、相机外参、安全状态和图像路径，含初态与悬停末帧。 |
| `episodes/<episode_id>/rgb/000000.png`、`thermal/000000.png` | 同帧号、同仿真时刻的两路无损观测，文件名按六位十进制补零。当前每帧各有一张 320 × 240 像素的三通道 8 位 PNG。两路相机的内参和视场各自独立。 |
| `episodes/<episode_id>/decisions.jsonl` | 每个高层决策一行，记录决策前能看到什么、教师提供什么监督、接下来实际执行了哪一段。 |
| `episodes/<episode_id>/rgb_thermal_fpv.mp4` | 从同一批 PNG 对应的渲染数组同步编码的第一视角视频。20 fps，640 × 240，左 RGB、右伪彩红外，供人连续检查。训练读取 PNG 可保留像素原值。 |
| `scenes/<scene_version>/calibration.json`、`scenes/room65_selected_anchored/landmarks.json` | 跨回合复用的标定及资产摘要。会议室另有地标坐标与来源，供教师和语义审核使用。 |
| `splits.json` | 数据集唯一的训练、验证、测试清单，按路线族分组。 |
| `README.md`、`verification.json`、`reference_verification.json` | 依次为回合索引、逐帧校验报告、航点坐标转换和走廊速度校验报告。 |
| `room_evidence.json`、`evidence/`、`review_media/`、`review.html` | 会议室任务证据、对应逐条记录与拼图、所有回合的预览图、人工审核入口。报告与人工审核导出字段见下文。 |
| `run_logs/`、`work_room/`、`work_corridor/` | 本次路线设计、预检和运行日志等过程材料，可用于诊断。正式回合的实飞状态和速度以 `episodes/` 为准。 |

同一任务的**三种轨迹**可以分别读取。画预设航线取 `route.json → knots`。画实际飞行轨迹取 `frames.jsonl → position_w_m`。训练局部航点预测取 `decisions.jsonl → teacher.chunk`。查无人机如何跟上航线，则读 `frames.jsonl` 中逐帧控制量。

<a id="route"></a>

## 专家航线 `route.json`

会议室和走廊使用同一格式。会议室路线常附地标和云台俯仰。走廊路线通过节点的位置和时间控制飞行速度。[笔记本抵近路线](outputs/expert_rgbt_20261123/episodes/room_inspect_laptop/route.json) · [走廊全程路线](outputs/expert_rgbt_20261123/episodes/corridor_full_transit/route.json)

| 字段 | 含义与使用方式 |
|---|---|
| `id`、`scene`、`family` | `id` 标识本次飞行，`family` 标识母路线。场景键 `room` 和 `corridor` 分别映射到 `room65_selected_anchored` 和 `corridor_native`。 |
| `instruction`、`instruction_zh` | 本批回合配套的英文原指令及中文译文。采集器按预设航线控制飞行，两份文本描述同一任务。 |
| `title`、`route_description` | 可选的任务标题和路线设计说明，如走廊只在终点减速。 |
| `knots` | 每行 `[t_s, x_w_m, y_w_m, z_w_m, yaw_deg]`，依次为仿真时刻、机体中心世界系位置、机头水平偏航。至少 3 行，从 0 秒起时间严格递增。连续绕圈可用超过 360° 的展开航向。末两行位姿相同且间隔至少 0.5 s，末时刻对齐控制时钟。 |
| `interpolation` | 参考节点间的插值。缺省 `quintic` 为分段五次插值，每段两端的参考速度均为零。走廊使用 `pchip` 形状保持分段三次插值，以便连续通过中间节点。 |
| `camera_pitch_keys` | 可选 `[时刻 s, 附加俯仰角 °]` 序列。采集器线性插值，相对已有相机安装姿态绕相机局部 x 轴转动，正值抬头、负值低头。本批会议室使用此项。 |
| `camera_targets` | 采集器另支持的可选写法，每行 `[时刻 s, 世界系注视点 x, y, z (m)]`。插值注视点后计算俯仰。与 `camera_pitch_keys` 二选一。两者均省略时使用固定安装姿态，如本批走廊。 |
| `subgoals[]` | 按开始时刻排列的任务阶段。`id` 是阶段名。`at_s` 是参考路线规定的开始秒数。`predicate` 是应完成的行为。可选 `landmark` 指向场景地标 ID。首项从 0 秒开始。省略时采集器生成单段 `follow_reference`。 |
| `object_observation` | 物体观察任务的核验范围。`landmark` 指定要绕行的物体锚点，`kind` 为 `arc`（一段弧）或 `full_orbit`（闭合一周），`arc_start_s` 和 `arc_end_s` 给出参考时钟上的观察段起止秒数。实际绕角根据实飞机体围绕锚点的位置计算。 |
| `inspection_landmarks` | 需抵近观察的地标 ID 列表，如 `["ac","laptop","sofa"]`，用于组织任务和审核。 |
| `search_evidence` | 搜索任务的取证索引。`target` 为被搜索地标 ID。`search_stage`、`approach_stage` 为搜索、抵近阶段的子目标 ID。目标何时进入图像另由实飞帧测得。 |
| `orbit_center_w_m`、`orbit_start_s`、`orbit_closed_at_s` | 绕行中心的世界坐标（米）、开始与计划闭环时刻（秒）。纯绕桌任务可省略开始时刻，核验从 0 秒起算。复合任务给出实际需检查的时段。 |
| `related_route_families` | 可选，列出共享前缀或部分几何的其他路线族，供分组时一并考虑。分组校验器直接检查的是 `route_family_id`。 |
| `initial_velocity_w_mps`、`source_episode_id` | 可选。前者为重置时施加的世界系初速度向量 [vx,vy,vz]（m/s），省略时从静止开始。后者为同指令来源回合 ID。这两个字段随新路线存档，初速度可从第 0 帧的 `velocity_w_mps` 独立查证。 |
| `seed`、`task_id`、`task_spec` | 采集器支持的可选字段。随机种子缺省 65，任务 ID 缺省回合 ID，附加任务规格按键合入回合的 `task_spec`。本批的停止门槛见下表。仅改路线中的容差不会更改采集循环里的停止判据。 |

## 回合元数据 `episode.json`

下表按 JSON 对象分组。同一行以对象路径开头时，后续并列字段位于同一对象内，例如 `task_spec.goal_w_m` 和 `task_spec.goal_yaw_deg`。

| 字段 | 含义与取值 |
|---|---|
| `schema_version`、`episode_id`、`scene_version` | 分别给出格式版本（当前 `rgbt_episode_v1`）、本次飞行的唯一 ID、能定位共享标定和场景资产的版本目录。 |
| `task_id`、`route_family_id`、`source_episode_id` | 任务标识、母路线分组、派生来源回合 ID。首批 15 条的来源为 `null`。追加的 8 条保留原指令和任务 ID，来源值指向同指令的原回合。不同曝光、同义指令或纠错变体可用来源字段追溯。 |
| `seed`、`instruction`、`instruction_lang`、`instruction_zh` | 重置随机种子、供语言策略读取的完整指令、其语言代码（本批为 `en`）、中文译文。训练如选用中文，应显式选取译文字段。 |
| `task_spec.ordered_subgoals` | 指令的有序子目标。元素中的 `id`、`at_s`、`predicate`、可选 `landmark` 见航线表。行为包括升高搜索、扫视、抵近观察、接近桌缘、绕桌一周和沿参考飞行。用于分阶段制标与评分。 |
| `task_spec.goal_w_m`、`goal_yaw_deg` | 终点世界坐标 `[x,y,z]`（米）和终端航向（度），由参考末节点取得。航向误差按最短有符号角计算。 |
| `task_spec.position_tol_m`、`yaw_tol_deg`、`speed_tol_mps` | 三维终点位置误差、绝对偏航误差、机体三维速率各自的上限。本批同时满足 0.12 m、8°、0.10 m/s 才能记录终端 STOP。 |
| `behavior_policy`、`teacher_version`、`controller_version` | 分别追溯谁驾驶、谁制标、谁执行底层控制。本批取 `closed_loop_reference_expert`、`continuous_reference_chunk_v1`、`fly.controller.nominal_v1`。 |
| `observation_config.rgb_ev`、`thermal_encoding` | RGB 曝光档位（本批 0 EV）和热像编码（本批 `pseudocolor_rgb8`）。热图是相对热外观的三通道伪彩。定量测温需要另外保存辐射标定与原始测量。 |
| `observation_config.rgb_wh`、`thermal_wh` | 两路各自的 `[宽, 高]`，单位像素。相同尺寸下，地标在两路可投影到不同像素。 |
| `waypoint_spec.horizon_steps`、`frame`、`units` | 预测槽数 3。`per_step_yaw_flu` 表示在每一段起点水平航向对齐的“前、左、上”坐标系表达。`m_deg` 表示平移用米、偏航用度。 |
| `waypoint_spec.sampling`、`control_dt_s` | 高层决策划段规则文本与低层固定控制周期（0.05 s）。两次高层决策的实际间隔见相应运动槽的 `duration_s`。 |
| `waypoint_spec.stop_at`、`stop_frame_id`、`terminal_hold_s` | `step_start` 表示 STOP 在该步开始生效。后两项是进入末端保持段后首次满足终端条件的实际帧号、从该帧到采集末帧的悬停记录秒数。 |
| `initial_frame_id`、`final_frame_id` | 记录的首、末状态帧号，含两端。终端悬停仍记录图像，因此末帧通常晚于 STOP。 |
| `termination_reason`、`task_success`、`human_review` | 结束原因、本次采集给出的程序性成功标记、写入时的人审状态。本批前两项为 `expert_stop`、`true`。`pending` 表示待真人签核。子目标的语义完成情况看关键帧与审核记录。 |
| `search_evidence`、`flight_metrics` | 前者复制搜索取证索引（普通回合为 `null`），后者从实际状态序列计算，字段见下表。 |
| `video`、`video_layout`、`video_fps` | 相对回合目录的视频路径、左右模态排列、编码帧率。 |

下列字段均位于 `flight_metrics` 对象中。

| 字段 | 如何从实飞记录理解 |
|---|---|
| `frames`、`decisions` | 分别为状态总帧数（含首帧）和高层决策总数（含末次 STOP）。 |
| `path_m`、`duration_s` | 相邻实飞位置的三维欧氏距离逐段求和，单位米。首末仿真时刻之差，单位秒。绕行一周回到起点也有完整航程。 |
| `z_min_m`、`z_max_m` | 实飞世界坐标 z 的最小、最大值，单位米。 |
| `peak_speed_mps`、`final_speed_mps` | 整条轨迹最大以及末帧的三维线速度模，单位 m/s。 |
| `min_safety_clearance_m` | 各控制区间已采样安全净空中的最小值，单位米，关联机体安全包络与场景碰撞几何。 |
| `final_error_m`、`final_yaw_error_deg` | 末帧距参考终点的位置误差模（米）、与终端航向的最短夹角绝对值（度）。 |
| `collision_count`、`outside_known_frames` | 碰撞计数、机体安全包络越出已观测自由空间的帧数。这批成功回合均为 0。采集器遇到这两种情况会中止该条飞行。 |

<a id="frames"></a>

## 观测帧 `frames.jsonl`

一行对应一次仿真状态回执。先执行从上一帧到本帧的控制，再在新位姿上渲染双模态图像。本帧如被某个决策引用，同时充当下一段动作开始前的观测。[走廊实飞帧文件](outputs/expert_rgbt_20261123/episodes/corridor_low_slow/frames.jsonl)

| 字段 | 含义、单位与空值 |
|---|---|
| `frame_id`、`t_s`、`dt_actual_s` | 本回合帧号、仿真时钟（秒）、距上帧实际推进秒数。重置首帧为 `0, 0.0, 0.0`。处理其他采集来源的短步时应使用实记时间。 |
| `position_w_m`、`q_w_from_b_wxyz` | 无人机中心在 Z 向上的世界系位置 `[x,y,z]`（米）。将机体系向量旋转到世界系的单位四元数，顺序 `[w,x,y,z]`。 |
| `velocity_w_mps`、`omega_b_radps` | 实际世界系线速度 `[vx,vy,vz]`（m/s）与机体系角速度 `[ωx,ωy,ωz]`（rad/s）。机体轴为前、左、上。要读机体系线速度可用姿态旋转矩阵的转置左乘世界系速度。 |
| `rgb_path`、`thermal_path` | 本时刻输入图像路径，相对本回合目录，按同一行配对。当前每一行均有两路 PNG。 |
| `commanded_body_rates_thrust` | 专家控制器送入本次仿真步的四元控制向量，前三项是机体 x/y/z 角速度指令（rad/s），末项为归一化推力。首帧无入帧命令，值为 `null`。 |
| `applied_body_rates_thrust` | 上一控制区间实际送入动力学的四元指令，与同帧的 `commanded_body_rates_thrust` 顺序、单位相同。当前 `body_rates_thrust` 接口直通角速度，将推力限制在 [0,1]。实际响应的角速度另读 `omega_b_radps`。首帧为 `null`。 |
| `collision`、`min_safety_clearance_m`、`in_known_space` | 接触标记、刚结束的控制区间内最小采样安全净空（米）、本帧旋转后的机体安全包络是否完全处在已观测自由空间。首帧没有前一区间，净空为 `null`。 |
| `rgb_w2c_native`、`thermal_w2c_native` | 各自的 4 × 4 齐次矩阵，将 3DGS **原生世界点**变换为对应 OpenCV 相机坐标，平移沿用重建单位。矩阵已包含本帧机体姿态、相机安装位姿和附加俯仰。世界系米制地标投影前须先用共享标定逆变换回原生坐标。 |
| `rig_pitch_deg` | 基础安装角上叠加的相机俯仰（度）。数值记录本帧渲染采用的角度，走廊为 0。机身姿态变化仍会改变镜头实际朝向。 |

对于异步实拍扩展，应另存两路传感器各自的原始时间戳、配对误差与有效标记。对于原生单通道热测量，应另记位深、标定、增益与处理版本。现有回合的 `t_s` 对两路同时有效。

## 导航决策 `decisions.jsonl`

| 字段 | 含义与关联对象 |
|---|---|
| `decision_id`、`observed_frame_id`、`t_s` | 本回合递增决策号、执行本步前实际观测的帧号、该帧仿真时刻。一个控制帧至多是一次决策的起点。 |
| `history_frame_ids` | 按时间升序列出此前最多两次决策用过的帧号。首个决策为空，第二次通常只有一项。读取时如需补帧，须遵守这个时间边界。 |
| `teacher.source`、`teacher.subgoal_id` | 监督来源与当前任务阶段 ID。本批来源 `known_reference_from_actual_pose`。以当前时刻已到达的最后一个参考阶段起始时刻选取阶段，标签来源可追溯至路线。 |
| `teacher.chunk` | 长度固定为 3 的有序槽数组，容纳从当前决策开始的最多三段未来教师动作，槽字段如下。 |
| `executed.source`、`executed.kind` | 实际驾驶方与执行方式。本批为 `expert`，飞行中跟踪 `reference_segment`，终端为 `stop`。 |
| `executed.reference_file`、`executed.segment_id` | 本回合参考文件 `route.json` 和本次开始时所在相邻节点区间的零基索引。某个执行时段可能跨过下一个节点，完整范围由起止时间给出。 |
| `executed.ref_start_s`、`executed.ref_end_s`、`executed.end_frame_id` | 本段执行在参考时钟上的起止秒数、执行后对应的实飞回执帧号。下一行决策从该帧开始。最后一次 STOP 关联到悬停结束后的末帧。 |

| `teacher.chunk[]` 中的字段 | 含义与使用条件 |
|---|---|
| `kind` | `move` 为移动或原地转向，`stop` 为从该步开始悬停，`pad` 为超出可标注范围。终端 STOP 之后的槽均填 pad。 |
| `delta_flu_m_yaw_deg` | move 的连续监督 `[Δ前, Δ左, Δ上, Δ偏航]`（m、m、m、°），向左转为正。stop 和 pad 均为 JSON `null`。水平限幅检验用前两项的模 ≤ 0.70 m，竖直绝对值 ≤ 0.30 m，偏航绝对值 ≤ 35°。 |
| `target_frame_id` | move 的目标时刻在本回合控制时钟上对应的帧号。该时刻的**目标位姿**取自参考轨迹。相同帧号在 `frames.jsonl` 中保存实际跟踪结果，可据此计算跟踪误差。 |
| `duration_s`、`target_reference_speed_mps` | 该段从起点到目标的秒数、在目标时刻对参考位置曲线求导所得的三维速率模（m/s）。两者仅随 move 写入。 |
| `target_rig_pitch_deg` | 可选，move 目标时刻的附加云台俯仰角（度），仅在航线提供 `camera_pitch_keys` 时写入。走廊 move 省略此键。 |

**三个槽按各段起点换系。** 第一槽从当前**实际**位置和航向到下一决策边界的**参考**位姿。第二、三槽从上一段**参考**末位姿到再下一段参考位姿。设世界系位置差为 `[dx,dy,dz]`，起点航向为 ψ，则 `Δ前 = cos(ψ)dx + sin(ψ)dy`，`Δ左 = −sin(ψ)dx + cos(ψ)dy`，`Δ上 = dz`，偏航差折回到 [−180°, 180°)。计算三角函数时将度换成弧度。叠加航点时要随每段更新原点和航向。这里仅用水平航向，机体滚转、俯仰和云台角不参与此换系。[换系与制标实现](scripts/collect_expert_episodes.py#L152)

**训练掩码由 `kind` 生成。** move 对应 STOP 标签 0，同时监督连续量。stop 对应 STOP 标签 1，关闭连续量回归。pad 同时关闭两类损失。例如 `[move,stop,pad]` 的运动掩码为 `[1,0,0]`，停止损失掩码为 `[1,1,0]`。本批起飞前已给专家完整参考，实飞由低层控制器连续跟踪。用这些样本训练策略后，闭环推理可执行预测首槽，再采样重算。

<a id="example"></a>

## 用一条真实决策构造训练样本

以 `room_inspect_laptop` 的第 3 号决策为例。语言来自 `episode.json` 的 `instruction`。历史输入为第 30、68 帧的两路图像，当前输入为第 103 帧（5.15 s）的两路图像，以及该时刻可用的机体状态（如线速度和角速度）。比如当前 RGB 路径为 `rgb/000103.png`，对应热像为 `thermal/000103.png`。[回合元数据](outputs/expert_rgbt_20261123/episodes/room_inspect_laptop/episode.json) · [实际决策记录](outputs/expert_rgbt_20261123/episodes/room_inspect_laptop/decisions.jsonl)

| 决策 3 的槽 | 教师给出的答案 | 如何使用 |
|---|---|---|
| 0 · move | 航点约为 `[0.038, −0.084, −0.005, 4.39]`，时长 0.50 s，目标为第 113 帧时刻的参考位姿 | 从第 103 帧实飞姿态出发，略向前、向右移动并左转。该段末参考速率为 0 m/s。此槽仍是 move。 |
| 1 · stop | `delta_flu_m_yaw_deg = null` | 表示在下一决策边界应开始悬停。 |
| 2 · pad | `delta_flu_m_yaw_deg = null` | 忽略所有损失。 |

实飞从第 103 帧执行至第 113 帧。下一条决策（4 号）才以第 113 帧为当前输入，监督为 `[stop,pad,pad]`，悬停至第 143 帧（7.15 s）结束。训练第 3 号样本时，输入限于第 103 帧及以前。未来帧、世界系终点、地图、地标坐标、预设子目标时刻、参考速度和完成情况留在教师或评分侧。

<a id="workflow"></a>

## 专家轨迹如何采集

本数据集的轨迹由**已知地图和地标的脚本专家**示范。专家设计路线并给出目标，飞行控制器读取实时机体状态，驱动同一台仿真无人机完成动作。OpenFly 可用这些示范学习指令到航点的映射。[采集器](scripts/collect_expert_episodes.py) · [飞行控制器](scenes/room65_selected_anchored/fly.py)

1. **选任务并设计参考航线。** 在所选场景的米制坐标中确定起点、朝向、有序子目标和终点悬停，写入中英文指令与带时间的世界系航点。会议室以桌面、笔记本、沙发、空调为地标，观察任务同时排定云台俯仰。搜索任务从目标在初始 RGB 视野外的朝向起飞，依次升高、扫转、抵近，再执行后续子目标。走廊用同一路径方向上的位置和到达时间设计快慢、升降与末端减速。路线末两点位姿相同、间隔至少 0.5 s，末点时间对齐 0.05 s 控制时钟。[会议室搜索路线](outputs/expert_rgbt_20261123/routes/room_search_full_survey.json) · [走廊变速路线](outputs/expert_rgbt_20261123/routes/corridor_climb_accelerate.json)
2. **重置并执行。** 采集器按场景版本加载 RGB、热 3DGS、米制碰撞几何、已观测自由空间和相机标定，用路线首点和种子重置。会议室按分段五次曲线插值，走廊用保形三次插值（PCHIP）维持中途运动的连续性。每 0.05 s，控制器读取上一回执的位置、姿态和速度，结合参考位置、速度、加速度和偏航，输出三个机体角速度指令与归一化推力。动力学每步积分 10 个子步，返回实飞位姿。碰撞、机体安全包络超出已观测空域或异常终止均使该次采集失败。
3. **同步记录。** 保存重置首帧和此后每次控制回执。每个 `frame_id` 对应实飞状态、刚结束的控制区间、同一仿真时刻的 RGB 与伪彩热图、两相机各自的外参。两路图像分别无损写入 PNG，同一对图像拼成左 RGB／右红外的 20 fps 第一视角视频。参考轨迹进入最终保持段后，取首个同时满足位置误差 ≤0.12 m、最短偏航误差 ≤8°、速度 ≤0.10 m/s 的实飞帧作为 STOP，继续记录到计划末帧，并再次核验终点条件。
4. **由参考路径与实飞状态生成监督。** 写完飞行记录后，采集器沿参考路径累计水平路程、升降量和绝对偏航变化。任一量达到 0.50 m、0.17 m 或 22°，或经历 2 s、遇到子任务切换或 STOP，便在该控制帧建立新的决策边界。在该边界取当前图像和至多两次先前决策的帧，计算三槽航点。第一槽从实飞位姿指向下一参考目标，后续槽连接相邻参考目标，每槽采用各自起点的水平航向。单段水平、垂直、偏航增量上限分别为 0.70 m、0.30 m、35°。这些边界用于生成监督，控制器在边界之间连续工作。[分段和制标实现](scripts/collect_expert_episodes.py)
5. **入库和验证。** 写完并完成采集器检查后，临时回合目录才重命名为 `episodes/<episode_id>/`。重跑使用新目录或新回合 ID。同一母路线的再采集、曝光变体或改写指令应归入同一分组。全库校验读取已有的 `splits.json`，核对时间、历史索引、动作和终点，解码全部 PNG 与视频，并将每一视频帧的两侧分别与对应 PNG 比对。会议室另用地标和实飞相机外参检查绕桌角度、闭合误差、抵近距离和目标进出视野的先后。走廊用实飞速度核对中段是否持续向前。[逐帧校验器](scripts/verify_expert_episodes.py) · [会议室几何核验](scripts/audit_room_expert_semantics.py) · [本批航点与走廊连续性核验](outputs/expert_rgbt_20261123/reference_verification.json)
6. **人工审核并留档。** 在审核页逐条对照指令播放视频，检查目标能否辨认、观察顺序是否正确、两模态是否同步、终点是否悬停。搜索任务重点看首帧、扫转后首次可辨认的画面和抵近画面。绕桌任务核对闭环与高度变化。地标中心投影进图只能说明其几何方向落在视野内，遮挡和目标可辨性需要回看原图。填入审核人、通过或退回及带视频秒数的意见，导出审核 JSON。页面将草稿暂存在当前浏览器，导出文件需另行归档。采集器初始写入 `human_review="pending"`，页面不会回写回合元数据。[本批审核页](outputs/expert_rgbt_20261123/review.html)

**复现一条轨迹。** 从项目根目录执行。以下命令使用已存档的笔记本抵近路线，将复采结果写入新目录，再对该回合做独立校验。单回合校验可在建立全库分组前执行。

```bash
sample_root=outputs/expert_rgbt_20261123
replay_root=outputs/expert_replay_laptop
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUDA_HOME=/usr/local/cuda-12.8 \
  PATH="/usr/local/cuda-12.8/bin:$PATH" \
  /home/zr_fan/miniconda3/envs/thermal_gaussian_5070/bin/python -B \
  scripts/collect_expert_episodes.py "$sample_root/routes/room_inspect_laptop.json" \
  --dataset "$replay_root" --resolution 0.5

/home/zr_fan/miniconda3/envs/gradnav_5070/bin/python -B \
  scripts/verify_expert_episodes.py "$replay_root" --episode room_inspect_laptop
```

**生成整批审核材料。** 新数据集先在根目录建立覆盖全部回合的 `splits.json`。如需生成会议室语义证据，另须放入对应的场景地标表，并采用带会议室子任务和地标字段的路线。以下以发布批次为例，先生成语义证据，再生成包含证据链接的审核页。全库校验和语义脚本会重写对应的派生报告。

```bash
sample_root=outputs/expert_rgbt_20261123
audit_python=/home/zr_fan/miniconda3/envs/gradnav_5070/bin/python
"$audit_python" -B scripts/audit_room_expert_semantics.py "$sample_root"
"$audit_python" -B scripts/verify_expert_episodes.py "$sample_root"
```

<a id="shared"></a>

## 场景标定、地标与数据划分

每个场景共享一份 `scenes/<scene_version>/calibration.json`。要将目标投到某一模态的图像上，先用场景变换的逆变换把米制世界坐标还原为重建原生坐标，再用该帧、该模态的 `*_w2c_native` 和内参投影。RGB 与热相机的焦距、视场和光心位置有差异，同步拍摄的两张图应分别投影。

| 标定字段 | 含义和读取方法 |
|---|---|
| `scene_version`、`wh` | 场景标识，与回合关联。本批两路共用的图像宽、高，单位为像素。 |
| `rgb_K`、`thermal_K` | 各自 3×3 针孔内参矩阵，含以像素表示的焦距 `fx, fy` 和主点 `cx, cy`。 |
| `distortion`、`convention`、`optical_axes` | 畸变处理方式、逐帧矩阵/机体/世界轴定义、安装矩阵的 OpenGL 光学轴定义。本批为无畸变针孔渲染。逐帧外参采用 OpenCV 相机轴（+x 向右、+y 向下、+z 向前）。安装矩阵使用 OpenGL 相机轴（+x 向右、+y 向上、−z 向前）。 |
| `body_from_rgb_opengl_at_zero_pitch_m` | 4×4 刚体变换，把零附加云台俯仰时的 RGB OpenGL 相机点变到机体系。平移列单位为米，安装角已经计入。 |
| `thermal_from_rgb_native`、`rig_translation_unit` | 从 RGB 相机坐标到热相机坐标的 4×4 变换及其平移单位。本批平移使用重建原生单位，可左乘 `rgb_w2c_native` 得到热相机外参。 |
| `native_to_sim`、`source_to_sim.rotation`、`source_to_sim.origin`、`source_to_sim.meters_per_source_unit` | 变换公式、3×3 旋转矩阵、原生空间中的仿真原点、每原生单位对应的米数。按列向量计算 `p_sim = s * R @ (p_native - origin)`。 |
| `source_to_sim.bounds_m`、`assumed_ceiling_height_m`、`scale_status` | 该变换块内的世界轴对齐边界 `[xyz_min, xyz_max]`、用于定标的假设层高和尺度来源。本批层高取 2.80 m，净空与距离均按此尺度计算。自由空间还需查询对应几何。 |
| `source_to_sim.floor_src`、`ceiling_src` | 会议室地面、天花板沿竖直向上轴在原生坐标中的投影值。 |
| `source_to_sim.floor_ceiling_source_offsets`、`wall_source_offsets`、`wall_source_normal` | 走廊地面/天花板在向下法向上的偏移、右/左侧墙沿墙法向的偏移（原生单位）、侧墙单位法向。 |
| `source_to_sim.plane_fit_median_abs_error_src`、`frame_plane_sample_counts`、`source` | 走廊的水平面组/侧墙组拟合中位绝对误差（原生单位），地/顶/右/左四组样本数，以及拟合所用深度与方法说明。 |
| `per_frame_extrinsics` | 指明云台运动已纳入每帧图像的外参。 |
| `assets.<资产键>.path`、`sha256` | 资产的可定位源路径与内容摘要。键 `models/rgb.ply`、`models/thermal.ply` 指两路 3DGS。`collision/rgb_surface_m.ply` 和 `collision/free_volume.npz` 指米制碰撞表面和已观测自由空间。`inputs/manifest.json`、`inputs/frame.json`、`inputs/rig.json` 指场景视图清单、坐标标定、双相机标定。迁移路径后可用摘要核对版本。 |

会议室的 `scenes/<scene_version>/landmarks.json` 以地标 ID 为键。本批有 `conference_table`（会议桌）、`laptop`（笔记本）、`sofa`（沙发）、`ac`（壁挂空调）。每项的 `center_w_m` 是选定锚点的仿真世界坐标 `[x,y,z]`，`description` 指明物体及取点部位，`evidence` 记录取点来源，如多视角三角化、参照帧或表面探测。围绕物体的距离和视场检验均针对该锚点。走廊本批使用连续飞行指令，未提供单独的对象地标表。

`splits.json` 的 `scope` 描述划分适用范围，`related_routes` 说明哪些相关路线一起分组，`splits.train`、`splits.val`、`splits.test` 分别列出各自的回合 ID。各组的当前数量见[数据总表](outputs/expert_rgbt_20261123/README.md)，两处物理场景均出现在训练集中。评测未见场景时，应按物理场所重新划分，并将同一母路线的曝光版、扰动版、指令改写及纠错派生样本保持在同一组。

## 数据集级校验与人工审核文件

审核用的 JPEG 是派生图像。`review_media/<episode_id>.jpg` 并排显示各时刻的 RGB 和热图，为审核页提供视频封面。`evidence/<episode_id>.jpg` 在每个取证时刻上排放 RGB，下排放热图，空心环标出地标中心投影。`evidence/<episode_id>.json` 保存该图的数值依据。`evidence/assistant_frame_notes.json` 以回合 ID 为键保存助手抽查关键帧的文字。[会议室证据汇总](outputs/expert_rgbt_20261123/room_evidence.json) · [搜索笔记本的关键帧](outputs/expert_rgbt_20261123/evidence/room_search_laptop_sofa.jpg)

### 自动校验报告的字段

`verification.json` 中 `dataset` 是数据批次名，`verified_at` 是带时区的校验时间，`passed` 是整体结果，`episode_count` 是发现的回合数。`decoded_video_frames`、`decoded_lossless_images` 分别汇总成功核查记录中的视频帧和单路 PNG 张数，`methods` 列出检查方法，`episodes` 存放成功逐条核查的记录，`errors` 用回合 ID 或 `dataset` 作键存放错误信息。

| `episodes[]` 字段 | 含义 |
|---|---|
| `episode_id`、`scene_version`、`route_family_id`、`instruction`、`instruction_zh` | 回合、场景和母路线标识，以及指令原文和中文说明，对应回合元数据。 |
| `decoded_frames`、`decoded_pngs`、`fps` | 解码的视频帧数、两路独立 PNG 总数、视频帧率。 |
| `decisions`、`move_decisions` | 决策总数，以及第一教师槽为移动的决策数。 |
| `seconds`、`length_m`、`horizontal_net_m`、`altitude_range_m` | 首尾仿真时差、相邻实飞位置三维距离之和、首尾位置的水平直线距离、世界 Z 的最小/最大值。 |
| `mean_speed_mps`、`peak_speed_mps`、`speed_p50_p90_mps`、`final_speed_mps` | 总航程除以时长、逐帧实际速率峰值、实际速率的第 50/90 百分位、末帧速率。 |
| `position_error_m`、`yaw_error_deg` | 末帧到任务目标的三维距离、与目标偏航的最小绝对角差。若未指定目标偏航，角误差为 `null`。 |
| `minimum_sampled_safety_clearance_m`、`collision_frames`、`unknown_space_frames` | 全回合最小采样安全净空、报告碰撞的帧数、报告机体超出已观测空域的帧数。 |
| `max_video_png_mae_u8`、`video`、`poster`、`passed` | 分别对 RGB/热图计算视频对应半幅与无损 PNG 的逐帧平均绝对像素误差，再各取全片最大值（8 位尺度）。`video` 和 `poster` 是相对于数据集根的 MP4 和预览图路径，`passed` 是该回合的自动检查结果。 |

`reference_verification.json` 以回合 ID 为键。`reconstruction_error_m` 是逐段把教师的 FLU 增量还原成世界坐标后，到对应参考航点的最大三维位置误差。走廊记录另含 `core_min_speed_mps`、`core_median_speed_mps`、`core_max_speed_mps`，为累计实飞航程 10% 至 90% 区间的最小、中位和最大速率。`core_min_forward_speed_mps` 是该区间沿飞行方向的最小纵向速度（世界 X 速度按全程行进方向取符号）。

`variant_verification.json` 汇总同指令各回合的 `start`（第 0 帧）、`entry`（首次进入共同距离范围或穿过同一走廊截面的实飞帧）及 `pairwise`（组内两两差值）。比较保持同一场景、原指令、任务 ID、路线族和数据分组。起点和进场位置均为世界系米制坐标，速度向量单位为 m/s，航向单位为度。可在[审核页的进场对比](outputs/expert_rgbt_20261123/review.html#variants)直接播放。

### 会议室证据与人工审核字段

对新增物体，`scenes/room65_selected_anchored/object_landmarks.json` 补充标定锚点和原始 RGB 多视角标注。`object_observations.json` 汇总各条轨迹的接近距离、实飞绕角、RGB/热图投影入画比例和连续稳定悬停时长。`signed_sweep_deg` 带运动方向，`closure_m` 是观察段两端的三维距离，`stable_hover_s` 取 STOP 之后直至回合结束持续满足位置、航向和速率门限的末段时长。同指令的两次飞行按各自入弧帧比较。

`room_evidence.json` 的 `method` 描述取证方法，`panels` 说明拼图排列与标记，`human_review` 是生成报告时的人工审核状态，`episodes[]` 内的各项与单条证据 JSON 对应。每条含 `episode_id`、`evidence_frames[]`（按帧号排列的取证画面）、`panel`（关键帧图相对数据集根的路径）、`inspection[]`（无抵近子目标时为空）。`orbit`、`search` 按任务出现，`assistant_keyframe_observations` 若存在则保存助手抽看的文字。

| 证据块 | 字段含义 |
|---|---|
| `evidence_frames[]`、`inspection[].end_of_stage_view` | `role` 为该帧的取证用途，`landmark` 为地标 ID，`frame_id`、`t_s`、`body_height_m` 为实际帧号、仿真秒数和机体世界高度。`rgb`、`thermal` 各含 `center_in_image`（点在相机前方且其中心投影在画幅内）、`optical_depth_m`（沿该相机光轴的有符号米制深度）、`uv_px`（原图的横/纵像素坐标，非正深度时为 `null`）。画面内的点仍可能受遮挡，目标可辨性需看原图。 |
| `orbit` | `start_frame`、`end_frame` 为绕桌段起止帧。`signed_azimuth_change_deg` 是机体位置相对桌中心的累计有符号方位角变化。`closure_error_m` 是起终点三维距离。`height_range_m` 为绕桌段最低/最高世界 Z。`complete` 表示绝对转角达到 350° 且闭合误差不超过 0.15 m。变高度回合另有 `higher_then_lower`，要求最高点先于最低点，且最高/最低相对起始高度的变化均超过 0.12 m。 |
| `search` | `landmark` 为目标，`start_frame`、`scan_start_frame`、`approach_start_frame` 是回合开始、旋转扫描开始和抵近开始帧。`center_initially_outside`、`initial_optical_depth_m` 是首帧 RGB 的地标中心出画判定和光轴深度。`climb_before_scan_m`、`climb_to_reacquisition_m` 为从起点到扫描开始、到几何确认帧的高度增量。`scan_yaw_range_deg` 是扫描中展开后的机体偏航角极差。`first_projected_frame`、`first_projected_t_s` 是截至抵近开始第一次投进 RGB 的帧号与时间。`search_confirm_frame` 是扫描期间中心投影最接近画面中心的帧。无对应帧时为 `null`。`projection_sequence_pass` 表示首帧中心在画外且扫描期间重新入画。 |
| `inspection[]` | `landmark`、`subgoal_id`、`start_frame`、`end_frame` 为地标、抵近子目标和阶段起止帧。`range_at_episode_start_m`、`range_at_stage_start_m`、`range_at_inspection_m`、`range_at_stage_end_m` 分别是回合起点、阶段起点、所选观察帧、阶段终点的机体到地标三维距离。`closest_central_frame` 在 RGB 中央区域（横纵各去掉两边 10%）内选距离最近的帧，缺失时帧号和距离为 `null`。`frames_center_in_rgb` 为中心投影在 RGB 内的帧数，`frames_observed` 是含首尾的阶段帧数。 |

人工审核表单的草稿暂存在当前浏览器。点击导出后，浏览器下载 `expert_episodes_human_review.json`，其中包含 `dataset`（批次名）、`reviewer`（填报人）、`exported_at`（导出时刻）和按回合 ID 索引的 `episodes`。每条结果含 `status`（`pending` 待审核、`accepted` 通过、`rejected` 退回）、`notes`（意见，可附视频秒数）和 `updated_at`（最近一次修改该回合表单的时间，未修改为 `null`）。浏览器导出不自动回写各回合元数据。当前批次的人工结论应以审核人实际导出的记录为准。

### 采集过程材料

`work_room/design.py` 是本批会议室路线设计脚本，`preview_room.py`、`refsheet.jpg` 用于预览，`work_room/landmarks.json` 是设计时的地标表，对外读取以场景目录中的版本为准。`run_logs/*.log` 和 `work_corridor/capture.log` 是运行文本日志。`work_corridor/preflight.json` 是早期走廊动力学与几何预检的记录数组，各项含义如下。

| 预检字段 | 含义 |
|---|---|
| `episode_id`、`check`、`pass_checks` | 试飞 ID、预检内容说明、该次预检的判定。 |
| `frames`、`unknown_frames` | 预检帧数、越出已观测空域的帧数。 |
| `min_safety_clearance_m`、`final_error_m` | 预检的最小采样净空与末帧到参考终点的位置误差，单位米。 |
| `speed_max_mps`、`speed_min_at_10_to_90_percent_path_mps` | 预检峰值速率、预检航程 10% 至 90% 内的最低速率，单位 m/s。正式复飞后的连续性读 `reference_verification.json`。 |

<a id="extensions"></a>

## 策略驾驶、DAgger 与早期回合

沿用 v1 时，应把实际驾驶方和教师分开记录。策略驾驶可将 `executed.source` 记为 `learner`，在 `executed.kind=move` 时以 `executed.delta_flu_m_yaw_deg` 写入实际发出的高层命令，`end_frame_id` 仍对应实飞回执。若被专家接管，可另记 `policy_proposal`（策略原提议）和接管原因。DAgger 的 `teacher.chunk` 需从每次偏离后的实际状态、按尚未完成的指令重新规划。教师无法可靠延伸时可用 pad 并记录原因，碰撞或越界时应按实际终止原因处理。当前成功专家回合校验器要求每个首槽有标签，导入带全 pad 或失败的纠错数据时须使用相应的校验规则。

早期方案的 `waypoint_spec.move_s` 和 `settle_s` 表示固定运动与稳定阶段时长。本批连续飞行以每个 move 的 `duration_s` 表示实际划段时长。侧向经过地标等任务可在子目标中增加 `side_of_body`（通过时地标位于机体哪一侧）。需要热属性的任务可记录相对冷/热条件及场景变体。旧格式允许仅记录状态的时刻将两路图像路径留空，所有被决策及历史引用的输入帧均需能解析出配对图像。要复查目标何时入画，应像本批一样保存全部帧。

<details>
<summary>查看原 v1 中固定时长、策略驾驶和教师纠错的 JSON 示例（示例数值）</summary>

以下是可扩展的记录写法。回合类型、场景 ID、控制时长和标签来源均为示例，已采集批次的真实值见上方字段表。

回合元数据和侧向通过任务。

```json
{
  "schema_version": "rgbt_episode_v1",
  "episode_id": "corridor_bin_017",
  "scene_version": "corridor_native_example_v1",
  "task_id": "pass_board_stop_before_bin",
  "route_family_id": "corridor_board_bin_east",
  "source_episode_id": null,
  "seed": 17,
  "instruction": "Pass the display board on your right, then stop before the white bin.",
  "instruction_lang": "en",
  "task_spec": {
    "ordered_subgoals": [
      {"id": "pass_board", "landmark": "board_02", "predicate": "pass", "side_of_body": "right"},
      {"id": "stop_before_bin", "landmark": "bin_01", "predicate": "stop_before"}
    ],
    "goal_w_m": [-23.13, 0.0, 1.42], "goal_yaw_deg": 0.0,
    "position_tol_m": 0.30, "yaw_tol_deg": 20, "speed_tol_mps": 0.10
  },
  "behavior_policy": "learner_example_v1", "teacher_version": "planner_example_v1",
  "controller_version": "controller_example_v1",
  "observation_config": {"rgb_ev": 0, "thermal_encoding": "pseudocolor_rgb8", "rgb_wh": [320,240], "thermal_wh": [320,240]},
  "waypoint_spec": {"horizon_steps": 3, "frame": "per_step_yaw_flu", "units": "m_deg", "move_s": 1.5, "settle_s": 0.5, "stop_at": "step_start"},
  "initial_frame_id": 0, "final_frame_id": 350, "termination_reason": "policy_stop", "task_success": true
}
```

入帧状态记录。

```json
{"frame_id":280,"t_s":14.0,"dt_actual_s":0.05,"position_w_m":[-23.39,0.03,1.41],"q_w_from_b_wxyz":[1,0,0,0],"velocity_w_mps":[0.02,0,0],"omega_b_radps":[0,0,0],"rgb_path":"rgb/000280.png","thermal_path":"thermal/000280.png","applied_body_rates_thrust":[0,0,0,0.415],"collision":false,"min_safety_clearance_m":0.54,"in_known_space":true}
```

教师重新规划与策略实际执行记录。

```json
{
  "decision_id": 8,
  "observed_frame_id": 280,
  "t_s": 14.0,
  "history_frame_ids": [200, 240],
  "teacher": {
    "source": "replan_from_observed_state",
    "subgoal_id": "stop_before_bin",
    "chunk": [
      {"kind": "move", "delta_flu_m_yaw_deg": [0.26, -0.03, 0.01, 4.0]},
      {"kind": "stop", "delta_flu_m_yaw_deg": null},
      {"kind": "pad",  "delta_flu_m_yaw_deg": null}
    ]
  },
  "executed": {
    "source": "learner",
    "kind": "move",
    "delta_flu_m_yaw_deg": [0.20, -0.07, 0.00, 6.0],
    "end_frame_id": 320
  }
}
```

</details>

## 兼容既有飞行记录

- 会议室专家日志用 `frame`，走廊专家日志用 `frame_id`，部分策略日志使用 `observed_frame` 关联决策。转换时以回合内原始 ID 和仿真时间交叉核对。[会议室专家记录](scenes/room65_selected_anchored/flight/frames.jsonl) · [走廊决策记录代码](openfly_rgb/corridor.py#L198)
- 走廊脚本的 `reference_waypoints.csv` 是预设路线，脚本专家的 `waypoints.csv` 是稀疏实飞位姿，OpenFly 回合里的同名文件是模型发出的世界坐标目标。按来源解释三种数据，并用统一的教师规则生成标签。[专家目录](outputs/corridor_native_flights/board_approach_final_map) · [策略记录](outputs/corridor_openfly_control_ev0/waypoints.csv)
- 旧回合仅有拼接视频，或仅有逐决策 RGB PNG 和首张热 PNG 时，可按原场景、标定、曝光和实飞位姿重渲染无损输入，和已有无损帧核对，并注明图像来源与渲染版本。MP4、轨迹 CSV、规划图及缓存的 OpenFly 特征可用作派生文件。[现有输入保存代码](openfly_rgb/corridor.py#L215)

新建标签版本时，可从一条终端样本与一条偏离示范的样本分别试读训练张量，检查停止掩码、机体系的航点方向，以及读取的历史帧是否全部早于当前决策。对相机外参或几何处理有改动时，可选已知地标做双路重投影，并按版本核对资产摘要。
