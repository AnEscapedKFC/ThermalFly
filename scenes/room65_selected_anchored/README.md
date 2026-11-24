# 会议室 6＋5 双模态飞行场景

当前 RGB 使用 room65 的 PlanarGS 官方流程第 30,000 步，红外使用 `room_vid65_selected_anchored_20261119_081202_4099` 第 24,000 步。两者共享原相机支架标定。碰撞表面和自由空间网格来自锚定版 RGB 24k。原 RGB 文件保存在 `models/rgb_anchored_24000.ply`，各模型的来源与校验值见 `inputs/source.json`。`flight/` 和 `runs/20261119_*/` 保存切换前的飞行记录。

## 观看与复现

此前的动作可直接在当前场景的 7890 回放台执行，7891 双模态查看器用于自由观察。[无人机回放台与启动命令](../../README.md#启动会议室-rgb-t-无人机第一视角查看器) · [双模态查看器与启动命令](../../../../ThermalGaussian_Surface/README.md#room65-双模态场景查看器)

切换 RGB 后复飞了桌面、墙面、地板、沙发左侧和沙发右侧五项碰撞任务。五次首次碰撞的帧号、时间和接触点均与原记录一致。每项任务的子目录内保存了新录像、逐帧动作与位姿、航点和对比结果。[本次复飞目录](../../outputs/room65_planargs_contacts_20261120_171620/) · [五项碰撞对比](../../outputs/room65_planargs_contacts_20261120_171620/comparison.json)

`flight/rgb_thermal_fpv.mp4` 左侧为 RGB，右侧为同一步的红外伪彩。两路原始单独编码的视频在同一目录，`frames.jsonl` 提供每帧动作、机体状态与两路 W2C，`trajectory.csv` 提供飞行轨迹和净空。`route.json` 保存模拟驾驶使用的语言指令与航点。

以下命令根据 `route.json` 重新执行示范巡航，写入新的输出目录。默认时间轴为 20 Hz，动力学积分为 200 Hz，全程开启扫掠碰撞检测。

```bash
cd /home/zr_fan/3DGS_env/gs_gradnav_simulator/scenes/room65_selected_anchored
conda activate thermal_gaussian_5070
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python -B fly.py --output flight_repeat
python -B inspect_flight.py --output flight_repeat
```

## 坐标和碰撞

- `models/rgb.ply` 与 `models/thermal.ply` 保留训练的原坐标、球谐和高斯属性。机身的刚性第一视角通过模拟器的 `fpv_camera` 得到 RGB 相机，先转回重建坐标，再使用 `inputs/rig.json` 中的 `W_thermal = T_rgb_to_thermal @ W_rgb`。各传感器使用各自内参。RGB 为原生渲染，红外为观测伪彩。
- `inputs/frame.json` 给出从重建坐标到模拟器 Z 向上的变换。`p_sim = meters_per_source_unit * rotation @ (p_source - origin)`。地面参考取训练深度中低于摄影轨迹的主高度峰，顶面取经多视角验证的平面。层高假定为 2.80 m，缺少实测标尺。
- `collision/rgb_surface_m.ply` 是模拟器米制坐标中的普通表面点云，直接以 `up_axis=z, scale=1` 加载。4 cm 体素内融合 RGB 渲染深度与 SfM 对齐的预测深度，使用透明度、深度离散度、至少三个相隔 22 cm 的观测机位和自由空间反证过滤，加入经多视角认证的墙与顶面局部。这里的空间尺度均为上述假定尺度。
- `collision/free_volume.npz` 在 10 cm 网格中记录经至少三个分离机位支持的空域。巡航录制脚本用它检查机身包络。动作回放通过 `collision/rgb_surface_m.ply` 对旋转机体及其 5 cm 安全余量做连续扫掠，并在首次接触时停止。未覆盖区域仍属未知，细线和局部缺测表面可能遗漏。
- 查看 `checks/scene.json` 中的墙面、天花板和桌面扫掠测试，及 `flight/summary.json` 中穿过被排除高斯中心的负例。`flight/alignment_check.json` 记录每帧相机支架一致性和终点空调在两个传感器中的投影位置。

动作回放使用随包的 `collision/rgb_surface_m.ply`。`prepare.py` 会基于当前 `models/rgb.ply` 重写碰撞图、自由空间网格和坐标范围。如需比较新图，请在独立场景副本内运行 `python -B prepare.py` 和 `python -B verify_scene.py`。训练导航策略可复用 `room.RigScene` 及飞行回执。
