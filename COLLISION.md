# 保留的旋转长方体碰撞

pointcloud.py 从现有模拟器原样复制，碰撞几何没有修改。

- 机体以动力学位置为中心，随完整四元数姿态旋转。
- 默认全长/宽/高 0.60×0.60×0.12 m，六面安全余量各 0.05 m。
- Open3D 读取普通表面点云；KDTree 外接球只筛选候选点，最终使用长方体判定。
- 每个 0.005 s 子步检查平移和旋转扫掠；旋转区间自适应细分至各轴额外外扩不超过微米级。
- 首次接触前退至安全位置，清零线速度和角速度，锁定碰撞事件。
  在新按步 API 中表现为 terminated=true，后续动作返回 409，需要显式 reset。
- 不模拟冲量、摩擦或反弹，不是精确机体网格碰撞。

clearance 为点云到机体表面的最小有符号距离，safety_clearance 为到外扩长方体的距离。
center_distance 为中心至最近点距离，可能对应不同采样点。
停止在安全位置时 in_collision 可以为 false，必须检查 collision.event。

点云与高斯 PLY 应在同一源模型坐标系，统一应用朝上轴和米制尺度。
没有普通点云时，沿用多视角深度反投影、透明背景过滤、多视角一致性与 Open3D 降采样。
不直接用高斯中心作表面。点云孔洞仍可能漏检，漂浮点仍可能误检。

```bash
bash run.sh --scene /path/scene.ply --point-cloud /path/obstacles.ply \
  --drone-size 0.60 0.60 0.12 --collision-margin 0.05
```

--no-collision 仅为显式调试选项；默认自动飞行和视频录制均启用碰撞。
