"""Audit a local RGB-T expert dataset and generate an offline human-review page."""
import argparse
import json
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image, ImageDraw


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path):
    with path.open(encoding="utf-8") as src:
        return [json.loads(line) for line in src if line.strip()]


def atomic_write(path, text):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as tmp:
        tmp.write(text)
    os.replace(tmp.name, path)


def vector(value, length):
    array = np.asarray(value, dtype=float)
    require(array.shape == (length,) and np.isfinite(array).all(), f"无效向量 {value}")
    return array


def has_keys(record, names):
    missing = set(names.split()) - set(record)
    require(not missing, f"缺少字段 {sorted(missing)}")


def yaw_deg(q):
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)))


def check_decisions(decisions, frame_by_id, episode):
    require(decisions, "无决策记录")
    frameids = list(frame_by_id)
    last_id = None
    expected_start = frameids[0]
    stop_decision = None
    move_count = 0
    subgoals = {m["id"] for m in episode["task_spec"]["ordered_subgoals"]}
    for row in decisions:
        has_keys(row, "decision_id observed_frame_id history_frame_ids t_s teacher executed")
        fid = row["observed_frame_id"]
        require(row["decision_id"] != last_id and (last_id is None or row["decision_id"] > last_id), "决策 ID 非单调")
        last_id = row["decision_id"]
        require(fid == expected_start and fid in frame_by_id, f"决策起点不连续，frame {fid}")
        require(abs(frame_by_id[fid]["t_s"] - row["t_s"]) < 1e-6, f"决策时钟不一致，frame {fid}")
        hist = row["history_frame_ids"]
        require(hist == sorted(set(hist)) and all(x < fid and x in frame_by_id for x in hist),
                f"历史观测包含未来或未知帧，frame {fid}")
        teacher, act = row["teacher"], row["executed"]
        has_keys(teacher, "source subgoal_id chunk")
        has_keys(act, "source kind end_frame_id")
        require(teacher["source"] and act["source"], "未记录教师或驾驶方来源")
        if subgoals:
            require(teacher["subgoal_id"] in subgoals, f"未知子目标 {teacher['subgoal_id']}")
        chunk = teacher["chunk"]
        require(len(chunk) == 3, "航点监督槽数不是 3")
        padding = False
        for slot in chunk:
            has_keys(slot, "kind delta_flu_m_yaw_deg")
            kind, delta = slot["kind"], slot["delta_flu_m_yaw_deg"]
            require(kind in ("move", "stop", "pad"), f"未知教师动作 {kind}")
            if padding:
                require(kind == "pad", "STOP 或 PAD 后出现有效动作")
            if kind == "move":
                x, y, z, yaw = vector(delta, 4)
                require(math.hypot(x, y) <= .70001 and abs(z) <= .30001 and abs(yaw) <= 35.0001,
                        f"航点超出物理范围，frame {fid}, {delta}")
            else:
                require(delta is None, "STOP 或 PAD 带有运动标签")
                padding = True
        require(chunk[0]["kind"] != "pad", "成功专家回合含未标注决策")
        require(act["end_frame_id"] >= fid and act["end_frame_id"] in frame_by_id, f"执行结束帧无效，frame {fid}")
        if chunk[0]["kind"] == "stop":
            require(act["kind"] == "stop" and stop_decision is None, f"停止决策执行不一致，frame {fid}")
            stop_decision = row["decision_id"]
        else:
            require(stop_decision is None, "停止后仍有动作")
            require(act["kind"] in ("move", "reference_segment"), f"未记录执行动作，frame {fid}")
            if act["kind"] == "move":
                vector(act["delta_flu_m_yaw_deg"], 4)
            move_count += 1
        expected_start = act["end_frame_id"]
    require(expected_start == frameids[-1], "回合末帧未关联执行记录")
    require(stop_decision == decisions[-1]["decision_id"], "回合末次决策没有专家 STOP")
    return move_count


def check_episode(root, folder, media_dir):
    meta = read_json(folder / "episode.json")
    has_keys(meta, "episode_id schema_version scene_version route_family_id task_id source_episode_id seed instruction instruction_lang "
                   "task_spec initial_frame_id final_frame_id behavior_policy teacher_version controller_version "
                   "termination_reason task_success observation_config waypoint_spec")
    require(meta["episode_id"] == folder.name and meta["instruction"].strip(), "目录身份或指令无效")
    require(meta["task_success"] is True, "入库回合任务未成功")
    has_keys(meta["waypoint_spec"], "horizon_steps frame units stop_at")
    require(meta["waypoint_spec"]["horizon_steps"] == 3 and meta["waypoint_spec"]["frame"] == "per_step_yaw_flu",
            "教师坐标或三步约定不符")
    require(meta["waypoint_spec"]["stop_at"] == "step_start", "停止时序定义不符")
    obs = meta["observation_config"]
    has_keys(obs, "rgb_wh thermal_wh rgb_ev thermal_encoding")
    ws, hs = zip(obs["rgb_wh"], obs["thermal_wh"])
    require(hs[0] == hs[1], "并排视频高度不一致")
    frames = read_lines(folder / "frames.jsonl")
    require(frames, "缺少飞行帧")
    ids = [f["frame_id"] for f in frames]
    require(ids == list(range(ids[0], ids[-1] + 1)), "仿真状态有缺帧或重复 ID")
    require(ids[0] == meta["initial_frame_id"] and ids[-1] == meta["final_frame_id"], "起止帧与回合元数据不符")
    frame_by_id = dict(zip(ids, frames))
    decisions = read_lines(folder / "decisions.jsonl")
    moves = check_decisions(decisions, frame_by_id, meta)
    video = cv2.VideoCapture(str(folder / "rgb_thermal_fpv.mp4"), cv2.CAP_FFMPEG,
                             [cv2.CAP_PROP_N_THREADS, 1])
    require(video.isOpened(), "无法打开双模态视频")
    fps = video.get(cv2.CAP_PROP_FPS)
    last_t = None
    positions, speeds, clearances = [], [], []
    is_contact = outside = 0
    max_pair_mae = np.zeros(2)
    paths_seen = [set(), set()]
    preview_indices = set(np.linspace(0, len(frames)-1, min(4, len(frames))).round().astype(int))
    preview = []
    try:
        for i, f in enumerate(frames):
            has_keys(f, "frame_id t_s dt_actual_s position_w_m q_w_from_b_wxyz velocity_w_mps omega_b_radps "
                        "rgb_path thermal_path applied_body_rates_thrust collision min_safety_clearance_m in_known_space")
            where = f"frame {f['frame_id']}"
            t = f["t_s"]
            dt = f["dt_actual_s"]
            require(math.isfinite(t) and math.isfinite(dt), f"时间无效，{where}")
            if last_t is None:
                require(abs(dt) < 1e-7, "首帧 dt 非零")
            else:
                require(t > last_t and abs(t - last_t - dt) < 1e-6, f"仿真时间或步长错误，{where}")
            last_t = t
            positions.append(vector(f["position_w_m"], 3))
            speeds.append(np.linalg.norm(vector(f["velocity_w_mps"], 3)))
            require(abs(np.linalg.norm(vector(f["q_w_from_b_wxyz"], 4)) - 1) < 1e-4, f"姿态未归一化，{where}")
            vector(f["omega_b_radps"], 3)
            if i or f["applied_body_rates_thrust"] is not None:
                vector(f["applied_body_rates_thrust"], 4)
            require(type(f["collision"]) is bool and type(f["in_known_space"]) is bool, f"缺失显式碰撞状态，{where}")
            is_contact += int(f["collision"])
            outside += int(not f["in_known_space"])
            if i or f["min_safety_clearance_m"] is not None:
                clearances.append(float(f["min_safety_clearance_m"]))
                require(math.isfinite(clearances[-1]), f"无效净空，{where}")
            ok, rendered = video.read()
            require(ok and rendered.shape[:2] == (hs[0], sum(ws)), f"视频帧缺失或尺寸错误，{where}")
            tiles = []
            offset = 0
            for j, (spec, w) in enumerate(zip(("rgb", "thermal"), ws)):
                relative = f[spec + "_path"]
                require(isinstance(relative, str) and Path(relative).parent == Path(spec)
                        and Path(relative).suffix.lower() == ".png", f"无效 {spec} 路径，{where}")
                require(relative not in paths_seen[j], f"图像路径重复，{where}")
                paths_seen[j].add(relative)
                pixels = cv2.imread(str(folder / relative), cv2.IMREAD_COLOR)
                require(pixels is not None and pixels.shape[:2] == (hs[j], w), f"图像缺失或尺寸不符，{relative}")
                half = rendered[:, offset:offset+w]
                offset += w
                # Each lossless eye must match its H.264 view in the same temporal slot.
                mae = np.mean(cv2.mean(cv2.absdiff(half, pixels))[:3])
                max_pair_mae[j] = max(max_pair_mae[j], mae)
                require(mae <= 22.0, f"视频和无损 {spec} 图像内容偏差大，{where}: {mae:.1f}/255")
                if i in preview_indices:
                    tiles.append(pixels)
            if tiles:
                preview.append((f["t_s"], cv2.cvtColor(np.hstack(tiles), cv2.COLOR_BGR2RGB)))
        require(not video.read()[0], "视频比状态序列长")
    finally:
        video.release()
    for j, subdir in enumerate(("rgb", "thermal")):
        require({str(p.relative_to(folder)) for p in (folder/subdir).glob("*.png")} == paths_seen[j],
                f"{subdir} 文件与帧引用集合不一致")
    duration = frames[-1]["t_s"] - frames[0]["t_s"]
    require(abs((len(frames)-1) / fps - duration) < 1e-3, "视频速率与仿真时钟不符")
    require(clearances and min(clearances) > 0 and not is_contact and not outside, "回合有碰撞、非正净空或越界")
    pos = np.stack(positions)
    speed = np.array(speeds)
    dist = np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()
    task = meta["task_spec"]
    has_keys(task, "ordered_subgoals goal_w_m goal_yaw_deg position_tol_m yaw_tol_deg speed_tol_mps")
    goal_error = np.linalg.norm(pos[-1] - vector(task["goal_w_m"], 3))
    goal_yaw = task["goal_yaw_deg"]
    yaw_error = None if goal_yaw is None else abs((goal_yaw - yaw_deg(frames[-1]["q_w_from_b_wxyz"]) + 180) % 360 - 180)
    require(goal_error <= task["position_tol_m"] and speed[-1] <= task["speed_tol_mps"], "终点距离或悬停速度超差")
    if yaw_error is not None:
        require(yaw_error <= task["yaw_tol_deg"], "终点机头朝向超差")
    media_dir.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (480 * 2, 208 * 2), "#f3f5f5")
    draw = ImageDraw.Draw(sheet)
    for j, (t, pix) in enumerate(preview):
        tile = Image.fromarray(pix)
        tile.thumbnail((476, 181), Image.Resampling.LANCZOS)
        x, y = (j % 2)*480, (j // 2)*208
        sheet.paste(tile, (x + (480 - tile.width)//2, y+2))
        draw.text((x+12, y+184), f"{t:.2f} s     RGB | Thermal", fill="#253838")
    poster = media_dir / (folder.name + ".jpg")
    sheet.save(poster, quality=88)
    return dict(episode_id=folder.name, scene_version=meta["scene_version"], route_family_id=meta["route_family_id"],
                instruction=meta["instruction"], instruction_zh=meta.get("instruction_zh", ""),
                decoded_frames=len(frames), decoded_pngs=2*len(frames), fps=fps, decisions=len(decisions),
                move_decisions=moves, seconds=round(duration, 3), length_m=round(float(dist), 3),
                horizontal_net_m=round(float(np.linalg.norm((pos[-1]-pos[0])[:2])),3),
                altitude_range_m=[round(float(z),3) for z in (pos[:,2].min(),pos[:,2].max())],
                mean_speed_mps=round(float(dist/duration),3), peak_speed_mps=round(float(speed.max()),3),
                speed_p50_p90_mps=[round(float(x),3) for x in np.percentile(speed,[50,90])],
                final_speed_mps=round(float(speed[-1]),5), position_error_m=round(float(goal_error),4),
                yaw_error_deg=None if yaw_error is None else round(float(yaw_error),3),
                minimum_sampled_safety_clearance_m=round(min(clearances),4), collision_frames=is_contact,
                unknown_space_frames=outside, max_video_png_mae_u8=max_pair_mae.round(2).tolist(),
                video="episodes/" + folder.name + "/rgb_thermal_fpv.mp4",
                poster=poster.relative_to(root).as_posix(), passed=True)


def check_splits(root, rows):
    splitfile = read_json(root / "splits.json")
    groups = splitfile.get("splits", splitfile)
    owners = {}
    families = {}
    index = {r["episode_id"]: r for r in rows}
    for part in ("train", "val", "test"):
        for episode_id in groups[part]:
            require(episode_id in index, f"分组出现未知回合 {episode_id}")
            require(episode_id not in owners, f"回合重复分组 {episode_id}")
            family = index[episode_id]["route_family_id"]
            require(families.get(family, part) == part, f"母路线跨分组 {family}")
            owners[episode_id] = families[family] = part
    require(set(index) == set(owners), "训练/验证/测试分组未包含所有回合")
    return owners


def reviewer_link(root, relative, name):
    return (f'<a href="{escape(quote(relative, safe="/"))}">{escape(name)}</a>'
            if (root / relative).is_file() else "")


def variant_comparison(root):
    report = root / "variant_verification.json"
    if not report.is_file():
        return ""
    tables = []
    for group in read_json(report)["groups"]:
        gate = group["entry_gate"]
        if gate["kind"] == "landmark_range":
            name = {"laptop": "笔记本屏幕", "sofa": "沙发座面"}.get(gate["landmark"], gate["landmark"])
            gate_text = f"首次进入距{escape(name)}地标 {gate['radius_m']:.2f} m 范围"
        else:
            gate_text = f"首次沿航线穿过世界坐标 x = {gate['x_m']:.2f} m 的截面"
        body = []
        for i, eid in enumerate([group["source_episode_id"], *group["variant_episode_ids"]]):
            member = group["members"][eid]
            start, entry = member["start"], member["entry"]
            xyz = ", ".join(f"{v:.2f}" for v in start["position_w_m"])
            label = "原轨迹" if i == 0 else f"新增 {i}"
            anchor = escape(eid, quote=True)
            body.append(f'<tr><td><a href="#{anchor}">{label}</a></td><td>{xyz}</td>'
                        f'<td>{start["speed_mps"]:.2f}</td><td>{entry["speed_mps"]:.2f}</td>'
                        f'<td>{entry["yaw_deg"]:.1f}°</td><td><a href="#{anchor}" '
                        f'data-entry-time="{entry["t_s"]:.2f}">播放 {entry["t_s"]:.2f} s</a></td></tr>')
        tables.append(f'<h3>{escape(group["instruction_zh"])}</h3><p>{gate_text}</p><div class="scroll"><table>'
                      '<thead><tr><th>记录</th><th>起点 XYZ (m)</th><th>初速 (m/s)</th>'
                      '<th>进场速度 (m/s)</th><th>进场航向</th><th>进场画面</th></tr></thead>'
                      '<tbody>' + ''.join(body) + '</tbody></table></div>')
    return ('<section id="variants"><h2>同一指令的不同进场</h2>'
            + ''.join(tables) + '<p><a href="variant_verification.json">查看逐帧取样和两两差异</a></p></section>')


def object_observation_section(root, rows):
    path = root / "room_evidence.json"
    if not path.is_file():
        return ""
    episodes = {row["episode_id"]:row for row in rows}
    items = []
    for record in read_json(path)["episodes"]:
        motion = record.get("object_observation")
        if not motion:
            continue
        eid = record["episode_id"]
        instruction = episodes[eid]["instruction_zh"]
        anchor = escape(eid, quote=True)
        items.append(f'<tr><td><a href="#{anchor}">{escape(instruction)}</a></td>'
                     f'<td>{motion["signed_sweep_deg"]:+.1f}°</td><td>{motion["stable_hover_s"]:.2f} s</td>'
                     f'<td><a href="#{anchor}" data-entry-time="{motion["arc_start_s"]:.2f}">观察</a> · '
                     f'<a href="#{anchor}" data-entry-time="{motion["stable_hover_start_s"]:.2f}">悬停</a> · '
                     f'{reviewer_link(root,record["panel"],"关键帧")}</td></tr>')
    if not items:
        return ""
    return ('<section id="objects"><h2>会议室物体观察</h2>'
            '<p>角度由机体相对物体的位置计算。从上方看，正值为逆时针。点击阶段名称可从对应画面播放。</p>'
            '<div class="scroll"><table><thead><tr><th>任务</th><th>实际绕行角</th><th>稳定悬停</th>'
            '<th>查看阶段</th></tr></thead><tbody>' + ''.join(items) + '</tbody></table></div></section>')


def write_gallery(root, rows, split_lookup):
    sections = []
    for row in rows:
        eid = row["episode_id"]
        e_id = escape(eid)
        folder = "episodes/" + eid + "/"
        heading = escape(row["instruction_zh"] or row["instruction"])
        evidence = reviewer_link(root, f"evidence/{eid}.jpg","查看目标或绕桌关键帧")
        evidence_json = reviewer_link(root, f"evidence/{eid}.json","查看地标与轨迹核对记录")
        low, high = row["altitude_range_m"]
        sections.append(f'''<article data-episode="{e_id}" id="{e_id}">
<h2>{heading}</h2><p class="meta">{e_id} · {escape(split_lookup[eid])}</p>
<div class="grid"><div><video controls preload="none" playsinline poster="{escape(quote(row["poster"]))}" src="{escape(quote(row["video"]))}"></video>
<p class="meta">左侧 RGB　右侧热红外　·　{row["seconds"]:.1f} 秒　·　{row["fps"]:.0f} 帧/秒</p></div>
<div><p class="instruction" lang="en">{escape(row["instruction"])}</p>
<dl><div><dt>实飞航程</dt><dd>{row["length_m"]:.2f} m</dd></div>
<div><dt>实飞高度范围</dt><dd>{low:.2f}–{high:.2f} m</dd></div>
<div><dt>均速 / 峰速</dt><dd>{row["mean_speed_mps"]:.2f} / {row["peak_speed_mps"]:.2f} m/s</dd></div>
<div><dt>最小采样安全净空</dt><dd>{row["minimum_sampled_safety_clearance_m"]:.3f} m</dd></div>
<div><dt>碰撞 / 越出已观测空域</dt><dd>{row["collision_frames"]} / {row["unknown_space_frames"]} 帧</dd></div></dl>
<div class="links">{reviewer_link(root,row["video"],"视频文件")}{reviewer_link(root,folder+"episode.json","回合元数据")}{reviewer_link(root,row["poster"],"双模态关键帧")}{evidence}{evidence_json}</div>
</div></div><div class="review"><label>人工审核<select data-key="status"><option value="pending">待审核</option><option value="accepted">通过</option><option value="rejected">退回</option></select></label>
<label class="notes">意见（可记视频秒数）<input type="text" data-key="notes" placeholder="例如 12.5 秒，目标进入画面"></label></div></article>''')
    scope = json.dumps(root.name)
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>专家飞行人工审核</title><style>*{box-sizing:border-box}body{margin:0;background:#f2f6f5;color:#162f32;font:16px/1.5 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:1190px;margin:auto;padding:40px 22px}header,article{background:white;border:1px solid #c5d6d1;border-radius:10px;padding:22px;margin:17px 0}h1,h2{line-height:1.4;margin:.15em 0 .4em}h1{font-size:2rem}h2{font-size:1.22rem}.meta,dt{color:#506363}p{margin:.5em 0}
.grid{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(260px,.8fr);gap:26px;align-items:start}video{width:100%;aspect-ratio:8 / 3;background:#1b292b}
dl{margin:10px 0}dl div{display:flex;justify-content:space-between;border-top:1px solid #e4ece9;padding:4px 0}dd{margin:0}
.links,.review,.tools{display:flex;flex-wrap:wrap;align-items:center;gap:12px}a{color:#006c5a}.review{border-top:1px solid #d5e0db;margin-top:15px;padding-top:12px}
#variants,#objects{background:white;border:1px solid #c5d6d1;border-radius:10px;margin:17px 0;padding:22px}#variants h3{margin:1.5em 0 .2em}.scroll table{border-collapse:collapse;font-variant-numeric:tabular-nums}.scroll th,.scroll td{padding:6px 12px;border-bottom:1px solid #d5e0db;text-align:left;white-space:nowrap}#objects td:first-child{white-space:normal;min-width:12em}.scroll{overflow-x:auto}
input,select,button{font:inherit;padding:5px 8px;margin-left:8px}button{color:white;background:#116354;border:0;border-radius:4px;cursor:pointer}
.notes{flex:1}.notes input{width:min(95%,34em)}@media(max-width:760px){main{padding:10px}.grid{grid-template-columns:1fr}.notes input{width:100%;margin-left:0}.review{align-items:start}}</style>
<main><header><h1>室内 RGB-T 专家飞行人工审核</h1>
<p>请逐条观看双模态视频，核对是否完成指令、目标是否可辨认、两路画面是否同步，并记录需要返工的时刻。搜索任务另请核对起飞时目标是否在画面外，搜索后是否找到并接近指定物体。</p>
<div class="tools"><label>审核人<input id="reviewer" autocomplete="name"></label><button id="save">导出人工审核记录</button><span id="count"></span></div>
<p class="meta">表单在当前浏览器中暂存。人工审核结束后请导出 JSON 留档。<a href="verification.json">完整自动核查结果</a></p></header>''' + object_observation_section(root, rows) + variant_comparison(root) + "\n".join(sections) + '''
</main><script>
const scope="expert_review_" + ''' + scope + ''', articles=[...document.querySelectorAll("article")];
let data={reviewer:"",episodes:{}};
try {const saved=localStorage.getItem(scope);if(saved)data=JSON.parse(saved);}catch(e){}
const reviewer=document.getElementById("reviewer");reviewer.value=data.reviewer||"";
for(const link of document.querySelectorAll("[data-entry-time]")){
 link.addEventListener("click",()=>{
   const video=document.getElementById(link.hash.slice(1)).querySelector("video");
   const playEntry=()=>{video.currentTime=Number(link.dataset.entryTime);video.play().catch(()=>{})};
   if(video.readyState>=1)playEntry();
   else{video.addEventListener("loadedmetadata",playEntry,{once:true});video.load()}
 });
}
function persist(){try{localStorage.setItem(scope,JSON.stringify(data))}catch(e){}}
function update(){let done=0;for(const a of articles){if(a.querySelector("select").value!=="pending")done++}
document.getElementById("count").textContent="已审核 "+done+" / "+articles.length+" 条";}
reviewer.oninput=()=>{data.reviewer=reviewer.value;persist()};
for(const a of articles){const k=a.dataset.episode;let row=data.episodes[k]||{status:"pending",notes:"",updated_at:null};data.episodes[k]=row;
 for(const el of a.querySelectorAll("[data-key]")){el.value=row[el.dataset.key]||"";el.addEventListener("input",()=>{row[el.dataset.key]=el.value;row.updated_at=new Date().toISOString();persist();update()})}}update();
document.getElementById("save").onclick=()=>{const name=reviewer.value.trim();if(!name){reviewer.reportValidity();reviewer.focus();document.getElementById("count").textContent="请填写审核人后导出";return}
const contents={dataset:scope.slice("expert_review_".length),reviewer:name,exported_at:new Date().toISOString(),episodes:Object.fromEntries(articles.map(a=>[a.dataset.episode,data.episodes[a.dataset.episode]]))};
const url=URL.createObjectURL(new Blob([JSON.stringify(contents,null,2)],{type:"application/json"}));const link=document.createElement("a");link.href=url;link.download="expert_episodes_human_review.json";link.click();setTimeout(()=>URL.revokeObjectURL(url),5000)};
</script></html>'''
    atomic_write(root / "review.html", html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episode", action="append", help="Audit a subset without publishing dataset-wide results")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    root = args.dataset.resolve(strict=True)
    folders = ([root / "episodes" / i for i in args.episode] if args.episode
               else sorted(x for x in (root / "episodes").iterdir() if x.is_dir() and not x.name.startswith(".")))
    require(folders, "无拟审核回合")
    rows, errors = [], {}
    def inspect(folder):
        try:
            row = check_episode(root, folder, root/"review_media")
            print(f"{folder.name}: {row['decoded_frames']} frames, {row['length_m']} m", flush=True)
            return row
        except Exception as exc:
            return {"episode_id":folder.name, "error":str(exc)}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(inspect, folders):
            if "error" in result:
                errors[result["episode_id"]] = result["error"]
            else:
                rows.append(result)
    if args.episode:
        if errors:
            raise SystemExit(json.dumps(errors, ensure_ascii=False, indent=2))
        return
    split_lookup = {}
    if not errors:
        try:
            split_lookup = check_splits(root, rows)
            for scene in {row["scene_version"] for row in rows}:
                calibration = read_json(root/"scenes"/scene/"calibration.json")
                require(isinstance(calibration, dict) and calibration, f"无标定记录 {scene}")
        except Exception as exc:
            errors["dataset"] = str(exc)
    report = {
        "dataset": root.name, "verified_at": datetime.now(timezone.utc).isoformat(), "passed": not errors,
        "episode_count": len(folders), "decoded_video_frames": sum(r["decoded_frames"] for r in rows),
        "decoded_lossless_images": sum(r["decoded_pngs"] for r in rows),
        "methods": ["full H.264 and PNG decode with OpenCV", "all video frames compared against both PNG eyes",
                    "causal frame and 3-step target checks", "state integrity, clearance, known space, final goal, stop",
                    "route-family grouped splits"], "episodes": rows, "errors": errors}
    atomic_write(root / "verification.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if errors:
        raise SystemExit(json.dumps(errors, ensure_ascii=False, indent=2))
    write_gallery(root, rows, split_lookup)


if __name__ == "__main__":
    main()
