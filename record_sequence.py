"""Execute explicit GRaD-Nav actions and record synchronized FPV/state samples.

No navigation policy, position controller, or waypoint planner is hidden here:
every 50 ms step uses the four values from the selected JSON sequence.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
CONTROL_DT = .05
ACTION_FORMATS = ('gradnav_normalized', 'gradnav_logits', 'body_rates_thrust')


def json_value(value):
    """Preserve structured receipts while accepting NumPy parameter values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def load_sequence(path):
    payload = json.loads(Path(path).expanduser().read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get('segments'), list) or not payload['segments']:
        raise ValueError('Sequence must contain a nonempty segments list')
    if float(payload.get('dt', CONTROL_DT)) != CONTROL_DT:
        raise ValueError('GRaD-Nav sequence dt must be exactly 0.05 simulation seconds')
    default_format = payload.get('format', 'gradnav_normalized')
    segments = []
    for index, item in enumerate(payload['segments']):
        if not isinstance(item, dict):
            raise ValueError(f'Segment {index} must be an object')
        steps = item.get('steps')
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError(f'Segment {index} steps must be a positive integer')
        action = item.get('action')
        if not isinstance(action, list) or len(action) != 4:
            raise ValueError(f'Segment {index} action must contain four values')
        action_format = item.get('format', default_format)
        if action_format not in ACTION_FORMATS:
            raise ValueError(f'Unsupported action format: {action_format}')
        for component, value in enumerate(action):
            if component == 3 and value == 'hover':
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f'Segment {index} action values must be finite numbers; only thrust may be "hover"')
            if action_format == 'gradnav_normalized' and not -1 <= value <= 1:
                raise ValueError(f'Segment {index} normalized actions must be in [-1,1]')
        segments.append({'name': str(item.get('name', f'segment_{index}')),
                         'steps': steps, 'action': action.copy(), 'format': action_format})
    return {**payload, 'dt': CONTROL_DT, 'segments': segments}


def resolve_action(action, action_format, parameters):
    """Resolve symbolic hover once from this episode's actual mass/thrust."""
    result = action.copy()
    if result[3] == 'hover':
        thrust = float(parameters['mass']) * 9.81 / float(parameters['max_thrust'])
        if action_format == 'body_rates_thrust':
            result[3] = thrust
        else:
            normalized = thrust / .25 - 1
            if not -1 <= normalized <= 1:
                raise ValueError('This episode cannot hover within the GRaD-Nav normalized thrust range')
            if action_format == 'gradnav_logits':
                if abs(normalized) >= 1:
                    raise ValueError('Hover at a tanh boundary has no finite logit')
                result[3] = float(np.arctanh(normalized))
            else:
                result[3] = normalized
    return np.asarray(result, dtype=np.float64).tolist()


def ffmpeg_executable():
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError('Video recording requires imageio-ffmpeg: python -m pip install imageio-ffmpeg') from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


class FpvVideoWriter:
    """Pipe exact-size RGB frames to the bundled FFmpeg H.264 encoder."""

    def __init__(self, path, width, height, fps=20.):
        self.path = Path(path)
        if self.path.exists():
            raise ValueError(f'Refusing to overwrite video: {self.path}')
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError('H.264 yuv420p recording requires positive even image width and height')
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Recording FPS must be positive and finite')
        self.width, self.height, self.fps = int(width), int(height), float(fps)
        self.frames = 0
        self.closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.errors = tempfile.TemporaryFile(mode='w+b')
        command = [ffmpeg_executable(), '-hide_banner', '-loglevel', 'error', '-n',
                   '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s:v', f'{width}x{height}',
                   '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                   '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', str(self.path)]
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                            stdout=subprocess.DEVNULL, stderr=self.errors)
        except Exception:
            self.errors.close()
            raise

    def _message(self):
        self.errors.flush()
        self.errors.seek(0)
        return self.errors.read().decode('utf-8', errors='replace').strip()

    def write(self, rgb):
        if self.closed:
            raise RuntimeError('Video writer is already closed')
        frame = np.asarray(rgb)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f'Expected RGB frame {(self.height, self.width, 3)}, got {frame.shape}')
        if frame.dtype != np.uint8:
            if not np.isfinite(frame).all() or np.any(frame < 0) or np.any(frame > 1):
                raise ValueError('Float RGB frames must be finite and within [0,1]')
            frame = np.rint(frame * 255).astype(np.uint8)
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            self.process.wait(timeout=30)
            raise RuntimeError('FFmpeg encoding failed: ' + self._message()) from exc
        self.frames += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                result = self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise RuntimeError('FFmpeg did not finish the video within 30 seconds')
            if result != 0:
                raise RuntimeError(f'FFmpeg exited with code {result}: {self._message()}')
            if not self.frames or not self.path.is_file() or self.path.stat().st_size == 0:
                raise RuntimeError('FFmpeg produced no video frames')
        finally:
            self.errors.close()


def verify_video(path, expected_frames):
    """Decode the whole video and compare FFmpeg's final output frame count."""
    command = [ffmpeg_executable(), '-hide_banner', '-loglevel', 'error',
               '-progress', 'pipe:1', '-nostats', '-i', str(path), '-map', '0:v:0',
               '-f', 'null', '-']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=60)
    if result.returncode:
        raise RuntimeError('Video verification failed: ' + result.stderr.strip())
    frames = [int(line.split('=', 1)[1]) for line in result.stdout.splitlines() if line.startswith('frame=')]
    if not frames or frames[-1] != expected_frames:
        raise RuntimeError(f'Video frame count mismatch: expected {expected_frames}, decoded {frames[-1] if frames else None}')
    return {'method': 'full_ffmpeg_decode', 'decoded_frames': frames[-1], 'passed': True}


CSV_FIELDS = ['video_frame', 'step_id', 'frame_id', 'sim_time', 'dt_actual',
              'x', 'y', 'z', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz', 'wx', 'wy', 'wz',
              'cmd_wx', 'cmd_wy', 'cmd_wz', 'normalized_thrust', 'body_clearance_m',
              'safety_clearance_m', 'collision', 'terminated', 'truncated', 'reason', 'segment']


def trajectory_row(receipt, video_frame, segment):
    state = receipt['state']
    collision = receipt.get('collision', {})
    return [video_frame, receipt['step_id'], receipt['frame_id'], receipt['sim_time'], receipt.get('dt_actual', 0.),
            *state['position'], *state['velocity'], *state['quaternion_wxyz'], *state['angular_velocity'],
            *receipt.get('applied_control', [0, 0, 0, 0]), collision.get('clearance'), collision.get('safety_clearance'),
            bool(collision.get('event')), bool(receipt.get('terminated')), bool(receipt.get('truncated')),
            receipt.get('reason'), segment]


def check_output_directory(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f'Output directory must be empty: {path}')
    return path


def record_sequence(simulator, sequence, output_dir, limit_steps=None, run_metadata=None):
    """Record initial and post-step receipts from a newly reset simulator."""
    if limit_steps is not None and (isinstance(limit_steps, bool) or not isinstance(limit_steps, int) or limit_steps < 1):
        raise ValueError('Recording step budget must be a positive integer')
    output_dir = check_output_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial = simulator.last_receipt
    image = np.asarray(simulator.last_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Simulator has no initial RGB observation; reset it before recording')
    requested_steps = sum(segment['steps'] for segment in sequence['segments'])
    budget = requested_steps if limit_steps is None else min(requested_steps, limit_steps)
    if isinstance(budget, bool) or budget < 1:
        raise ValueError('Recording step budget must be positive')
    # Snapshot the exact executed inputs, including automatically generated
    # geometry that otherwise disappears with the engine's temporary directory.
    (output_dir/'sequence.json').write_text(json.dumps(sequence, ensure_ascii=False, indent=2, default=json_value, allow_nan=False)+'\n')
    profile = getattr(simulator, 'config', None)
    if profile is not None:
        (output_dir/'profile.json').write_text(json.dumps(profile, ensure_ascii=False, indent=2, default=json_value, allow_nan=False)+'\n')
    collision = initial.get('collision', {})
    cloud_record = {'file': None, 'metadata_file': None, 'source': collision.get('source'),
                    'points': collision.get('points'), 'enabled': bool(collision.get('enabled', False))}
    point_cloud_file = getattr(simulator, 'point_cloud_file', None)
    if point_cloud_file is not None:
        cloud_path = Path(point_cloud_file)
        if not cloud_path.is_file():
            raise ValueError(f'Obstacle point cloud disappeared before recording: {cloud_path}')
        shutil.copy2(cloud_path, output_dir/'obstacles.ply')
        cloud_record['file'] = 'obstacles.ply'
        metadata = cloud_path.with_suffix('.json')
        if metadata.is_file():
            shutil.copy2(metadata, output_dir/'obstacles.json')
            cloud_record['metadata_file'] = 'obstacles.json'
    metadata = dict(run_metadata or {})
    metadata.update(point_cloud_source=cloud_record['source'], point_cloud_points=cloud_record['points'],
                    saved_point_cloud=cloud_record['file'])
    video_path = output_dir/'fpv.mp4'
    writer = FpvVideoWriter(video_path, image.shape[1], image.shape[0], 1/CONTROL_DT)
    executed = 0
    final = initial
    clearances = []
    status = 'sequence_completed'
    try:
        with (output_dir/'states.jsonl').open('w') as jsonl, (output_dir/'trajectory.csv').open('w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(CSV_FIELDS)

            def sample(receipt, label, command=None):
                frame_index = writer.frames
                writer.write(simulator.last_rgb)
                row = {**receipt, 'video_frame': frame_index, 'segment': label, 'sequence_action': command}
                jsonl.write(json.dumps(row, ensure_ascii=False, default=json_value, allow_nan=False)+'\n')
                csv_writer.writerow(trajectory_row(receipt, frame_index, label))
                for clearance in (receipt.get('min_sampled_safety_clearance'),
                                  receipt.get('collision', {}).get('safety_clearance')):
                    if clearance is not None:
                        clearances.append(float(clearance))

            sample(initial, 'initial')
            if initial.get('terminated') or initial.get('truncated'):
                status = initial.get('reason') or 'initial_state_terminated'
            else:
                for segment in sequence['segments']:
                    action = resolve_action(segment['action'], segment['format'], simulator.parameters)
                    for _ in range(segment['steps']):
                        if executed >= budget:
                            status = 'step_limit' if budget < requested_steps else 'sequence_completed'
                            break
                        final = simulator.step(action, format=segment['format'], repeat=1)
                        executed += 1
                        sample(final, segment['name'], action)
                        if final.get('terminated') or final.get('truncated'):
                            status = final.get('reason') or 'terminated'
                            break
                    if executed >= budget or final.get('terminated') or final.get('truncated'):
                        break
                if executed >= budget and budget < requested_steps and not (final.get('terminated') or final.get('truncated')):
                    status = 'step_limit'
    finally:
        writer.close()
    verification = verify_video(video_path, writer.frames)
    summary = {
        'schema': 'gradnav-fpv-recording/v1',
        'status': status,
        'video': video_path.name,
        'states': 'states.jsonl',
        'trajectory': 'trajectory.csv',
        'codec': 'H.264 / yuv420p',
        'width': writer.width, 'height': writer.height, 'fps': writer.fps,
        'frames': writer.frames,
        'video_duration_seconds': writer.frames/writer.fps,
        'simulation_duration_seconds': final['sim_time']-initial['sim_time'],
        'timeline_note': 'Frame 0 is reset. Each later frame is the exact post-step snapshot; an early collision may shorten the final 0.05 s step.',
        'requested_steps': requested_steps, 'executed_steps': executed,
        'collision': bool(final.get('collision', {}).get('event')),
        'minimum_sampled_safety_clearance_m': min(clearances) if clearances else None,
        'initial_state': initial['state'], 'final_state': final['state'],
        'final_step_id': final['step_id'], 'final_frame_id': final['frame_id'],
        'parameters': simulator.parameters,
        'run': metadata,
        'point_cloud': cloud_record,
        'profile_file': 'profile.json' if profile is not None else None,
        'sequence_file': 'sequence.json',
        'sequence': sequence,
        'verification': verification,
    }
    (output_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_value, allow_nan=False)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=Path.home()/'3DGS_env/no_train_outputs/export_gate_mid/gate_mid.ply')
    parser.add_argument('--point-cloud', type=Path, help='Optional ordinary surface PLY aligned with the scene')
    parser.add_argument('--sequence', type=Path, default=BASE_DIR/'configs/demo_actions.json')
    parser.add_argument('--profile', default='nominal', help='nominal, training, or a training-profile JSON path')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--spawn', nargs=3, type=float, default=(-6., 0., 1.25))
    parser.add_argument('--yaw', type=float, default=0.)
    parser.add_argument('--resolution', type=float, default=.75)
    parser.add_argument('--drone-size', nargs=3, type=float, default=(.60, .60, .12))
    parser.add_argument('--collision-margin', type=float, default=.05)
    parser.add_argument('--no-collision', action='store_true', help='Explicit debugging option; collision is enabled by default')
    parser.add_argument('--limit-steps', type=int)
    parser.add_argument('--output-dir', type=Path, default=BASE_DIR/'outputs'/datetime.now(timezone.utc).strftime('demo_%Y%m%dT%H%M%S_%fZ'))
    args = parser.parse_args()
    if args.limit_steps is not None and args.limit_steps < 1:
        parser.error('--limit-steps must be positive')
    try:
        # Reject output mistakes before loading CUDA, the scene, or exporting a
        # collision cloud; record_sequence rechecks against concurrent writes.
        check_output_directory(args.output_dir)
        sequence = load_sequence(args.sequence)
        # Import lazily: sequence/video utilities and --help need no CUDA scene.
        from config import load_config
        from engine import PolicyFlightSimulator
        simulator = PolicyFlightSimulator(args.scene.expanduser(), point_cloud=args.point_cloud,
                                         config=load_config(args.profile), seed=args.seed,
                                         spawn=args.spawn, yaw=args.yaw, resolution=args.resolution,
                                         drone_size=args.drone_size, collision_margin=args.collision_margin,
                                         no_collision=args.no_collision)
        try:
            simulator.reset()
            metadata = {'scene': str(args.scene.expanduser().resolve()),
                        'point_cloud': str(args.point_cloud.expanduser().resolve()) if args.point_cloud else None,
                        'profile': args.profile, 'seed': args.seed,
                        'sequence_file': str(args.sequence.expanduser().resolve()),
                        'spawn': args.spawn, 'yaw': args.yaw, 'collision_enabled': not args.no_collision}
            summary = record_sequence(simulator, sequence, args.output_dir, args.limit_steps, metadata)
        finally:
            simulator.close()
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'Flight recording failed: {exc}\n')
    print(json.dumps({'output_dir': str(args.output_dir.resolve()),
                      **{key: summary[key] for key in ('status', 'frames', 'executed_steps', 'simulation_duration_seconds', 'collision', 'verification')}},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
