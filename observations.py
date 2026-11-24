"""GRaD-Nav observation layout without manufacturing learned features.

The 57-value layout follows the released code, not the paper's differently
dimensioned description. A visual TorchScript encoder receives [1,3,224,224]
RGB in [0,1], using adaptive average pooling and no ImageNet normalization.
A context TorchScript encoder receives [1,5*57] history and must return its
selected latent tensor [1,24] (export the desired mean/sampling wrapper).

Without these assets, structured truth/base observations and aligned RGB remain
usable by an external navigation process, but no ready-made 57-vector is claimed.
"""

from collections import deque
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


BASE_DIM = 17
GRADNAV_SCHEMA = {
    'height': [0, 1], 'vertical_velocity': [1, 2], 'quaternion_xyzw': [2, 6],
    'current_applied_control': [6, 10], 'previous_applied_control': [10, 14],
    'world_linear_velocity': [14, 17], 'visual_features': [17, 33],
    'context_latent': [33, 57],
}


def _finite_vector(value, width, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain {width} finite values')
    return array


class ObservationBuilder:
    """Build each observation once, retaining source-code CENet history timing.

    History is zero-padded at reset. On post-step t, context inference uses the
    history returned at t-1, then the prior frame's VAE observation is appended
    for the next context inference. VAE latent slots are explicitly zero as in
    the source. Missing learned visual history is marked invalid, never passed
    to a context encoder as fabricated features. Re-reading a frame is cached.

    assets may be a JSON filename or a mapping with visual_encoder,
    context_encoder, and obs_rms paths. Relative paths in JSON are relative to
    that JSON. obs_rms is a JSON mapping {mean, var, epsilon?}; epsilon defaults
    to the original RunningMeanStd's 1e-5. Ordinary Python pickle is not loaded.
    """

    def __init__(self, config, seed=0, device='cpu', assets=None):
        options = config.get('observation', config)
        self.history_length = int(options.get('history', 5))
        self.visual_dim = int(options.get('visual_dim', 16))
        self.latent_dim = int(options.get('latent_dim', 24))
        self.noise_level = float(options.get('noise_level', 0.))
        self.require_obs_rms = bool(options.get('require_obs_rms', False))
        if (self.history_length < 1 or self.visual_dim < 1 or self.latent_dim < 1 or
                not np.isfinite(self.noise_level) or self.noise_level < 0):
            raise ValueError('Invalid observation dimensions, history, or noise level')
        self.dimension = BASE_DIM + self.visual_dim + self.latent_dim
        self.seed = int(seed)
        self.device = torch.device(device)
        self.visual_encoder = self.context_encoder = None
        self.rms = None
        self.asset_sources = {}
        self._load_assets(assets)
        self.reset()

    def _load_assets(self, assets):
        base = Path.cwd()
        if isinstance(assets, (str, Path)):
            path = Path(assets).expanduser().resolve()
            assets = json.loads(path.read_text())
            base = path.parent
        if assets is None:
            return
        if not isinstance(assets, dict):
            raise ValueError('Observation assets must be a JSON object or JSON path')
        for key in ('visual_encoder', 'context_encoder'):
            if assets.get(key) is not None:
                path = Path(assets[key]).expanduser()
                path = (base / path).resolve() if not path.is_absolute() else path.resolve()
                module = torch.jit.load(str(path), map_location=self.device).eval()
                setattr(self, key, module)
                self.asset_sources[key] = str(path)
        if assets.get('obs_rms') is not None:
            value = assets['obs_rms']
            if isinstance(value, (str, Path)):
                path = Path(value).expanduser()
                path = (base / path).resolve() if not path.is_absolute() else path.resolve()
                value = json.loads(path.read_text())
                self.asset_sources['obs_rms'] = str(path)
            if not isinstance(value, dict):
                raise ValueError('obs_rms must be JSON with mean and var arrays')
            mean = _finite_vector(value.get('mean'), self.dimension, 'obs_rms.mean')
            var = _finite_vector(value.get('var'), self.dimension, 'obs_rms.var')
            epsilon = float(value.get('epsilon', 1e-5))
            if np.any(var < 0) or not np.isfinite(epsilon) or epsilon <= 0:
                raise ValueError('obs_rms requires nonnegative variance and positive epsilon')
            self.rms = (mean, var, epsilon)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._history = deque([np.zeros(self.dimension) for _ in range(self.history_length)], maxlen=self.history_length)
        self._history_ids = deque([None] * self.history_length, maxlen=self.history_length)
        self._history_valid = deque([True] * self.history_length, maxlen=self.history_length)
        self._previous_vae = None
        self._previous_ids = None
        self._previous_valid = False
        self._cache = None
        self._signature = None

    def _feature(self, value, width, name, frame_id, step_id):
        if value is None:
            return None
        if isinstance(value, dict):
            if value.get('frame_id') != frame_id or value.get('step_id') != step_id:
                raise ValueError(f'{name} frame_id/step_id does not match the observation')
            value = value.get('values')
        return _finite_vector(value, width, name)

    def _run_encoder(self, encoder, tensor, width, name):
        with torch.no_grad():
            output = encoder(tensor.to(self.device))
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, width):
            raise ValueError(f'{name} TorchScript must return tensor [1,{width}]')
        return _finite_vector(output.detach().cpu().numpy()[0], width, name)

    def _history_result(self):
        ready = all(self._history_valid)
        return {
            'shape': [self.history_length, self.dimension], 'order': 'oldest_to_newest',
            'flat': np.asarray(self._history).reshape(-1).tolist() if ready else None,
            'frames': list(self._history_ids), 'valid_rows': list(self._history_valid),
            'ready': ready, 'zero_padding_at_reset': True, 'latent_slots_zero': True,
        }

    def build(self, state, current_control, previous_control, rgb, frame_id, step_id,
              visual_features=None, latent_features=None):
        state = _finite_vector(state, 13, 'state')
        current = _finite_vector(current_control, 4, 'current_control')
        previous = _finite_vector(previous_control, 4, 'previous_control')
        if (not isinstance(frame_id, (int, np.integer)) or not isinstance(step_id, (int, np.integer)) or
                frame_id < 0 or step_id < 0):
            raise ValueError('frame_id and step_id must be nonnegative integers')
        frame_id, step_id = int(frame_id), int(step_id)
        image = np.asarray(rgb)
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.
        else:
            image = image.astype(np.float32)
        if (image.ndim != 3 or image.shape[2] != 3 or not image.shape[0] or not image.shape[1] or
                not np.isfinite(image).all() or image.min() < 0 or image.max() > 1):
            raise ValueError('RGB must be finite HxWx3 RGB in [0,1], or uint8')
        q = state[6:10]
        if np.linalg.norm(q) < 1e-12:
            raise ValueError('State quaternion must be nonzero')
        q = q / np.linalg.norm(q)
        visual = self._feature(visual_features, self.visual_dim, 'visual_features', frame_id, step_id)
        latent = self._feature(latent_features, self.latent_dim, 'latent_features', frame_id, step_id)
        # A read of the same snapshot is pure: no additional noise or history.
        signature = (state.tobytes(), current.tobytes(), previous.tobytes(), image.tobytes(),
                     None if visual is None else visual.tobytes(), None if latent is None else latent.tobytes())
        if self._cache is not None:
            old = self._cache
            if (frame_id, step_id) == (old['frame_id'], old['step_id']):
                if signature != self._signature:
                    raise ValueError('An existing frame_id/step_id cannot identify different observation data')
                return deepcopy(old)
            if frame_id <= old['frame_id'] or step_id <= old['step_id']:
                raise ValueError('Observation IDs must increase; call reset() for a new episode')
            if step_id != old['step_id'] + 1:
                # Do not invent the skipped visual/control observations.
                self._history_valid = deque([False] * self.history_length, maxlen=self.history_length)
        context_history = self._history_result()
        reasons = []
        sources = {'visual': None, 'context': None}
        if visual is not None:
            sources['visual'] = 'external_matching_frame'
        elif self.visual_encoder is not None:
            tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())[None]
            tensor = F.adaptive_avg_pool2d(tensor, (224, 224))
            visual = self._run_encoder(self.visual_encoder, tensor, self.visual_dim, 'visual_encoder')
            sources['visual'] = 'torchscript'
        else:
            reasons.append('No trained visual encoder or matching-frame visual features supplied')
        if latent is not None:
            sources['context'] = 'external_matching_frame'
        elif self.context_encoder is not None and context_history['ready']:
            if self._previous_ids is None and step_id == 0:
                # Source env.reset initializes latent_vect to zero. This is only
                # a valid reset context when the actual context asset is present.
                latent = np.zeros(self.latent_dim)
                sources['context'] = 'source_reset_zero_with_configured_encoder'
            else:
                tensor = torch.tensor([context_history['flat']], dtype=torch.float32)
                latent = self._run_encoder(self.context_encoder, tensor, self.latent_dim, 'context_encoder')
                sources['context'] = 'torchscript_previous_history'
        elif self.context_encoder is not None:
            reasons.append('Context history contains unavailable visual frames')
        else:
            reasons.append('No trained context encoder or matching-frame latent features supplied')
        # Match source noise locations: world v[0:3] remains noise-free; the
        # separately repeated vertical-velocity field receives sensor noise.
        pos_noise = self.rng.uniform(-.5, .5, 3) * self.noise_level
        velocity_noise = self.rng.uniform(-.5, .5, 3) * self.noise_level
        visual_noise = self.rng.uniform(-.5, .5, self.visual_dim) * self.noise_level
        latent_noise = self.rng.uniform(-.5, .5, self.latent_dim) * self.noise_level
        q_noise = self.rng.uniform(-.5, .5, 4) * self.noise_level
        clean = np.r_[state[2], state[5], q[[1, 2, 3, 0]], current, previous, state[3:6]]
        base = clean.copy()
        base[0] += pos_noise[2]; base[1] += velocity_noise[2]; base[2:6] += q_noise
        noisy_visual = visual + visual_noise if visual is not None else None
        noisy_latent = latent + latent_noise if latent is not None else None
        raw = np.r_[base, noisy_visual, noisy_latent] if visual is not None and latent is not None else None
        normalized = (raw - self.rms[0]) / np.sqrt(self.rms[1] + self.rms[2]) if raw is not None and self.rms is not None else raw
        if self.require_obs_rms and self.rms is None:
            reasons.append('This profile requires the trained obs_rms normalizer')
        # Source env.step appends its PRE-step observation, after selecting the
        # latent used by this post-step snapshot; never append the current frame.
        if self._previous_vae is not None:
            self._history.append(self._previous_vae.copy())
            self._history_ids.append(self._previous_ids.copy())
            self._history_valid.append(self._previous_valid)
        history_for_next = self._history_result()
        self._previous_vae = np.r_[base, noisy_visual if noisy_visual is not None else np.zeros(self.visual_dim), np.zeros(self.latent_dim)]
        self._previous_valid = noisy_visual is not None
        self._previous_ids = {'frame_id': frame_id, 'step_id': step_id}
        schema = {**GRADNAV_SCHEMA, 'visual_features': [17, 17 + self.visual_dim],
                  'context_latent': [17 + self.visual_dim, self.dimension]}
        result = {
            'schema': 'gradnav_released_code', 'dimension': self.dimension, 'offsets': schema,
            'frame_id': frame_id, 'step_id': step_id,
            'policy_ready': raw is not None and not (self.require_obs_rms and self.rms is None),
            'policy_observation': normalized.tolist() if normalized is not None else None,
            'raw_policy_observation': raw.tolist() if raw is not None else None,
            'normalization_applied': self.rms is not None and raw is not None,
            'base_observation': base.tolist(), 'base_observation_clean': clean.tolist(),
            'visual_features': noisy_visual.tolist() if noisy_visual is not None else None,
            'latent_features': noisy_latent.tolist() if noisy_latent is not None else None,
            'feature_sources': sources, 'assets': dict(self.asset_sources), 'unavailable_reasons': reasons,
            'context_history': context_history, 'history_for_next': history_for_next,
            'context_timing': 'use_previous_returned_history_then_append_pre_step_frame',
            'image_preprocessing': {'color': 'RGB', 'range': [0, 1], 'shape': [1, 3, 224, 224],
                                    'resize': 'adaptive_average_pool', 'imagenet_normalization': False},
            'noise_level': self.noise_level,
        }
        self._cache = deepcopy(result); self._signature = signature
        return result
