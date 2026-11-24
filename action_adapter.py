"""GRaD-Nav policy-action conventions, with explicit filter/history profiles.

The normalized policy action is NOT a physical control: its first three
components are scaled by .5 rad/s and its thrust component maps [-1, 1] to
[0, .5]. The dynamics themselves accept [wx, wy, wz, normalized_total_thrust].
"""

from copy import deepcopy

import numpy as np

from dynamics import DEFAULT_DYNAMICS_PARAMETERS, validate_dynamics_parameters

DEFAULT_ACTION_CONFIG = {
    'lpf': {'enabled': True, 'alpha': .5},
    'delay': {'enabled': True, 'body_rate_alpha': .8, 'thrust_alpha': .7,
              'history': 'legacy_two_step'},
}


def make_dynamics_parameters(config, seed=0):
    """Sample the source-code training ranges without changing global RNG state.

    These ranges reproduce the checked-in training code, not the different
    ranges printed in the paper's Table III. Sampling happens once per reset.
    """
    parameters = deepcopy(DEFAULT_DYNAMICS_PARAMETERS)
    parameters.update(deepcopy(config.get('dynamics', {})))
    parameters = validate_dynamics_parameters(parameters)
    randomization = config.get('randomization', {})
    if randomization.get('enabled', False):
        rng = np.random.default_rng(seed)
        for key, default in [('mass', (1., 1.2)), ('max_thrust', (24., 28.))]:
            bounds = np.asarray(randomization.get(key, default), dtype=float)
            if (bounds.shape != (2,) or not np.isfinite(bounds).all() or
                    bounds[0] <= 0 or bounds[1] < bounds[0]):
                raise ValueError(f'Invalid randomization range for {key}')
            parameters[key] = float(rng.uniform(*bounds))
        fraction = float(randomization.get('inertia_fraction', .2))
        if not np.isfinite(fraction) or not 0 <= fraction < 1:
            raise ValueError('inertia_fraction must be in [0, 1)')
        parameters['inertia'] = (np.asarray(parameters['inertia']) *
                                  rng.uniform(1-fraction, 1+fraction, 3)).tolist()
    return validate_dynamics_parameters(parameters)


class ActionAdapter:
    """Convert one policy action per 50 ms outer step to physical controls.

    ``legacy_two_step`` deliberately matches the original environment's
    history-update order. After adaptation, current_control is u[t] and
    previous_control is u[t-1]; the legacy delay read u[t-2] before that shift.
    ``previous_step`` selects an ordinary one-step exponential delay instead.
    """

    FORMATS = ('gradnav_normalized', 'gradnav_logits', 'body_rates_thrust')

    def __init__(self, config, hover_thrust):
        self.config = deepcopy(config)
        for section, defaults in DEFAULT_ACTION_CONFIG.items():
            self.config[section] = {**defaults, **self.config.get(section, {})}
        lpf = self.config.get('lpf', {})
        delay = self.config.get('delay', {})
        self.lpf_enabled = bool(lpf.get('enabled', False))
        self.lpf_alpha = float(lpf.get('alpha', .5))
        self.delay_enabled = bool(delay.get('enabled', True))
        self.body_rate_alpha = float(delay.get('body_rate_alpha', .8))
        self.thrust_alpha = float(delay.get('thrust_alpha', .7))
        self.history_mode = delay.get('history', 'legacy_two_step')
        if self.history_mode not in ('legacy_two_step', 'previous_step'):
            raise ValueError('Unknown delay history mode')
        for name in ('lpf_alpha', 'body_rate_alpha', 'thrust_alpha'):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be in [0, 1]')
        self.hover_thrust = float(hover_thrust)
        self.reset()

    def reset(self, hover_thrust=None):
        if hover_thrust is not None:
            self.hover_thrust = float(hover_thrust)
        if not np.isfinite(self.hover_thrust) or not 0 <= self.hover_thrust <= 1:
            raise ValueError('hover_thrust must be in [0, 1]')
        self.current_control = np.array([0., 0., 0., self.hover_thrust])
        self.previous_control = self.current_control.copy()
        self.filtered_action = None
        self.action_index = 0
        self.last_info = {}

    def adapt(self, action, format='gradnav_normalized'):
        if format not in self.FORMATS:
            raise ValueError(f'Unknown action format: {format}')
        raw = np.asarray(action, dtype=float)
        if raw.shape != (4,) or not np.isfinite(raw).all():
            raise ValueError('action must contain four finite numbers')
        normalized = filtered = None
        if format == 'body_rates_thrust':
            # An explicit physical-control interface avoids accidental repeated
            # tanh, scaling, LPF or delay after an external controller did them.
            target = raw.copy()
            target[3] = np.clip(target[3], 0., 1.)
            applied = target.copy()
            clipped = bool(target[3] != raw[3])
            self.filtered_action = None
        else:
            normalized = np.tanh(raw) if format == 'gradnav_logits' else np.clip(raw, -1., 1.)
            clipped = bool(format == 'gradnav_normalized' and np.any(normalized != raw))
            filtered = normalized.copy()
            if self.lpf_enabled:
                # Original BatchLowPassFilter starts with the first action,
                # rather than blending it against a synthetic zero action.
                if self.filtered_action is not None:
                    filtered = self.lpf_alpha*normalized + (1-self.lpf_alpha)*self.filtered_action
                self.filtered_action = filtered.copy()
            target = np.r_[.5*np.clip(filtered[:3], -1., 1.), .25*(np.clip(filtered[3], -1., 1.)+1.)]
            if self.delay_enabled:
                history = self.previous_control if self.history_mode == 'legacy_two_step' else self.current_control
                applied = np.r_[self.body_rate_alpha*target[:3] + (1-self.body_rate_alpha)*history[:3],
                                self.thrust_alpha*target[3] + (1-self.thrust_alpha)*history[3]]
            else:
                applied = target.copy()
            applied[:3] = np.clip(applied[:3], -.5, .5)
            applied[3] = np.clip(applied[3], 0., 1.)
        self.previous_control = self.current_control.copy()
        self.current_control = applied.copy()
        self.action_index += 1
        self.last_info = {
            'format': format, 'input_action': raw.tolist(),
            'normalized_action': None if normalized is None else normalized.tolist(),
            'filtered_action': None if filtered is None else filtered.tolist(),
            'target_control': target.tolist(), 'applied_control': applied.tolist(),
            'input_clipped': clipped, 'history_mode': self.history_mode,
            'lpf_applied': self.lpf_enabled and format != 'body_rates_thrust',
            'delay_applied': self.delay_enabled and format != 'body_rates_thrust',
            'action_index': self.action_index,
        }
        return applied.copy()
