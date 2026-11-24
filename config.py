"""Self-contained profiles and an adapter for the original training YAML."""
from copy import deepcopy
import json
from pathlib import Path


def _merge(base, override):
    for key,value in override.items():
        if isinstance(value,dict) and isinstance(base.get(key),dict):
            _merge(base[key],value)
        else:base[key]=deepcopy(value)
    return base


def load_config(value=None):
    root=Path(__file__).resolve().parent/'configs'
    base=json.loads((root/'nominal.json').read_text())
    if value is None:return base
    if isinstance(value,dict):return _merge(base,value)
    if str(value) in ('nominal','training'):
        return json.loads((root/(str(value)+'.json')).read_text())
    path=Path(value).expanduser().resolve()
    if path.suffix.lower() in ('.yaml','.yml'):
        import yaml
        data=yaml.safe_load(path.read_text())
    else:data=json.loads(path.read_text())
    if not isinstance(data,dict):raise ValueError('Configuration must be an object')
    if 'params' not in data:return _merge(base,data)
    # These source parameters live in the original environment, not its YAML.
    # Start from the checked-in source training profile, then honor YAML flags.
    result=json.loads((root/'training.json').read_text())
    cfg=data['params'].get('config',{});player=cfg.get('player',{})
    env=data['params'].get('env_hyper',{})
    result['name']=str(path.name)
    result['lpf']={'enabled':bool(player.get('LPF_eval',False)),
                   'alpha':float(player.get('LPF_val',.5))}
    result['randomization']['enabled']=bool(cfg.get('domain_randomization',False))
    result['observation']['require_obs_rms']=bool(cfg.get('obs_rms',False))
    for dest,source in [('history','HISTORY_BUFFER_NUM'),('visual_dim','SINGLE_VISUAL_INPUT_SIZE'),('latent_dim','LATENT_VECT_NUM')]:
        if source in env:result['observation'][dest]=int(env[source])
    result['source']['training_yaml']=str(path)
    return result
