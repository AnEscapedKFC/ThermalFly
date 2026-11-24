import builtins
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from config import load_config


class ConfigurationChecks(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)

    def tearDown(self):self.temporary.cleanup()

    def test_named_profiles_and_default_are_independent_copies(self):
        first=load_config();second=load_config('nominal');training=load_config('training')
        self.assertEqual(first,second)
        self.assertFalse(first['randomization']['enabled'])
        self.assertTrue(training['randomization']['enabled'])
        self.assertEqual(first['dynamics']['rotor_noise_std'],0.)
        self.assertEqual(training['dynamics']['rotor_noise_std'],.01)
        self.assertTrue(training['observation']['require_obs_rms'])
        first['dynamics']['inertia'][0]=99.
        first['lpf']['alpha']=.1
        self.assertEqual(second['dynamics']['inertia'][0],.01)
        self.assertEqual(load_config()['lpf']['alpha'],.5)
        training['randomization']['mass'][0]=999.
        self.assertEqual(load_config('training')['randomization']['mass'],[1.,1.2])

    def test_partial_dict_override_keeps_defaults_without_aliasing_input(self):
        override={'name':'custom','dynamics':{'mass':1.4},'observation':{'require_obs_rms':True},
                  'limits':{'position_min':[-1,-2,-3]}}
        original=copy.deepcopy(override)
        result=load_config(override)
        self.assertEqual(override,original)
        self.assertEqual(result['name'],'custom');self.assertEqual(result['dynamics']['mass'],1.4)
        self.assertEqual(result['dynamics']['max_thrust'],26.)
        self.assertTrue(result['observation']['require_obs_rms'])
        self.assertEqual(result['observation']['history'],5)
        result['limits']['position_min'][0]=100
        self.assertEqual(override['limits']['position_min'],[-1,-2,-3])
        self.assertNotIn('position_min',load_config()['limits'])

    def test_json_and_plain_yaml_files_merge_nested_overrides(self):
        values={'name':'file_profile','lpf':{'enabled':False},'dynamics':{'mass':1.3},
                'observation':{'visual_dim':12}}
        json_path=self.root/'profile.json';json_path.write_text(json.dumps(values))
        yaml_path=self.root/'profile.yaml'
        yaml_path.write_text('name: file_profile\nlpf:\n  enabled: false\ndynamics:\n  mass: 1.3\nobservation:\n  visual_dim: 12\n')
        from_json=load_config(json_path);from_yaml=load_config(yaml_path)
        self.assertEqual(from_json,from_yaml)
        self.assertFalse(from_json['lpf']['enabled'])
        self.assertEqual(from_json['lpf']['alpha'],.5)
        self.assertEqual(from_json['observation']['visual_dim'],12)
        self.assertEqual(from_json['observation']['latent_dim'],24)

    def test_original_yaml_uses_player_inference_lpf_and_source_training_parameters(self):
        path=self.root/'original_training.yaml'
        path.write_text('''params:
  config:
    LPF_train: true
    LPF_val: 0.1
    domain_randomization: true
    obs_rms: true
    player:
      LPF_eval: true
      LPF_val: 0.75
  env_hyper:
    HISTORY_BUFFER_NUM: 7
    SINGLE_VISUAL_INPUT_SIZE: 20
    LATENT_VECT_NUM: 12
''')
        result=load_config(path)
        self.assertEqual(result['name'],'original_training.yaml')
        self.assertEqual(result['lpf'],{'enabled':True,'alpha':.75})
        self.assertTrue(result['randomization']['enabled'])
        self.assertTrue(result['observation']['require_obs_rms'])
        self.assertEqual(result['observation']['history'],7)
        self.assertEqual(result['observation']['visual_dim'],20)
        self.assertEqual(result['observation']['latent_dim'],12)
        self.assertEqual(result['dynamics']['br_noise_std'],.01)
        self.assertEqual(result['randomization']['mass'],[1.,1.2])
        self.assertEqual(result['source']['training_yaml'],str(path.resolve()))

    def test_original_yaml_can_disable_lpf_randomization_and_normalizer(self):
        path=self.root/'disabled.yml'
        path.write_text('''params:
  config:
    LPF_train: true
    LPF_val: 0.9
    domain_randomization: false
    obs_rms: false
    player:
      LPF_eval: false
      LPF_val: 0.2
''')
        result=load_config(path)
        self.assertEqual(result['lpf'],{'enabled':False,'alpha':.2})
        self.assertFalse(result['randomization']['enabled'])
        self.assertFalse(result['observation']['require_obs_rms'])
        self.assertEqual(result['observation']['history'],5)
        self.assertEqual(result['observation']['visual_dim'],16)
        self.assertEqual(result['observation']['latent_dim'],24)

    def test_yaml_conversion_has_no_original_source_import_dependency(self):
        path=self.root/'self_contained.yaml';path.write_text('params:\n  config:\n    obs_rms: true\n')
        real_import=builtins.__import__
        blocked={'grad_nav','envs','algorithms','utils','dflex'}
        def import_guard(name,*args,**kwargs):
            if name.split('.')[0] in blocked:
                raise AssertionError('Original repository dependency: '+name)
            return real_import(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=import_guard):
            result=load_config(path)
        self.assertTrue(result['observation']['require_obs_rms'])
        self.assertFalse(result['lpf']['enabled'])
        self.assertFalse(result['randomization']['enabled'])

    def test_non_object_configuration_file_is_rejected(self):
        path=self.root/'bad.json';path.write_text('[1,2,3]')
        with self.assertRaisesRegex(ValueError,'object'):load_config(path)


if __name__=='__main__':unittest.main()
