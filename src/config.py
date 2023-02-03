import os
import yaml

class Config(dict):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.load(self._yaml, Loader=yaml.FullLoader)
            self._dict['PATH'] = os.path.dirname(config_path)

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]
        if DEFAULT_CONFIG.get(name) is not None:
            return DEFAULT_CONFIG[name]
        return None


DEFAULT_CONFIG = {
    'GPU': [0],
    'DATA_ROOT': './SCUT-EnsText/trainlmdb',
    'DIS_LR': 0.00001,
    'DIS_BETA1': 0.5,
    'DIS_BETA2': 0.9,
    'GEN_LR': 0.0001,
    'GEN_BETA1': 0.0,
    'GEN_BETA2': 0.9,
    'BATCH_SIZE': 4,
    'INPUT_SIZE': 512,
    'NUMOFWORKERS': 0,
    'L1_LOSS_WEIGHT': 1,
    'FM_LOSS_WEIGHT': 10,
    'STYLE_LOSS_WEIGHT': 1,
    'CONTENT_LOSS_WEIGHT': 1,
    'INPAINT_ADV_LOSS_WEIGHT': 0.01,
    'NUM_EPOCHES': 500,
    'SAVE_INTERVAL': 10,
}