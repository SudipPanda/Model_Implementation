import torch 
import math
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import yaml

"""
Loading the config file here
"""
class MistralConfig(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}") from None


def load_mistral_config(config_path = None):
    if config_path is None:
        config_path = (
            Path(__file__).resolve().parents[2]
            / "Config"
            / "mistral_config.yaml"
        )
    
    with open(config_path ,encoding="utf-8") as config_file:
        return MistralConfig(yaml.safe_load(config_file))



