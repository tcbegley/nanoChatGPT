import os
from distutils.util import strtobool

from trainers.rl_trainer import GumbelTrainer, PolicyGradientTrainer
from utils import load_config

USE_GPU = strtobool(os.environ.get("USE_GPU", "true"))

if USE_GPU:
    config_path = "config/config_rl.yaml"
else:
    config_path = "config/config_rl_cpu.yaml"

config = load_config(config_path)

if config["method"] == "gumbel":
    print("Using Gumbel method")
    assert (
        config["hard_code_reward"] == False
    ), "hard_code_reward must be False for Gumbel method"
    trainer = GumbelTrainer(config_path)
elif config["method"] == "pg":
    print("Using Policy Gradient method")
    trainer = PolicyGradientTrainer(config_path)
else:
    raise NotImplementedError

trainer.train()
