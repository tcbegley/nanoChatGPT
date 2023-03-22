import os
from distutils.util import strtobool
from trainers.trainer import Trainer

USE_GPU = strtobool(os.environ.get("USE_GPU", "true"))

# Note: the two models are not equivalent. use cpu only for debug
if USE_GPU:
    config = "config/config.yaml"
else:
    config = "config/config_cpu.yaml"

trainer = Trainer(config)

trainer.train()
