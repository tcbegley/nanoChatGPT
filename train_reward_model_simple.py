import os
from distutils.util import strtobool

from trainers.reward_trainer import ProbRewardModelTrainer

USE_GPU = strtobool(os.environ.get("USE_GPU", "true"))

if USE_GPU:
    config = "config/config_reward.yaml"
else:
    config = "config/config_reward_cpu.yaml"

trainer = ProbRewardModelTrainer(config, discrete_reward=True)
trainer.train()
