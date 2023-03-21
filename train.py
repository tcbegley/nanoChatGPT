from trainers.trainer import Trainer

USE_GPU = False

# Note: the two models are not equivalent. use cpu only for debug
if USE_GPU:
    config = "config/config.yaml"
else:
    config = "config/config_cpu.yaml"

trainer = Trainer(config)

trainer.train()
