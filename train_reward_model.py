import tiktoken
import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from tensordict import MemmapTensor
from tensordict.prototype import tensorclass
from tqdm import tqdm

from trainers.reward_trainer import RewardModelTrainer

# with inspiration from CarperAI's trlx library


def load_config():
    with open("config/config_reward.yaml") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        # nested dictionary structure
        config = {}
        for v in conf.values():
            for k2, v2 in v.items():
                config[k2] = v2

    return config


@tensorclass
class PairwiseDataset:
    chosens: torch.Tensor
    rejecteds: torch.Tensor

    @classmethod
    def from_dataset(cls, dataset, max_length):
        data = cls(
            chosens=MemmapTensor(len(dataset), max_length, dtype=torch.int16),
            rejecteds=MemmapTensor(len(dataset), max_length, dtype=torch.int16),
            batch_size=[len(dataset)],
        )
        enc = tiktoken.get_encoding("gpt2")
        i = 0

        for sample in tqdm(dataset, total=len(dataset)):
            prompt = sample["prompt"]
            chosen = sample["chosen"]
            rejected = sample["rejected"]

            if (
                chosen == rejected
                or len(chosen.split()) < 5
                or len(rejected.split()) < 5
            ):
                continue

            chosen = "\n".join([prompt, chosen])
            rejected = "\n".join([prompt, rejected])

            chosen = enc.encode(
                "<|startoftext|>" + chosen + "<|endoftext|>", allowed_special="all"
            )[-max_length:]
            rejected = enc.encode(
                "<|startoftext|>" + rejected + "<|endoftext|>", allowed_special="all"
            )[-max_length:]

            data[i] = cls(chosens=chosen, rejecteds=rejected, batch_size=[])
            i += 1

        # TODO: we might have skipped some samples and hence not fully populated the
        # memmap, so we need to cut off the unused rows to avoid training with them
        return data


class Collate(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

    def __call__(self, batch):
        if self.device.type == "cuda":
            out = batch.apply(lambda x: x.as_tensor()).pin_memory()
        else:
            out = batch.apply(lambda x: x.as_tensor())

        return out.to(self.device)


if __name__ == "__main__":
    print("Loading config")
    config = load_config()
    print(config, end="\n\n")

    # Make pairwise datasets for training
    print("Creating pairwise datasets")
    data_path = "CarperAI/openai_summarize_comparisons"
    train_dataset = PairwiseDataset.from_dataset(
        load_dataset(data_path, split="train"), max_length=config["block_size"]
    )
    val_dataset = PairwiseDataset.from_dataset(
        load_dataset(data_path, split="test"), max_length=config["block_size"]
    )

    print("Creating trainer")
    trainer = RewardModelTrainer(
        config, train_dataset, val_dataset, collate_fn=Collate()
    )

    print("Training")
    trainer.train()
