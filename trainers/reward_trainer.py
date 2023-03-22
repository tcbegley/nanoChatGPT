import os
import time

import numpy as np
import tiktoken
import torch
import wandb
from torch.distributed import destroy_process_group
from torch.utils.data import DataLoader

from model import RLHF
from trainers.trainer import Trainer

from tensordict.prototype import tensorclass


# This one for reward models similar to InstructGPT paper (rewards based on comparisons)
class RewardModelTrainer(Trainer):
    def __init__(self, config, train_data, val_data, collate_fn):
        super().__init__(config)
        self.enc = tiktoken.get_encoding("gpt2")
        self.mode = "reward"
        train_dataloader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn
        )
        val_dataloader = DataLoader(
            val_data, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn
        )
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def get_batch(self, split):
        dataloader = self.train_dataloader if split == "train" else self.val_dataloader
        return next(iter(dataloader))

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self, model, ctx):
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                batch = self.get_batch(split)
                with ctx:
                    reward_chosen = model(batch.chosens)
                    reward_rejected = model(batch.rejecteds)
                    loss = -torch.log(
                        torch.sigmoid(reward_chosen - reward_rejected)
                    ).mean()
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def evaluate(self, model, ctx):
        losses = self.estimate_loss(model, ctx)
        print(
            f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

        if self.wandb_log:
            wandb.log(
                {
                    "iter": self.iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": self.lr,
                    "mfu": self.running_mfu * 100,  # convert to percentage
                }
            )
        if losses["val"] < self.best_val_loss or self.always_save_checkpoint:
            self.best_val_loss = losses["val"]
            raw_model = model.module if self.ddp else model
            if self.iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "model_args": self.model_args,
                    "iter_num": self.iter_num,
                    "best_val_loss": self.best_val_loss,
                    "config": self.config,
                }
                print(f"saving checkpoint to {self.config['out_dir_multihead']}")
                torch.save(
                    checkpoint,
                    os.path.join(self.config["out_dir_multihead"], "ckpt.pt"),
                )

    def train(self):
        # set up distributed training
        self.setup_ddp()

        ctx, meta_vocab_size = self.setup()

        # model init

        model = self.init_model()
        model = RLHF(model, self.mode)
        print("Config of model: ", model.config)

        if self.config["init_multihead_from"] == "scratch":
            print("initializing multihead from scratch")
        else:
            if self.config["init_multihead_from"] == "resume":
                print(f"Resuming training from {self.config['out_dir_multihead']}")
                # resume training from a checkpoint.
                ckpt_path = os.path.join(self.config["out_dir_multihead"], "ckpt.pt")
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                state_dict = checkpoint["model"]
                # fix the keys of the state dictionary :(
                # honestly no idea how checkpoints sometimes get this prefix, have to debug more
                unwanted_prefix = "_orig_mod."
                for k, v in list(state_dict.items()):
                    if k.startswith(unwanted_prefix):
                        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
                model.load_state_dict(state_dict)

        model.to(self.device)

        # self.optimizer = torch.optim.AdamW(model.model.reward_head.parameters(), lr=1e-3)
        self.optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-4)

        model = self.setup_model(model)

        # logging
        if self.wandb_log and self.master_process:
            wandb.init(
                project=self.wandb_project, name=self.wandb_run_name, config=self.config
            )

        # training loop
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        self.running_mfu = -1.0
        loss = None
        while True:

            # determine and set the learning rate for this iteration
            lr = self.get_lr(self.iter_num) if self.decay_lr else self.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # # every once in a while evaluate the loss on train and val sets
            if self.iter_num % self.eval_interval == 0 and self.master_process:
                self.evaluate(model, ctx)

            if self.iter_num == 0 and self.eval_only:
                break

            # sample a batch of data
            batch = self.get_batch("train")

            # evaluate the loss
            reward_chosen = model(batch.chosens)
            reward_rejected = model(batch.rejecteds)
            loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # timing and logging
            t1 = time.time()
            # dt = t1 - t0
            t0 = t1
            self.iter_num += 1
            local_iter_num += 1

            # termination conditions
            if self.iter_num > self.max_iters:
                break

        if self.ddp:
            destroy_process_group()


@tensorclass
class RewardData:
    prompt: torch.Tensor
    reward: torch.Tensor


# This one is for reward models which output a probability of reward directly from a
# given text (no comparison)
class ProbRewardModelTrainer(Trainer):
    def __init__(self, config, discrete_reward=False):
        super().__init__(config)
        self.enc = tiktoken.get_encoding("gpt2")
        self.mode = "reward"
        self.discrete_reward = discrete_reward

    def get_batch(self, split):
        batch = super().get_batch(split)
        return RewardData(
            prompt=batch.prompt,
            reward=torch.stack(
                [self.reward(row) for row in batch.target.unbind(0)], dim=0
            ),
            batch_size=[len(batch)],
        )

    def reward(self, sequence, t="and"):
        if t in self.enc.decode(sequence.tolist()):
            # print('hello')
            return torch.tensor([0.0, 1.0])
        else:
            return torch.tensor([1.0, 0.0])

    def evaluate(self, model, ctx, X, lr):
        losses = self.estimate_loss(model, ctx)
        print(
            f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

        text = self.enc.decode(X[self.iter_num % self.eval_interval].tolist())

        try:
            reward_probs, _ = model(X[self.iter_num % self.eval_interval].unsqueeze(0))
            actual_reward_probs = self.reward(X[self.iter_num % self.eval_interval])[1]

            print(
                "input: ",
                text[:30],
                f"expect {actual_reward_probs}, reward: {reward_probs[0][-1]} \n",
            )
        except:
            pass

        # test_text = text[:4] + 'z' + text[4 + 1:-1]
        test_text = list(text)
        test_text[3] = " "
        test_text[4] = "a"
        test_text[5] = "n"
        test_text[6] = "d"
        test_text[7] = " "
        test_text = "".join(test_text)
        try:
            test_text_enc = torch.tensor(
                self.enc.encode(test_text)[: self.block_size]
            ).unsqueeze(0)
            test_reward_probs, _ = model(test_text_enc.to(self.device))
            actual_reward_probs = self.reward(test_text_enc[0].to(self.device))[1]

            print(
                "input: ",
                test_text[:30],
                f"expect {actual_reward_probs}, reward: {test_reward_probs[0][-1]} \n",
            )
        except:
            pass

        if self.wandb_log:
            wandb.log(
                {
                    "iter": self.iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    # "mfu": self.running_mfu*100, # convert to percentage
                }
            )
        if losses["val"] < self.best_val_loss or self.always_save_checkpoint:
            self.best_val_loss = losses["val"]
            raw_model = model.module if self.ddp else model
            if self.iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "model_args": self.model_args,
                    "iter_num": self.iter_num,
                    "best_val_loss": self.best_val_loss,
                    "config": self.config,
                }
                print(f"saving checkpoint to {self.config['out_dir_multihead']}")
                torch.save(
                    checkpoint,
                    os.path.join(self.config["out_dir_multihead"], "ckpt.pt"),
                )

    def train(self):
        # set up distributed training
        self.setup_ddp()

        ctx, meta_vocab_size = self.setup()

        # model init

        if self.master_process:
            os.makedirs(self.config["out_dir_multihead"], exist_ok=True)

        model = self.init_model()
        model = RLHF(model, self.mode, discrete_reward=self.discrete_reward)

        if self.config["init_multihead_from"] == "scratch":
            print("initializing multihead from scratch")
        else:
            if self.config["init_multihead_from"] == "resume":
                print(f"Resuming training from {self.config['out_dir_multihead']}")
                # resume training from a checkpoint.
                ckpt_path = os.path.join(self.config["out_dir_multihead"], "ckpt.pt")
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                state_dict = checkpoint["model"]
                # fix the keys of the state dictionary :(
                # honestly no idea how checkpoints sometimes get this prefix, have to debug more
                unwanted_prefix = "_orig_mod."
                for k, v in list(state_dict.items()):
                    if k.startswith(unwanted_prefix):
                        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
                model.load_state_dict(state_dict)

        model.to(self.device)

        # self.optimizer = torch.optim.AdamW(model.model.reward_head.parameters(), lr=1e-3)
        self.optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)

        model = self.setup_model(model)

        # logging
        if self.wandb_log and self.master_process:
            wandb.init(
                project=self.wandb_project, name=self.wandb_run_name, config=self.config
            )

        # training loop
        batch = self.get_batch("train")  # fetch the very first batch
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        self.running_mfu = -1.0
        while True:

            # determine and set the learning rate for this iteration
            lr = self.get_lr(self.iter_num) if self.decay_lr else self.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # every once in a while evaluate the loss on train and val sets
            if self.iter_num % self.eval_interval == 0 and self.master_process:
                self.evaluate(model, ctx, batch.prompt, lr)

            if self.iter_num == 0 and self.eval_only:
                break

            # sample a batch of data
            batch = self.get_batch("train")

            # evaluate the loss
            logits, loss = model(batch.prompt, batch.reward)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # timing and logging
            t1 = time.time()
            # dt = t1 - t0
            t0 = t1
            self.iter_num += 1
            local_iter_num += 1

            # termination conditions
            if self.iter_num > self.max_iters:
                break

        if self.ddp:
            destroy_process_group()

    # TODO: this won't be necessary once model has been wrapped with TensorDictModule
    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self, model, ctx):
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                batch = self.get_batch(split)
                with ctx:
                    logits, loss = model(batch.prompt, batch.reward)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
