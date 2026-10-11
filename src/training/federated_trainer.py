# src/training/federated_trainer.py
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd


class FederatedTrainer:
    def __init__(
        self,
        model,
        options,
        train_dataset,
        valid_dataset=None,
        seed=42,
        save_dir=None,
        model_name=None,
    ):
        self.model = model.to(options["device"])
        self.device = options["device"]

        self.batch_size = options["batch_size"]
        self.accumulation_step = options["accumulation_step"]
        self.max_epoch = options["max_epoch"]

        self.lr_step = options["lr_step"]
        self.lr_decay_ratio = options["lr_decay_ratio"]

        self.seed = seed

        self.save_dir = save_dir
        self.model_name = model_name or options.get("model_name", "model")
        self.best_loss = float("inf")

        self.log = {
            "train": {key: [] for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: [] for key in ["epoch", "lr", "time", "loss"]},
        }

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        g = torch.Generator()
        g.manual_seed(self.seed)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            generator=g,
        )

        self.valid_loader = None
        if valid_dataset is not None:
            self.valid_loader = DataLoader(
                valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
            )

        if options["optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=options["lr"],
                momentum=0.9,
            )
        elif options["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=options["lr"],
                betas=(0.9, 0.999),
            )
        else:
            raise NotImplementedError

    def save_checkpoint(self, epoch, save_optimizer=True, suffix=""):
        if self.save_dir is None:
            return

        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "best_loss": self.best_loss,
            "log": self.log,
        }
        if save_optimizer:
            checkpoint["optimizer"] = self.optimizer.state_dict()

        ckpt_path = os.path.join(
            self.save_dir,
            f"{self.model_name}_{suffix}.pth",
        )
        torch.save(checkpoint, ckpt_path)
        print(f"Save model checkpoint at {ckpt_path}")

    def save_log(self):
        if self.save_dir is None:
            return
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(
                    os.path.join(self.save_dir, f"{key}_log.csv"),
                    index=False,
                )
            print("Log saved")
        except Exception:
            print("Failed to save logs")

    def train_one_epoch(self, epoch):
        self.log["train"]["epoch"].append(epoch)
        self.log["train"]["lr"].append(self.optimizer.state_dict()["param_groups"][0]["lr"])
        self.log["train"]["time"].append(time.strftime("%H:%M:%S"))

        self.model.train()
        self.optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()

        total_losses = 0.0
        num_batch = len(self.train_loader)

        tbar = tqdm(self.train_loader, desc=f"Client train epoch {epoch}", leave=False)
        for i, (log, label) in enumerate(tbar):
            features = [value.clone().detach().to(self.device) for value in log.values()]
            output = self.model(features=features, device=self.device)
            loss = criterion(output, label.to(self.device))
            total_losses += float(loss)

            loss = loss / self.accumulation_step
            loss.backward()

            if (i + 1) % self.accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            tbar.set_description(f"Train loss: {total_losses / (i + 1):.5f}")

        avg_loss = total_losses / max(num_batch, 1)
        self.log["train"]["loss"].append(avg_loss)
        return {"train_loss": avg_loss}

    def evaluate(self, epoch=None):
        if self.valid_loader is None:
            return {"val_loss": 0.0, "val_examples": 0}

        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_losses = 0.0
        num_batch = len(self.valid_loader)

        if epoch is not None:
            self.log["valid"]["epoch"].append(epoch)
            self.log["valid"]["lr"].append(self.optimizer.state_dict()["param_groups"][0]["lr"])
            self.log["valid"]["time"].append(time.strftime("%H:%M:%S"))

        with torch.no_grad():
            for log, label in self.valid_loader:
                features = [value.clone().detach().to(self.device) for value in log.values()]
                output = self.model(features=features, device=self.device)
                loss = criterion(output, label.to(self.device))
                total_losses += float(loss)

        val_loss = total_losses / max(num_batch, 1)

        if epoch is not None:
            self.log["valid"]["loss"].append(val_loss)

        return {
            "val_loss": val_loss,
            "val_examples": len(self.valid_loader.dataset),
        }

    def fit(self):
        metrics = {}

        for epoch in range(self.max_epoch):
            if epoch == 0:
                self.optimizer.param_groups[0]["lr"] /= 32
            if epoch in [1, 2, 3, 4, 5]:
                self.optimizer.param_groups[0]["lr"] *= 2
            if epoch in self.lr_step:
                self.optimizer.param_groups[0]["lr"] *= self.lr_decay_ratio

            train_metrics = self.train_one_epoch(epoch)
            metrics.update(train_metrics)

            if self.valid_loader is not None:
                eval_metrics = self.evaluate(epoch=epoch)
                metrics.update(eval_metrics)

                if eval_metrics["val_loss"] < self.best_loss:
                    self.best_loss = eval_metrics["val_loss"]
                    self.save_checkpoint(
                        epoch=epoch,
                        save_optimizer=False,
                        suffix="bestloss",
                    )

            self.save_checkpoint(
                epoch=epoch,
                save_optimizer=True,
                suffix="last",
            )
            self.save_log()

        return metrics