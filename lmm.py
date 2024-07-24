# Copyright © Scott Workman. 2024.

import torch
import torchmetrics
import lightning as L
from torch.utils.data import DataLoader

import models
from data import DTSDataset


class LMM(L.LightningModule):

  def __init__(self, **kwargs):
    super().__init__()

    self.save_hyperparameters()

    self.net, self.criterion = models.build_model(self.hparams)

    self.train_f1 = torchmetrics.F1Score(task="binary")
    self.train_acc = torchmetrics.Accuracy(task="multiclass",
                                           num_classes=16,
                                           ignore_index=-1)
    self.train_mse = torchmetrics.MeanSquaredError()
    self.val_f1 = torchmetrics.F1Score(task="binary")
    self.val_acc = torchmetrics.Accuracy(task="multiclass",
                                         num_classes=16,
                                         ignore_index=-1)
    self.val_mse = torchmetrics.MeanSquaredError()

  def forward(self, x):
    return self.net(x)

  def _compute_metrics(self, data, mode):
    f1 = eval(f"self.{mode}_f1")
    acc = eval(f"self.{mode}_acc")
    mse = eval(f"self.{mode}_mse")

    if "road" in data.keys():
      out_road, tar_road = data["road"]
      f1(out_road, tar_road)
    else:
      f1(torch.zeros(1, dtype=int), torch.zeros(1, dtype=int))

    if "angle" in data.keys():
      out_angle, tar_angle = data["angle"]
      acc(out_angle, tar_angle)
    else:
      acc(torch.zeros(1, dtype=int), torch.zeros(1, dtype=int))

    out_speed, tar_speed = data["speed"]
    mse(out_speed, tar_speed)

  def training_step(self, batch):
    inputs, targets = batch
    outputs = self(inputs)
    loss, metric_data = self.criterion(outputs, targets)

    self._compute_metrics(metric_data, "train")

    self.log_dict(
        {
            "train_loss": loss["total"],
            "train_loss_road": loss["road"],
            "train_loss_angle": loss["angle"],
            "train_loss_speed": loss["speed"],
            "train_f1_score": self.train_f1,
            "train_acc": self.train_acc,
            "train_mse": self.train_mse
        },
        prog_bar=True,
        on_step=True,
        on_epoch=True,
        logger=True,
        sync_dist=True)

    return loss["total"]

  def validation_step(self, batch, batch_idx):
    inputs, targets = batch
    outputs = self(inputs)
    loss, metric_data = self.criterion(outputs, targets)

    self._compute_metrics(metric_data, "val")

    self.log_dict(
        {
            "val_loss": loss["total"],
            "val_loss_road": loss["road"],
            "val_loss_angle": loss["angle"],
            "val_loss_speed": loss["speed"],
            "val_f1_score": self.val_f1,
            "val_acc": self.val_acc,
            "val_mse": self.val_mse
        },
        prog_bar=True,
        on_step=False,
        on_epoch=True,
        logger=True,
        sync_dist=True)

    return loss["total"]

  def configure_optimizers(self):
    opt = torch.optim.Adam(self.net.parameters(), lr=self.hparams.learning_rate)
    return opt

  def train_dataloader(self):
    city = "cincinnati" if self.hparams.adapt else "new_york"
    return DataLoader(DTSDataset('train', city=city),
                      batch_size=self.hparams.batch_size,
                      shuffle=True,
                      num_workers=6,
                      pin_memory=True)

  def val_dataloader(self):
    city = "cincinnati" if self.hparams.adapt else "new_york"
    return DataLoader(DTSDataset('val', city=city),
                      batch_size=self.hparams.batch_size,
                      shuffle=False,
                      num_workers=6,
                      pin_memory=True)

  def on_load_checkpoint(self, checkpoint: dict) -> None:
    state_dict = checkpoint["state_dict"]
    model_state_dict = self.state_dict()
    is_changed = False
    for k in state_dict:
      if k in model_state_dict:
        if state_dict[k].shape != model_state_dict[k].shape:
          print(f"Skip loading parameter: {k}, "
                f"required shape: {model_state_dict[k].shape}, "
                f"loaded shape: {state_dict[k].shape}")
          state_dict[k] = model_state_dict[k]
          is_changed = True
      else:
        print(f"Dropping parameter {k}")
        is_changed = True

    if is_changed:
      checkpoint.pop("optimizer_states", None)


if __name__ == "__main__":
  m = LMM(
      **{
          "method": "multitask",
          "loss": "huber",
          "decoder": "mlp",
          "aggregate": True,
          "batch_size": 1,
          "learning_rate": 1e-3
      })
  print(m)
