# Copyright © Scott Workman. 2024.

import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import utils
from lmm import LMM

import argparse

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", default=2, type=int)
  parser.add_argument('--batch_size', default=2, type=int)
  parser.add_argument('--save_dir', default='./logs/', type=str)
  parser.add_argument('--learning_rate', default=1e-4, type=float)
  parser.add_argument('--method', default='multitask', type=str)
  parser.add_argument('--loss', default='student', type=str)
  parser.add_argument('--decoder', default='mlp', type=str)
  parser.add_argument('--adapt', default=False, type=utils.boolean_string)
  parser.add_argument('--pretrain', default=None, type=str)
  parser.add_argument('--resume', default=None, type=str)
  args = parser.parse_args()

  L.seed_everything(args.seed, workers=True)

  if args.pretrain != None:
    model = LMM.load_from_checkpoint(args.pretrain, **vars(args), strict=False)
  else:
    model = LMM(**vars(args))

  if args.adapt == True:
    for name, param in model.named_parameters():
      if "context" not in name:
        param.requires_grad = False
      print(name, param.requires_grad)

  checkpoint_callback = ModelCheckpoint(monitor="val_mse",
                                        mode="min",
                                        save_last=True,
                                        every_n_train_steps=1000)
  lr_monitor_callback = LearningRateMonitor(logging_interval='step')

  job_dir = "{}geo_{}_{}_{}".format(args.save_dir, args.decoder, args.loss,
                                    args.method)
  logger = TensorBoardLogger(job_dir)

  trainer = L.Trainer(accelerator="gpu",
                      devices=1,
                      max_epochs=50,
                      logger=logger,
                      num_sanity_val_steps=1,
                      val_check_interval=1000,
                      default_root_dir=job_dir,
                      callbacks=[checkpoint_callback, lr_monitor_callback],
                      profiler="simple",
                      precision="16-mixed",
                      accumulate_grad_batches=8)
  trainer.fit(model, ckpt_path=args.resume)
