# Copyright © Scott Workman. 2024.

import _init_paths

import torch
import torchmetrics
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lmm import LMM
from nets import ops
from data import DTSDataset

import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--method', default='multitask', type=str)
parser.add_argument('--loss', default='student', type=str)
parser.add_argument('--decoder', default='mlp', type=str)
parser.add_argument('--save_dir', default='../logs/', type=str)
args = parser.parse_args()

job_dir = "{}geo_{}_{}_{}".format(args.save_dir, args.decoder, args.loss,
                                  args.method)

if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  dts = DTSDataset('test')
  dataset = DataLoader(dts,
                       batch_size=args.batch_size,
                       shuffle=False,
                       num_workers=4)

  ckpt_fname = glob.glob(
      f"{job_dir}/lightning_logs/version_0/checkpoints/epoch*.ckpt")[0]
  model = LMM.load_from_checkpoint(ckpt_fname, **vars(args))
  model.to(device)
  model.eval()

  # metrics
  f1 = torchmetrics.F1Score(task="binary").to(device)
  acc = torchmetrics.Accuracy(task="multiclass",
                              num_classes=16,
                              average="micro",
                              ignore_index=-1).to(device)
  mse = torchmetrics.MeanSquaredError().to(device)

  # helper for formatting
  f = lambda x: x.compute().cpu().numpy()

  for batch_idx, data in enumerate(dataset):
    inputs, targets = [[y.to(device) for y in x] for x in data]
    tar_road, tar_speed, tar_angle, tar_angle_bin, tar_count, tar_id, _ = targets

    with torch.no_grad():
      outputs = model(inputs)
      out_road, out_angle, out_speed = outputs
      out_angle = torch.argmax(F.softmax(out_angle, dim=1), dim=1)

    if args.loss == "student":
      out_speed = out_speed[:, 0, ...].unsqueeze(1)

    f1(out_road.squeeze(1), tar_road)
    acc(out_angle, tar_angle_bin)

    # aggregation
    out_speed = ops.aggregate(out_speed, tar_id)
    tar_speed = ops.aggregate(tar_speed, tar_id)

    mse(out_speed, tar_speed)

    template = "[{:4d}/{:4d}] F1: {:.4f} Acc: {:.4f} RMSE: {:.4f}"
    print(template.format(batch_idx + 1, len(dataset), f(f1),
                          100 * np.mean(f(acc)), np.sqrt(f(mse))),
          end='\r')

  print("\033[K", end='')
  print("F1: {:.4f} Acc: {:.4f} RMSE: {:.4f}".format(f(f1),
                                                     100 * np.mean(f(acc)),
                                                     np.sqrt(f(mse))),
        flush=True)
