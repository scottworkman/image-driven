# Copyright © Scott Workman. 2024.

import _init_paths

import torch
import torchmetrics
from torch.utils.data import DataLoader

import utils
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
parser.add_argument('--city', default='new_york', type=str)
parser.add_argument('--dense', default=False, type=utils.boolean_string)
args = parser.parse_args()

job_dir = "{}geo_{}_{}_{}".format(args.save_dir, args.decoder, args.loss,
                                  args.method)

if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  dts = DTSDataset('test', city=args.city, dense=args.dense)
  dataset = DataLoader(dts,
                       batch_size=args.batch_size,
                       shuffle=False,
                       num_workers=4)

  ckpt_fname = glob.glob(
      f"{job_dir}/lightning_logs/version_0/checkpoints/epoch*.ckpt")[0]
  model = LMM.load_from_checkpoint(ckpt_fname, **vars(args))
  model.to(device)
  model.eval()

  # average speed of training set across time
  if args.city == "new_york":
    avg_speed_mph = 18.93318
  elif args.city == "cincinnati":
    avg_speed_mph = 31.51379
  else:
    raise ValueError
  avg_speed = avg_speed_mph * utils.MPH_TO_KMH

  # metrics
  mse = torchmetrics.MeanSquaredError().to(device)
  mae = torchmetrics.MeanAbsoluteError().to(device)
  mse_baseline = torchmetrics.MeanSquaredError().to(device)

  # helper for formatting
  f = lambda x: x.compute().cpu().numpy()

  for batch_idx, data in enumerate(dataset):
    inputs, targets = [[y.to(device) for y in x] for x in data]
    tar_road, tar_speed, tar_angle, tar_angle_bin, tar_count, tar_samp_id, tar_road_id = targets

    if args.dense:
      tar_id = tar_road_id
    else:
      tar_id = tar_samp_id

    with torch.no_grad():
      outputs = model(inputs)
      _, _, out_speed = outputs

    if args.loss == "student":
      out_speed = out_speed[:, 0, ...].unsqueeze(1)

    # aggregation
    out_speed = ops.aggregate(out_speed, tar_id)
    tar_speed = ops.aggregate(tar_speed, tar_id)

    mse.update(out_speed, tar_speed)
    mae.update(out_speed, tar_speed)
    mse_baseline.update(
        torch.zeros_like(tar_speed, device=device) + avg_speed, tar_speed)

    template = "[{:4d}/{:4d}] MSE: {:.4f} MAE: {:.4f} MSE (baseline): {:.4f}"
    print(template.format(batch_idx + 1, len(dataset), f(mse), f(mae),
                          f(mse_baseline)),
          end='\r')

  r2 = 1.0 - (f(mse) / f(mse_baseline))

  print("\033[K", end='')
  print("MSE: {:.2f} RMSE: {:.2f} MAE: {:.2f} R2: {:.2f}".format(
      f(mse), np.sqrt(f(mse)), f(mae), r2),
        flush=True)
