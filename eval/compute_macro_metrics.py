# Copyright © Scott Workman. 2024.

import _init_paths

import torch
from torch.utils.data import DataLoader

import utils
from lmm import LMM
from data_eval import DTSDataset

import glob
import argparse
import numpy as np
import pandas as pd

# average speed of training set across time (new_york)
avg_speed = 18.93318 * utils.MPH_TO_KMH

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=3, type=int)
parser.add_argument('--method', default='multitask', type=str)
parser.add_argument('--loss', default='student', type=str)
parser.add_argument('--decoder', default='mlp', type=str)
parser.add_argument('--save_dir', default='../logs/', type=str)
args = parser.parse_args()

job_dir = "{}geo_{}_{}_{}".format(args.save_dir, args.decoder, args.loss,
                                  args.method)


def aggregate(predictions, target_ids):
  b, _, _, _ = predictions.shape
  device = predictions.device

  ids = []
  values = []

  for idx in range(b):
    local_id = target_ids[idx, ...].squeeze()
    local_pred = predictions[idx, ...].squeeze()

    num_ids = torch.unique(local_id).max() + 1
    index = torch.flatten(local_id, start_dim=0).unsqueeze(0)
    local_pred = torch.flatten(local_pred, start_dim=0).unsqueeze(0)

    # pixel count per road segment
    num_pixels = torch.zeros(1, num_ids, device=device)
    num_pixels = num_pixels.scatter_add(1, index, torch.ones_like(local_pred))

    # aggregate predictions
    pred = torch.zeros(1, num_ids, device=device)
    pred = pred.scatter_add(1, index, local_pred)
    pred = pred / num_pixels

    idx = torch.nonzero(num_pixels).split(1, dim=1)[1][1:]

    ids.append(idx)
    values.append(pred[:, idx])

  return values, ids


if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  dts = DTSDataset('test', days=[1, 6], hours=[3, 7, 11, 16, 19, 23])
  dataset = DataLoader(dts,
                       batch_size=args.batch_size,
                       shuffle=False,
                       num_workers=4)

  ckpt_fname = glob.glob(
      f"{job_dir}/lightning_logs/version_0/checkpoints/epoch*.ckpt")[0]
  model = LMM.load_from_checkpoint(ckpt_fname, **vars(args))
  model.to(device)
  model.eval()

  tmp = []
  for batch_idx, data in enumerate(dataset):
    inputs, targets = [[y.to(device) for y in x] for x in data]
    tar_road, tar_speed, tar_angle, tar_angle_bin, tar_count, tar_id, _, tile, time = targets

    with torch.no_grad():
      outputs = model(inputs)
      _, _, out_speed = outputs

    if args.loss == "student":
      out_speed = out_speed[:, 0, ...].unsqueeze(1)

    # aggregation
    pp = lambda x: x.squeeze().cpu().numpy().tolist()
    tiles = pp(tile)
    times = pp(time)
    if inputs[0].shape[0] == 1:
      tiles = [tiles]
      times = [times]
    out_speed, ids = [[pp(y) for y in x] for x in aggregate(out_speed, tar_id)]
    tar_speed, _ = [[pp(y) for y in x] for x in aggregate(tar_speed, tar_id)]

    for tile, time, os, ts, ids in zip(tiles, times, out_speed, tar_speed, ids):
      dow, hour = time
      if not isinstance(os, list):
        os = [os]
        ts = [ts]
        ids = [ids]
      for o, t, i in zip(os, ts, ids):
        tmp.append(
            pd.Series(
                data={
                    'tile': tile,
                    'id': i,
                    'dow': dow,
                    'hour': hour,
                    'prediction': o,
                    'target': t,
                    'mse': np.square(np.subtract(t, o)),
                    'mae': np.abs(np.subtract(t, o)),
                    'mse_baseline': np.square(np.subtract(t, avg_speed)),
                }))

    template = "[{:4d}/{:4d}] "
    print(template.format(batch_idx + 1, len(dataset), end='\r'))

  df = pd.concat(tmp, axis=1).T
  df = df.astype({
      'prediction': 'float',
      'target': 'float',
      'mse': 'float',
      'mae': 'float',
      'mse_baseline': 'float'
  })
  df.to_pickle("macro_results.pkl")

  # grouped
  print("Grouped:")
  df_g = df[['dow', 'hour', 'mse', 'mae',
             'mse_baseline']].groupby(['dow', 'hour']).mean()
  df_g['rmse'] = np.sqrt(df_g['mse'])
  df_g['R2'] = 1.0 - (df_g['mse'] / df_g['mse_baseline'])
  print(df_g)
  print("rmse: ", np.sqrt(df_g['mse'].mean()))
  print("mae: ", df_g['mae'].mean())
  print("R2: ", df_g['R2'].mean())
  print()

  # overall
  print("Overall:")
  df_a = df[['mse', 'mae', 'mse_baseline']].mean()
  df_a['rmse'] = np.sqrt(df_a['mse'])
  df_a['R2'] = 1.0 - (df_a['mse'] / df_a['mse_baseline'])
  print(df_a)
