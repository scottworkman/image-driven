# Copyright © Scott Workman. 2024.

import _init_paths

import torch
from torch.utils.data import DataLoader

from lmm import LMM
from nets import ops
from data_extract import DTSDataset

import glob
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd

base_dir = "../dts++/labels/new_york/road/"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=3, type=int)
parser.add_argument('--method', default='multitask', type=str)
parser.add_argument('--loss', default='student', type=str)
parser.add_argument('--decoder', default='mlp', type=str)
parser.add_argument('--save_dir', default='../logs/', type=str)
args = parser.parse_args()

job_dir = "{}geo_{}_{}_{}".format(args.save_dir, args.decoder, args.loss,
                                  args.method)

if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  dts = DTSDataset(days=[1], hours=[3], full=True)
  dataset = DataLoader(dts,
                       batch_size=args.batch_size,
                       shuffle=False,
                       num_workers=12)

  ckpt_fname = glob.glob(
      f"{job_dir}/lightning_logs/version_0/checkpoints/epoch*.ckpt")[0]
  model = LMM.load_from_checkpoint(ckpt_fname, **vars(args))
  model.to(device)
  model.eval()

  results_df = pd.DataFrame(columns=[
      "osmwayid", "osmstartnode", "osmendnode", "prediction", "target"
  ])
  for batch_idx, data in enumerate(dataset):
    inputs, targets = [[y.to(device) for y in x] for x in data]
    tar_road, tar_speed, tar_angle, tar_angle_bin, tar_count, tar_id, _, tile, time = targets

    with torch.no_grad():
      outputs = model(inputs)
      _, _, out_speed = outputs

    if args.loss == "student":
      out_speed = out_speed[:, 0, ...].unsqueeze(1)

    # aggregation
    out_speeds, out_ids = ops.aggregate_list(out_speed, tar_id)
    tar_speeds, _ = ops.aggregate_list(tar_speed, tar_id)

    convert = lambda x: x.cpu().numpy().squeeze()
    make_list = lambda x: [x] if not isinstance(x[0], list) else x
    tiles = make_list(convert(tile).tolist())
    times = make_list(convert(time).tolist())
    for tile, time, speeds, targets, ids in zip(tiles, times, out_speeds,
                                                tar_speeds, out_ids):
      dow, hour = time
      speeds = convert(speeds)
      targets = convert(targets)
      ids = convert(ids)
      targets[targets == 0] = np.nan

      # load geometry file for this tile
      z, x, y = tile
      df = gpd.read_file("{}{}/{}/{}.geojson".format(base_dir, z, x, y))
      df = df[['osmwayid', 'osmstartnode', 'osmendnode']].iloc[ids - 1]

      df['prediction'] = speeds
      df['target'] = targets

      results_df = pd.concat([results_df, df], ignore_index=True)

    template = "[{:4d}/{:4d}] "
    print(template.format(batch_idx + 1, len(dataset), end='\r'))

  results_df = results_df.astype({
      "osmwayid": str,
      "osmstartnode": str,
      "osmendnode": str,
      "prediction": float,
      "target": float
  })

  results_df = results_df.groupby(['osmwayid', 'osmstartnode',
                                   'osmendnode'])[['prediction', 'target'
                                                  ]].mean().reset_index()
  results_df.to_pickle('results.pkl')
  print(results_df)
  print(len(results_df))
