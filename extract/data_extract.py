# Copyright © Scott Workman. 2024.

import _init_paths

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import utils

import os
import sparse
import numpy as np
import pandas as pd
from pathlib import Path


class DTSDataset(Dataset):
  """Dynamic Traffic Speeds Dataset."""

  def __init__(self,
               days=[1],
               hours=[11],
               full=False,
               city="new_york",
               data_dir=None):
    if data_dir is None:
      data_dir = f"{Path(os.path.abspath(__file__)).parent.parent}/dts++/"
    self.data_dir = data_dir
    self.full = full
    self.city = city

    dfs = []
    for mode in ["train", "test", "val"]:
      dfs.append(self.__load_df(mode))
    self.df_raw = pd.concat(dfs, ignore_index=True)

    tmp = []
    for index, row in self.df_raw.iterrows():
      for day in days:
        for hour in hours:
          row['time'] = [day, hour]
          tmp.append(row.copy())
    self.df = pd.concat(tmp, axis=1, ignore_index=True).T

  def __load_df(self, mode):
    df = pd.read_csv("{}{}_{}.txt".format(self.data_dir, mode, self.city),
                     sep=',',
                     header=None,
                     usecols=[0, 1, 2])
    df.columns = ['image', 'label_mask', 'label_speed']
    df['tile'] = df.apply(
        lambda row: [int(x) for x in row['image'][:-4].strip().split('/')[-3:]],
        axis=1)
    absolute = lambda x: os.path.join(self.data_dir, str(x))
    df['image'] = df['image'].apply(absolute)
    df['label_mask'] = df['label_mask'].apply(absolute)
    df['label_speed'] = df['label_speed'].apply(absolute)
    return df

  def __getitem__(self, idx):
    image = utils.preprocess(utils.imread(self.df['image'][idx]))
    mask = sparse.load_npz(self.df['label_mask'][idx]).todense()
    df_speed = pd.read_csv(self.df['label_speed'][idx])

    sliced = [x.squeeze() for x in np.array_split(mask, 6, axis=2)]
    l_road, l_road_id, _, l_samp_id, l_angle, l_angle_bin = sliced
    l_angle_bin = l_angle_bin - 1  # invalid pixels are now -1, 16 bins

    # parameterize location
    location = utils.compute_location(self.df['tile'][idx],
                                      out_shape=image.shape,
                                      city=self.city)

    # generate speed mask
    time = np.asarray(self.df['time'][idx])
    result = df_speed.query('dow == {} and hour == {}'.format(time[0], time[1]))
    l_speed, _ = utils.generate_speed_mask(result['id'],
                                           result['speed_mph_mean'], l_samp_id)
    if self.full:
      l_count, _ = utils.generate_speed_mask(result['id'], result['count'],
                                             l_samp_id)
    else:
      l_count, l_samp_id = utils.generate_speed_mask(result['id'],
                                                     result['count'], l_samp_id)

    # use updated ids to zero out invalid spots
    l_angle[l_samp_id == 0] = 0

    # normalize time (dow, hour)
    time_param = time.astype(float)
    time_param[0] = ((time_param[0] + 1) / 7 * 2) - 1
    time_param[1] = ((time_param[1] + 1) / 24 * 2) - 1

    t_image = TF.to_tensor(image).float()
    t_loc = torch.from_numpy(location).float()
    t_time = torch.from_numpy(time_param).float()
    t_road = torch.from_numpy(l_road).long()
    t_speed = torch.from_numpy(l_speed[np.newaxis, :, :]).float()
    t_angle = torch.from_numpy(l_angle[np.newaxis, :, :]).float()
    t_angle_bin = torch.from_numpy(l_angle_bin).long()
    t_count = torch.from_numpy(l_count[np.newaxis, :, :]).float()
    t_samp_id = torch.from_numpy(l_samp_id[np.newaxis, :, :]).long()
    t_road_id = torch.from_numpy(l_road_id[np.newaxis, :, :]).long()
    t_tile = torch.from_numpy(np.asarray(self.df['tile'][idx])).long()
    t_time_raw = torch.from_numpy(time).long()

    inputs = [t_image, t_loc, t_time]
    targets = [
        t_road, t_speed, t_angle, t_angle_bin, t_count, t_samp_id, t_road_id,
        t_tile, t_time_raw
    ]

    return inputs, targets

  def __len__(self):
    return len(self.df)


if __name__ == "__main__":
  take = 8

  dataset = DTSDataset()
  print(len(dataset))

  for n, data in zip(range(take), dataset):
    inputs, targets = [[y.numpy() for y in x] for x in data]
    im, loc, time = inputs
    l_road, l_speed, l_angle, l_angle_bin, _, l_samp_id, _, _, _ = targets
    print(
        n, loc.shape, time, im.shape, l_road.shape, np.quantile(im, [0, 1]),
        np.unique(l_road), np.quantile(l_angle, [0, 1]), np.unique(l_angle_bin),
        np.quantile(l_speed[l_speed > 0], [0, 1])
        if np.count_nonzero(l_speed) > 0 else [0, 0])
