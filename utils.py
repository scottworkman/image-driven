# Copyright © Scott Workman. 2024.

import pyproj
import imageio
import mercantile
import numpy as np

MPH_TO_KMH = 1.60934


def imread(path):
  image = imageio.imread(path)
  return image


def preprocess(image):
  image = image / 255.
  return image


def get_bbox(z, x, y):
  tile = mercantile.Tile(x, y, z)
  bounds = mercantile.xy_bounds(tile)
  return bounds


def convert_bbox(bounds, transformer):
  left, bottom, right, top = bounds
  result = transformer.transform([left, right], [bottom, top])
  left_new, right_new = result[0]
  bottom_new, top_new = result[1]
  return [left_new, bottom_new, right_new, top_new]


def get_transform(city):

  if city == "new_york":
    x_min, x_max = [278717.8020024174, 325189.23155259725]
    y_min, y_max = [37184.57555182383, 82939.78847541365]

    # Web Mercator to NAD83 / New York Long Island
    transformer = pyproj.Transformer.from_crs("EPSG:3857",
                                              "EPSG:32118",
                                              always_xy=True)
  elif city == "cincinnati":
    x_min, x_max = [366806.0002835511, 465075.9545187966]
    y_min, y_max = [69310.90364514636, 176893.4001424716]

    # Web Mercator to NAD83 / Ohio South
    transformer = pyproj.Transformer.from_crs("EPSG:3857",
                                              "EPSG:32123",
                                              always_xy=True)
  else:
    raise ValueError

  return x_min, x_max, y_min, y_max, transformer


def compute_location(tile, out_shape, city):
  x_min, x_max, y_min, y_max, transformer = get_transform(city)
  bbox = convert_bbox(get_bbox(*tile), transformer)
  left, bottom, right, top = bbox

  left_norm, right_norm = (2 * (np.array([left, right]) - x_min) /
                           (x_max - x_min)) - 1
  top_norm, bottom_norm = (2 * (np.array([top, bottom]) - y_min) /
                           (y_max - y_min)) - 1

  h, w = out_shape[:2]
  xs = np.linspace(left_norm, right_norm, w)
  ys = np.linspace(top_norm, bottom_norm, h)
  grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
  return np.dstack((grid_x, grid_y)).transpose(2, 0, 1)


def sample_time(df):
  times = np.unique(df[['dow', 'hour']].values, axis=0)
  ind = np.random.choice(len(times))
  return times[ind]


def generate_speed_mask(ids, speeds, id_mask, metric=True):
  if ids.empty:
    return np.zeros_like(id_mask), np.zeros_like(id_mask)

  id_mask = id_mask.copy()
  max_id = np.maximum(id_mask.max(), ids.to_numpy().max()).astype(int)

  speed_key = np.zeros(max_id + 1)
  speed_key[ids] = speeds

  speed_mask = speed_key[id_mask.astype(int)]
  id_mask[speed_mask == 0] = 0

  if metric:
    speed_mask = speed_mask * MPH_TO_KMH

  return speed_mask, id_mask


def boolean_string(s):
  if s not in {'False', 'True'}:
    raise ValueError('Not a valid boolean string')
  return s == 'True'
