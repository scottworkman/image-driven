# Copyright © Scott Workman. 2024.

import torch


def aggregate(predictions, target_ids):
  b, _, _, _ = predictions.shape

  num_ids = torch.unique(target_ids).max() + 1
  index = torch.flatten(target_ids, start_dim=1)
  predictions = torch.flatten(predictions, start_dim=1)

  # pixel count per image, per road segment
  num_pixels = torch.zeros(b, num_ids).type_as(predictions)
  num_pixels = num_pixels.scatter_add(1, index, torch.ones_like(predictions))

  # aggregate predictions
  pred = torch.zeros(b, num_ids).type_as(predictions)
  pred = pred.scatter_add(1, index, predictions)
  pred = pred / num_pixels

  # ignore background (target_id == 0, non-road)
  pred = pred[:, 1:, ...]

  # extract valid road segments
  valid_preds = pred[~torch.isnan(pred)]

  return valid_preds


def aggregate_list(predictions, target_ids):
  b, _, _, _ = predictions.shape

  ids = []
  values = []

  for idx in range(b):
    local_id = target_ids[idx, ...].squeeze()
    local_pred = predictions[idx, ...].squeeze()

    num_ids = torch.unique(local_id).max() + 1
    index = torch.flatten(local_id, start_dim=0).unsqueeze(0)
    local_pred = torch.flatten(local_pred, start_dim=0).unsqueeze(0)

    # pixel count per road segment
    num_pixels = torch.zeros(1, num_ids).type_as(predictions)
    num_pixels = num_pixels.scatter_add(1, index, torch.ones_like(local_pred))

    # aggregate predictions
    pred = torch.zeros(1, num_ids).type_as(predictions)
    pred = pred.scatter_add(1, index, local_pred)
    pred = pred / num_pixels

    idx = torch.unique(local_id)[1:]

    ids.append(idx)
    values.append(pred[:, idx])

  return values, ids
