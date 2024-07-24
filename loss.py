# Copyright © Scott Workman. 2024.

import torch
import torch.nn as nn

from nets import ops


class TDiceLoss(nn.Module):
  """
  From: https://github.com/snakers4/spacenet-three
  """

  def __init__(self, dice_weight=1):
    super().__init__()
    self.nll_loss = nn.BCEWithLogitsLoss()
    self.dice_weight = dice_weight

  def forward(self, outputs, targets):
    loss = self.nll_loss(outputs, targets)
    if self.dice_weight:
      eps = 1e-15
      dice_target = (targets == 1).float()
      dice_output = torch.sigmoid(outputs)
      intersection = (dice_output * dice_target).sum()
      union = dice_output.sum() + dice_target.sum() + eps
      loss += 1 - torch.log(2 * intersection / union)

    return loss


class StudentTLoss(nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, mean, var, counts, targets):
    dist = torch.distributions.studentT.StudentT(df=counts,
                                                 loc=mean,
                                                 scale=torch.sqrt(var))
    loss = -torch.mean(dist.log_prob(targets))

    return loss


class CombinedHuberLoss(nn.Module):

  def __init__(self, aggregate=True):
    super().__init__()
    self.aggregate = aggregate
    self.dice = TDiceLoss()
    self.ce = nn.CrossEntropyLoss(ignore_index=-1)
    self.huber = nn.SmoothL1Loss(beta=2)

  def forward(self, outputs, targets):
    out_road, out_angle, out_speed = outputs
    tar_road, tar_speed, tar_angle, tar_angle_bin, _, tar_id, _ = targets

    loss_road = loss_angle = loss_speed = 0

    data = {}

    if out_road is not None:
      loss_road = self.dice(out_road.squeeze(1), tar_road.float())
      data["road"] = [out_road.squeeze(1), tar_road]

    if out_angle is not None:
      loss_angle = self.ce(out_angle, tar_angle_bin)
      data["angle"] = [out_angle, tar_angle_bin]

    # aggregate the speeds
    if self.aggregate:
      out_speed = ops.aggregate(out_speed, tar_id)
      tar_speed = ops.aggregate(tar_speed, tar_id)
    else:
      valid_inds = torch.nonzero(tar_speed).split(1, dim=1)
      out_speed = out_speed[valid_inds]
      tar_speed = tar_speed[valid_inds]

    loss_speed = self.huber(out_speed, tar_speed)
    data["speed"] = [out_speed, tar_speed]

    loss = loss_road + loss_angle + (loss_speed * 1)

    return {
        "total": loss,
        "road": loss_road,
        "angle": loss_angle,
        "speed": loss_speed
    }, data


class CombinedStudentTLoss(nn.Module):

  def __init__(self, aggregate=True):
    super().__init__()
    self.aggregate = aggregate
    self.dice = TDiceLoss()
    self.ce = nn.CrossEntropyLoss(ignore_index=-1)
    self.student = StudentTLoss()

  def forward(self, outputs, targets):
    out_road, out_angle, out_speed = outputs
    tar_road, tar_speed, tar_angle, tar_angle_bin, tar_count, tar_id, _ = targets

    loss_road = loss_angle = loss_speed = 0

    data = {}

    if out_road is not None:
      loss_road = self.dice(out_road.squeeze(1), tar_road.float())
      data["road"] = [out_road.squeeze(1), tar_road]

    if out_angle is not None:
      loss_angle = self.ce(out_angle, tar_angle_bin)
      data["angle"] = [out_angle, tar_angle_bin]

    mean = out_speed[:, 0, ...].unsqueeze(1)
    var = out_speed[:, 1, ...].unsqueeze(1)

    # aggregate the speeds
    if self.aggregate:
      tar_speed = ops.aggregate(tar_speed, tar_id)
      tar_count = ops.aggregate(tar_count, tar_id)
      mean = ops.aggregate(mean, tar_id)
      var = ops.aggregate(var, tar_id) + 1e-8
    else:
      raise NotImplementedError

    loss_speed = self.student(mean, var, tar_count, tar_speed)
    data["speed"] = [mean, tar_speed]

    loss = loss_road + loss_angle + (loss_speed * 1)

    return {
        "total": loss,
        "road": loss_road,
        "angle": loss_angle,
        "speed": loss_speed
    }, data
