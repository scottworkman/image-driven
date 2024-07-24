# Copyright © Scott Workman. 2024.

import torch.nn as nn
import torch.nn.functional as F

import loss
from nets.base import *
from nets.context import ContextEncoder


def build_model(args):
  print("[*] building model from {}, {}, {}".format(args.method, args.loss,
                                                    args.decoder))
  method, objective, decoder = [args.method, args.loss, args.decoder]

  if objective == "huber":
    criterion = loss.CombinedHuberLoss()
    num_outputs = 1
  elif objective == "student":
    criterion = loss.CombinedStudentTLoss()
    num_outputs = 2
  else:
    raise ValueError("Unknown objective function.")

  if method == "multitask":
    model = Multi(num_outputs, decoder, context="both")
  elif method == "multitask_loc":
    model = Multi(num_outputs, decoder, context="location")
  elif method == "multitask_time":
    model = Multi(num_outputs, decoder, context="time")
  elif method == "multitask_image":
    model = Multi(num_outputs, decoder, context=None)
  elif method == "doubletask_road":
    model = Multi(num_outputs, decoder, context="both", task="road")
  elif method == "doubletask_angle":
    model = Multi(num_outputs, decoder, context="both", task="angle")
  elif method == "singletask":
    model = Single(num_outputs, decoder, context="both")
  elif method == "metadata":
    model = Metadata(num_outputs, context="both")
  elif method == "metadata_loc":
    model = Metadata(num_outputs, context="location")
  elif method == "metadata_time":
    model = Metadata(num_outputs, context="time")
  else:
    raise ValueError('Unknown method.')

  return model, criterion


class Multi(nn.Module):

  def __init__(self, num_outputs=1, decoder="mlp", context="both", task=None):
    super().__init__()

    self.task = task
    self.encoder = Encoder(context)
    self.d = DecoderMulti([1, 16, num_outputs], decoder_type=decoder)

  def forward(self, inputs):
    features = self.encoder(inputs)
    _road, _angle, speed = self.d(features)

    road = angle = None
    if self.task is None or self.task == "road":
      road = _road
    if self.task is None or self.task == "angle":
      angle = _angle

    return road, angle, F.softplus(speed)


class Single(nn.Module):

  def __init__(self, num_outputs=1, decoder="mlp", context="both"):
    super().__init__()

    self.encoder = Encoder(context)
    if decoder == "mlp":
      self.d = SegFormerDecoder(num_outputs)
    else:
      raise NotImplementedError

  def forward(self, inputs):
    features = self.encoder(inputs)
    speed = self.d(features)

    return None, None, F.softplus(speed)


class Metadata(nn.Module):

  def __init__(self, num_outputs=16, decoder="mlp", context="both"):
    super().__init__()

    self.context = context
    self.encoder = ContextEncoder(context)

    if decoder == "mlp":
      self.d = DecoderMetadata(num_outputs)
    else:
      raise NotImplementedError

  def forward(self, inputs):
    image, location, time = inputs
    if self.context == "both":
      features = self.encoder(location, time)
    elif self.context == "location":
      features = self.encoder(location, None)
    elif self.context == "time":
      features = self.encoder(None, time)
    else:
      raise ValueError('Unknown metadata.')

    speed = self.d(features)

    return None, None, F.softplus(speed)
