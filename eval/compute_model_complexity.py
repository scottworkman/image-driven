# Copyright © Scott Workman. 2024.
"""
  References:
    https://github.com/MrYxJ/calculate-flops.pytorch
    https://github.com/xiamenwcy/Evaluation-Metrics-PyTorch/blob/main/test_model_complexity.py
"""

import _init_paths

import torch
from lmm import LMM

from timeit import default_timer as timer


def get_fake_data(device):
  im = torch.rand((1, 3, 1024, 1024))
  loc = torch.rand((1, 2, 1024, 1024))
  time = torch.tensor([0, 0]).unsqueeze(0).float()
  return [im.to(device), loc.to(device), time.to(device)]


def compute_gflops_and_model_size(model):
  from thop import profile
  input_ = get_fake_data("cpu")
  macs, params = profile(model, inputs=(input_,), verbose=False)

  GFlops = macs * 2.0 / pow(10, 9)
  model_size = params * 4.0 / 1024 / 1024
  params_M = params / pow(10, 6)
  return params_M, model_size, GFlops


@torch.no_grad()
def compute_fps(model, epoch=100, device=None):
  """
    frames per second
    """
  total_time = 0.0

  if device:
    model = model.to(device)

  data = get_fake_data(device)

  start = timer()
  for i in range(epoch):
    outputs = model(data)
  end = timer()
  total_time = (end - start)

  print(f"time={total_time}, epochs={epoch}, fps={epoch / total_time}")
  print(f"one frame: {total_time / epoch}")
  return epoch / total_time


def test_model_flops():
  model = LMM(
      **{
          "method": "multitask",
          "loss": "student",
          "decoder": "mlp",
          "aggregate": True,
          "batch_size": 1,
          "learning_rate": 1e-4
      })
  params_M, model_size, gflops = compute_gflops_and_model_size(model)

  print('Number of parameters: {:.2f} M '.format(params_M))
  print('Size of model: {:.2f} MB'.format(model_size))
  print('Computational complexity: {:.2f} GFlops'.format(gflops))


def test_fps():
  model = LMM(
      **{
          "method": "multitask",
          "loss": "student",
          "decoder": "mlp",
          "aggregate": True,
          "batch_size": 1,
          "learning_rate": 1e-4
      })

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  fps = compute_fps(model, device=device)
  print('device: {} - fps: {:.3f}s'.format(device.type, fps))


def test_model_flops_v2():
  from calflops import calculate_flops

  model = LMM(
      **{
          "method": "multitask",
          "loss": "student",
          "decoder": "mlp",
          "aggregate": True,
          "batch_size": 1,
          "learning_rate": 1e-4
      })

  im = torch.rand((1, 3, 1024, 1024))
  loc = torch.rand((1, 2, 1024, 1024))
  time = torch.tensor([0, 0]).unsqueeze(0).float()

  # Note: update the forward pass to support this operation
  # (i.e., individual arguments instead of list)
  flops, macs, params = calculate_flops(model=model,
                                        kwargs={
                                            "im": im,
                                            "loc": loc,
                                            "time": time
                                        })
  print("FLOPs:%s  MACs:%s  Params:%s \n" % (flops, macs, params))


if __name__ == '__main__':
  test_model_flops()
  test_fps()
  #test_model_flops_v2()
