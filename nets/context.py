# Copyright © Scott Workman. 2024.

import torch
import torch.nn as nn

from nets import siren


class LocationEncoder(nn.Module):

  def __init__(self, dim_out=64):
    super().__init__()

    self.dim_out = dim_out

    self.net = siren.SirenNet(dim_in=2,
                              dim_hidden=64,
                              num_layers=3,
                              dim_out=self.dim_out,
                              dropout=False)

  def forward(self, coords):
    b, c, h, w = coords.shape

    output = self.net(coords.permute(0, 2, 3, 1).flatten(end_dim=2))
    return output.reshape(b, h, w, self.dim_out).permute(0, 3, 1, 2)


class TimeEncoder(nn.Module):

  def __init__(self, dim_out=64):
    super().__init__()

    self.pe = PresenceAware()
    self.net = siren.SirenNet(dim_in=self.pe.embedding_dim,
                              dim_hidden=64,
                              num_layers=3,
                              dim_out=dim_out,
                              dropout=False)

  def forward(self, time):
    embedding = self.pe(time)
    return self.net(embedding)


class LocationTimeEncoder(nn.Module):

  def __init__(self, dim_out=64):
    super().__init__()

    self.dim_out = dim_out

    self.pe_time = PresenceAware()
    self.net = siren.SirenNet(dim_in=2 + self.pe_time.embedding_dim,
                              dim_hidden=128,
                              num_layers=3,
                              dim_out=self.dim_out,
                              dropout=False)

  def forward(self, loc, time):
    embedding_time = self.pe_time(time)

    # replicate time feat in spatial dimensions
    b, c, h, w = loc.shape
    embedding_time = embedding_time.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)
    embedding = torch.cat((loc, embedding_time), 1)

    output = self.net(embedding.permute(0, 2, 3, 1).flatten(end_dim=2))
    return output.reshape(b, h, w, self.dim_out).permute(0, 3, 1, 2)


class ContextEncoder(nn.Module):

  def __init__(self, context="both"):
    super().__init__()

    self.context = context

    if context is not None:
      if context in ["both"]:
        self.enc_loctime = LocationTimeEncoder()
      if context in ["both", "location"]:
        self.enc_loc = LocationEncoder()
      if context in ["both", "time"]:
        self.enc_time = TimeEncoder()

  def forward(self, loc, time):
    if self.context == "both":
      feat_loc = self.enc_loc(loc)
      feat_time = self.enc_time(time)
      feat_loctime = self.enc_loctime(loc, time)

      # replicate time feat in spatial dimensions
      _, _, h, w = feat_loc.shape
      feat_time = feat_time.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)

      return feat_loc + feat_time + feat_loctime
    elif self.context == "location":
      return self.enc_loc(loc)
    elif self.context == "time":
      feat_time = self.enc_time(time)

      # replicate time feat in spatial dimensions
      return feat_time.unsqueeze(2).unsqueeze(3).repeat(1, 1, 1024, 1024)
    else:
      return None


class PresenceAware(nn.Module):

  def __init__(self):
    super().__init__()

    self.embedding_dim = 4

  def forward(self, time):
    dow = time[:, 0].unsqueeze(1)
    hour = time[:, 1].unsqueeze(1)

    cos_dow = torch.cos(torch.pi * dow)
    sin_dow = torch.sin(torch.pi * dow)
    cos_hour = torch.cos(torch.pi * hour)
    sin_hour = torch.sin(torch.pi * hour)

    return torch.cat((sin_dow, cos_dow, sin_hour, cos_hour), 1)


if __name__ == "__main__":
  loc = torch.rand((1, 2, 1024, 1024))
  time = torch.tensor([0, 0]).unsqueeze(0).float()

  enc_loc = LocationEncoder()
  enc_time = TimeEncoder()
  enc_loctime = LocationTimeEncoder()
  enc = ContextEncoder()

  feat_loc = enc_loc(loc)
  print(feat_loc.shape)

  feat_time = enc_time(time)
  print(feat_time.shape)

  feat_loctime = enc_loctime(loc, time)
  print(feat_loctime.shape)

  output = enc(loc, time)
  print(output.shape)
