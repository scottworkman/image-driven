# Copyright © Scott Workman. 2024.

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets import meta

nonlinearity = nn.ReLU


class DecoderBlock(nn.Module):

  def __init__(self, in_channels, n_filters):
    super().__init__()

    # B, C, H, W -> B, C/4, H, W
    self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
    self.norm1 = nn.GroupNorm(16, in_channels // 4)
    self.relu1 = nonlinearity(inplace=True)

    # B, C/4, H, W -> B, C/4, H, W
    self.deconv2 = nn.ConvTranspose2d(in_channels // 4,
                                      in_channels // 4,
                                      3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1)
    self.norm2 = nn.GroupNorm(16, in_channels // 4)
    self.relu2 = nonlinearity(inplace=True)

    # B, C/4, H, W -> B, C, H, W
    self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
    self.norm3 = nn.GroupNorm(16, n_filters)
    self.relu3 = nonlinearity(inplace=True)

  def forward(self, x):
    x = self.conv1(x)
    x = self.norm1(x)
    x = self.relu1(x)
    x = self.deconv2(x)
    x = self.norm2(x)
    x = self.relu2(x)
    x = self.conv3(x)
    x = self.norm3(x)
    x = self.relu3(x)
    return x


class Encoder(nn.Module):

  def __init__(self, context="both", img_size=1024):
    super().__init__()

    extra_token_num = 0
    meta_dims = []

    self.context = context
    self.net = meta.MetaFormer(img_size=img_size,
                               extra_token_num=extra_token_num,
                               meta_dims=meta_dims,
                               context=context)

  def forward(self, inputs):
    image, location, time = inputs

    if self.context == "both":
      features = self.net(image, location, time)
    elif self.context == "location":
      features = self.net(image, location, None)
    elif self.context == "time":
      features = self.net(image, None, time)
    elif self.context is None:
      features = self.net(image, None, None)
    else:
      raise ValueError('Unknown context.')

    return features


class Decoder(nn.Module):

  def __init__(self, num_outputs):
    super().__init__()

    filters = [64, 128, 256, 512]

    self.decoder4 = DecoderBlock(filters[3], filters[2])
    self.decoder3 = DecoderBlock(filters[2], filters[1])
    self.decoder2 = DecoderBlock(filters[1], filters[0])
    self.decoder1 = DecoderBlock(filters[0], filters[0])

    self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
    self.finalrelu1 = nonlinearity(inplace=True)
    self.finalconv2 = nn.Conv2d(32, 32, 3)
    self.finalrelu2 = nonlinearity(inplace=True)
    self.finalconv3 = nn.Conv2d(32, num_outputs, 2, padding=1)

  def forward(self, encodings):
    e1, e2, e3, e4 = encodings

    d4 = self.decoder4(e4) + e3
    d3 = self.decoder3(d4) + e2
    d2 = self.decoder2(d3) + e1
    d1 = self.decoder1(d2)

    f1 = self.finaldeconv1(d1)
    f2 = self.finalrelu1(f1)
    f3 = self.finalconv2(f2)
    f4 = self.finalrelu2(f3)
    f5 = self.finalconv3(f4)

    return f5


class MLP(nn.Module):
  """
    Linear Embedding
    """

  def __init__(self, input_dim=2048, embed_dim=768):
    super().__init__()
    self.proj = nn.Linear(input_dim, embed_dim)

  def forward(self, x):
    x = x.flatten(2).permute(0, 2, 1)
    x = self.proj(x)
    return x


class SegFormerDecoder(nn.Module):
  """
    From SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." arXiv preprint arXiv:2105.15203 (2021).
    Args:
        num_classes (int): The unique number of target classes.
        embedding_dim (int): The MLP decoder channel dimension.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature.
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
    """

  def __init__(self, num_classes, embedding_dim=512, align_corners=False):
    super(SegFormerDecoder, self).__init__()

    self.align_corners = align_corners
    self.num_classes = num_classes

    feat_channels = [64, 128, 256, 512]

    self.linear_c4 = MLP(input_dim=feat_channels[3], embed_dim=embedding_dim)
    self.linear_c3 = MLP(input_dim=feat_channels[2], embed_dim=embedding_dim)
    self.linear_c2 = MLP(input_dim=feat_channels[1], embed_dim=embedding_dim)
    self.linear_c1 = MLP(input_dim=feat_channels[0], embed_dim=embedding_dim)

    self.dropout = nn.Dropout2d(0.1)
    self.linear_fuse = nn.Sequential(
        nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
        nn.BatchNorm2d(embedding_dim), nonlinearity())

    self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

  def forward(self, encodings):
    c1, c2, c3, c4 = encodings
    B, _, H, W = c1.shape

    f_c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(B, -1, c4.shape[2],
                                                       c4.shape[3])
    f_c4 = F.interpolate(f_c4,
                         size=c1.shape[2:],
                         mode='bilinear',
                         align_corners=self.align_corners)

    f_c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(B, -1, c3.shape[2],
                                                       c3.shape[3])
    f_c3 = F.interpolate(f_c3,
                         size=c1.shape[2:],
                         mode='bilinear',
                         align_corners=self.align_corners)

    f_c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(B, -1, c2.shape[2],
                                                       c2.shape[3])
    f_c2 = F.interpolate(f_c2,
                         size=c1.shape[2:],
                         mode='bilinear',
                         align_corners=self.align_corners)

    f_c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(B, -1, c1.shape[2],
                                                       c1.shape[3])

    feats = self.linear_fuse(torch.cat([f_c4, f_c3, f_c2, f_c1], dim=1))

    x = self.dropout(feats)
    x = self.linear_pred(x)
    x = F.interpolate(x,
                      size=(4 * H, 4 * W),
                      mode='bilinear',
                      align_corners=self.align_corners)
    return x


class DecoderMetadata(nn.Module):

  def __init__(self, num_outputs, num_context=64):
    super().__init__()

    self.num_context = num_context
    self.finalconv2 = nn.Conv2d(num_context, 32, 3, padding=1)
    self.finalrelu2 = nonlinearity(inplace=True)
    self.finalconv3 = nn.Conv2d(32 + num_context, num_outputs, 3, padding=1)

  def forward(self, context):
    f3 = self.finalconv2(context)
    f4 = self.finalrelu2(f3)
    fused = torch.cat((f4, context), dim=1)
    f5 = self.finalconv3(fused)

    return f5


class DecoderMulti(nn.Module):

  def __init__(self, num_outputs=[1, 16, 1], decoder_type="mlp"):
    super().__init__()

    if decoder_type == "cnn":
      self.decoder1 = Decoder(num_outputs[0])
      self.decoder2 = Decoder(num_outputs[1])
      self.decoder3 = Decoder(num_outputs[2])
    elif decoder_type == "mlp":
      self.decoder1 = SegFormerDecoder(num_outputs[0])
      self.decoder2 = SegFormerDecoder(num_outputs[1])
      self.decoder3 = SegFormerDecoder(num_outputs[2])
    else:
      raise ValueError('Unknown decoder type.')

  def forward(self, encodings):
    d1 = self.decoder1(encodings)
    d2 = self.decoder2(encodings)
    d3 = self.decoder3(encodings)
    return d1, d2, d3


if __name__ == "__main__":
  im = torch.rand((1, 3, 1024, 1024))
  loc = torch.rand((1, 2, 1024, 1024))
  time = torch.tensor([0, 0]).unsqueeze(0).float()

  encoder = Encoder()
  decoder = DecoderMulti([1, 16, 1])

  outputs = decoder(encoder([im, loc, time]))
  for output in outputs:
    print(output.shape)
