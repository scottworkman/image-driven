# Copyright © Scott Workman. 2024.
""" 
  Modified from: https://github.com/dqshuai/MetaFormer
"""

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import math


class Identity(nn.Module):

  def __init__(self,):
    super().__init__()

  def forward(self, input):
    return input


class SwishImplementation(torch.autograd.Function):

  @staticmethod
  def forward(ctx, i):
    result = i * torch.sigmoid(i)
    ctx.save_for_backward(i)
    return result

  @staticmethod
  def backward(ctx, grad_output):
    i = ctx.saved_variables[0]
    sigmoid_i = torch.sigmoid(i)
    return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):

  def forward(self, x):
    return SwishImplementation.apply(x)


def drop_connect(inputs, p, training):
  if not training:
    return inputs
  batch_size = inputs.shape[0]
  keep_prob = 1 - p
  random_tensor = keep_prob
  random_tensor += torch.rand([batch_size, 1, 1, 1],
                              dtype=inputs.dtype,
                              device=inputs.device)
  binary_tensor = torch.floor(random_tensor)
  output = inputs / keep_prob * binary_tensor
  return output


class MBConvBlock(nn.Module):

  def __init__(self,
               ksize,
               input_filters,
               output_filters,
               expand_ratio=1,
               stride=1,
               image_size=224,
               drop_connect_rate=0.):
    super().__init__()

    self._se_ratio = 0.25
    self._input_filters = input_filters
    self._output_filters = output_filters
    self._expand_ratio = expand_ratio
    self._kernel_size = ksize
    self._stride = stride
    self._drop_connect_rate = drop_connect_rate
    inp = self._input_filters
    oup = self._input_filters * self._expand_ratio

    if self._expand_ratio != 1:
      self._expand_conv = nn.Conv2d(in_channels=inp,
                                    out_channels=oup,
                                    kernel_size=1,
                                    bias=False)
      self._bn0 = nn.BatchNorm2d(num_features=oup)

    # depthwise convolution
    k = self._kernel_size
    s = self._stride
    self._depthwise_conv = nn.Conv2d(in_channels=oup,
                                     out_channels=oup,
                                     groups=oup,
                                     kernel_size=k,
                                     stride=s,
                                     padding=1,
                                     bias=False)

    self._bn1 = nn.BatchNorm2d(num_features=oup)

    # squeeze and excitation layer
    num_squeezed_channels = max(1, int(self._input_filters * self._se_ratio))
    self._se_reduce = nn.Conv2d(in_channels=oup,
                                out_channels=num_squeezed_channels,
                                kernel_size=1)
    self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels,
                                out_channels=oup,
                                kernel_size=1)

    # output phase
    final_oup = self._output_filters
    self._project_conv = nn.Conv2d(in_channels=oup,
                                   out_channels=final_oup,
                                   kernel_size=1,
                                   bias=False)
    self._bn2 = nn.BatchNorm2d(num_features=final_oup)
    self._swish = MemoryEfficientSwish()

  def forward(self, inputs):
    # expansion and depthwise convolution
    x = inputs
    if self._expand_ratio != 1:
      expand = self._expand_conv(inputs)
      bn0 = self._bn0(expand)
      x = self._swish(bn0)
    depthwise = self._depthwise_conv(x)
    bn1 = self._bn1(depthwise)
    x = self._swish(bn1)

    # squeeze and excitation
    x_squeezed = F.adaptive_avg_pool2d(x, 1)
    x_squeezed = self._se_reduce(x_squeezed)
    x_squeezed = self._swish(x_squeezed)
    x_squeezed = self._se_expand(x_squeezed)
    x = torch.sigmoid(x_squeezed) * x

    x = self._bn2(self._project_conv(x))

    # skip connection and drop connect
    input_filters, output_filters = self._input_filters, self._output_filters
    if self._stride == 1 and input_filters == output_filters:
      if self._drop_connect_rate != 0:
        x = drop_connect(x, p=self._drop_connect_rate, training=self.training)
      x = x + inputs
    return x


class Mlp(nn.Module):

  def __init__(self,
               in_features,
               hidden_features=None,
               out_features=None,
               act_layer=nn.GELU,
               drop=0.):
    super().__init__()

    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.act = act_layer()
    self.fc2 = nn.Linear(hidden_features, out_features)
    self.drop = nn.Dropout(drop)

  def forward(self, x, H=None, W=None):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.fc2(x)
    x = self.drop(x)
    return x


class Relative_Attention(nn.Module):

  def __init__(self,
               dim,
               img_size,
               extra_token_num=1,
               num_heads=8,
               qkv_bias=False,
               qk_scale=None,
               attn_drop=0.,
               proj_drop=0.):
    super().__init__()

    self.num_heads = num_heads
    self.extra_token_num = extra_token_num
    head_dim = dim // num_heads
    self.img_size = img_size  # h,w
    self.scale = qk_scale or head_dim**-0.5

    # define a parameter table of relative position bias
    self.relative_position_bias_table = nn.Parameter(
        torch.zeros((2 * img_size[0] - 1) * (2 * img_size[1] - 1),
                    num_heads))  # 2*h-1 * 2*w-1 + 1, nH

    # get pair-wise relative position index for each token
    coords_h = torch.arange(self.img_size[0])
    coords_w = torch.arange(self.img_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w],
                                        indexing="ij"))  # 2, h, w
    coords_flatten = torch.flatten(coords, 1)  # 2, h*w
    relative_coords = coords_flatten[:, :,
                                     None] - coords_flatten[:,
                                                            None, :]  # 2, h*w, h*w
    relative_coords = relative_coords.permute(1, 2,
                                              0).contiguous()  # h*w, h*w, 2
    relative_coords[:, :, 0] += self.img_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += self.img_size[1] - 1
    relative_coords[:, :, 0] *= 2 * self.img_size[1] - 1
    relative_position_index = relative_coords.sum(-1)  # h*w, h*w
    relative_position_index = F.pad(relative_position_index,
                                    (extra_token_num, 0, extra_token_num, 0))
    relative_position_index = relative_position_index.long()
    self.register_buffer("relative_position_index", relative_position_index)
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(proj_drop)
    trunc_normal_(self.relative_position_bias_table, std=.02)
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x):
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                              C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    q = q * self.scale
    attn = (q @ k.transpose(-2, -1))

    # h*w+1, h*w+1, nH
    relative_position_bias = self.relative_position_bias_table[
        self.relative_position_index.view(-1)].view(
            self.img_size[0] * self.img_size[1] + self.extra_token_num,
            self.img_size[0] * self.img_size[1] + self.extra_token_num, -1)

    # nH, h*w+1, h*w+1
    relative_position_bias = relative_position_bias.permute(2, 0,
                                                            1).contiguous()

    attn = attn + relative_position_bias.unsqueeze(0)

    attn = self.softmax(attn)

    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class OverlapPatchEmbed(nn.Module):

  def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
    super().__init__()

    patch_size = to_2tuple(patch_size)
    self.patch_size = patch_size
    self.proj = nn.Conv2d(in_chans,
                          embed_dim,
                          kernel_size=patch_size,
                          stride=stride,
                          padding=(patch_size[0] // 2, patch_size[1] // 2))
    self.norm = nn.LayerNorm(embed_dim)

    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      trunc_normal_(m.weight, std=.02)
      if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      nn.init.constant_(m.bias, 0)
      nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
      fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
      fan_out //= m.groups
      m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
      if m.bias is not None:
        m.bias.data.zero_()

  def forward(self, x):
    x = self.proj(x)
    _, _, H, W = x.shape
    x = x.flatten(2).transpose(1, 2)
    x = self.norm(x)

    return x, H, W


class MHSABlock(nn.Module):

  def __init__(self,
               input_dim,
               output_dim,
               image_size,
               stride,
               num_heads,
               extra_token_num=1,
               mlp_ratio=4.,
               qkv_bias=False,
               qk_scale=None,
               drop=0.,
               attn_drop=0.,
               drop_path=0.,
               act_layer=nn.GELU,
               norm_layer=nn.LayerNorm):
    super().__init__()

    self.extra_token_num = extra_token_num

    if stride != 1:
      self.patch_embed = OverlapPatchEmbed(patch_size=3,
                                           stride=stride,
                                           in_chans=input_dim,
                                           embed_dim=output_dim)
      self.img_size = image_size // 2
    else:
      self.patch_embed = None
      self.img_size = image_size
    self.img_size = to_2tuple(self.img_size)

    self.norm1 = norm_layer(output_dim)
    self.attn = Relative_Attention(output_dim,
                                   self.img_size,
                                   extra_token_num=extra_token_num,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   qk_scale=qk_scale,
                                   attn_drop=attn_drop,
                                   proj_drop=drop)
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    self.norm2 = norm_layer(output_dim)
    mlp_hidden_dim = int(output_dim * mlp_ratio)
    self.mlp = Mlp(in_features=output_dim,
                   hidden_features=mlp_hidden_dim,
                   act_layer=act_layer,
                   drop=drop)

  def forward(self, x, H, W, geo_bias, extra_tokens=None):
    if self.patch_embed is not None:
      x, _, _ = self.patch_embed(x)
      extra_tokens = [
          token.expand(x.shape[0], -1, -1) for token in extra_tokens
      ]
      extra_tokens.append(x)
      x = torch.cat(extra_tokens, dim=1)

    x = x + geo_bias

    x = x + self.drop_path(self.attn(self.norm1(x)))
    x = x + self.drop_path(self.mlp(self.norm2(x), H // 2, W // 2))
    return x


class ResNormLayer(nn.Module):

  def __init__(self, linear_size):
    super().__init__()

    self.l_size = linear_size
    self.nonlin1 = nn.ReLU(inplace=True)
    self.nonlin2 = nn.ReLU(inplace=True)
    self.norm_fn1 = nn.LayerNorm(self.l_size)
    self.norm_fn2 = nn.LayerNorm(self.l_size)
    self.w1 = nn.Linear(self.l_size, self.l_size)
    self.w2 = nn.Linear(self.l_size, self.l_size)

  def forward(self, x):
    y = self.w1(x)
    y = self.nonlin1(y)
    y = self.norm_fn1(y)
    y = self.w2(y)
    y = self.nonlin2(y)
    y = self.norm_fn2(y)
    out = x + y
    return out
