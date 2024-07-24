# Copyright © Scott Workman. 2024.
""" 
  Modified from: https://github.com/dqshuai/MetaFormer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from nets.context import ContextEncoder
from nets.meta_ops import MBConvBlock, MHSABlock, ResNormLayer


def make_blocks(stage_index,
                depths,
                embed_dims,
                img_size,
                dpr,
                extra_token_num=1,
                num_heads=8,
                mlp_ratio=4.,
                stage_type='conv'):

  stage_name = f'stage_{stage_index}'
  blocks = []
  for block_idx in range(depths[stage_index]):
    stride = 2 if block_idx == 0 and stage_index != 1 else 1
    in_chans = embed_dims[stage_index] if block_idx != 0 else embed_dims[
        stage_index - 1]
    out_chans = embed_dims[stage_index]
    image_size = img_size if block_idx == 0 or stage_index == 1 else img_size // 2
    drop_path_rate = dpr[sum(depths[1:stage_index]) + block_idx]
    if stage_type == 'conv':
      blocks.append(
          MBConvBlock(ksize=3,
                      input_filters=in_chans,
                      output_filters=out_chans,
                      image_size=image_size,
                      expand_ratio=int(mlp_ratio),
                      stride=stride,
                      drop_connect_rate=drop_path_rate))
    elif stage_type == 'mhsa':
      blocks.append(
          MHSABlock(input_dim=in_chans,
                    output_dim=out_chans,
                    image_size=image_size,
                    stride=stride,
                    num_heads=num_heads,
                    extra_token_num=extra_token_num,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_rate))
    else:
      raise NotImplementedError("We only support conv and mhsa")
  return blocks


class MetaFormer(nn.Module):

  def __init__(self,
               img_size=512,
               in_chans=3,
               conv_embed_dims=[64, 64, 128],
               attn_embed_dims=[256, 512],
               conv_depths=[2, 2, 3],
               attn_depths=[5, 2],
               num_heads=8,
               extra_token_num=0,
               mlp_ratio=4.,
               attn_norm_layer=nn.LayerNorm,
               conv_act_layer=nn.ReLU,
               attn_act_layer=nn.GELU,
               qkv_bias=False,
               qk_scale=None,
               drop_rate=0.,
               attn_drop_rate=0.,
               drop_path_rate=0.,
               meta_dims=[],
               context="both"):
    super().__init__()

    self.img_size = img_size
    self.num_heads = num_heads
    self.meta_dims = meta_dims
    self.attn_embed_dims = attn_embed_dims
    self.extra_token_num = extra_token_num
    self.context = context

    for ind, meta_dim in enumerate(meta_dims):
      meta_head_1 = nn.Sequential(
          nn.Linear(meta_dim, attn_embed_dims[0]),
          nn.ReLU(inplace=True),
          nn.LayerNorm(attn_embed_dims[0]),
          ResNormLayer(attn_embed_dims[0]),
      ) if meta_dim > 0 else nn.Identity()
      meta_head_2 = nn.Sequential(
          nn.Linear(meta_dim, attn_embed_dims[1]),
          nn.ReLU(inplace=True),
          nn.LayerNorm(attn_embed_dims[1]),
          ResNormLayer(attn_embed_dims[1]),
      ) if meta_dim > 0 else nn.Identity()
      setattr(self, f"meta_{ind+1}_head_1", meta_head_1)
      setattr(self, f"meta_{ind+1}_head_2", meta_head_2)

    stem_chs = (3 * (conv_embed_dims[0] // 4), conv_embed_dims[0])
    dpr = [
        x.item() for x in torch.linspace(0, drop_path_rate,
                                         sum(conv_depths[1:] + attn_depths))
    ]

    #stage_0
    self.stage_0 = nn.Sequential(*[
        nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(num_features=stem_chs[0]),
        conv_act_layer(inplace=True),
        nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(num_features=stem_chs[1]),
        conv_act_layer(inplace=True),
        nn.Conv2d(
            stem_chs[1], conv_embed_dims[0], 3, stride=1, padding=1, bias=False)
    ])
    self.bn1 = nn.BatchNorm2d(num_features=conv_embed_dims[0])
    self.act1 = conv_act_layer(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    #stage_1
    self.stage_1 = nn.ModuleList(
        make_blocks(1,
                    conv_depths + attn_depths,
                    conv_embed_dims + attn_embed_dims,
                    img_size // 4,
                    dpr=dpr,
                    num_heads=num_heads,
                    extra_token_num=extra_token_num,
                    mlp_ratio=mlp_ratio,
                    stage_type='conv'))

    #stage_2
    self.stage_2 = nn.ModuleList(
        make_blocks(2,
                    conv_depths + attn_depths,
                    conv_embed_dims + attn_embed_dims,
                    img_size // 4,
                    dpr=dpr,
                    num_heads=num_heads,
                    extra_token_num=extra_token_num,
                    mlp_ratio=mlp_ratio,
                    stage_type='conv'))

    #stage_3
    self.stage_3 = nn.ModuleList(
        make_blocks(3,
                    conv_depths + attn_depths,
                    conv_embed_dims + attn_embed_dims,
                    img_size // 8,
                    dpr=dpr,
                    num_heads=num_heads,
                    extra_token_num=extra_token_num,
                    mlp_ratio=mlp_ratio,
                    stage_type='mhsa'))

    #stage_4
    self.stage_4 = nn.ModuleList(
        make_blocks(4,
                    conv_depths + attn_depths,
                    conv_embed_dims + attn_embed_dims,
                    img_size // 16,
                    dpr=dpr,
                    num_heads=num_heads,
                    extra_token_num=extra_token_num,
                    mlp_ratio=mlp_ratio,
                    stage_type='mhsa'))

    if context in ["both", "location", "time"]:
      self.context_emb = ContextEncoder(context)

      num_chans = 64
      self.context_conv1 = nn.Conv2d(num_chans, 256, 3, padding=1)
      self.context_conv2 = nn.Conv2d(num_chans, 512, 3, padding=1)

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
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
      nn.init.ones_(m.weight)
      nn.init.zeros_(m.bias)

  def forward_features(self, x, loc, time=None):
    B = x.shape[0]
    extra_tokens_1 = []
    extra_tokens_2 = []

    if time is None:
      metas = []
    else:
      #metas = (time,)
      # ignore time as a token logic
      metas = []

    for ind, cur_meta in enumerate(metas):
      meta_head_1 = getattr(self, f"meta_{ind+1}_head_1")
      meta_head_2 = getattr(self, f"meta_{ind+1}_head_2")
      meta_1 = meta_head_1(cur_meta)
      meta_1 = meta_1.reshape(B, -1, self.attn_embed_dims[0])
      meta_2 = meta_head_2(cur_meta)
      meta_2 = meta_2.reshape(B, -1, self.attn_embed_dims[1])
      extra_tokens_1.append(meta_1)
      extra_tokens_2.append(meta_2)

    x = self.stage_0(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.maxpool(x)

    features = []
    for blk in self.stage_1:
      x = blk(x)
    features.append(x)

    for blk in self.stage_2:
      x = blk(x)
    features.append(x)

    # process context
    if self.context is not None:
      feat_context = self.context_emb(loc, time)

      h = w = (self.img_size // 16)
      context_bias = F.interpolate(self.context_conv1(feat_context), (h, w),
                                   mode="bilinear")
      context_bias = torch.flatten(context_bias, start_dim=2).permute(0, 2, 1)
      context_bias = F.pad(context_bias, (0, 0, self.extra_token_num, 0))
    else:
      context_bias = torch.tensor(0).to(x.device)

    H0, W0 = self.img_size // 8, self.img_size // 8
    for ind, blk in enumerate(self.stage_3):
      if ind == 0:
        x = blk(x, H0, W0, context_bias, extra_tokens_1)
      else:
        x = blk(x, H0, W0, context_bias)
    tmp = x[:, self.extra_token_num:, :]
    features.append(tmp.reshape(B, H0 // 2, W0 // 2, -1).permute(0, 3, 1, 2))

    x = x[:, self.extra_token_num:, :]
    H1, W1 = self.img_size // 16, self.img_size // 16
    x = x.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

    # process context
    if self.context is not None:
      h = w = (self.img_size // 32)
      context_bias = F.interpolate(self.context_conv2(feat_context), (h, w),
                                   mode="bilinear")
      context_bias = torch.flatten(context_bias, start_dim=2).permute(0, 2, 1)
      context_bias = F.pad(context_bias, (0, 0, self.extra_token_num, 0))
    else:
      context_bias = torch.tensor(0).to(x.device)

    for ind, blk in enumerate(self.stage_4):
      if ind == 0:
        x = blk(x, H1, W1, context_bias, extra_tokens_2)
      else:
        x = blk(x, H1, W1, context_bias)
    tmp = x[:, self.extra_token_num:, :]
    features.append(tmp.reshape(B, H1 // 2, W1 // 2, -1).permute(0, 3, 1, 2))

    return features

  def forward(self, x, loc, time=None):
    x = self.forward_features(x, loc, time)
    return x


if __name__ == "__main__":
  x = torch.randn([4, 3, 1024, 1024])
  loc = torch.randn([4, 2, 1024, 1024])
  time = torch.randn([4, 2])

  model = MetaFormer(img_size=x.shape[2])

  output = model.forward_features(x, loc, time)
  for item in output:
    print(item.shape)
