
import torch
from torch import nn

class DownSampleConv(nn.Module):

   def __init__(self, in_channel, out_channels, \
                kernel=4, strides=2, padding=1, activation=True, batchnorm=True):
       super().__init__()
       self.activation = activation
       self.batchnorm = batchnorm

       self.conv = nn.Conv2d(in_channel, out_channels, kernel, strides, padding)

       if batchnorm:
           self.bn = nn.BatchNorm2d(out_channels)

       if activation:
           self.act = nn.LeakyReLU(0.2)

   def forward(self, x):
       x = self.conv(x)
       if self.batchnorm:
           x = self.bn(x)
       if self.activation:
           x = self.act(x)
       return x


class PatchGAN(nn.Module):

   def __init__(self, input_channels):
       super().__init__()
       self.d1 = DownSampleConv(input_channels, 64, batchnorm=False)
       self.d2 = DownSampleConv(64, 128)
       self.d3 = DownSampleConv(128, 256)
       self.d4 = DownSampleConv(256, 512)
       self.final = nn.Conv2d(512, 1, kernel_size=1)

   def forward(self, x, y):
       x = torch.cat([x, y], axis=1)
       x0 = self.d1(x)  # output_size: 64*128*128
       x1 = self.d2(x0)  # output_size: 128*64*64
       x2 = self.d3(x1)  # output_size: 256*32*32
       x3 = self.d4(x2)  # output_size: 512*16*16
       xn = self.final(x3)  # output_size: 1*16*16
       return xn

class UpSampleConv(nn.Module):

   def __init__(
       self,
       in_channels,
       out_channels,
       kernel=4,
       strides=2,
       padding=1,
       activation=True,
       batchnorm=True,
       dropout=False
   ):
       super().__init__()
       self.activation = activation
       self.batchnorm = batchnorm
       self.dropout = dropout

       self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel, strides, padding)

       if batchnorm:
           self.bn = nn.BatchNorm2d(out_channels)

       if activation:
           self.act = nn.ReLU(True)

       if dropout:
           self.drop = nn.Dropout2d(0.5)

   def forward(self, x):
       x = self.deconv(x)
       if self.batchnorm:
           x = self.bn(x)

       if self.dropout:
           x = self.drop(x)
       return x

class Generator(nn.Module):

   def __init__(self, in_channels, out_channels_cholec80, out_channels_depth, \
                out_channels_seg, metics = 1.11):
       """
       Paper details:
       - Encoder: C64-C128-C256-C512-C512-C512-C512-C512
       - All convolutions are 4×4 spatial filters applied with stride 2
       - Convolutions in the encoder downsample by a factor of 2
       - Decoder: CD512-CD1024-CD1024-C1024-C1024-C512 -C256-C128
       """
       super().__init__()

       # encoder/donwsample convs
       self.encoders = [
           DownSampleConv(in_channels, 64, batchnorm=False),  # bs x 64 x 128 x 128
           DownSampleConv(64, 128),  # bs x 128 x 64 x 64
           DownSampleConv(128, 256),  # bs x 256 x 32 x 32
           DownSampleConv(256, 512),  # bs x 512 x 16 x 16
           DownSampleConv(512, 512),  # bs x 512 x 8 x 8
           DownSampleConv(512, 512),  # bs x 512 x 4 x 4
           DownSampleConv(512, 512),  # bs x 512 x 2 x 2
           DownSampleConv(512, 512, batchnorm=False),  # bs x 512 x 1 x 1
       ]

       # decoder/upsample convs
       self.decoders_cholec80 = [
           UpSampleConv(512, 512, dropout=True),  # bs x 512 x 2 x 2
           UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 4 x 4
           UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 8 x 8
           UpSampleConv(1024, 512),  # bs x 512 x 16 x 16
           UpSampleConv(1024, 256),  # bs x 256 x 32 x 32
           UpSampleConv(512, 128),  # bs x 128 x 64 x 64
           UpSampleConv(256, 64),  # bs x 64 x 128 x 128
       ]
       self.decoders_depth = [
           UpSampleConv(512, 512, dropout=True),  # bs x 512 x 2 x 2
           UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 4 x 4
           UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 8 x 8
           UpSampleConv(1024, 512),  # bs x 512 x 16 x 16
           UpSampleConv(1024, 256),  # bs x 256 x 32 x 32
           UpSampleConv(512, 128),  # bs x 128 x 64 x 64
           UpSampleConv(256, 64),  # bs x 64 x 128 x 128
       ]
       self.decoders_seg = [
           UpSampleConv(512, 512, dropout=True),  # bs x 512 x 2 x 2
           UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 4 x 4
           UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 8 x 8
           UpSampleConv(1024, 512),  # bs x 512 x 16 x 16
           UpSampleConv(1024, 256),  # bs x 256 x 32 x 32
           UpSampleConv(512, 128),  # bs x 128 x 64 x 64
           UpSampleConv(256, 64),  # bs x 64 x 128 x 128
       ]
       self.decoder_channels = [512, 512, 512, 512, 256, 128, 64]
       self.final_conv_cholec80 = nn.ConvTranspose2d(64, out_channels_cholec80, kernel_size=4, stride=2, padding=1)
       self.final_conv_depth = nn.ConvTranspose2d(64, out_channels_depth, kernel_size=4, stride=2, padding=1)
       self.final_conv_seg = nn.ConvTranspose2d(64, out_channels_seg, kernel_size=4, stride=2, padding=1)
       self.tanh = nn.Tanh()
       self.sigmoid = nn.Sigmoid()

       self.encoders = nn.ModuleList(self.encoders)
       self.decoders_cholec80 = nn.ModuleList(self.decoders_cholec80)
       self.decoders_depth = nn.ModuleList(self.decoders_depth)
       self.decoders_seg = nn.ModuleList(self.decoders_seg)
       self.metrics = metics
   def forward(self, x):
       skips_cons_cholec80 = []
       for encoder in self.encoders:
           x = encoder(x)

           skips_cons_cholec80.append(x)

       skips_cons_cholec80 = list(reversed(skips_cons_cholec80[:-1]))
       decoders_cholec80 = self.decoders_cholec80[:-1]
       decoders_depth = self.decoders_depth[:-1]
       decoders_seg = self.decoders_seg[:-1]
       x_depth = torch.clone(x)
       x_seg = torch.clone(x)
       for decoder, skip in zip(decoders_cholec80, skips_cons_cholec80):
           x = decoder(x)
           # print(x.shape, skip.shape)
           x = torch.cat((x, skip), axis=1)
       for decoder, skip in zip(decoders_depth, skips_cons_cholec80):
           x_depth = decoder(x_depth)
           # print(x.shape, skip.shape)
           x_depth = torch.cat((x_depth, skip), axis=1)
       for decoder, skip in zip(decoders_seg, skips_cons_cholec80):
           x_seg = decoder(x_seg)
           # print(x.shape, skip.shape)
           x_seg = torch.cat((x_seg, skip), axis=1)
       x = self.decoders_cholec80[-1](x)
       x_depth = self.decoders_depth[-1](x_depth)
       x_seg = self.decoders_seg[-1](x_seg)
       # print(x.shape)
       x = self.final_conv_cholec80(x)
       x_depth = self.final_conv_depth(x_depth)
       x_seg = self.final_conv_seg(x_seg)
       return self.metrics*self.tanh(x), self.metrics*self.tanh(x_depth), self.sigmoid(x_seg)

def _weights_init(m):
   if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
       torch.nn.init.normal_(m.weight, 0.0, 0.02)
   if isinstance(m, nn.BatchNorm2d):
       torch.nn.init.normal_(m.weight, 0.0, 0.02)
       torch.nn.init.constant_(m.bias, 0)

