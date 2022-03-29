from __future__ import print_function, division, absolute_import
import os
import numpy as np
import scipy
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

import network.hg2 as hg2
import network.ve2 as ve2
import network.cg2 as cg2


import torch.autograd.profiler as profiler
class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        with profiler.record_function("positional_enc"):
            embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
            embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
            embed = embed.view(x.shape[0], -1)
            if self.include_input:
                embed = torch.cat((x, embed), dim=-1)
            return embed

    @classmethod
    def from_conf(cls, conf, d_in=3):
        # PyHocon construction
        return cls(
            conf.get_int("num_freqs", 6),
            d_in,
            conf.get_float("freq_factor", np.pi),
            conf.get_bool("include_input", True),
        )
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='xavier', gain=0.02):
        '''
        initializes network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                         or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class MLP(BaseNetwork):
    """
    MLP implemented using 2D convolution
    Neuron number: (257, 1024, 512, 256, 128, 1)
    """
    def __init__(self, in_channels=257, out_channels=1, bias=True, out_sigmoid=True, weight_norm=False):
        super(MLP, self).__init__()
        inter_channels = (1024, 512, 256, 128)
        norm_fn = lambda x: x
        if weight_norm:
            norm_fn = lambda x: nn.utils.weight_norm(x)

        self.conv0 = nn.Sequential(
            norm_fn(nn.Conv2d(in_channels=in_channels, out_channels=inter_channels[0],
                              kernel_size=1, stride=1, padding=0, bias=bias)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv1 = nn.Sequential(
            norm_fn(nn.Conv2d(in_channels=inter_channels[0] + in_channels,
                              out_channels=inter_channels[1],
                              kernel_size=1, stride=1, padding=0, bias=bias)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            norm_fn(nn.Conv2d(in_channels=inter_channels[1] + in_channels,
                              out_channels=inter_channels[2],
                              kernel_size=1, stride=1, padding=0, bias=bias)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            norm_fn(nn.Conv2d(in_channels=inter_channels[2] + in_channels,
                              out_channels=inter_channels[3],
                              kernel_size=1, stride=1, padding=0, bias=bias)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        if out_sigmoid:
            self.conv4 = nn.Sequential(
                nn.Conv2d(in_channels=inter_channels[3], out_channels=out_channels,
                          kernel_size=1, stride=1, padding=0, bias=bias),
                nn.Sigmoid()
            )
        else:
            self.conv4 = nn.Conv2d(in_channels=inter_channels[3], out_channels=out_channels,
                                   kernel_size=1, stride=1, padding=0, bias=bias)
        self.init_weights()

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(torch.cat([x, out], dim=1))
        out = self.conv2(torch.cat([x, out], dim=1))
        out = self.conv3(torch.cat([x, out], dim=1))
        out = self.conv4(out)
        return out

    def forward0(self, x):
        out = self.conv0(x)
        out = self.conv1(torch.cat([x, out], dim=1))
        out = self.conv2(torch.cat([x, out], dim=1))
        out = self.conv3(torch.cat([x, out], dim=1))
        return out

    def forward1(self, x, out):
        out = self.conv4(out)
        return out


class MLP_NeRF(BaseNetwork):
    """
    MLP implemented using 2D convolution
    Neuron number: (257, 1024, 512, 256, 128, 1)
    """
    def __init__(self, in_channels=257, occ_channels=128, out_channels=5, bias=True, weight_norm=False):
        super(MLP_NeRF, self).__init__()
        inter_channels = (1024, 512, 256, 128)
        norm_fn = lambda x: x
        if weight_norm:
            norm_fn = lambda x: nn.utils.weight_norm(x)

        self.conv0 = nn.Sequential(
            norm_fn(nn.Conv2d(in_channels=in_channels, out_channels=inter_channels[0],
                              kernel_size=1, stride=1, padding=0, bias=bias)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv1 = nn.Sequential(
            norm_fn(nn.Conv2d(in_channels=inter_channels[0] + in_channels,
                              out_channels=inter_channels[1],
                              kernel_size=1, stride=1, padding=0, bias=bias)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            norm_fn(nn.Conv2d(in_channels=inter_channels[1] + in_channels,
                              out_channels=inter_channels[2],
                              kernel_size=1, stride=1, padding=0, bias=bias)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            norm_fn(nn.Conv2d(in_channels=inter_channels[2] + in_channels,
                              out_channels=inter_channels[3],
                              kernel_size=1, stride=1, padding=0, bias=bias)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv_color = nn.Conv2d(in_channels=inter_channels[3], out_channels=out_channels-1,
                               kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv_sigma = nn.Sequential(
            norm_fn(nn.Conv2d(in_channels=inter_channels[3],# + occ_channels,
                              out_channels=inter_channels[3],
                              kernel_size=1, stride=1, padding=0, bias=bias)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=inter_channels[3], out_channels=1,
                      kernel_size=1, stride=1, padding=0, bias=bias)
        )
        self.init_weights()

    def forward(self, x):#, feat_occupancy):
        out = self.conv0(x)
        out = self.conv1(torch.cat([x, out], dim=1))
        out = self.conv2(torch.cat([x, out], dim=1))
        out = self.conv3(torch.cat([x, out], dim=1))
        color = self.conv_color(out)
        sigma = self.conv_sigma(out)#torch.cat([out, feat_occupancy], dim=1))
        return torch.cat([color, sigma], dim=1), out

    def forward_sigma(self, x, feat_occupancy):
        out = self.conv0(x)
        out = self.conv1(torch.cat([x, out], dim=1))
        out = self.conv2(torch.cat([x, out], dim=1))
        out = self.conv3(torch.cat([x, out], dim=1))
        sigma = self.conv_sigma(torch.cat([out, feat_occupancy], dim=1))
        return sigma

    def forward_color(self, x):
        out = self.conv0(x)
        out = self.conv1(torch.cat([x, out], dim=1))
        out = self.conv2(torch.cat([x, out], dim=1))
        out = self.conv3(torch.cat([x, out], dim=1))
        color = self.conv_color(out)
        return color

class MLPShallow(BaseNetwork):
    def __init__(self, in_channels, median_channels, out_channels, bias=True):
        super(MLPShallow, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=median_channels,
                      kernel_size=1, stride=1, padding=0, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=median_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=bias),
        )

    def forward(self, x):
        return self.layers(x)


class PamirNet(BaseNetwork):
    """PIVOIF implementation with multi-stage output"""

    def __init__(self):
        super(PamirNet, self).__init__()
        # self.hg = hg.HourglassNet(4, 4, 128, 64)
        # self.mlp = MLP()
        self.feat_ch_2D = 256
        self.feat_ch_3D = 32
        self.add_module('hg', hg2.HourglassNet(2, 3, 128, self.feat_ch_2D))
        self.add_module('ve', ve2.VolumeEncoder(3, self.feat_ch_3D))
        self.add_module('mlp', MLP(self.feat_ch_2D + self.feat_ch_3D, 1, weight_norm=False))

        logging.info('#trainable params of hourglass = %d' %
                     sum(p.numel() for p in self.hg.parameters() if p.requires_grad))
        logging.info('#trainable params of 3d encoder = %d' %
                     sum(p.numel() for p in self.ve.parameters() if p.requires_grad))
        logging.info('#trainable params of mlp = %d' %
                     sum(p.numel() for p in self.mlp.parameters() if p.requires_grad))

    def forward(self, img, vol, pts, pts_proj):
        """
        img: [batchsize * 3 (RGB) * img_h * img_w]
        pts: [batchsize * point_num * 3 (XYZ)]
        """
        batch_size = pts.size()[0]
        point_num = pts.size()[1]
        img_feats = self.hg(img)
        vol_feats = self.ve(vol)
        img_feats = img_feats[-len(vol_feats):]
        pt_sdf_list = []
        h_grid = pts_proj[:, :, 0].view(batch_size, point_num, 1, 1)
        v_grid = pts_proj[:, :, 1].view(batch_size, point_num, 1, 1)
        grid_2d = torch.cat([h_grid, v_grid], dim=-1)
        pts=pts*2.0  # corrects coordinates for torch in-network sampling
        x_grid = pts[:, :, 0].view(batch_size, point_num, 1, 1, 1)
        y_grid = pts[:, :, 1].view(batch_size, point_num, 1, 1, 1)
        z_grid = pts[:, :, 2].view(batch_size, point_num, 1, 1, 1)
        grid_3d = torch.cat([x_grid, y_grid, z_grid], dim=-1)

        for img_feat, vol_feat in zip(img_feats, vol_feats):
            pt_feat_2D = F.grid_sample(input=img_feat, grid=grid_2d,align_corners=False,
                                       mode='bilinear', padding_mode='border')
            pt_feat_3D = F.grid_sample(input=vol_feat, grid=grid_3d,align_corners=False,
                                       mode='bilinear', padding_mode='border')
            pt_feat_3D = pt_feat_3D.view([batch_size, -1, point_num, 1])
            pt_feat = torch.cat([pt_feat_2D, pt_feat_3D], dim=1)
            pt_output = self.mlp(pt_feat)  # shape = [batch_size, channels, point_num, 1]
            pt_output = pt_output.permute([0, 2, 3, 1])
            pt_sdf = pt_output.view([batch_size, point_num, 1])
            pt_sdf_list.append(pt_sdf)
        return pt_sdf_list

    def get_img_feature(self, img, no_grad=True):
        if no_grad:
            with torch.no_grad():
                f = self.hg(img)[-1]
            return f
        else:
            return self.hg(img)[-1]

    def get_vol_feature(self, vol, no_grad=True):
        if no_grad:
            with torch.no_grad():
                f = self.ve(vol, intermediate_output=False)
            return f
        else:
            return self.ve(vol, intermediate_output=False)

    def get_mlp_feature(self, img, vol, pts, pts_proj ):
        batch_size = pts.size()[0]
        point_num = pts.size()[1]
        img_feats = self.hg(img)
        vol_feats = self.ve(vol)
        img_feats = img_feats[-len(vol_feats):]
        pt_sdf_list = []
        h_grid = pts_proj[:, :, 0].view(batch_size, point_num, 1, 1)
        v_grid = pts_proj[:, :, 1].view(batch_size, point_num, 1, 1)
        grid_2d = torch.cat([h_grid, v_grid], dim=-1)
        pts=pts*2.0  # corrects coordinates for torch in-network sampling
        x_grid = pts[:, :, 0].view(batch_size, point_num, 1, 1, 1)
        y_grid = pts[:, :, 1].view(batch_size, point_num, 1, 1, 1)
        z_grid = pts[:, :, 2].view(batch_size, point_num, 1, 1, 1)
        grid_3d = torch.cat([x_grid, y_grid, z_grid], dim=-1)


        pt_feat_2D = F.grid_sample(input=img_feats[-1], grid=grid_2d, align_corners=False,
                                   mode='bilinear', padding_mode='border')
        pt_feat_3D = F.grid_sample(input=vol_feats[-1], grid=grid_3d, align_corners=False,
                                   mode='bilinear', padding_mode='border')
        pt_feat_3D = pt_feat_3D.view([batch_size, -1, point_num, 1])
        pt_feat = torch.cat([pt_feat_2D, pt_feat_3D], dim=1)
        pt_feature = self.mlp.forward0(pt_feat)  # shape = [batch_size, channels, point_num, 1]
        return pt_feature



class PamirNetMultiview(BaseNetwork):
    def __init__(self):
        super(PamirNetMultiview, self).__init__()
        # self.hg = hg.HourglassNet(4, 4, 128, 64)
        # self.mlp = MLP()
        self.feat_ch_2D = 256
        self.feat_ch_3D = 32
        self.add_module('hg', hg2.HourglassNet(4, 3, 128, self.feat_ch_2D))
        self.add_module('ve', ve2.VolumeEncoder(3, self.feat_ch_3D))
        self.add_module('mlp', MLP(self.feat_ch_2D + self.feat_ch_3D, 1))

        logging.info('#trainable params of hourglass = %d' %
                     sum(p.numel() for p in self.hg.parameters() if p.requires_grad))
        logging.info('#trainable params of 3d encoder = %d' %
                     sum(p.numel() for p in self.ve.parameters() if p.requires_grad))
        logging.info('#trainable params of mlp = %d' %
                     sum(p.numel() for p in self.mlp.parameters() if p.requires_grad))

    def forward(self, img, vol, pts, pts_proj, mean_num=3, attention_net=None):
        """
        img: [batchsize * 3 (RGB) * img_h * img_w]
        pts: [batchsize * point_num * 3 (XYZ)]
        """
        view_num = img.size()[0]
        point_num = pts.size()[1]
        img_feats = self.hg(img)
        vol_feats = self.ve(vol)
        img_feats = img_feats[-len(vol_feats):]
        pts = pts.expand(view_num, -1, -1)
        pts_proj = pts_proj.expand(view_num, -1, -1)
        pt_sdf_list = []
        h_grid = pts_proj[:, :, 0].view(view_num, point_num, 1, 1)
        v_grid = pts_proj[:, :, 1].view(view_num, point_num, 1, 1)
        grid_2d = torch.cat([h_grid, v_grid], dim=-1)
        pts=pts*2.0  # corrects coordinates for torch in-network sampling
        x_grid = pts[:, :, 0].view(view_num, point_num, 1, 1, 1)
        y_grid = pts[:, :, 1].view(view_num, point_num, 1, 1, 1)
        z_grid = pts[:, :, 2].view(view_num, point_num, 1, 1, 1)
        grid_3d = torch.cat([x_grid, y_grid, z_grid], dim=-1)

        for img_feat, vol_feat in zip(img_feats, vol_feats):
            pt_feat_2D = F.grid_sample(input=img_feat, grid=grid_2d, align_corners=False,
                                       mode='bilinear', padding_mode='border')
            pt_feat_3D = F.grid_sample(input=vol_feat, grid=grid_3d, align_corners=False,
                                       mode='bilinear', padding_mode='border')
            pt_feat_3D = pt_feat_3D.view([view_num, -1, point_num, 1])
            pt_feat = torch.cat([pt_feat_2D, pt_feat_3D], dim=1)
            pt_out0 = self.mlp.forward0(pt_feat)
            if attention_net is None:
                pt_out0_mean = torch.mean(pt_out0[:mean_num], dim=0, keepdim=True)
                pt_feat_mean = torch.mean(pt_feat[:mean_num], dim=0, keepdim=True)
            else:
                pt_feat_mean, pt_out0_mean = self.weighted_avg(
                    pt_feat[:mean_num], pt_out0[:mean_num], attention_net)
            pt_output = self.mlp.forward1(pt_feat_mean, pt_out0_mean)
            pt_output = pt_output.permute([0, 2, 3, 1])
            pt_sdf = pt_output.view([1, point_num, 1])
            pt_sdf_list.append(pt_sdf)
        return pt_sdf_list

    def get_img_feature(self, img, no_grad=True):
        if no_grad:
            with torch.no_grad():
                f = self.hg(img)[-1]
            return f
        else:
            return self.hg(img)[-1]

    def get_vol_feature(self, vol, no_grad=True):
        if no_grad:
            with torch.no_grad():
                f = self.ve(vol, intermediate_output=False)
            return f
        else:
            return self.ve(vol, intermediate_output=False)

    def weighted_avg(self, pt_feat, pt_intermediate_out, attention_network):
        x = torch.cat([pt_feat, pt_intermediate_out], dim=1)
        y = attention_network(x)
        w = F.softmax(y, dim=0)
        pt_feat_mean = torch.mean(pt_feat * w[:, :pt_feat.size(1)], dim=0, keepdim=True)
        pt_intermediate_out_mean = torch.mean(
            pt_intermediate_out * w[:, pt_feat.size(1):], dim=0, keepdim=True)
        return pt_feat_mean, pt_intermediate_out_mean


class TexPamirNet(BaseNetwork):
    def __init__(self, out_channel=3):
        super(TexPamirNet, self).__init__()
        self.feat_ch_2D = 256
        self.feat_ch_3D = 32
        self.out_channel = out_channel
        self.add_module('cg', cg2.CycleGANEncoder(3, self.feat_ch_2D))
        self.add_module('ve', ve2.VolumeEncoder(6, self.feat_ch_3D))
        self.add_module('mlp', MLP(256 + self.feat_ch_2D + self.feat_ch_3D, out_channel))

    def forward(self, img, vol, pts, pts_proj, img_feat_geo):
        """
        img: [batchsize * 3 (RGB) * img_h * img_w]
        pts: [batchsize * point_num * 3 (XYZ)]
        pts: [batchsize * 256 * point_num ]
        """
        batch_size = pts.size()[0]
        point_num = pts.size()[1]
        img_feat_tex = self.cg(img)
        img_feat = torch.cat([img_feat_tex, img_feat_geo], dim=1)

        h_grid = pts_proj[:, :, 0].view(batch_size, point_num, 1, 1)
        v_grid = pts_proj[:, :, 1].view(batch_size, point_num, 1, 1)
        grid_2d = torch.cat([h_grid, v_grid], dim=-1)

        pts = pts * 2.0  # corrects coordinates for torch in-network sampling
        x_grid = pts[:, :, 0].view(batch_size, point_num, 1, 1, 1)
        y_grid = pts[:, :, 1].view(batch_size, point_num, 1, 1, 1)
        z_grid = pts[:, :, 2].view(batch_size, point_num, 1, 1, 1)
        grid_3d = torch.cat([x_grid, y_grid, z_grid], dim=-1)
        vol_feat = self.ve(vol, intermediate_output=False)

        pt_feat_2D = F.grid_sample(input=img_feat, grid=grid_2d, align_corners=False,
                                   mode='bilinear', padding_mode='border')
        pt_feat_3D = F.grid_sample(input=vol_feat, grid=grid_3d, align_corners=False,
                                   mode='bilinear', padding_mode='border')
        pt_feat_3D = pt_feat_3D.view([batch_size, -1, point_num, 1])

        pt_feat = torch.cat([pt_feat_2D, pt_feat_3D], dim=1)
        pt_tex = self.mlp(pt_feat)
        pt_tex = pt_tex.permute([0, 2, 3, 1])
        pt_tex = pt_tex.view(batch_size, point_num, self.out_channel)
        return pt_tex, pt_feat_3D.squeeze()


class TexPamirNetAttention(BaseNetwork):
    def __init__(self):
        super(TexPamirNetAttention, self).__init__()
        self.feat_ch_2D = 256
        self.feat_ch_3D = 32
        self.add_module('cg', cg2.CycleGANEncoder(3, self.feat_ch_2D))
        self.add_module('ve', ve2.VolumeEncoder(3, self.feat_ch_3D))
        self.add_module('mlp', MLP(256 + self.feat_ch_2D + self.feat_ch_3D, 4))

        logging.info('#trainable params of 2d encoder = %d' %
                     sum(p.numel() for p in self.cg.parameters() if p.requires_grad))
        logging.info('#trainable params of 3d encoder = %d' %
                     sum(p.numel() for p in self.ve.parameters() if p.requires_grad))
        logging.info('#trainable params of mlp = %d' %
                     sum(p.numel() for p in self.mlp.parameters() if p.requires_grad))

    def forward(self, img, vol, pts, pts_proj, img_feat_geo):
        """
        img: [batchsize * 3 (RGB) * img_h * img_w]
        pts: [batchsize * point_num * 3 (XYZ)]
        pts: [batchsize * 256 * point_num ]
        """
        batch_size = pts.size()[0]
        point_num = pts.size()[1]
        img_feat_tex = self.cg(img)
        img_feat = torch.cat([img_feat_tex, img_feat_geo], dim=1)

        h_grid = pts_proj[:, :, 0].view(batch_size, point_num, 1, 1)
        v_grid = pts_proj[:, :, 1].view(batch_size, point_num, 1, 1)
        grid_2d = torch.cat([h_grid, v_grid], dim=-1)

        pts = pts * 2.0  # corrects coordinates for torch in-network sampling
        x_grid = pts[:, :, 0].view(batch_size, point_num, 1, 1, 1)
        y_grid = pts[:, :, 1].view(batch_size, point_num, 1, 1, 1)
        z_grid = pts[:, :, 2].view(batch_size, point_num, 1, 1, 1)
        grid_3d = torch.cat([x_grid, y_grid, z_grid], dim=-1)
        vol_feat = self.ve(vol, intermediate_output=False)

        pt_feat_2D = F.grid_sample(input=img_feat, grid=grid_2d, align_corners=False,
                                   mode='bilinear', padding_mode='border')
        pt_feat_3D = F.grid_sample(input=vol_feat, grid=grid_3d, align_corners=False,
                                   mode='bilinear', padding_mode='border')
        pt_feat_3D = pt_feat_3D.view([batch_size, -1, point_num, 1])

        pt_feat = torch.cat([pt_feat_2D, pt_feat_3D], dim=1)
        pt_out = self.mlp(pt_feat)
        pt_out = pt_out.permute([0, 2, 3, 1])
        pt_out = pt_out.view(batch_size, point_num, 4)
        pt_tex_pred = pt_out[:, :, :3]
        pt_tex_att = pt_out[:, :, 3:]
        pt_tex_sample = F.grid_sample(input=img, grid=grid_2d, align_corners=False,
                                      mode='bilinear', padding_mode='border')
        pt_tex_sample = pt_tex_sample.permute([0, 2, 3, 1]).squeeze(2)
        pt_tex = pt_tex_att * pt_tex_sample + (1 - pt_tex_att) * pt_tex_pred
        return pt_tex_pred, pt_tex, pt_tex_att, pt_feat_3D.squeeze()


import constant as const
class TexPamirNetAttention_nerf(BaseNetwork):
    def __init__(self):
        super(TexPamirNetAttention_nerf, self).__init__()
        self.feat_ch_2D = 256
        self.feat_ch_3D = 32
        self.feat_ch_out = 3 + 1+ 1 #+1
        self.feat_ch_occupancy = 128
        #self.add_module('cg', cg2.CycleGANEncoder(3+2, self.feat_ch_2D))
        self.add_module('cg', hg2.HourglassNet(2, 3, 128, self.feat_ch_2D))
        self.add_module('ve', ve2.VolumeEncoder(3, self.feat_ch_3D))
        num_freq= 10
        self.pe = PositionalEncoding(num_freqs=num_freq, d_in=3, freq_factor=np.pi, include_input=True)
        self.add_module('mlp', MLP(self.feat_ch_2D + self.feat_ch_3D + num_freq*2*3+3, self.feat_ch_out, out_sigmoid=False))
        #self.add_module('mlp',  MLP_NeRF(self.feat_ch_2D  + 3, self.feat_ch_occupancy, self.feat_ch_out))

        logging.info('#trainable params of 2d encoder = %d' %
                     sum(p.numel() for p in self.cg.parameters() if p.requires_grad))
        logging.info('#trainable params of 3d encoder = %d' %
                     sum(p.numel() for p in self.ve.parameters() if p.requires_grad))
        logging.info('#trainable params of mlp = %d' %
                     sum(p.numel() for p in self.mlp.parameters() if p.requires_grad))


    def forward(self, img, vol, pts, pts_proj, return_flow_feature=False):#, img_feat_geo):#, feat_occupancy):
        """
        img: [batchsize * 3 (RGB) * img_h * img_w]
        pts: [batchsize * point_num * 3 (XYZ)]
        pts: [batchsize * 256 * point_num ]
        """
        batch_size = pts.size()[0]
        point_num = pts.size()[1]

        #_2d_grid = self.generate_2d_grids(img.shape[2])
        #_2d_grid = torch.from_numpy(_2d_grid).permute(2, 0, 1).unsqueeze(0).repeat(batch_size, 1, 1, 1).cuda()[:,
        #           [1, 0], :, :]
        #img_gridconcat = torch.cat([img, _2d_grid], 1)
        img_feat = self.cg( img)[-1]#_gridconcat)#[-1]


        h_grid = pts_proj[:, :, 0].view(batch_size, point_num, 1, 1)
        v_grid = pts_proj[:, :, 1].view(batch_size, point_num, 1, 1)
        grid_2d = torch.cat([h_grid, v_grid], dim=-1)

        pts = pts * 2.0  # corrects coordinates for torch in-network sampling
        x_grid = pts[:, :, 0].view(batch_size, point_num, 1, 1, 1)
        y_grid = pts[:, :, 1].view(batch_size, point_num, 1, 1, 1)
        z_grid = pts[:, :, 2].view(batch_size, point_num, 1, 1, 1)
        grid_3d = torch.cat([x_grid, y_grid, z_grid], dim=-1)
        vol_feat = self.ve(vol, intermediate_output=False)

        pt_feat_2D = F.grid_sample(input=img_feat, grid=grid_2d, align_corners=False,
                                   mode='bilinear', padding_mode='border')
        pt_feat_3D = F.grid_sample(input=vol_feat, grid=grid_3d, align_corners=False,
                                   mode='bilinear', padding_mode='border')
        pt_feat_3D = pt_feat_3D.view([batch_size, -1, point_num, 1])

        pt_feat = torch.cat([pt_feat_2D, pt_feat_3D], dim=1) #batch_size, ch, point_num, 1
        #pt_feat = pt_feat_2D

        ##add coordinate
        pts_pe= self.pe(pts.reshape(-1, 3)).reshape(batch_size, point_num, -1) #batch_size, point_num, ch

        #import pdb; pdb.set_trace()
        pt_out= self.mlp(torch.cat([pt_feat, pts_pe.permute(0, 2, 1).unsqueeze(-1)], dim=1))
        #pt_out = self.mlp(torch.cat([pt_feat, pts.permute(0, 2, 1).unsqueeze(-1)], dim=1))

        pt_out = pt_out.permute([0, 2, 3, 1])
        #import pdb; pdb.set_trace()
        pt_out = pt_out.view(batch_size, point_num, self.feat_ch_out)
        pt_tex_pred = pt_out[:, :, :3].sigmoid()
        # pt_tex_coord = pt_out[:, :, 3:5].unsqueeze(2)
        pt_tex_att = pt_out[:, :, 3:4].sigmoid()
        pt_tex_sigma = pt_out[:, :, -1:].sigmoid()

        ##
        #grid_2d_offset = pt_tex_coord + grid_2d


        pt_tex_sample = F.grid_sample(input=img, grid=grid_2d, align_corners=False,
                                      mode='bilinear', padding_mode='border')
        pt_tex_sample = pt_tex_sample.permute([0, 2, 3, 1]).squeeze(2)
        pt_tex = pt_tex_att * pt_tex_sample + (1 - pt_tex_att) * pt_tex_pred
        if return_flow_feature:
            return torch.cat([grid_2d.squeeze(-2), pt_tex_pred], dim=-1), torch.cat([feature.permute(0,2,1,3).squeeze(-1), pt_tex_att],dim=-1), pt_tex_att, None, pt_tex_sigma

        return pt_tex_pred, pt_tex, pt_tex_att, pt_feat_3D.squeeze(), pt_tex_sigma

    def generate_2d_grids(self, res):
        x_coords = np.array(range(0, res), dtype=np.float32)
        y_coords = np.array(range(0, res), dtype=np.float32)

        yv, xv= np.meshgrid(x_coords, y_coords)
        xv = np.reshape(xv, (res, res, 1))
        yv = np.reshape(yv, (res, res, 1))

        xv = xv / res - 0.5 + 0.5 / res
        yv = yv / res - 0.5 + 0.5 / res

        pts = np.concatenate([xv, yv], axis=-1)
        pts = np.float32(pts)
        pts *= 2.0

        return pts
class TexPamirNetAttention_nerf1(BaseNetwork):
    def __init__(self):
        super(TexPamirNetAttention_nerf1, self).__init__()
        self.feat_ch_2D = 256
        self.feat_ch_3D = 32
        self.feat_ch_out = 3 + 1+ 1 #+1
        self.feat_ch_occupancy = 128
        self.add_module('cg', cg2.CycleGANEncoder(3+2, self.feat_ch_2D))
        #self.add_module('cg', hg2.HourglassNet(3+2 ,2, 3, 128, self.feat_ch_2D))
        self.add_module('ve', ve2.VolumeEncoder(3, self.feat_ch_3D))
        num_freq= 10
        self.pe = PositionalEncoding(num_freqs=num_freq, d_in=3, freq_factor=np.pi, include_input=True)
        self.add_module('mlp', MLP_NeRF(self.feat_ch_2D + self.feat_ch_3D + num_freq*2*3+3, self.feat_ch_occupancy, self.feat_ch_out))
        #self.add_module('mlp',  MLP_NeRF(self.feat_ch_2D  + 3, self.feat_ch_occupancy, self.feat_ch_out))

        logging.info('#trainable params of 2d encoder = %d' %
                     sum(p.numel() for p in self.cg.parameters() if p.requires_grad))
        logging.info('#trainable params of 3d encoder = %d' %
                     sum(p.numel() for p in self.ve.parameters() if p.requires_grad))
        logging.info('#trainable params of mlp = %d' %
                     sum(p.numel() for p in self.mlp.parameters() if p.requires_grad))


    def forward(self, img, vol, pts, pts_proj, return_flow_feature=False):#, img_feat_geo):#, feat_occupancy):
        """
        img: [batchsize * 3 (RGB) * img_h * img_w]
        pts: [batchsize * point_num * 3 (XYZ)]
        pts: [batchsize * 256 * point_num ]
        """
        batch_size = pts.size()[0]
        point_num = pts.size()[1]

        _2d_grid = self.generate_2d_grids(img.shape[2])
        _2d_grid = torch.from_numpy(_2d_grid).permute(2, 0, 1).unsqueeze(0).repeat(batch_size, 1, 1, 1).cuda()[:,
                   [1, 0], :, :]
        img_gridconcat = torch.cat([img, _2d_grid], 1)
        img_feat_tex = self.cg( img_gridconcat)#[-1]
        img_feat = img_feat_tex#torch.cat([img_feat_tex, img_feat_geo], dim=1)

        h_grid = pts_proj[:, :, 0].view(batch_size, point_num, 1, 1)
        v_grid = pts_proj[:, :, 1].view(batch_size, point_num, 1, 1)
        grid_2d = torch.cat([h_grid, v_grid], dim=-1)

        pts = pts * 2.0  # corrects coordinates for torch in-network sampling
        x_grid = pts[:, :, 0].view(batch_size, point_num, 1, 1, 1)
        y_grid = pts[:, :, 1].view(batch_size, point_num, 1, 1, 1)
        z_grid = pts[:, :, 2].view(batch_size, point_num, 1, 1, 1)
        grid_3d = torch.cat([x_grid, y_grid, z_grid], dim=-1)
        vol_feat = self.ve(vol, intermediate_output=False)

        pt_feat_2D = F.grid_sample(input=img_feat, grid=grid_2d, align_corners=False,
                                   mode='bilinear', padding_mode='border')
        pt_feat_3D = F.grid_sample(input=vol_feat, grid=grid_3d, align_corners=False,
                                   mode='bilinear', padding_mode='border')
        pt_feat_3D = pt_feat_3D.view([batch_size, -1, point_num, 1])

        pt_feat = torch.cat([pt_feat_2D, pt_feat_3D], dim=1) #batch_size, ch, point_num, 1
        #pt_feat = pt_feat_2D

        ##add coordinate
        pts_pe= self.pe(pts.reshape(-1, 3)).reshape(batch_size, point_num, -1) #batch_size, point_num, ch

        #import pdb; pdb.set_trace()
        pt_out, feature = self.mlp(torch.cat([pt_feat, pts_pe.permute(0, 2, 1).unsqueeze(-1)], dim=1))
        #pt_out = self.mlp(torch.cat([pt_feat, pts.permute(0, 2, 1).unsqueeze(-1)], dim=1))

        pt_out = pt_out.permute([0, 2, 3, 1])
        pt_out = pt_out.view(batch_size, point_num, self.feat_ch_out)
        pt_tex_pred = pt_out[:, :, :3].sigmoid()
        # pt_tex_coord = pt_out[:, :, 3:5].unsqueeze(2)
        pt_tex_att = pt_out[:, :, 3:4].sigmoid()
        pt_tex_sigma = pt_out[:, :, -1:].sigmoid()

        ##
        #grid_2d_offset = pt_tex_coord + grid_2d


        pt_tex_sample = F.grid_sample(input=img, grid=grid_2d, align_corners=False,
                                      mode='bilinear', padding_mode='border')
        pt_tex_sample = pt_tex_sample.permute([0, 2, 3, 1]).squeeze(2)
        pt_tex = pt_tex_att * pt_tex_sample + (1 - pt_tex_att) * pt_tex_pred
        if return_flow_feature:
            return torch.cat([grid_2d.squeeze(-2), pt_tex_pred], dim=-1), torch.cat([feature.permute(0,2,1,3).squeeze(-1), pt_tex_att],dim=-1), pt_tex_att, None, pt_tex_sigma

        return pt_tex_pred, pt_tex, pt_tex_att, pt_feat_3D.squeeze(), pt_tex_sigma

    def generate_2d_grids(self, res):
        x_coords = np.array(range(0, res), dtype=np.float32)
        y_coords = np.array(range(0, res), dtype=np.float32)

        yv, xv= np.meshgrid(x_coords, y_coords)
        xv = np.reshape(xv, (res, res, 1))
        yv = np.reshape(yv, (res, res, 1))

        xv = xv / res - 0.5 + 0.5 / res
        yv = yv / res - 0.5 + 0.5 / res

        pts = np.concatenate([xv, yv], axis=-1)
        pts = np.float32(pts)
        pts *= 2.0

        return pts
class TexPamirNetAttention_nerf_multiview(BaseNetwork):
    def __init__(self):
        super(TexPamirNetAttention_nerf_multiview, self).__init__()
        self.feat_ch_2D = 256
        self.feat_ch_3D = 32
        self.feat_ch_out = 3 + 3+ 1 #+1
        self.feat_ch_occupancy = 128
        #self.add_module('cg', cg2.CycleGANEncoder(3, self.feat_ch_2D))
        self.add_module('cg', hg2.HourglassNet(2, 3, 128, self.feat_ch_2D))
        self.add_module('ve', ve2.VolumeEncoder(3, self.feat_ch_3D))
        num_freq= 10
        self.pe = PositionalEncoding(num_freqs=num_freq, d_in=3, freq_factor=np.pi, include_input=True)
        self.add_module('mlp',MLP( self.feat_ch_3D +(self.feat_ch_2D + num_freq * 2 * 3 + 3)*2, self.feat_ch_out, out_sigmoid=False))


        logging.info('#trainable params of 2d encoder = %d' %
                     sum(p.numel() for p in self.cg.parameters() if p.requires_grad))
        logging.info('#trainable params of 3d encoder = %d' %
                     sum(p.numel() for p in self.ve.parameters() if p.requires_grad))
        logging.info('#trainable params of mlp = %d' %
                     sum(p.numel() for p in self.mlp.parameters() if p.requires_grad))


    def forward(self, img, vol, pts, pts_proj, return_flow_feature=False):#, img_feat_geo):#, feat_occupancy):
        """
        img:  [batchsize * source_num* 3 (RGB) * img_h * img_w]
        pts: [batchsize * source_num*point_num * 3 (XYZ)]
        vol       [batchsize * 3 *vol_res*vol_res*vol_res]
        """
        batch_size, point_num = pts[:,0].size()[:2]

        pts_first = pts[:,0] * 2.0  # only use first source

        x_grid = pts_first[:, :, 0].view(batch_size, point_num, 1, 1, 1)
        y_grid = pts_first[:, :, 1].view(batch_size, point_num, 1, 1, 1)
        z_grid = pts_first[:, :, 2].view(batch_size, point_num, 1, 1, 1)
        grid_3d = torch.cat([x_grid, y_grid, z_grid], dim=-1)
        vol_feat = self.ve(vol, intermediate_output=False)

        pt_feat_3D = F.grid_sample(input=vol_feat, grid=grid_3d, align_corners=False,mode='bilinear', padding_mode='border')
        pt_feat_3D = pt_feat_3D.view([batch_size, -1, point_num, 1])



        #_2d_grid = self.generate_2d_grids(img.shape[2])
        #_2d_grid = torch.from_numpy(_2d_grid).permute(2, 0, 1).unsqueeze(0).repeat(batch_size, 1, 1, 1).cuda()[:,[1, 0], :, :]
        #img_gridconcat = torch.cat([img, _2d_grid], 1)

        pt_out0_list=[]
        pt_tex_sample_list=[]
        for i in range(img.size(1)) :

            img_feat = self.cg(img[:,i])[-1]
            h_grid = pts_proj[:,i][:, :, 0].view(batch_size, point_num, 1, 1)
            v_grid = pts_proj[:,i][:, :, 1].view(batch_size, point_num, 1, 1)
            grid_2d = torch.cat([h_grid, v_grid], dim=-1)

            #sample rgb
            pt_tex_sample = F.grid_sample(input=img[:,i], grid=grid_2d, align_corners=False,
                                          mode='bilinear', padding_mode='border')
            pt_tex_sample = pt_tex_sample.permute([0, 2, 3, 1]).squeeze(2)
            pt_tex_sample_list.append(pt_tex_sample)
            #sample feature
            pt_feat_2D = F.grid_sample(input=img_feat, grid=grid_2d, align_corners=False,
                                       mode='bilinear', padding_mode='border')

            pts_pe = self.pe(pts[:,i].reshape(-1, 3)).reshape(batch_size, point_num, -1)  # batch_size, point_num, ch
            pt_out0_list.append(torch.cat([pt_feat_2D, pts_pe.permute(0, 2, 1).unsqueeze(-1)], dim=1))


        pt_tex_sample = torch.stack(pt_tex_sample_list, 1)
        pt_out0 = torch.cat(pt_out0_list, 1)
        pt_out0 = torch.cat([pt_feat_3D, pt_out0],1)

        pt_out = self.mlp.forward(pt_out0)
        pt_out = pt_out.permute([0, 2, 3, 1])
        pt_out = pt_out.view(batch_size, point_num, self.feat_ch_out)
        pt_tex_pred = pt_out[:, :, :3].sigmoid()
        pt_tex_att = pt_out[:, :, 3:6]#.sigmoid() # b, num, 3 (sourcenum)
        pt_tex_sigma = pt_out[:, :, -1:].sigmoid()



        pt_tex_att= torch.softmax(pt_tex_att, dim=-1)
        pt_tex = pt_tex_pred*pt_tex_att[:,:, 0:1]
        for i in range(img.size(1)):
            pt_tex = pt_tex+ pt_tex_sample[:,i]*pt_tex_att[:,:, i+1:i+2]


        #pt_tex = pt_tex_att * pt_tex_sample[:,i] + (1 - pt_tex_att) * pt_tex_pred
        if return_flow_feature:
            return grid_2d.squeeze(-2), pt_feat_3D.permute(0,2,1,3).squeeze(-1), pt_tex_att, None, pt_tex_sigma


        return pt_tex_pred, pt_tex, pt_tex_att, pt_feat_3D.squeeze(), pt_tex_sigma

    def generate_2d_grids(self, res):
        x_coords = np.array(range(0, res), dtype=np.float32)
        y_coords = np.array(range(0, res), dtype=np.float32)

        yv, xv= np.meshgrid(x_coords, y_coords)
        xv = np.reshape(xv, (res, res, 1))
        yv = np.reshape(yv, (res, res, 1))

        xv = xv / res - 0.5 + 0.5 / res
        yv = yv / res - 0.5 + 0.5 / res

        pts = np.concatenate([xv, yv], axis=-1)
        pts = np.float32(pts)
        pts *= 2.0

        return pts

class TexPamirNetAttentionMultiview(BaseNetwork):
    def __init__(self):
        super(TexPamirNetAttentionMultiview, self).__init__()
        self.feat_ch_2D = 256
        self.feat_ch_3D = 32
        self.add_module('cg', cg2.CycleGANEncoder(3, self.feat_ch_2D))
        self.add_module('ve', ve2.VolumeEncoder(6, self.feat_ch_3D))
        self.add_module('mlp', MLP(256 + self.feat_ch_2D + self.feat_ch_3D, 4))

    def forward(self, img, vol, pts, pts_proj, img_feat_geo, mean_num=3):
        """
        img: [batchsize * 3 (RGB) * img_h * img_w]
        pts: [batchsize * point_num * 3 (XYZ)]
        pts: [batchsize * 256 * point_num ]
        """
        batch_size = pts.size()[0]
        point_num = pts.size()[1]
        img_feat_tex = self.cg(img)
        img_feat = torch.cat([img_feat_tex, img_feat_geo], dim=1)

        h_grid = pts_proj[:, :, 0].view(batch_size, point_num, 1, 1)
        v_grid = pts_proj[:, :, 1].view(batch_size, point_num, 1, 1)
        grid_2d = torch.cat([h_grid, v_grid], dim=-1)

        pts = pts * 2.0  # corrects coordinates for torch in-network sampling
        x_grid = pts[:, :, 0].view(batch_size, point_num, 1, 1, 1)
        y_grid = pts[:, :, 1].view(batch_size, point_num, 1, 1, 1)
        z_grid = pts[:, :, 2].view(batch_size, point_num, 1, 1, 1)
        grid_3d = torch.cat([x_grid, y_grid, z_grid], dim=-1)
        vol_feat = self.ve(vol, intermediate_output=False)

        pt_feat_2D = F.grid_sample(input=img_feat, grid=grid_2d, align_corners=False,
                                   mode='bilinear', padding_mode='border')
        pt_feat_3D = F.grid_sample(input=vol_feat, grid=grid_3d, align_corners=False,
                                   mode='bilinear', padding_mode='border')
        pt_feat_3D = pt_feat_3D.view([batch_size, -1, point_num, 1])

        pt_feat = torch.cat([pt_feat_2D, pt_feat_3D], dim=1)
        pt_out0 = self.mlp.forward0(pt_feat)
        pt_out0_mean = torch.mean(pt_out0[:mean_num], dim=0, keepdim=True)
        pt_feat_mean = torch.mean(pt_feat[:mean_num], dim=0, keepdim=True)
        pt_out_multiview = self.mlp.forward1(pt_feat_mean, pt_out0_mean)
        pt_out_multiview = pt_out_multiview.permute([0, 2, 3, 1]).reshape((1, -1, 4))
        pt_tex_pred_multiview = pt_out_multiview[:, :, :3]

        pt_out = self.mlp(pt_feat)
        pt_out = pt_out.permute([0, 2, 3, 1])
        pt_out = pt_out.view(batch_size, point_num, 4)
        pt_tex_att = pt_out[:, :, 3:]
        pt_tex_sample = F.grid_sample(input=img, grid=grid_2d, align_corners=False,
                                      mode='bilinear', padding_mode='border')
        pt_tex_sample = pt_tex_sample.permute([0, 2, 3, 1]).squeeze()
        pt_tex = pt_tex_att * pt_tex_sample + (1 - pt_tex_att) * pt_tex_pred_multiview
        return pt_tex_pred_multiview, pt_tex, pt_tex_att, pt_feat_3D.squeeze()

    def forward_testing(self, img, vol, pts, pts_proj, img_feat_geo, mean_num=3):
        """
        img: [batchsize * 3 (RGB) * img_h * img_w]
        pts: [batchsize * point_num * 3 (XYZ)]
        pts: [batchsize * 256 * point_num ]
        """
        batch_size = pts.size()[0]
        point_num = pts.size()[1]
        img_feat_tex = self.cg(img)
        img_feat = torch.cat([img_feat_tex, img_feat_geo], dim=1)

        h_grid = pts_proj[:, :, 0].view(batch_size, point_num, 1, 1)
        v_grid = pts_proj[:, :, 1].view(batch_size, point_num, 1, 1)
        grid_2d = torch.cat([h_grid, v_grid], dim=-1)

        pts = pts * 2.0  # corrects coordinates for torch in-network sampling
        x_grid = pts[:, :, 0].view(batch_size, point_num, 1, 1, 1)
        y_grid = pts[:, :, 1].view(batch_size, point_num, 1, 1, 1)
        z_grid = pts[:, :, 2].view(batch_size, point_num, 1, 1, 1)
        grid_3d = torch.cat([x_grid, y_grid, z_grid], dim=-1)
        vol_feat = self.ve(vol, intermediate_output=False)

        pt_feat_2D = F.grid_sample(input=img_feat, grid=grid_2d, align_corners=False,
                                   mode='bilinear', padding_mode='border')
        pt_feat_3D = F.grid_sample(input=vol_feat, grid=grid_3d, align_corners=False,
                                   mode='bilinear', padding_mode='border')
        pt_feat_3D = pt_feat_3D.view([batch_size, -1, point_num, 1])

        pt_feat = torch.cat([pt_feat_2D, pt_feat_3D], dim=1)
        pt_out0 = self.mlp.forward0(pt_feat)
        pt_out0_mean = torch.mean(pt_out0[:mean_num], dim=0, keepdim=True)
        pt_feat_mean = torch.mean(pt_feat[:mean_num], dim=0, keepdim=True)
        pt_out_multiview = self.mlp.forward1(pt_feat_mean, pt_out0_mean)
        pt_out_multiview = pt_out_multiview.permute([0, 2, 3, 1]).reshape((1, -1, 4))
        pt_tex_pred_multiview = pt_out_multiview[:, :, :3]

        pt_out = self.mlp(pt_feat)
        pt_out = pt_out.permute([0, 2, 3, 1])
        pt_out = pt_out.view(batch_size, point_num, 4)
        pt_tex_att = pt_out[:, :, 3:]
        pt_tex_sample = F.grid_sample(input=img, grid=grid_2d, align_corners=False,
                                      mode='bilinear', padding_mode='border')
        pt_tex_sample = pt_tex_sample.permute([0, 2, 3, 1]).squeeze()
        pt_tex_alpha_pred = torch.sum(pt_tex_att, dim=0).unsqueeze(0)
        pt_tex_alpha_pred = torch.max(torch.zeros_like(pt_tex_alpha_pred), pt_tex_alpha_pred)
        pt_tex = torch.sum(pt_tex_att * pt_tex_sample, dim=0).unsqueeze(0) + \
                 pt_tex_alpha_pred * pt_tex_pred_multiview
        pt_tex = pt_tex / (torch.sum(pt_tex_att, dim=0).unsqueeze(0) + pt_tex_alpha_pred)

        # pt_tex = pt_tex_att * pt_tex_sample + (1 - pt_tex_att) * pt_tex_pred_multiview
        return pt_tex_pred_multiview, pt_tex, pt_tex_att, pt_feat_3D.squeeze()


#from kornia.filters import filter2D
from math import log2

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2D(x, f, normalized=True)

class NeuralRenderer(nn.Module):
    ''' Neural renderer class
    Args:
        n_feat (int): number of features
        input_dim (int): input dimension; if not equal to n_feat,
            it is projected to n_feat with a 1x1 convolution
        out_dim (int): output dimension
        final_actvn (bool): whether to apply a final activation (sigmoid)
        min_feat (int): minimum features
        img_size (int): output image size
        use_rgb_skip (bool): whether to use RGB skip connections
        upsample_feat (str): upsampling type for feature upsampling
        upsample_rgb (str): upsampling type for rgb upsampling
        use_norm (bool): whether to use normalization
    '''

    def __init__(
            self, n_feat=128, input_dim=128, out_dim=3, final_actvn=True,
            min_feat=32, img_size=64, feature_size=32, use_rgb_skip=True,
            upsample_feat="nn", upsample_rgb="bilinear", use_norm=False,
            **kwargs):
        super().__init__()
        self.final_actvn = final_actvn
        self.input_dim = input_dim
        self.use_rgb_skip = use_rgb_skip
        self.use_norm = use_norm
        n_blocks = int(log2(img_size)) - int(log2(feature_size))


        assert(upsample_feat in ("nn", "bilinear"))
        if upsample_feat == "nn":
            self.upsample_2 = nn.Upsample(scale_factor=2.)
        elif upsample_feat == "bilinear":
            self.upsample_2 = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), Blur())

        assert(upsample_rgb in ("nn", "bilinear"))
        if upsample_rgb == "nn":
            self.upsample_rgb = nn.Upsample(scale_factor=2.)
        elif upsample_rgb == "bilinear":
            self.upsample_rgb = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), Blur())

        if n_feat == input_dim:
            self.conv_in = lambda x: x
        else:
            self.conv_in = nn.Conv2d(input_dim, n_feat, 1, 1, 0)

        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1)] +
            [nn.Conv2d(max(n_feat // (2 ** (i + 1)), min_feat),
                       max(n_feat // (2 ** (i + 2)), min_feat), 3, 1, 1)
                for i in range(0, n_blocks - 1)]
        )
        if use_rgb_skip:
            self.conv_rgb = nn.ModuleList(
                [nn.Conv2d(input_dim, out_dim, 3, 1, 1)] +
                [nn.Conv2d(max(n_feat // (2 ** (i + 1)), min_feat),
                           out_dim, 3, 1, 1) for i in range(0, n_blocks)]
            )
        else:
            self.conv_rgb = nn.Conv2d(
                max(n_feat // (2 ** (n_blocks)), min_feat), 3, 1, 1)

        if use_norm:
            self.norms = nn.ModuleList([
                nn.InstanceNorm2d(max(n_feat // (2 ** (i + 1)), min_feat))
                for i in range(n_blocks)
            ])
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):

        net = self.conv_in(x)

        if self.use_rgb_skip:
            rgb = self.upsample_rgb(self.conv_rgb[0](x))

        for idx, layer in enumerate(self.conv_layers):
            hid = layer(self.upsample_2(net))
            if self.use_norm:
                hid = self.norms[idx](hid)
            net = self.actvn(hid)

            if self.use_rgb_skip:
                rgb = rgb + self.conv_rgb[idx + 1](net)
                if idx < len(self.conv_layers) - 1:
                    rgb = self.upsample_rgb(rgb)

        if not self.use_rgb_skip:
            rgb = self.conv_rgb(net)

        if self.final_actvn:
            rgb = torch.sigmoid(rgb)
        return rgb


class NeuralRenderer_coord(nn.Module):
    ''' Neural renderer class
    Args:
        n_feat (int): number of features
        input_dim (int): input dimension; if not equal to n_feat,
            it is projected to n_feat with a 1x1 convolution
        out_dim (int): output dimension
        final_actvn (bool): whether to apply a final activation (sigmoid)
        min_feat (int): minimum features
        img_size (int): output image size
        use_rgb_skip (bool): whether to use RGB skip connections
        upsample_feat (str): upsampling type for feature upsampling
        upsample_rgb (str): upsampling type for rgb upsampling
        use_norm (bool): whether to use normalization
    '''

    def __init__(
            self, n_feat=64):
        super().__init__()

        layers = [nn. Linear(n_feat, n_feat), nn.LeakyReLU(0.2), nn.Linear(n_feat, 2), nn.Tanh()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        coord = self.layers(x)
        return coord



class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout,
                                3, stride=1, padding=1, bias=is_bias)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(
                self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def actvn(self, x):
        out = F.leaky_relu(x, 2e-1)
        return out

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(self.actvn(x))
        dx = self.conv_1(self.actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s



class ResDecoder(nn.Module):
    def __init__(self,):
        super().__init__()
        blocks = []
        nf = 128
        self.nlayers = 4
        for i in range(self.nlayers):
            nf0 = nf // (2 ** i)
            nf1 = nf // (2 ** (i + 1))

            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]
        blocks += [
            ResnetBlock(nf // (2 ** self.nlayers), nf // (2 ** self.nlayers))
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf // (2 ** self.nlayers), 3, 3, padding=1)

    def forward(self, x):

        pixels = x

        for block in self.resnet:
            pixels = block(pixels)

            # pixels = self.resnet(pixels)
        pixels = self.conv_img(F.leaky_relu(pixels, 2e-1))
        pixels = torch.sigmoid(pixels)

        return pixels
