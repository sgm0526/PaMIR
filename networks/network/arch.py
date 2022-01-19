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
        self.add_module('hg', hg2.HourglassNet(4, 3, 128, self.feat_ch_2D))
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
        pts *= 2.0  # corrects coordinates for torch in-network sampling
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
        pts *= 2.0  # corrects coordinates for torch in-network sampling
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
class TexPamirNetAttention_uv(BaseNetwork):
    def __init__(self):
        super(TexPamirNetAttention_uv, self).__init__()
        self.feat_ch_2D = 256
        self.feat_ch_3D = 32
        self.add_module('cg', cg2.CycleGANEncoder(3+2, self.feat_ch_2D))
        self.add_module('ve', ve2.VolumeEncoder(2, self.feat_ch_3D))
        self.add_module('mlp', MLP(256 + self.feat_ch_2D + self.feat_ch_3D, 2, out_sigmoid=False))
        self.add_module('mlp_refine', MLP(256 + self.feat_ch_2D, 4))

        logging.info('#trainable params of 2d encoder = %d' %
                     sum(p.numel() for p in self.cg.parameters() if p.requires_grad))
        logging.info('#trainable params of 3d encoder = %d' %
                     sum(p.numel() for p in self.ve.parameters() if p.requires_grad))
        logging.info('#trainable params of mlp = %d' %
                     sum(p.numel() for p in self.mlp.parameters() if p.requires_grad))

        #self.pe = PositionalEncoding(num_freqs=6, d_in=2, freq_factor=np.pi, include_input=True)

    def forward(self, img, vol, pts, pts_proj, img_feat_geo, uv):
        """
        img: [batchsize * 3 (RGB) * img_h * img_w]
        pts: [batchsize * point_num * 3 (XYZ)]
        pts: [batchsize * 256 * point_num ]
        """
        batch_size = pts.size()[0]
        point_num = pts.size()[1]

        _2d_grid = self.generate_2d_grids(img.shape[2])
        _2d_grid = torch.from_numpy(_2d_grid).permute(2, 0, 1).unsqueeze(0).repeat(batch_size, 1, 1, 1).cuda()[:,  [1, 0], :, :]
        #_H = _2d_grid.shape[2]
        #_W = _2d_grid.shape[3]
        #_2d_grid = self.pe(_2d_grid.permute(0, 2, 3, 1).reshape(-1, 2))
        #_2d_grid = _2d_grid.reshape(batch_size, _H, _W, 26).permute(0,3,1,2)

        img_gridconcat = torch.cat([img, _2d_grid], 1)
        img_feat_tex = self.cg(img_gridconcat )
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


        ##
        #pt_feat = torch.cat([grid_2d.permute(0,3,1,2), pt_feat],1)
        pt_out = self.mlp(pt_feat)
        offset = pt_out.permute([0, 2, 3, 1])

        grid_2d_offset =grid_2d + offset
        #import pdb; pdb.set_trace()

        pt_tex_sample = F.grid_sample(input=img, grid=grid_2d_offset, align_corners=False,
                                      mode='bilinear', padding_mode='border')

        pt_tex_sample = pt_tex_sample.permute([0, 2, 3, 1]).squeeze(2)


        return pt_tex_sample,pt_tex_sample, grid_2d_offset, pt_feat_3D.squeeze()#,  rgb_pred


    def generate_3d_grids(self, res):
        x_coords = np.array(range(0, res), dtype=np.float32)
        y_coords = np.array(range(0, res), dtype=np.float32)
        z_coords = np.array(range(0, res), dtype=np.float32)

        yv, xv, zv = np.meshgrid(x_coords, y_coords, z_coords)
        xv = np.reshape(xv, (res, res, res, 1))
        yv = np.reshape(yv, (res, res, res, 1))
        zv = np.reshape(zv, (res, res, res, 1))
        xv = xv / res - 0.5 + 0.5 / res
        yv = yv / res - 0.5 + 0.5 / res
        zv = zv / res - 0.5 + 0.5 / res

        pts = np.concatenate([xv, yv, zv], axis=-1)
        pts = np.float32(pts)
        pts *= 2.0

        return pts


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