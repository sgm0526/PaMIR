"""
This file includes the full training procedure.
"""
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure
from tqdm import tqdm
import scipy.io as sio
import datetime
import glob
import logging
import math
import mrcfile

from network.arch import PamirNet, TexPamirNetAttention,TexPamirNetAttention_nerf, NeuralRenderer, NeuralRenderer_coord
from pifu_lib.model import HGPIFuNet, ResBlkPIFuNet
from pifu_lib import options as opt_pifu
from neural_voxelization_layer.voxelize import Voxelization
from neural_voxelization_layer.smpl_model import TetraSMPL
from util.img_normalization import ImgNormalizerForResnet
from graph_cmr.models import GraphCNN, SMPLParamRegressor2, SMPL
from graph_cmr.utils import Mesh
import util.util as util
import util.obj_io as obj_io
import constant as const
from util.volume_rendering import *
from torchvision.utils import save_image



class EvaluatorTex(object):
    def __init__(self, device, pretrained_checkpoint_pamir, pretrained_checkpoint_pamir_tex, no_weight=False):
        super(EvaluatorTex, self).__init__()
        util.configure_logging(True, False, None)

        self.device = device


        # pamir_net
        opt = opt_pifu.BaseOptions().parse()
        self.pamir_net = HGPIFuNet( opt).to(self.device)
        self.pamir_tex_net = ResBlkPIFuNet( opt).to(self.device)
        self.graph_mesh = Mesh()


        self.models_dict = {'pamir_tex_net': self.pamir_tex_net}
        if not no_weight:
            self.load_pretrained_pamir_net(pretrained_checkpoint_pamir)
            self.load_pretrained(checkpoint_file=pretrained_checkpoint_pamir_tex)
        self.pamir_net.eval()
        self.pamir_tex_net.eval()

    def load_pretrained(self, checkpoint_file=None):
        """Load a pretrained checkpoint.
        This is different from resuming training using --resume.
        """
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            for model in self.models_dict:
                if model in checkpoint:
                    self.models_dict[model].load_state_dict(checkpoint[model])
                    logging.info('Loading pamir_tex_net from ' + checkpoint_file)

    def generate_point_grids(self, vol_res, cam_R, cam_t, cam_f, img_res):
        x_coords = np.array(range(0, vol_res), dtype=np.float32)
        y_coords = np.array(range(0, vol_res), dtype=np.float32)
        z_coords = np.array(range(0, vol_res), dtype=np.float32)

        yv, xv, zv = np.meshgrid(x_coords, y_coords, z_coords)
        xv = np.reshape(xv, (-1, 1))
        yv = np.reshape(yv, (-1, 1))
        zv = np.reshape(zv, (-1, 1))
        xv = xv / vol_res - 0.5 + 0.5 / vol_res
        yv = yv / vol_res - 0.5 + 0.5 / vol_res
        zv = zv / vol_res - 0.5 + 0.5 / vol_res
        pts = np.concatenate([xv, yv, zv], axis=-1)
        pts = np.float32(pts)
        pts_proj = np.dot(pts, cam_R.transpose()) + cam_t
        pts_proj[:, 0] = pts_proj[:, 0] * cam_f / pts_proj[:, 2] / (img_res / 2)
        pts_proj[:, 1] = pts_proj[:, 1] * cam_f / pts_proj[:, 2] / (img_res / 2)
        pts_proj = pts_proj[:, :2]
        return pts, pts_proj

    def test_tex_pifu(self, img, mesh_v):
        #self.pamir_net.eval()
        self.pamir_tex_net.eval()

        group_size = 512 * 128
        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)

        pts = mesh_v
        pts_proj = self.forward_project_points(
            pts, cam_r, cam_t, cam_f, img.size(2))
        clr = self.forward_infer_color_value_group(
            img, pts, pts_proj, group_size)
        return clr


    def forward_project_points(self, pts, cam_r, cam_t, cam_f, img_res):
        pts_proj = pts * cam_r.view((1, 1, -1)) + cam_t.view((1, 1, -1))
        pts_proj = pts_proj * (cam_f / (img_res / 2)) / pts_proj[:, :, 2:3]
        pts_proj = pts_proj[:, :, :2]
        return pts_proj

    def forward_infer_color_value_group(self, img, pts, pts_proj, group_size):
        pts_group_num = (pts.size()[1] + group_size - 1) // group_size
        pts_clr = []
        for gi in tqdm(range(pts_group_num), desc='Texture query'):
            # print('Testing point group: %d/%d' % (gi + 1, pts_group_num))
            pts_group = pts[:, (gi * group_size):((gi + 1) * group_size), :]
            pts_proj_group = pts_proj[:, (gi * group_size):((gi + 1) * group_size), :]
            outputs = self.forward_infer_color_value(
                img, pts_group, pts_proj_group)
            pts_clr.append(np.squeeze(outputs[0].detach().cpu().numpy()))
        pts_clr = np.concatenate(pts_clr)
        pts_clr = np.array(pts_clr)
        return pts_clr

    def forward_infer_color_value(self, img, pts, pts_proj):
        self.pamir_net.filter(img)
        img_feat_geo = self.pamir_net.get_im_feat()
        clr= self.pamir_tex_net.forward(img, img_feat_geo ,pts, pts_proj)#, img_feat_geo, feat_occupancy=None) ##
        return clr


    def load_pretrained_pamir_net(self, model_path):
        data = torch.load(model_path)
        #import pdb; pdb.set_trace()
        self.pamir_net.load_state_dict(data)
        # if os.path.isdir(model_path):
        #     tmp1 = glob.glob(os.path.join(model_path, 'pamir_net*.pt'))
        #     assert len(tmp1) == 1
        #     logging.info('Loading pamir_net from ' + tmp1[0])
        #     data = torch.load(tmp1[0])
        # else:
        #     logging.info('Loading pamir_net from ' + model_path)
        #     data = torch.load(model_path)
        # if 'pamir_net' in data:
        #     self.pamir_net.load_state_dict(data['pamir_net'])
        # else:
        #     raise IOError('Failed to load pamir_net model from the specified checkpoint!!')

    def rotate_points(self, pts, view_id, view_num_per_item=360):
        # rotate points to current view
        angle = 2 * np.pi * view_id / view_num_per_item
        pts_rot = torch.zeros_like(pts)
        if len(pts.size()) == 4:
            angle = angle[:, None, None]
        else:
            angle = angle[:, None]

        pts_rot[..., 0] = pts[..., 0] * angle.cos() - pts[..., 2] * angle.sin()
        pts_rot[..., 1] = pts[..., 1]
        pts_rot[..., 2] = pts[..., 0] * angle.sin() + pts[..., 2] * angle.cos()
        return pts_rot

    def project_points(self, sampled_points, cam_f, cam_c, cam_tz):

        sampled_points_proj = sampled_points.clone()
        sampled_points_proj[..., 1] *= -1
        sampled_points_proj[..., 2] *= -1

        sampled_points_proj[..., 2] += cam_tz  # add cam_t
        sampled_points_proj[..., 0] = sampled_points_proj[..., 0] * cam_f / sampled_points_proj[..., 2] / (cam_c)
        sampled_points_proj[..., 1] = sampled_points_proj[..., 1] * cam_f / sampled_points_proj[..., 2] / (cam_c)
        sampled_points_proj = sampled_points_proj[..., :2]
        return sampled_points_proj

    def project_points2(self, sampled_points, cam_f, cam_c, cam_tz):
        qq = sampled_points[..., 1] * -1
        ww = sampled_points[..., 2] * -1 + cam_tz
        ee = sampled_points[..., 0] * cam_f / ww / (cam_c)
        qq = qq * cam_f / ww / (cam_c)
        sampled_points_proj = torch.cat([ee[..., None], qq[..., None]], dim=-1)

        return sampled_points_proj


    def test_pifu(self, img,  vol_res):
        self.pamir_tex_net.eval()
        #self.graph_cnn.eval()  # lock BN and dropout
        #self.smpl_param_regressor.eval()  # lock BN and dropout

        group_size = 512 * 80
        grid_ov = self.forward_infer_occupancy_value_grid_octree(img,  vol_res, group_size)
        vertices, simplices, normals, _ = measure.marching_cubes_lewiner(grid_ov, 0.5)

        mesh = dict()
        mesh['v'] = vertices / vol_res - 0.5
        mesh['f'] = simplices[:, (1, 0, 2)]
        mesh['vn'] = normals
        return mesh


    def forward_infer_occupancy_feature_grid_naive(self, img, vol, test_res, group_size):
        pts, pts_proj = self.generate_point_grids(
            test_res, const.cam_R, const.cam_t, const.cam_f, img.size(2))
        pts_ov = self.forward_infer_occupancy_value_group(img, vol, pts, pts_proj, group_size)
        pts_ov = pts_ov.reshape([test_res, test_res, test_res])
        return pts_ov


    def forward_infer_occupancy_value_grid_octree(self, img, test_res, group_size,
                                                  init_res=64, ignore_thres=0.05):
        pts, pts_proj = self.generate_point_grids(
            test_res, const.cam_R, const.cam_t, const.cam_f, img.size(2))
        pts = np.reshape(pts, (test_res, test_res, test_res, 3))
        pts_proj = np.reshape(pts_proj, (test_res, test_res, test_res, 2))

        pts_ov = np.zeros([test_res, test_res, test_res])
        dirty = np.ones_like(pts_ov, dtype=np.bool)
        grid_mask = np.zeros_like(pts_ov, dtype=np.bool)

        reso = test_res // init_res
        while reso > 0:
            grid_mask[0:test_res:reso, 0:test_res:reso, 0:test_res:reso] = True
            test_mask = np.logical_and(grid_mask, dirty)

            pts_ = pts[test_mask]
            pts_proj_ = pts_proj[test_mask]
            pts_ov[test_mask] = self.forward_infer_occupancy_value_group(
                img, pts_, pts_proj_, group_size).squeeze()

            if reso <= 1:
                break
            for x in range(0, test_res - reso, reso):
                for y in range(0, test_res - reso, reso):
                    for z in range(0, test_res - reso, reso):
                        # if center marked, return
                        if not dirty[x + reso // 2, y + reso // 2, z + reso // 2]:
                            continue
                        v0 = pts_ov[x, y, z]
                        v1 = pts_ov[x, y, z + reso]
                        v2 = pts_ov[x, y + reso, z]
                        v3 = pts_ov[x, y + reso, z + reso]
                        v4 = pts_ov[x + reso, y, z]
                        v5 = pts_ov[x + reso, y, z + reso]
                        v6 = pts_ov[x + reso, y + reso, z]
                        v7 = pts_ov[x + reso, y + reso, z + reso]
                        v = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
                        v_min = v.min()
                        v_max = v.max()
                        # this cell is all the same
                        if (v_max - v_min) < ignore_thres:
                            pts_ov[x:x + reso, y:y + reso, z:z + reso] = (v_max + v_min) / 2
                            dirty[x:x + reso, y:y + reso, z:z + reso] = False
            reso //= 2
        return pts_ov

    def forward_infer_occupancy_value_group(self, img, pts, pts_proj, group_size):
        assert isinstance(pts, np.ndarray)
        assert len(pts.shape) == 2
        assert pts.shape[1] == 3
        pts_num = pts.shape[0]
        pts = torch.from_numpy(pts).unsqueeze(0).to(self.device)
        pts_proj = torch.from_numpy(pts_proj).unsqueeze(0).to(self.device)


        pts_group_num = (pts.size()[1] + group_size - 1) // group_size
        pts_ov = []
        for gi in tqdm(range(pts_group_num), desc='SDF query'):
            # print('Testing point group: %d/%d' % (gi + 1, pts_group_num))
            pts_group = pts[:,  (gi * group_size):((gi + 1) * group_size), :]
            pts_proj_group = pts_proj[:, (gi * group_size):((gi + 1) * group_size), :]
            outputs = self.forward_infer_occupancy_value(
                img, pts_group, pts_proj_group)
            pts_ov.append(np.squeeze(outputs.detach().cpu().numpy()))
        pts_ov = np.concatenate(pts_ov)
        pts_ov = np.array(pts_ov)
        return pts_ov

    def forward_infer_occupancy_value(self, img, pts, pts_proj):

        return self.pamir_net(img,  pts, pts_proj)



