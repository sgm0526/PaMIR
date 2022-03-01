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

from network.arch import PamirNet, TexPamirNetAttention
from neural_voxelization_layer.voxelize import Voxelization
from neural_voxelization_layer.smpl_model import TetraSMPL
from util.img_normalization import ImgNormalizerForResnet
from graph_cmr.models import GraphCNN, SMPLParamRegressor2, SMPL
from graph_cmr.utils import Mesh
import util.util as util
import util.obj_io as obj_io
import constant as const
from util.volume_rendering import *

class EvaluatorTex(object):
    def __init__(self, device, pretrained_checkpoint_pamir, pretrained_checkpoint_pamir_tex):
        super(EvaluatorTex, self).__init__()
        util.configure_logging(True, False, None)

        self.device = device

        # neural voxelization components
        self.smpl = SMPL('./data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl').to(self.device)
        self.tet_smpl = TetraSMPL('./data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
                                  './data/tetra_smpl.npz').to(self.device)
        smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = \
            util.read_smpl_constants('./data')
        self.smpl_faces = smpl_faces
        self.voxelization = Voxelization(smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras,
                                         volume_res=const.vol_res,
                                         sigma=const.semantic_encoding_sigma,
                                         smooth_kernel_size=const.smooth_kernel_size,
                                         batch_size=1).to(self.device)
        # pamir_net
        self.pamir_net = PamirNet().to(self.device)
        self.pamir_tex_net = TexPamirNetAttention().to(self.device)

        self.models_dict = {'pamir_tex_net': self.pamir_tex_net}
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

    def test_tex_pifu(self, img, mesh_v, betas, pose, scale, trans):
        self.pamir_net.eval()
        self.pamir_tex_net.eval()
        gt_vert_cam = scale * self.tet_smpl(pose, betas) + trans
        vol = self.voxelization(gt_vert_cam)

        group_size = 512 * 128
        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)

        pts = mesh_v
        pts_proj = self.forward_project_points(
            pts, cam_r, cam_t, cam_f, img.size(2))
        clr = self.forward_infer_color_value_group(
            img, vol, pts, pts_proj, group_size)
        return clr

    def test_att_pifu(self, img, mesh_v, betas, pose, scale, trans):
        self.pamir_net.eval()
        self.pamir_tex_net.eval()
        gt_vert_cam = scale * self.tet_smpl(pose, betas) + trans
        vol = self.voxelization(gt_vert_cam)
        group_size = 512 * 128
        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)

        pts = mesh_v
        pts_proj = self.forward_project_points(
            pts, cam_r, cam_t, cam_f, img.size(2))
        att = self.forward_infer_attention_value_group(
            img, vol, pts, pts_proj, group_size)
        att = np.squeeze(att).reshape((-1, 1))
        att = np.concatenate([att, att, att], axis=1)
        return att

    def forward_project_points(self, pts, cam_r, cam_t, cam_f, img_res):
        pts_proj = pts * cam_r.view((1, 1, -1)) + cam_t.view((1, 1, -1))
        pts_proj = pts_proj * (cam_f / (img_res / 2)) / pts_proj[:, :, 2:3]
        pts_proj = pts_proj[:, :, :2]
        return pts_proj

    def forward_infer_color_value_group(self, img, vol, pts, pts_proj, group_size):
        pts_group_num = (pts.size()[1] + group_size - 1) // group_size
        pts_clr = []
        for gi in tqdm(range(pts_group_num), desc='Texture query'):
            # print('Testing point group: %d/%d' % (gi + 1, pts_group_num))
            pts_group = pts[:, (gi * group_size):((gi + 1) * group_size), :]
            pts_proj_group = pts_proj[:, (gi * group_size):((gi + 1) * group_size), :]
            outputs = self.forward_infer_color_value(
                img, vol, pts_group, pts_proj_group)
            pts_clr.append(np.squeeze(outputs[0].detach().cpu().numpy()))
        pts_clr = np.concatenate(pts_clr)
        pts_clr = np.array(pts_clr)
        return pts_clr

    def forward_infer_color_value(self, img, vol, pts, pts_proj):
        img_feat_geo = self.pamir_net.get_img_feature(img, no_grad=True)
        _, clr, _, _ = self.pamir_tex_net.forward(img, vol, pts, pts_proj, img_feat_geo)
        return clr

    def forward_infer_attention_value_group(self, img, vol, pts, pts_proj, group_size):
        pts_group_num = (pts.size()[1] + group_size - 1) // group_size
        pts_att = []
        for gi in tqdm(range(pts_group_num), desc='Texture query'):
            # print('Testing point group: %d/%d' % (gi + 1, pts_group_num))
            pts_group = pts[:, (gi * group_size):((gi + 1) * group_size), :]
            pts_proj_group = pts_proj[:, (gi * group_size):((gi + 1) * group_size), :]
            outputs = self.forward_infer_attention_value(
                img, vol, pts_group, pts_proj_group)
            pts_att.append(np.squeeze(outputs.detach().cpu().numpy()))
        pts_att = np.concatenate(pts_att)
        pts_att = np.array(pts_att)
        return pts_att

    def forward_infer_attention_value(self, img, vol, pts, pts_proj):
        img_feat_geo = self.pamir_net.get_img_feature(img, no_grad=True)
        _, _, att, _ = self.pamir_tex_net.forward(img, vol, pts, pts_proj, img_feat_geo)
        return att

    def load_pretrained_pamir_net(self, model_path):
        if os.path.isdir(model_path):
            tmp1 = glob.glob(os.path.join(model_path, 'pamir_net*.pt'))
            assert len(tmp1) == 1
            logging.info('Loading pamir_net from ' + tmp1[0])
            data = torch.load(tmp1[0])
        else:
            logging.info('Loading pamir_net from ' + model_path)
            data = torch.load(model_path)
        if 'pamir_net' in data:
            self.pamir_net.load_state_dict(data['pamir_net'])
        else:
            raise IOError('Failed to load pamir_net model from the specified checkpoint!!')

    def run_Secant_method(self, img, vol, f_low, f_high, z_low, z_high,n_secant_steps, sampled_rays_d_world, view_diff, threshold):

        batch_size = img.size(0)
        num_rays = z_low.size(1)

        z_pred = - f_low * (z_high - z_low) / (f_high - f_low) + z_low
        for i in range(n_secant_steps):
            p_mid = sampled_rays_d_world.unsqueeze(-2) * z_pred
            p_mid[:, :, :, 2] += const.cam_tz  # batch, numray, z=1, 3

            z_point = self.rotate_points(p_mid, view_diff).reshape(batch_size, num_rays , 3)
            z_point_proj = self.project_points(z_point , const.cam_f, const.cam_c, const.cam_tz).reshape(batch_size, num_rays , 2)
            nerf_output_sigma = self.pamir_net.forward(
                img, vol, z_point, z_point_proj)[-1]
            alphas = nerf_output_sigma.reshape(batch_size, num_rays, 1, 1)

            f_mid = alphas.squeeze(1) - threshold
            inz_low = f_mid < 0
            if inz_low.sum() > 0:
                z_low[inz_low] = z_pred[inz_low]
                f_low[inz_low] = f_mid[inz_low]
            if (inz_low == 0).sum() > 0:
                z_high[inz_low == 0] = z_pred[inz_low == 0]
                f_high[inz_low == 0] = f_mid[inz_low == 0]

            z_pred = - f_low * (z_high - z_low) / (f_high - f_low) + z_low

        return z_pred.data

    def run_Bisection_method(self, img, vol, z_low, z_high, n_secant_steps, sampled_rays_d_world, view_diff, threshold):

        batch_size = img.size(0)
        num_rays = z_low.size(1)

        z_pred = (z_low + z_high) / 2.
        for i in range(n_secant_steps):

            p_mid = sampled_rays_d_world.unsqueeze(-2) * z_pred
            p_mid[:, :, :, 2] += const.cam_tz  # batch, numray, z=1, 3

            z_point = self.rotate_points(p_mid, view_diff).reshape(batch_size, num_rays, 3)
            z_point_proj = self.project_points(z_point, const.cam_f, const.cam_c, const.cam_tz).reshape(batch_size,
                                                                                                      num_rays, 2)
            nerf_output_sigma = self.pamir_net.forward(
                img, vol, z_point, z_point_proj)[-1]
            alphas = nerf_output_sigma.reshape(batch_size, num_rays, 1, 1)

            f_mid = alphas.squeeze(1) - threshold
            inz_low = f_mid < 0
            z_low[inz_low] = z_pred[inz_low]
            z_high[inz_low == 0] = z_pred[inz_low == 0]
            z_pred = 0.5 * (z_low + z_high)


        return z_pred.data
    def test_surface_rendering(self, img, betas, pose, scale, trans, view_diff):
        self.pamir_tex_net.eval()

        gt_vert_cam = scale * self.tet_smpl(pose, betas) + trans
        vol = self.voxelization(gt_vert_cam)

        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)

        batch_size = img.size(0)
        img_size = const.img_res
        fov = 2 * torch.atan(torch.Tensor([cam_c / cam_f])).item()
        fov_degree = fov * 180 / math.pi
        ray_start = const.ray_start  # cam_tz - 0.87  # (
        ray_end = const.ray_end  # cam_tz + 0.87
        num_steps = const.num_steps
        hierarchical = const.hierarchical

        ## todo hierarchical sampling

        points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps,
                                                               resolution=(const.img_res, const.img_res),
                                                               device=self.device, fov=fov_degree, ray_start=ray_start,
                                                               ray_end=ray_end)  # batch_size, pixels, num_steps, 1

        # 1, img_size*img_size, num_steps, 3
        points_cam[:, :, :, 2] += cam_tz
        points_cam_source = self.rotate_points(points_cam, view_diff)
        points_cam_source_proj = self.project_points(points_cam_source, cam_f, cam_c, cam_tz)
        # batch_size, 512*512, num_step, 3

        num_ray = 24000
        img_size = int(img_size )
        pts_group_num = (img_size * img_size + num_ray - 1) // num_ray
        pts_clr_pred = []
        pts_clr_warped = []

        for gi in tqdm(range(pts_group_num), desc='Texture query'):
            # print('Testing point group: %d/%d' % (gi + 1, pts_group_num))
            sampled_points = points_cam_source[:, (gi * num_ray):((gi + 1) * num_ray), :,
                             :]  # 1, group_size, num_step, 3
            sampled_points_proj = points_cam_source_proj[:, (gi * num_ray):((gi + 1) * num_ray), :, :]
            sampled_z_vals = z_vals[:, (gi * num_ray):((gi + 1) * num_ray), :, :]
            sampled_rays_d_world = rays_d_cam[:, (gi * num_ray):((gi + 1) * num_ray)]

            num_ray_part = sampled_points.size(1)
            # num_ray -> num_ray_part

            with torch.no_grad():
                sampled_points = sampled_points.reshape(batch_size, -1, 3)  # 1 group_size*num_step, 3
                sampled_points_proj = sampled_points_proj.reshape(batch_size, -1, 2)

                nerf_output_sigma = self.pamir_net.forward(
                    img, vol, sampled_points, sampled_points_proj)[-1]

                alphas = nerf_output_sigma.reshape(batch_size, num_ray_part, num_steps, -1)

                # max_index = abs(alphas - 0.5).argmin(dim=2)
                threshold = const.threshold
                sign = (alphas > threshold).int().squeeze(-1) * torch.linspace(2, 1, num_steps)[None,][None,].repeat(
                    batch_size, num_ray_part, 1).to(self.device)
                max_index = sign.unsqueeze(-1).argmax(dim=2)
                ray_mask = max_index != 0
                max_index[max_index == 0] += 1
                start_index = max_index - 1
                start_z_vals = torch.gather(sampled_z_vals, 2, start_index.unsqueeze(-1))
                end_z_vals = torch.gather(sampled_z_vals, 2, max_index.unsqueeze(-1))

                start_alphas_vals = torch.gather(alphas, 2, start_index.unsqueeze(-1))
                end_alphas_vals = torch.gather(alphas, 2, max_index.unsqueeze(-1))

                #z_pred = start_z_vals
                z_pred = self.run_Bisection_method(img, vol, start_z_vals, end_z_vals,
                                                3, sampled_rays_d_world, view_diff, threshold)
                #z_pred = self.run_Secant_method(img, vol, start_alphas_vals, end_alphas_vals, start_z_vals, end_z_vals,
                #                                3, sampled_rays_d_world, view_diff)

                p_mid = sampled_rays_d_world.unsqueeze(-2) * z_pred
                p_mid[:, :, :, 2] += const.cam_tz  # batch, numray, z=1, 3
                z_point = self.rotate_points(p_mid, view_diff).reshape(batch_size, num_ray_part, 3)
                z_point_proj = self.project_points(z_point, const.cam_f, const.cam_c, const.cam_tz).reshape(batch_size,
                                                                                                          num_ray_part, 2)

                img_feat_geo = self.pamir_net.get_img_feature(img, no_grad=True)
                pixels_pred, feature_pred, _, _= self.pamir_tex_net.forward(
                    img, vol, z_point ,z_point_proj,img_feat_geo)

                pixels_pred[~ray_mask[..., 0]] = 1
                feature_pred[~ray_mask[..., 0]] = 1


            pts_clr_pred.append(pixels_pred.detach().cpu())
            pts_clr_warped.append(feature_pred.detach().cpu())

        ##
        pts_clr_pred = torch.cat(pts_clr_pred, dim=1)
        pts_clr_pred = pts_clr_pred.permute(0, 2, 1).reshape(batch_size, pts_clr_pred.size(2), img_size, img_size)
        pts_clr_warped = torch.cat(pts_clr_warped, dim=1)
        pts_clr_warped = pts_clr_warped.permute(0, 2, 1).reshape(batch_size, pts_clr_warped.size(2), img_size, img_size)

        return pts_clr_pred, pts_clr_warped


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