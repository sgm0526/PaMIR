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
        #self.pamir_net = PamirNet().to(self.device)
        self.pamir_tex_net = TexPamirNetAttention_nerf().to(self.device)
        self.graph_mesh = Mesh()

        self.decoder_output_size = 128
        self.NR = NeuralRenderer_coord().to(self.device)

        self.models_dict = {'pamir_tex_net': self.pamir_tex_net,'pamir_tex_NR': self.NR}
        if not no_weight:
            #self.load_pretrained_pamir_net(pretrained_checkpoint_pamir)
            self.load_pretrained(checkpoint_file=pretrained_checkpoint_pamir_tex)
        #self.pamir_net.eval()
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

    def forward_infer_occupancy_feature_grid_naive(self, img, vol, test_res, group_size):
        pts, pts_proj = self.generate_point_grids(
            test_res, const.cam_R, const.cam_t, const.cam_f, img.size(2))
        pts_ov = self.forward_infer_occupancy_value_group(img, vol, pts, pts_proj, group_size)
        pts_ov = pts_ov.reshape([test_res, test_res, test_res])
        return pts_ov

    def test_nerf_target(self, img, betas, pose, scale, trans, view_diff, return_cam_loc=False):
        #self.pamir_net.eval()
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
        ray_start = const.ray_start#cam_tz - 0.87  # (
        ray_end = const.ray_end#cam_tz + 0.87
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
        #batch_size, 512*512, num_step, 3




        num_ray= 5000
        pts_group_num = (img_size *img_size + num_ray - 1) //num_ray
        pts_clr_pred = []
        pts_clr_warped = []
        for gi in tqdm(range(pts_group_num), desc='Texture query'):
            # print('Testing point group: %d/%d' % (gi + 1, pts_group_num))
            sampled_points = points_cam_source[:, (gi * num_ray):((gi + 1) * num_ray), :, :] # 1, group_size, num_step, 3
            sampled_points_proj=   points_cam_source_proj[:, (gi * num_ray):((gi + 1) * num_ray), :,:]
            sampled_z_vals = z_vals[:, (gi * num_ray):((gi + 1) * num_ray), :,:]
            sampled_rays_d_world  = rays_d_cam[:, (gi * num_ray):((gi + 1) * num_ray)]

            num_ray_part = sampled_points.size(1)
            #num_ray -> num_ray_part



            with torch.no_grad():
                sampled_points  =  sampled_points.reshape(batch_size, -1, 3) # 1 group_size*num_step, 3
                sampled_points_proj = sampled_points_proj.reshape(batch_size, -1, 2)

                nerf_output_clr_, nerf_output_clr, _, _, nerf_output_sigma = self.pamir_tex_net.forward(
                    img, vol, sampled_points, sampled_points_proj)
                nerf_output_clr_=nerf_output_clr_[-1]
                nerf_output_clr = nerf_output_clr[-1]
                nerf_output_sigma = nerf_output_sigma[-1]

                if const.hierarchical:
                    with torch.no_grad():
                        alphas = nerf_output_sigma.reshape(batch_size, num_ray_part , num_steps, -1)
                        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1 - alphas + 1e-10], -2)
                        weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :,
                                           :-1] + 1e-5  # batch, num_ray, 24, 1

                        sampled_z_vals_mid = 0.5 * (
                                    sampled_z_vals[:, :, :-1] + sampled_z_vals[:, :, 1:])  # batch, num_ray, 23, 1
                        fine_z_vals = sample_pdf(sampled_z_vals_mid.reshape(-1, num_steps - 1),
                                                 weights.reshape(-1, num_steps)[:, 1:-1], num_steps, det=False).detach()
                        fine_z_vals = fine_z_vals.reshape(batch_size, num_ray_part , num_steps)

                        sampled_rays_d_world = sampled_rays_d_world.unsqueeze(-2).repeat(1, 1, num_steps, 1)
                        fine_points = sampled_rays_d_world * fine_z_vals[..., None]
                        fine_points[:, :, :, 2] += cam_tz
                        fine_points = self.rotate_points(fine_points, view_diff)
                        fine_points_proj = self.project_points(fine_points, cam_f, cam_c, cam_tz)

                        all_z_vals = torch.cat([sampled_z_vals, fine_z_vals.unsqueeze(-1)], dim=2)

                        _, indices = torch.sort(all_z_vals, dim=2)


                    nerf_output_clr_fine_, nerf_output_clr_fine, _, _, nerf_output_sigma_fine = self.pamir_tex_net.forward(
                        img, vol, fine_points.reshape(batch_size, num_ray_part  * num_steps, 3),
                        fine_points_proj.reshape(batch_size, num_ray_part  * num_steps, 2))
                    nerf_output_clr_fine_ = nerf_output_clr_fine_[-1]
                    nerf_output_clr_fine = nerf_output_clr_fine[-1]
                    nerf_output_sigma_fine = nerf_output_sigma_fine[-1]

                    nerf_output_clr_ = torch.gather(torch.cat(
                        [nerf_output_clr_.reshape(batch_size, num_ray_part, num_steps, 3),
                         nerf_output_clr_fine_.reshape(batch_size, num_ray_part, num_steps, 3)], dim=2), 2,
                                                    indices.expand(-1, -1, -1, 3))
                    nerf_output_clr = torch.gather(torch.cat(
                        [nerf_output_clr.reshape(batch_size, num_ray_part, num_steps, 3),
                         nerf_output_clr_fine.reshape(batch_size, num_ray_part, num_steps, 3)], dim=2), 2,
                                                   indices.expand(-1, -1, -1, 3))
                    nerf_output_sigma = torch.gather(torch.cat(
                        [nerf_output_sigma.reshape(batch_size, num_ray_part, num_steps, 1),
                         nerf_output_sigma_fine.reshape(batch_size, num_ray_part, num_steps, 1)], dim=2), 2, indices)
                    sampled_z_vals =torch.gather(all_z_vals, 2, indices)

                all_outputs = torch.cat([nerf_output_clr_, nerf_output_sigma], dim=-1)
                pixels_pred, _, _ = fancy_integration2(all_outputs, sampled_z_vals, device=self.device, white_back=True)

                all_outputs = torch.cat([nerf_output_clr, nerf_output_sigma], dim=-1)
                feature_pred, _, _ = fancy_integration2(all_outputs, sampled_z_vals, device=self.device,
                                                        white_back=True)

            pts_clr_pred.append(pixels_pred.detach().cpu())
            pts_clr_warped.append(feature_pred.detach().cpu())
            #pts_clr_pred.append(pixels_pred)
            #pts_clr_warped.append(pixels_warped)
        ##
        pts_clr_pred= torch.cat(pts_clr_pred, dim=1)
        pts_clr_pred = pts_clr_pred.permute(0,2,1).reshape(batch_size, 3, img_size,img_size)
        pts_clr_warped= torch.cat(pts_clr_warped, dim=1)
        pts_clr_warped= pts_clr_warped.permute(0, 2, 1).reshape(batch_size, 3, img_size, img_size)
        # pts_clr = pts_clr.permute(2,0,1
        if return_cam_loc:
            return pts_clr, self.rotate_points(cam_t.unsqueeze(0), view_diff)

        return pts_clr_pred, pts_clr_warped


    def test_nerf_target_sigma(self, img, betas, pose, scale, trans, vol_res):
        #self.pamir_net.eval()
        self.pamir_tex_net.eval()

        gt_vert_cam = scale * self.tet_smpl(pose, betas) + trans
        vol = self.voxelization(gt_vert_cam)

        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)




        pts, pts_proj = self.generate_point_grids(
            vol_res, const.cam_R, const.cam_t, const.cam_f, img.size(2))

        pts = torch.from_numpy(pts).unsqueeze(0).to(self.device)
        pts_proj = torch.from_numpy(pts_proj).unsqueeze(0).to(self.device)

        group_size= 10000
        pts_group_num = (pts.shape[1] + group_size - 1) // group_size
        pts_clr = []
        for gi in tqdm(range(pts_group_num), desc='Sigma query'):
            # print('Testing point group: %d/%d' % (gi + 1, pts_group_num))
            sampled_points = pts[:, (gi * group_size):((gi + 1) * group_size), :] # 1, group_size, num_step, 3
            sampled_points_proj= pts_proj[:, (gi * group_size):((gi + 1) * group_size), :]
            # sampled_rays_d_world  = rays_d_cam[:, (gi * num_ray):((gi + 1) * num_ray)]


            with torch.no_grad():
                # sampled_points = sampled_points.reshape(batch_size, -1, 3) # 1 group_size*num_step, 3
                # sampled_points_proj = sampled_points_proj.reshape(batch_size, -1, 2)


                #img_feat_geo = self.pamir_net.get_img_feature(img, no_grad=True)
                #nerf_feat_occupancy = self.pamir_net.get_mlp_feature(img, vol, sampled_points , sampled_points_proj )

                nerf_output_clr_, nerf_output_clr, nerf_output_att, nerf_smpl_feat, nerf_output_sigma = self.pamir_tex_net.forward(
                    img, vol, sampled_points, sampled_points_proj)#, img_feat_geo, nerf_feat_occupancy)

            ##

            pts_clr.append(nerf_output_sigma.detach().cpu())
        # import pdb
        # pdb.set_trace()
        pts_clr2 = torch.cat(pts_clr, dim=1)[0]
        pts_clr2 = pts_clr2.reshape( vol_res,  vol_res,  vol_res)

        return pts_clr2

    def test_tex_featurenerf(self, img, mesh_v, betas, pose, scale, trans, qwe):

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
        depth_diff = []
        pred_img_list = []
        tex_sample_list = []
        for view_angle in [0, 90, 180, 270]:

            pts_2 = self.rotate_points(pts, torch.ones(1).to(self.device) * view_angle * -1)
            pts_2_proj = self.forward_project_points(
                pts_2, cam_r, cam_t, cam_f, img.size(2))
            # view_angle = 90
            # 카메라 위치 구하기
            # 카메라 위치부터 각각 포인트까지 레이 쏘기
            # 각각 포인트까지의 gt_depth구하기
            # 그걸로 fancy_integration해서 각각 nerf_depth 구하기
            # 각각의 차이를 포인트마다의 list에 append하기

            pred_img, cam_loc =  self.test_nerf_target(img, betas, pose, scale, trans,torch.ones(1).to(self.device)*view_angle, return_cam_loc=True)
            pred_img_list.append(pred_img)

            ray_d = pts - cam_loc
            ray_d = normalize_vecs(ray_d)

            z_vals = torch.linspace(const.ray_start, const.ray_end, const.num_steps, device=self.device).reshape(1, 1, const.num_steps, 1).repeat(1, pts.size(1), 1, 1)
            points = ray_d.unsqueeze(2).repeat(1, 1, const.num_steps, 1) * z_vals
            points = points.reshape(1, -1, 3)
            points = points + cam_loc
            #import pdb; pdb.set_trace()
            points_proj = self.forward_project_points(
                points, cam_r, cam_t, cam_f, img.size(2))
            h_grid = pts_2_proj[:, :, 0].view(1, pts.size(1), 1, 1)
            v_grid = pts_2_proj[:, :, 1].view(1, pts.size(1), 1, 1)
            grid_2d = torch.cat([h_grid, v_grid], dim=-1)
            if view_angle == 0:
                pred_img = img
            tex_sample = F.grid_sample(input=pred_img.to(self.device), grid=grid_2d.to(self.device), align_corners=False,
                                          mode='bilinear', padding_mode='border')

            points = points.reshape(1, -1, const.num_steps, 3)
            points_proj = points_proj.reshape(1, -1, const.num_steps, 2)


            gt_depth = pts_2[..., 2] + cam_tz
            # gt_depth = pts[..., 2] - cam_loc[..., 2] + cam_tz
            #points[..., 2] += cam_tz
            #points = points + cam_loc
            group_size = 1000 #5000
            pts_group_num = (pts.shape[1] + group_size - 1) // group_size
            points_sigma = []
            for gi in tqdm(range(pts_group_num), desc='Sigma query'):
                # print('Testing point group: %d/%d' % (gi + 1, pts_group_num))
                sampled_points = points[:, (gi * group_size):((gi + 1) * group_size), :, :]  # 1, group_size, num_step, 3
                sampled_points_proj = points_proj[:, (gi * group_size):((gi + 1) * group_size), :, :]
                with torch.no_grad():
                    sampled_points = sampled_points.reshape(1, -1, 3) # 1 group_size*num_step, 3
                    sampled_points_proj = sampled_points_proj.reshape(1, -1, 2)

                    img_feat_geo = self.pamir_net.get_img_feature(img, no_grad=True)
                    nerf_feat_occupancy = self.pamir_net.get_mlp_feature(img, vol, sampled_points, sampled_points_proj)

                    nerf_output_clr, nerf_output_feature, _, _, nerf_output_sigma = self.pamir_tex_net.forward(
                        img, vol, sampled_points, sampled_points_proj, img_feat_geo, nerf_feat_occupancy)

                ##

                points_sigma.append(nerf_output_sigma.detach())
            points_sigma2 = torch.cat(points_sigma, dim=1)
            points_sigma2 = points_sigma2.reshape(1, pts.size(1), const.num_steps, 1)
            _, nerf_depth, _= fancy_integration(points_sigma2.repeat(1, 1, 1, 2), z_vals, self.device, last_back=True)
            nerf_depth = nerf_depth[..., 0]
            depth_diff.append(nerf_depth - gt_depth)

            tex_sample_list.append(tex_sample)

        depth_diff = torch.cat(depth_diff, dim=0)
        pred_img = torch.cat(pred_img_list, dim=0)
        tex_sample = torch.cat(tex_sample_list, dim=0)
        value, ind = abs(depth_diff).min(dim=0)
        # import pdb; pdb.set_trace()
        # value = depth_diff[qwe]
        tex_sample_final = torch.gather(tex_sample, 0, ind[None,None, :, None].repeat(1, 3, 1, 1))
        # tex_sample_final = torch.gather(tex_sample, 0, torch.ones_like(ind[None,None, :, None].repeat(1, 3, 1, 1)).cuda() * qwe)
        tex_sample_final = tex_sample_final[0, :, :, 0].permute(1,0)
        tex_mask = (value < 1).unsqueeze(-1).cpu()
        tex_final = tex_sample_final.cpu() * tex_mask + torch.zeros_like(tex_sample_final).cpu() * ~tex_mask
        #tex_final = tex_sample_final.cpu() * tex_mask + torch.Tensor(clr) * ~tex_mask

        return tex_final

    def optm_smpl_param(self, img, mask, betas, pose, scale, trans, iter_num):
        assert iter_num > 0
        self.pamir_tex_net.eval()

        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)

        # convert rotmat to theta

        if pose.ndimension() ==2 :
            theta = pose
        else:
            rotmat_host = pose.detach().cpu().numpy().squeeze()
            theta_host = []
            for r in rotmat_host:
                theta_host.append(cv.Rodrigues(r)[0])
            theta_host = np.asarray(theta_host).reshape((1, -1))
            theta = torch.from_numpy(theta_host).to(self.device)

        # construct parameters

        theta_new = torch.nn.Parameter(theta)
        betas_new = torch.nn.Parameter(betas)
        theta_orig = theta_new.clone().detach()
        betas_orig = betas_new.clone().detach()
        optm = torch.optim.Adam(params=(theta_new,), lr=2e-3)

        vert_cam = scale * self.tet_smpl(theta, betas) + trans
        vol = self.voxelization(vert_cam)
        vert_cam = vert_cam[:, :6890]
        vert_cam_proj = self.project_points2(vert_cam, cam_f, cam_c, cam_tz)
        _, _, att_orig, _, smpl_sdf = self.pamir_tex_net(img, vol, vert_cam ,  vert_cam_proj)


        nerf_color_pred_before, _ = self.test_nerf_target(img, betas, theta, scale, trans,
                                                                   torch.ones(img.shape[0]).cuda() * 0)


        for i in tqdm(range(iter_num), desc='Body Fitting Optimization'):
            theta_new_ = torch.cat([theta_orig[:, :3], theta_new[:, 3:]], dim=1)
            vert_tetsmpl_new = self.tet_smpl(theta_new_, betas_new)
            vert_tetsmpl_new_cam = scale * vert_tetsmpl_new + trans

            vol = self.voxelization(vert_tetsmpl_new_cam.detach())
            pred_vert_new_cam = self.graph_mesh.downsample(vert_tetsmpl_new_cam[:, :6890], n2=1)
            #pred_vert_new_cam = vert_tetsmpl_new_cam[:, :6890]
            #pred_vert_new_cam = pred_vert_new_cam+torch.normal(0, 0.1, size=pred_vert_new_cam.shape).cuda()


            #pred_vert_new_proj = self.forward_project_points(pred_vert_new_cam, cam_r, cam_t, cam_f,2*cam_c)
            pred_vert_new_proj =self.project_points2(pred_vert_new_cam, cam_f, cam_c, cam_tz)



            _,_,att,_,smpl_sdf = self.pamir_tex_net(img, vol, pred_vert_new_cam, pred_vert_new_proj )
            loss_fitting = torch.mean(torch.abs(F.leaky_relu(0.5 - smpl_sdf, negative_slope=0.5)))


            # nerf_color_pred, nerf_color_warped = self.test_nerf_target(img, betas_new, theta_new_, scale, trans, torch.ones(img.shape[0]).cuda()*0)
            h_grid = pred_vert_new_proj[:, :, 0].view(1, pred_vert_new_proj .size(1), 1, 1)
            v_grid = pred_vert_new_proj[:, :, 1].view(1, pred_vert_new_proj .size(1), 1, 1)
            grid_2d = torch.cat([h_grid, v_grid], dim=-1)
            gt_clr_nerf = F.grid_sample(input=img.to(self.device), grid=grid_2d.to(self.device),
                                        align_corners=False, mode='bilinear', padding_mode='border').permute(0, 2, 3,
                                                                                                             1).squeeze(
                2)  # b,5000,3

            proj_mask = F.grid_sample(input=mask.permute(0, 3, 1, 2).to(self.device), grid=grid_2d.to(self.device),
                                      align_corners=False, mode='bilinear', padding_mode='border').permute(0, 2, 3,
                                                                                                           1).squeeze(
                2)  # b,5000,3
            loss_mask = nn.MSELoss()(torch.ones_like(proj_mask, device='cuda'), proj_mask)

            ray_d_target = pred_vert_new_cam - cam_t
            ray_d_target = normalize_vecs(ray_d_target)
            z_vals_ = torch.linspace(const.ray_start, const.ray_end, const.num_steps, device=self.device).reshape(1, 1,
                                                                                                                  const.num_steps,
                                                                                                                  1).repeat(1, pred_vert_new_cam.size(1), 1, 1)

            points = ray_d_target.unsqueeze(2).repeat(1, 1, const.num_steps, 1) * z_vals_
            # points = points.reshape(1, -1, 3)
            points = points + cam_t  # target view!!


            num_ray = points.size(1)
            sampled_z_vals = z_vals_
            sampled_points = points

            #sampled_points_proj = self.forward_project_points(sampled_points, cam_r, cam_t, cam_f,2*cam_c)
            sampled_points_proj = self.project_points2(sampled_points, cam_f, cam_c, cam_tz)
            sampled_points = sampled_points.reshape(1, -1, 3)
            sampled_points_proj = sampled_points_proj.reshape(1, -1, 2)

            nerf_output_clr_, nerf_output_clr, nerf_output_att, nerf_smpl_feat, nerf_output_sigma = self.pamir_tex_net(
                img, vol, sampled_points, sampled_points_proj)
            all_outputs = torch.cat([nerf_output_clr_, nerf_output_sigma], dim=-1)
            pixels_pred, _, _ = fancy_integration2(all_outputs.reshape(1, num_ray, const.num_steps, -1),
                                                   sampled_z_vals, device=self.device, white_back=False)# white_back=True)



            #import pdb; pdb.set_trace()


            loss_nerf = nn.L1Loss()(pixels_pred, gt_clr_nerf ) #+  nn.L1Loss()(img, gt_clr_nerf )
            #loss_att = torch.mean((att - att_orig.detach()) ** 2)

            loss_bias = torch.mean((theta_orig - theta_new) ** 2) + \
                        torch.mean((betas_orig - betas_new) ** 2) * 0.01

            loss = loss_fitting  + loss_nerf  #loss_fitting * 1.0 +loss_nerf #+ 10*loss_bias
            #loss = loss_nerf +loss_mask#+ 10*loss_bias #loss_fitting * 1.0 +loss_nerf * 1.0#+ loss_bias * 1.0

            optm.zero_grad()
            loss.backward()
            optm.step()
            print('loss:', loss)

            # print('Iter No.%d: loss_fitting = %f, loss_bias = %f, loss_kp = %f' %
            #       (i, loss_fitting.item(), loss_bias.item(), loss_kp.item()))
        nerf_color_pred, nerf_color_warped = self.test_nerf_target(img, betas_new, theta_new_, scale, trans,
                                                                   torch.ones(img.shape[0]).cuda() * 0)
        return theta_new, betas_new, vert_tetsmpl_new_cam[:, :6890], nerf_color_pred_before, nerf_color_pred


    def test_tex_pifu(self, img, mesh_v, betas, pose, scale, trans):
        #self.pamir_net.eval()
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
        #img_feat_geo = self.pamir_net.get_img_feature(img, no_grad=True)
        _,clr, _, _ , _= self.pamir_tex_net.forward(img, vol, pts, pts_proj)#, img_feat_geo, feat_occupancy=None) ##
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

