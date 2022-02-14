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
        self.pamir_net = PamirNet().to(self.device)
        self.pamir_tex_net = TexPamirNetAttention_nerf().to(self.device)

        self.decoder_output_size = 128
        self.NR = NeuralRenderer_coord().to(self.device)

        self.models_dict = {'pamir_tex_net': self.pamir_tex_net,'pamir_tex_NR': self.NR}
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

    def forward_infer_occupancy_feature_grid_naive(self, img, vol, test_res, group_size):
        pts, pts_proj = self.generate_point_grids(
            test_res, const.cam_R, const.cam_t, const.cam_f, img.size(2))
        pts_ov = self.forward_infer_occupancy_value_group(img, vol, pts, pts_proj, group_size)
        pts_ov = pts_ov.reshape([test_res, test_res, test_res])
        return pts_ov

    def test_nerf_target(self, img, betas, pose, scale, trans, view_diff, return_cam_loc=False, return_flow_feature=False):
        self.pamir_net.eval()
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
                                                               resolution=(int(const.img_res / const.down_scale), int(const.img_res / const.down_scale)),
                                                               device=self.device, fov=fov_degree, ray_start=ray_start,
                                                               ray_end=ray_end)  # batch_size, pixels, num_steps, 1

        # 1, img_size*img_size, num_steps, 3
        points_cam[:, :, :, 2] += cam_tz
        points_cam_source = self.rotate_points(points_cam, view_diff)
        points_cam_source_proj = self.project_points(points_cam_source, cam_f, cam_c, cam_tz)
        #batch_size, 512*512, num_step, 3




        num_ray= 5000
        pts_group_num = (int(img_size / const.down_scale) * int(img_size / const.down_scale) + num_ray - 1) //num_ray
        pts_clr_pred = []
        pts_clr_warped = []
        pts_flow = []
        pts_volume_feature = []
        img_feat_geo = self.pamir_net.get_img_feature(img, no_grad=True)
        img_feats = self.pamir_net.hg(img)
        vol_feats = self.pamir_net.ve(vol)
        _2d_grid = self.pamir_tex_net.generate_2d_grids(img.shape[2])
        _2d_grid = torch.from_numpy(_2d_grid).permute(2, 0, 1).unsqueeze(0).repeat(batch_size, 1, 1,
                                                                                   1).cuda()[:, [1, 0], :, :]
        img_feat_tex = self.pamir_tex_net.cg(torch.cat([img, _2d_grid], 1))
        vol_feat_tex = self.pamir_tex_net.ve(vol, intermediate_output=False)
        for gi in tqdm(range(pts_group_num), desc='Texture query'):
            # print('Testing point group: %d/%d' % (gi + 1, pts_group_num))
            sampled_points = points_cam_source[:, (gi * num_ray):((gi + 1) * num_ray), :, :] # 1, group_size, num_step, 3
            sampled_points_proj=   points_cam_source_proj[:, (gi * num_ray):((gi + 1) * num_ray), :,:]
            sampled_z_vals = z_vals[:, (gi * num_ray):((gi + 1) * num_ray), :,:]
            sampled_rays_d_world  = rays_d_cam[:, (gi * num_ray):((gi + 1) * num_ray)]

            num_ray_part = sampled_points.size(1)
            #num_ray -> num_ray_part

        ##

            if return_flow_feature:
                with torch.no_grad():
                    sampled_points = sampled_points.reshape(batch_size, -1, 3)  # 1 group_size*num_step, 3
                    sampled_points_proj = sampled_points_proj.reshape(batch_size, -1, 2)


                    flow, volume_feature = self.get_nerf(img, vol, img_feat_geo, sampled_points, sampled_points_proj,
                                                               sampled_z_vals, sampled_rays_d_world, hierarchical,
                                                               batch_size,
                                                               num_ray_part, num_steps, cam_f, cam_c, cam_tz, view_diff,
                                                        img_feats, vol_feats, img_feat_tex, vol_feat_tex, return_flow_feature=True)
                pts_flow.append(flow.detach().cpu())
                pts_volume_feature.append(volume_feature.detach().cpu())

            else:
                with torch.no_grad():
                    sampled_points  =  sampled_points.reshape(batch_size, -1, 3) # 1 group_size*num_step, 3
                    sampled_points_proj = sampled_points_proj.reshape(batch_size, -1, 2)

                    pixels_pred, pixels_warped = self.get_nerf(img, vol, img_feat_geo, sampled_points, sampled_points_proj,
                                                               sampled_z_vals, sampled_rays_d_world, hierarchical,
                                                               batch_size,
                                                               num_ray_part , num_steps, cam_f, cam_c, cam_tz, view_diff,
                                                               img_feats, vol_feats, img_feat_tex, vol_feat_tex)

                pts_clr_pred.append(pixels_pred.detach().cpu())
                pts_clr_warped.append(pixels_warped.detach().cpu())
        ##

        if return_flow_feature:
            import pdb;pdb.set_trace()
            pts_flow = torch.cat(pts_flow, dim=1).squeeze(-2)
            pts_flow = pts_flow.permute(0, 2, 1).reshape(batch_size, 2, img_size, img_size)
            pts_volume_feature = torch.cat(pts_volume_feature, dim=1)
            ch_num = pts_volume_feature.size(2)
            pts_volume_feature = pts_volume_feature.permute(0, 2, 1).reshape(batch_size, ch_num, img_size, img_size)
            # pts_clr = pts_clr.permute(2,0,1

            return pts_flow, pts_volume_feature

        else:
            pts_clr_pred= torch.cat(pts_clr_pred, dim=1)
            pts_clr_pred = pts_clr_pred.permute(0,2,1).reshape(batch_size, 3, img_size,img_size)
            pts_clr_warped= torch.cat(pts_clr_warped, dim=1)
            pts_clr_warped= pts_clr_warped.permute(0, 2, 1).reshape(batch_size, 3, img_size, img_size)
            # pts_clr = pts_clr.permute(2,0,1
            if return_cam_loc:
                return pts_clr, self.rotate_points(cam_t.unsqueeze(0), view_diff)

            return pts_clr_pred, pts_clr_warped


    def test_nerf_target_sigma(self, img, betas, pose, scale, trans, view_diff):
        self.pamir_net.eval()
        self.pamir_tex_net.eval()

        gt_vert_cam = scale * self.tet_smpl(pose, betas) + trans
        vol = self.voxelization(gt_vert_cam)

        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)

        vol_res= 128


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


                img_feat_geo = self.pamir_net.get_img_feature(img, no_grad=True)
                nerf_feat_occupancy = self.pamir_net.get_mlp_feature(img, vol, sampled_points , sampled_points_proj )

                nerf_output_clr_, nerf_output_clr, nerf_output_att, nerf_smpl_feat, nerf_output_sigma = self.pamir_tex_net.forward(
                    img, vol, sampled_points, sampled_points_proj, img_feat_geo, nerf_feat_occupancy)

            ##

            pts_clr.append(nerf_output_sigma.detach().cpu())
        # import pdb
        # pdb.set_trace()
        pts_clr2 = torch.cat(pts_clr, dim=1)[0]
        pts_clr2 = pts_clr2.reshape( vol_res,  vol_res,  vol_res)
        with mrcfile.new_mmap(os.path.join('./', f'{1}.mrc'), overwrite=True, shape=pts_clr2.shape, mrc_mode=2) as mrc:
            mrc.data[:] = pts_clr2

        vertices, simplices, normals, _ = measure.marching_cubes_lewiner(np.array(pts_clr2), 15)
        mesh = dict()
        mesh['v'] = vertices /  vol_res - 0.5
        mesh['f'] = simplices[:, (1, 0, 2)]
        mesh['vn'] = normals

        obj_io.save_obj_data(mesh, './sigma_mesh.obj')
        return

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

            pred_img, cam_loc = self.test_nerf_target(img, betas, pose, scale, trans,torch.ones(1).to(self.device)*view_angle, return_cam_loc=True)
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
        clr, _, _, _ , _= self.pamir_tex_net.forward(img, vol, pts, pts_proj, img_feat_geo, feat_occupancy=None) ##
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

    def get_nerf(self, img, vol, img_feat_geo, sampled_points, sampled_points_proj, sampled_z_vals,
                 sampled_rays_d_world, hierarchical, batch_size, num_ray, num_steps, cam_f, cam_c, cam_tz, view_diff,
                 img_feats, vol_feats, img_feat_tex, vol_feat_tex, return_flow_feature=False):

        with torch.no_grad():
            nerf_feat_occupancy = self.pamir_net.get_mlp_feature(img, vol, sampled_points, sampled_points_proj, img_feats=img_feats, vol_feats=vol_feats)

            ##for hierarchical sampling
            if hierarchical:
                ray_occupancy = self.pamir_net.forward(img, vol, sampled_points, sampled_points_proj)[-1]
                # batch_size, num_ray*num_step, 1
                ray_occupancy = ray_occupancy.reshape(batch_size, num_ray, num_steps, 1)
                ray_occupancy_diff = ray_occupancy[:, :, 1:] - ray_occupancy[:, :, :-1]
                max_index = ray_occupancy_diff.argmax(dim=2) + 1

                max_z_vals = torch.gather(sampled_z_vals, 2, max_index.unsqueeze(-1))

                std = 0.1
                std_line = torch.linspace(-std / 2, std / 2, num_steps)[None,][None,].repeat(batch_size, num_ray, 1)
                fine_z_vals = max_z_vals.squeeze(-1) + std_line.to(self.device)

                sampled_rays_d_world = sampled_rays_d_world.unsqueeze(-2).repeat(1, 1, num_steps, 1)
                fine_points = sampled_rays_d_world * fine_z_vals[..., None]
                fine_points[:, :, :, 2] += cam_tz
                fine_points = self.rotate_points(fine_points, view_diff)
                # sampled_rays_d = sampled_rays_d.unsqueeze(-2).repeat(1, 1, num_steps, 1)
                # fine_points = sampled_rays_d * fine_z_vals[..., None]
                # fine_points[:, :, :, 2] += cam_tz

                fine_points_proj = self.project_points(fine_points, cam_f, cam_c, cam_tz)
                fine_points = fine_points.reshape(batch_size, num_ray * num_steps, 3)
                fine_points_proj = fine_points_proj.reshape(batch_size, num_ray * num_steps, 2)
                fine_nerf_feat_occupancy = self.pamir_net.get_mlp_feature(img, vol, fine_points, fine_points_proj)
                sampled_points = sampled_points.reshape(batch_size, num_ray, num_steps, 3)
                fine_points = fine_points.reshape(batch_size, num_ray, num_steps, 3)

                ch_mlp_feat = nerf_feat_occupancy.size(1)
                all_nerf_feat_occupancy = torch.cat(
                    [nerf_feat_occupancy.reshape(batch_size, ch_mlp_feat, num_ray, num_steps, 1),
                     fine_nerf_feat_occupancy.reshape(batch_size, ch_mlp_feat, num_ray, num_steps, 1)], dim=3)
                all_points = torch.cat([sampled_points, fine_points], dim=2)
                all_points_proj = torch.cat([sampled_points_proj.reshape(batch_size, num_ray, num_steps, 2),
                                             fine_points_proj.reshape(batch_size, num_ray, num_steps, 2)], dim=2)
                all_z_vals = torch.cat([sampled_z_vals, fine_z_vals.unsqueeze(-1)], dim=2)
                _, indices = torch.sort(all_z_vals, dim=2)
                all_z_vals = torch.gather(all_z_vals, 2, indices)
                all_points = torch.gather(all_points, 2, indices.expand(-1, -1, -1, 3))
                all_points_proj = torch.gather(all_points_proj, 2, indices.expand(-1, -1, -1, 2))
                all_nerf_feat_occupancy = torch.gather(all_nerf_feat_occupancy, 3,
                                                       indices.unsqueeze(1).expand(-1, ch_mlp_feat, -1, -1, -1))

        if hierarchical:
            nerf_output_clr_, nerf_output_clr, nerf_output_att, nerf_smpl_feat, nerf_output_sigma = self.pamir_tex_net.forward(
                img, vol, all_points.reshape(batch_size, num_ray * num_steps * 2, 3),
                all_points_proj.reshape(batch_size, num_ray * num_steps * 2, 2), img_feat_geo,
                all_nerf_feat_occupancy.reshape(batch_size, ch_mlp_feat, num_ray * num_steps * 2, 1))
            # all_outputs = torch.cat([nerf_output_clr_, nerf_output_sigma], dim=-1)
            num_steps = num_steps * 2


        else:
            nerf_output_clr_, nerf_output_clr, nerf_output_att, nerf_smpl_feat, nerf_output_sigma = self.pamir_tex_net.forward(
                img, vol, sampled_points, sampled_points_proj, img_feat_geo, nerf_feat_occupancy, img_feat_tex=img_feat_tex, vol_feat=vol_feat_tex)

        all_outputs = torch.cat([nerf_output_clr_, nerf_output_sigma], dim=-1)
        pixels_pred, _, _ = fancy_integration(all_outputs.reshape(batch_size, num_ray, num_steps, -1),
                                              sampled_z_vals, device=self.device, white_back=True)

        all_outputs = torch.cat([nerf_output_clr, nerf_output_sigma], dim=-1)
        feature_pred, _, _ = fancy_integration(all_outputs.reshape(batch_size, num_ray, num_steps, -1),
                                               sampled_z_vals, device=self.device, white_back=True)
        if False:
            _, depth, _ = fancy_integration(all_outputs.reshape(batch_size, num_ray, num_steps, -1),
                                                   sampled_z_vals, device=self.device, last_back=True)
            surface = sampled_rays_d_world * depth
            surface[..., 2] += cam_tz
            surface_proj = self.project_points(surface, cam_f, cam_c, cam_tz)
            surface_nerf_feat_occupancy = self.pamir_net.get_mlp_feature(img, vol, surface, surface_proj)
            flow, volume_feature, sigma = self.pamir_tex_net.forward(
                img, vol, surface, surface_proj, img_feat_geo, surface_nerf_feat_occupancy, return_flow_feature=True)
            return flow, volume_feature

        if return_flow_feature:
            nerf_flow, nerf_volume_feat, nerf_sigma = self.pamir_tex_net.forward(
                img, vol, sampled_points, sampled_points_proj, img_feat_geo, nerf_feat_occupancy, img_feat_tex=img_feat_tex, vol_feat=vol_feat_tex, return_flow_feature=True)
            all_outputs = torch.cat([nerf_flow.squeeze(-2), nerf_volume_feat, nerf_sigma], dim=-1)
            outputs, _, _ = fancy_integration(all_outputs.reshape(batch_size, num_ray, num_steps, -1),
                                                   sampled_z_vals, device=self.device)
            flow = outputs[:, :, :2]
            volume_feature = outputs[:, :, 2:]
            return flow, volume_feature
        # pred_img = pixels_pred.permute(0, 2, 1).reshape(batch_size, 3, const.feature_res, const.feature_res)
        # source_warped_img = feature_pred.reshape(batch_size, const.feature_res, const.feature_res, -1).permute(0, 3, 1, 2)

        return pixels_pred, feature_pred
