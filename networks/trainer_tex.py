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
import scipy.io as sio
import datetime
import glob
import logging
import math

from util.base_trainer import BaseTrainer
from dataloader.dataloader_tex import TrainingImgDataset
from network.arch import PamirNet, TexPamirNetAttention, TexPamirNetAttention_nerf
from neural_voxelization_layer.smpl_model import TetraSMPL
from neural_voxelization_layer.voxelize import Voxelization
from util.img_normalization import ImgNormalizerForResnet
from graph_cmr.models import GraphCNN, SMPLParamRegressor, SMPL
from graph_cmr.utils import Mesh
from graph_cmr.models.geometric_layers import rodrigues, orthographic_projection
import util.obj_io as obj_io
import util.util as util
import constant as const
from util.volume_rendering import *

from network.swapAE_networks.patch_discriminator import StyleGAN2PatchDiscriminator, PatchDiscriminator


class Trainer(BaseTrainer):
    def __init__(self, options):
        super(Trainer, self).__init__(options)

    def init_fn(self):
        super(BaseTrainer, self).__init__()
        # dataset
        self.train_ds = TrainingImgDataset(
            self.options.dataset_dir, img_h=const.img_res, img_w=const.img_res,
            training=True, testing_res=256,
            view_num_per_item=self.options.view_num_per_item,
            point_num=self.options.point_num,
            load_pts2smpl_idx_wgt=True,
            smpl_data_folder='./data')
        self.val_ds = TrainingImgDataset(
            self.options.dataset_dir, img_h=const.img_res, img_w=const.img_res,
            training=False, testing_res=256,
            view_num_per_item=self.options.view_num_per_item,
            point_num=self.options.point_num,
            load_pts2smpl_idx_wgt=True,
            smpl_data_folder='./data')

        # neural voxelization components
        self.smpl = SMPL('./data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl').to(self.device)
        self.tet_smpl = TetraSMPL('./data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
                                  './data/tetra_smpl.npz').to(self.device)
        smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = util.read_smpl_constants('./data')
        self.smpl_faces = smpl_faces
        self.voxelization = Voxelization(smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras,
                                         volume_res=const.vol_res,
                                         sigma=const.semantic_encoding_sigma,
                                         smooth_kernel_size=const.smooth_kernel_size,
                                         batch_size=self.options.batch_size).to(self.device)

        # pamir_net
        self.pamir_net = PamirNet().to(self.device)
        self.pamir_tex_net =  TexPamirNetAttention_nerf().to(self.device)


        # optimizers
        self.optm_pamir_tex_net = torch.optim.Adam(
            params=list(self.pamir_tex_net.parameters()), lr=float(self.options.lr)
        )

        # loses
        self.criterion_tex = nn.L1Loss().to(self.device)

        self.TrainGAN = False

        if self.TrainGAN:
            ## add for discriminator
            self.pamir_tex_discriminator = PatchDiscriminator().to(self.device)
            self.optm_pamir_tex_discriminator = torch.optim.Adam(
                params=list(self.pamir_tex_discriminator.parameters()), lr=float(self.options.lr)
            )

            # Pack models and optimizers in a dict - necessary for checkpointing
            self.models_dict = {'pamir_tex_net': self.pamir_tex_net, 'pamir_tex_discriminator': self.pamir_tex_discriminator}
            self.optimizers_dict = {'optimizer_pamir_net': self.optm_pamir_tex_net, 'optimizer_pamir_discriminator': self.optm_pamir_tex_discriminator}
        else:
            # Pack models and optimizers in a dict - necessary for checkpointing
            self.models_dict = {'pamir_tex_net': self.pamir_tex_net}
            self.optimizers_dict = {'optimizer_pamir_net': self.optm_pamir_tex_net}

        # Optionally start training from a pretrained checkpoint
        # Note that this is different from resuming training
        # For the latter use --resume
        assert self.options.pretrained_pamir_net_checkpoint is not None, 'You must provide a pretrained PaMIR geometry model!'
        self.load_pretrained_pamir_net(self.options.pretrained_pamir_net_checkpoint)

        # read energy weights
        self.loss_weights = {
            'tex': 1.0,
            'att': 0.005,
            'g_loss': 0.01,
        }

        logging.info('#trainable_params = %d' %
                     sum(p.numel() for p in self.pamir_tex_net.parameters() if p.requires_grad))

        # meta results
        now = datetime.datetime.now()
        self.log_file_path = os.path.join(
            self.options.log_dir, 'log_%s.npz' % now.strftime('%Y_%m_%d_%H_%M_%S'))

    def train_step(self, input_batch):
        self.pamir_tex_net.train()
        self.pamir_net.eval()

        # some constants
        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)

        # training data
        img = input_batch['img']
        pts = input_batch['pts']    # [:, :-self.options.point_num]
        pts_proj = input_batch['pts_proj']  # [:, :-self.options.point_num]
        gt_clr = input_batch['pts_clr']   # [:, :-self.options.point_num]
        gt_betas = input_batch['betas']
        gt_pose = input_batch['pose']
        gt_scale = input_batch['scale']
        gt_trans = input_batch['trans']

        target_img = input_batch['target_img']

        mask = input_batch['mask']
        target_mask = input_batch['target_mask']

        ###
        batch_size = pts.size(0)
        img_size = const.img_res
        fov = 2 * torch.atan(torch.Tensor([cam_c / cam_f])).item()
        fov_degree = fov * 180 / math.pi
        ray_start = cam_tz - 0.87
        ray_end = cam_tz + 0.87


        num_steps = self.options.num_steps
        hierarchical= self.options.hierarchical

        ## todo hierarchical sampling
        gt_clr_nerf = target_img.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)

        points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps,
                                                               resolution=(const.feature_res, const.feature_res),
                                                               device=self.device, fov=fov_degree, ray_start=ray_start,
                                                               ray_end=ray_end)  # batch_size, pixels, num_steps, 1

        #points_cam_check = rays_d_cam.unsqueeze(-2).repeat(1,1,num_steps,1)*z_vals


        view_diff = input_batch['view_id'] - input_batch['target_view_id']


        # 1, img_size*img_size, num_steps, 3
        points_cam[:,:,:,2] +=cam_tz


        points_cam_source = self.rotate_points(points_cam, view_diff)

        batch_size, pts_num = pts.size()[:2]
        losses = dict()

        with torch.no_grad():
            gt_vert_cam = gt_scale * self.tet_smpl(gt_pose, gt_betas) + gt_trans
            vol = self.voxelization(gt_vert_cam)    # we simply use ground-truth SMPL for when training texture module
            img_feat_geo = self.pamir_net.get_img_feature(img, no_grad=True)
            feat_occupancy = self.pamir_net.get_mlp_feature(img, vol, pts, pts_proj)

        output_clr_, output_clr, output_att, smpl_feat, output_sigma = self.pamir_tex_net.forward(
            img, vol, pts, pts_proj, img_feat_geo, feat_occupancy)

        # import pdb; pdb.set_trace
        losses['tex'] = self.tex_loss(output_clr, gt_clr) + self.tex_loss(output_clr_, gt_clr)
        losses['att'] = self.attention_loss(output_att)



        if False:
            patch_size = 32
            num_ray = patch_size * patch_size
            ray_index = self.sample_ray_index(img_size, target_mask, patch_size=patch_size)

            sampled_points = torch.gather(points_cam_source, 1,
                                          ray_index[:, :, None, None].repeat(1, 1, num_steps, 3))
            sampled_z_vals = torch.gather(z_vals, 1,
                                          ray_index[:, :, None, None].repeat(1, 1, num_steps, 1))
            sampled_rays_d_world = torch.gather(rays_d_cam, 1,
                                                ray_index[:, :, None].repeat(1, 1, 3))
            gt_clr_nerf = torch.gather(gt_clr_nerf, 1,
                                       ray_index[:, :, None].repeat(1, 1, 3))

            # gt_clr_nerf2 = gt_clr_nerf.reshape(3, patch_size, patch_size, 3)
            # gt_clr_nerf2 = gt_clr_nerf2.permute(0, 3, 1, 2)
            # from torchvision.utils import save_image
            # save_image(gt_clr_nerf2, './patch_300.png')

        if False:
            num_ray = 1000
            ray_index = np.random.randint(0, img_size * img_size, num_ray)
            sampled_points = points_cam_source[:, ray_index]
            sampled_z_vals = z_vals[:, ray_index]
            sampled_rays_d_world = rays_d_cam[:, ray_index]

            gt_clr_nerf = gt_clr_nerf[:, ray_index]

            # rays_d_cam_source = self.rotate_points(rays_d_cam, view_diff)
            # sampled_rays_d = rays_d_cam_source[:, ray_index]

        if True:
            num_ray = const.feature_res*const.feature_res
            sampled_points = points_cam_source
            sampled_z_vals = z_vals
            sampled_rays_d_world = rays_d_cam

        sampled_points_proj = self.project_points(sampled_points, cam_f, cam_c, cam_tz)

        sampled_points = sampled_points.reshape(batch_size, -1, 3)
        sampled_points_proj = sampled_points_proj.reshape(batch_size, -1, 2)

        pixels_pred, pixels_final = self.get_nerf(img, vol, img_feat_geo, sampled_points, sampled_points_proj,
                                                  sampled_z_vals, sampled_rays_d_world, hierarchical, batch_size,
                                                  num_ray, num_steps, cam_f, cam_c, cam_tz, view_diff)


        points_cam_proj = self.project_points(points_cam, cam_f, cam_c, cam_tz)
        feature_res_grid = points_cam_proj[:, :, 0].reshape(batch_size, const.feature_res, const.feature_res, 2)
        down_target_img = F.grid_sample(input=target_img, grid=feature_res_grid, align_corners=False, mode='nearest',
                                        padding_mode='border')
        # save_image(down_target_img, './down_target_img.png')

        pred_img = pixels_pred.permute(0,2,1).reshape(batch_size, 3, const.feature_res, const.feature_res)
        final_img = pixels_final.permute(0, 2, 1).reshape(batch_size, 3, const.feature_res, const.feature_res)

        losses['nerf_tex'] = self.tex_loss(pred_img, down_target_img)#pixels_pred, gt_clr_nerf)
        losses['nerf_tex_final'] = self.tex_loss(final_img, down_target_img)#pixels_final, gt_clr_nerf)

        if self.TrainGAN:

            ## GAN loss
            #pixels_pred : b, patch_size*patch_size, 3
            patch_img = pixels_pred.permute(0,2,1).reshape(batch_size,3, patch_size, patch_size)
            fake_score = self.pamir_tex_discriminator(patch_img)
            losses['g_loss'] = self.gan_loss(fake_score, should_be_classified_as_real=True ).mean()


        # calculates total loss
        total_loss = 0.
        for ln in losses.keys():
            w = self.loss_weights[ln] if ln in self.loss_weights else 1.0
            total_loss += w * losses[ln]
        losses.update({'total_loss': total_loss})

        # Do backprop
        self.optm_pamir_tex_net.zero_grad()
        total_loss.backward()

        self.optm_pamir_tex_net.step()


        ## update discriminator

        if self.TrainGAN:
            if False:
                ray_index = self.sample_ray_index(img_size, target_mask, patch_size=patch_size)

                sampled_points = torch.gather(points_cam_source, 1,
                                              ray_index[:, :, None, None].repeat(1, 1, num_steps, 3))
                sampled_z_vals = torch.gather(z_vals, 1,
                                              ray_index[:, :, None, None].repeat(1, 1, num_steps, 1))
                sampled_rays_d_world = torch.gather(rays_d_cam, 1,
                                                    ray_index[:, :, None].repeat(1, 1, 3))
                gt_clr_nerf = torch.gather(gt_clr_nerf, 1,
                                           ray_index[:, :, None].repeat(1, 1, 3))

                sampled_points_proj = self.project_points(sampled_points, cam_f, cam_c, cam_tz)

                sampled_points = sampled_points.reshape(batch_size, -1, 3)
                sampled_points_proj = sampled_points_proj.reshape(batch_size, -1, 2)

                pixels_pred, pixels_final = self.get_nerf(img, vol, img_feat_geo, sampled_points, sampled_points_proj,
                                                          sampled_z_vals, sampled_rays_d_world, hierarchical,
                                                          batch_size,
                                                          num_ray, num_steps, cam_f, cam_c, cam_tz, view_diff)
                patch_img = pixels_pred.permute(0, 2, 1).reshape(batch_size, 3, patch_size, patch_size)

            gt_patch_img = gt_clr_nerf.permute(0, 2, 1).reshape(batch_size, 3, patch_size, patch_size)
            fake_score_d = self.pamir_tex_discriminator(patch_img.detach())
            real_score_d = self.pamir_tex_discriminator(gt_patch_img)
            total_loss_d = 0.
            losses['d_loss'] =self.gan_loss(real_score_d, should_be_classified_as_real=True ).mean() + self.gan_loss(fake_score_d, should_be_classified_as_real=False ).mean()
            total_loss_d+= losses['d_loss']

            self.optm_pamir_tex_discriminator.zero_grad()
            total_loss_d.backward()
            self.optm_pamir_tex_discriminator.step()


        # save
        self.write_logs(losses)

        # update learning rate
        if self.step_count % 10000 == 0:
            learning_rate = self.options.lr * (0.9 ** (self.step_count//10000))
            logging.info('Epoch %d, LR = %f' % (self.step_count, learning_rate))
            for param_group in self.optm_pamir_tex_net.param_groups:
                param_group['lr'] = learning_rate
        return losses


    def get_nerf(self, img, vol,img_feat_geo, sampled_points, sampled_points_proj, sampled_z_vals, sampled_rays_d_world, hierarchical, batch_size, num_ray, num_steps, cam_f, cam_c, cam_tz, view_diff):

        with torch.no_grad():
            nerf_feat_occupancy = self.pamir_net.get_mlp_feature(img, vol, sampled_points, sampled_points_proj)

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
            all_outputs = torch.cat([nerf_output_clr_, nerf_output_clr, nerf_output_sigma], dim=-1)
            pixels, depth, weights = fancy_integration(all_outputs.reshape(batch_size, num_ray, num_steps * 2, -1),
                                                       all_z_vals, device=self.device, white_back=True)
            pixels_pred = pixels[..., :3]
            pixels_final = pixels[..., 3:6]
        else:
            nerf_output_clr_, nerf_output_clr, nerf_output_att, nerf_smpl_feat, nerf_output_sigma = self.pamir_tex_net.forward(
                img, vol, sampled_points, sampled_points_proj, img_feat_geo, nerf_feat_occupancy)

            all_outputs = torch.cat([nerf_output_clr_, nerf_output_clr, nerf_output_sigma], dim=-1)
            pixels, depth, weights = fancy_integration(all_outputs.reshape(batch_size, num_ray, num_steps, -1),
                                                       sampled_z_vals, device=self.device, white_back=True)
            pixels_pred = pixels[..., :3]
            pixels_final = pixels[..., 3:6]

        return pixels_pred, pixels_final



    def sample_ray_index(self, img_size, mask, patch_size=32):
        """Computes per-sample loss of the occupancy value"""
        batch_size = mask.shape[0]
        x, y = torch.meshgrid(torch.linspace(0,img_size-1, img_size), torch.linspace(0,img_size-1, img_size))
        grid = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1).cuda()
        grid_for_max = grid * mask[...,:1]
        x_max = grid_for_max[...,0].max(1)[0].max(1)[0]
        y_max = grid_for_max[...,1].max(1)[0].max(1)[0]
        grid_for_min = grid * mask[...,:1] + img_size * (~mask[...,:1].bool()).float()
        x_min = grid_for_min[...,0].min(1)[0].min(1)[0]
        y_min = grid_for_min[...,1].min(1)[0].min(1)[0]
        ray_grid = torch.range(0, img_size * img_size - 1)
        ray_grid = ray_grid.reshape(img_size, img_size)
        ray_index_list = []
        for i in range(batch_size):
            tl_x = torch.randint(int(x_min[i].item()), int((x_max[i]-patch_size).item()), [1])
            tl_y = torch.randint(int(y_min[i].item()), int((y_max[i]-patch_size).item()), [1])
            ray_index_list.append(ray_grid[tl_x:tl_x + patch_size, tl_y:tl_y + patch_size].unsqueeze(0))
        ray_index = torch.cat(ray_index_list,dim=0)

        return ray_index.reshape(batch_size, -1).type(torch.int64).cuda()

    def gan_loss(self, pred, should_be_classified_as_real):
        bs = pred.size(0)
        if should_be_classified_as_real:
            return F.softplus(-pred).view(bs, -1).mean(dim=1)
        else:
            return F.softplus(pred).view(bs, -1).mean(dim=1)

    def tex_loss(self, pred_clr, gt_clr, att=None):
        """Computes per-sample loss of the occupancy value"""
        if att is None:
            att = torch.ones_like(gt_clr)
        if len(att.size()) != len(gt_clr.size()):
            att = att.unsqueeze(0)
        loss = self.criterion_tex(pred_clr * att, gt_clr * att)
        return loss

    def attention_loss(self, pred_att):
        return torch.mean(-torch.log(pred_att + 1e-4))
        # return torch.mean((pred_att - 0.9) ** 2)

    def train_summaries(self, input_batch, losses=None):
        assert losses is not None
        for ln in losses.keys():
            self.summary_writer.add_scalar(ln, losses[ln].item(), self.step_count)

    def write_logs(self, losses):
        data = dict()
        if os.path.exists(self.log_file_path):
            data = dict(np.load(self.log_file_path))
            for k in losses.keys():
                data[k] = np.append(data[k], losses[k].item())
        else:
            for k in losses.keys():
                data[k] = np.array([losses[k].item()])
        np.savez(self.log_file_path, **data)

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
        if len(pts.size())==4:
            angle = angle[:, None, None]
        else:
            angle= angle[:, None]

        pts_rot[..., 0] = pts[..., 0] * angle.cos() - pts[..., 2] * angle.sin()
        pts_rot[..., 1] = pts[..., 1]
        pts_rot[..., 2] = pts[..., 0] * angle.sin() + pts[..., 2] * angle.cos()
        return pts_rot


    def project_points(self, sampled_points, cam_f, cam_c, cam_tz):


        sampled_points_proj = sampled_points.clone()
        sampled_points_proj[..., 1]*=-1
        sampled_points_proj[..., 2]*=-1

        sampled_points_proj[..., 2] += cam_tz  # add cam_t
        sampled_points_proj[..., 0] = sampled_points_proj[..., 0] * cam_f / sampled_points_proj[..., 2] / (cam_c)
        sampled_points_proj[..., 1] = sampled_points_proj[..., 1] * cam_f / sampled_points_proj[..., 2] / (cam_c)
        sampled_points_proj = sampled_points_proj[..., :2]
        return sampled_points_proj