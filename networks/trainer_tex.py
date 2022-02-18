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
from network.arch import PamirNet, TexPamirNetAttention, TexPamirNetAttention_nerf, NeuralRenderer, ResDecoder, NeuralRenderer_coord
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
        #self.pamir_net = PamirNet().to(self.device)
        self.pamir_tex_net =  TexPamirNetAttention_nerf().to(self.device)

        # neural renderer

        self.UseGCMR= False
        self.depthaware = False
        if self.UseGCMR:
            self.graph_mesh = Mesh()
            self.graph_cnn = GraphCNN(self.graph_mesh.adjmat, self.graph_mesh.ref_vertices.t(),
                                      const.cmr_num_layers, const.cmr_num_channels).to(self.device)
            self.smpl_param_regressor = SMPLParamRegressor().to(self.device)

            self.load_pretrained_gcmr(self.options.pretrained_gcmr_checkpoint)


        # optimizers
        self.optm_pamir_tex_net = torch.optim.Adam(
            params=list(self.pamir_tex_net.parameters()), lr=float(self.options.lr)
        )

        # loses
        self.criterion_geo = nn.MSELoss().to(self.device)
        self.criterion_tex = nn.L1Loss().to(self.device)

        self.TrainGAN = False

        if self.TrainGAN:
            ## add for discriminator
            self.pamir_tex_discriminator = PatchDiscriminator().to(self.device)
            self.optm_pamir_tex_discriminator = torch.optim.Adam(
                params=list(self.pamir_tex_discriminator.parameters()), lr=float(self.options.lr)
            )

            # Pack models and optimizers in a dict - necessary for checkpointing
            self.models_dict = {'pamir_tex_net': self.pamir_tex_net, 'pamir_tex_NR': self.NR,'pamir_tex_discriminator': self.pamir_tex_discriminator}
            self.optimizers_dict = {'optimizer_pamir_net': self.optm_pamir_tex_net, 'optimizer_pamir_discriminator': self.optm_pamir_tex_discriminator}
        else:
            # Pack models and optimizers in a dict - necessary for checkpointing
            self.models_dict = {'pamir_tex_net': self.pamir_tex_net}
            self.optimizers_dict = {'optimizer_pamir_net': self.optm_pamir_tex_net}

        # Optionally start training from a pretrained checkpoint
        # Note that this is different from resuming training
        # For the latter use --resume
        #assert self.options.pretrained_pamir_net_checkpoint is not None, 'You must provide a pretrained PaMIR geometry model!'
        #self.load_pretrained_pamir_net(self.options.pretrained_pamir_net_checkpoint)

        # read energy weights
        self.loss_weights = {
            'tex': 1.0,
            'att': 0.005,
            'g_loss': 0.01,
            'geo': 1.0,
        }

        logging.info('#trainable_params = %d' %
                     sum(p.numel() for p in self.pamir_tex_net.parameters() if p.requires_grad))

        # meta results
        now = datetime.datetime.now()
        self.log_file_path = os.path.join(
            self.options.log_dir, 'log_%s.npz' % now.strftime('%Y_%m_%d_%H_%M_%S'))

    def train_step(self, input_batch):
        self.pamir_tex_net.train()
        #self.pamir_net.eval()

        # some constants
        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)

        # training data
        img = input_batch['img']
        pts = input_batch['pts']    # [:, :-self.options.point_num]
        pts_proj = input_batch['pts_proj']  # [:, :-self.options.point_num]

        #pts_occ = input_batch['pts_occ']  # [:, :-self.options.point_num]
        #pts_occ_proj = input_batch['pts_occ_proj']  # [:, :-self.options.point_num]
        #gt_ov = input_batch['pts_ov']  # [:, :-self.options.point_num]
        #pts2smpl_idx = input_batch['pts2smpl_idx']  # [:, :-self.options.point_num]
        #pts2smpl_wgt = input_batch['pts2smpl_wgt']  # [:, :-self.options.point_num]
        pts_occ_in = input_batch['pts_occ_in']  # [:, :-self.options.point_num]
        pts_occ_proj_in = input_batch['pts_occ_proj_in']  # [:, :-self.options.point_num]
        gt_ov_in = input_batch['pts_ov_in']  # [:, :-self.options.point_num]
        pts_occ_out = input_batch['pts_occ_out']  # [:, :-self.options.point_num]
        pts_occ_proj_out= input_batch['pts_occ_proj_out']  # [:, :-self.options.point_num]
        gt_ov_out = input_batch['pts_ov_out']  # [:, :-self.options.point_num]


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
        ray_start = const.ray_start#cam_tz - 0.87
        ray_end = const.ray_end#cam_tz + 0.87

        num_steps = const.num_steps
        hierarchical = const.hierarchical

        ## todo hierarchical sampling


        batch_size, pts_num = pts.size()[:2]
        losses = dict()

        if self.UseGCMR:
            gt_vert_cam = gt_scale * self.tet_smpl(gt_pose, gt_betas) + gt_trans

            with torch.no_grad():
                pred_cam, pred_rotmat, pred_betas, pred_vert_sub, \
                pred_vert, pred_vert_tetsmpl, pred_keypoints_2d = self.forward_gcmr(img)

                # camera coordinate conversion
                scale_, trans_ = self.forward_coordinate_conversion(
                    pred_vert_tetsmpl, cam_f, cam_tz, cam_c, cam_r, cam_t, pred_cam, gt_trans)
                # pred_vert_cam = scale_ * pred_vert + trans_
                # pred_vert_tetsmpl_cam = scale_ * pred_vert_tetsmpl + trans_

                pred_vert_tetsmpl_gtshape_cam = \
                    scale_ * self.tet_smpl(pred_rotmat, pred_betas.detach()) + trans_

            # randomly replace one predicted SMPL with ground-truth one
            rand_id = np.random.randint(0, batch_size, size=[batch_size//3])
            rand_id = torch.from_numpy(rand_id).long()
            pred_vert_tetsmpl_gtshape_cam[rand_id] = gt_vert_cam[rand_id]

            vol = self.voxelization(pred_vert_tetsmpl_gtshape_cam)

            if self.depthaware:  ##not yet
                pts_occ = self.forward_warp_gt_field(
                    pred_vert_tetsmpl_gtshape_cam, gt_vert_cam, pts_occ, pts2smpl_idx, pts2smpl_wgt)

        else:
            with torch.no_grad():
                gt_vert_cam = gt_scale * self.tet_smpl(gt_pose, gt_betas) + gt_trans
                vol = self.voxelization(gt_vert_cam)  # we simply use ground-truth SMPL for when training texture module
                # img_feat_geo = self.pamir_net.get_img_feature(img, no_grad=True)
                # feat_occupancy = self.pamir_net.get_mlp_feature(img, vol, pts, pts_proj)





        ## 1 train geo loss

        #_,_,_,_,output_sdf = self.pamir_tex_net.forward(img, vol, pts_occ, pts_occ_proj)
        #losses['geo'] = self.geo_loss(output_sdf, gt_ov)
        _, _, _, _, output_sdf_in = self.pamir_tex_net.forward(img, vol, pts_occ_in, pts_occ_proj_in)
        losses['geo_in'] = self.geo_loss(output_sdf_in, gt_ov_in)
        _, _, _, _, output_sdf_out = self.pamir_tex_net.forward(img, vol, pts_occ_out, pts_occ_proj_out)
        losses['geo_out'] = self.geo_loss(output_sdf_out, gt_ov_out)
        #self.loss_weights['geo'] =0

        ## 2 train tex loss

        output_clr_, output_clr, output_att, smpl_feat, output_sigma = self.pamir_tex_net.forward(img, vol, pts, pts_proj)#, img_feat_geo, feat_occupancy)

        # import pdb; pdb.set_trace
        losses['tex'] = self.tex_loss(output_clr_, gt_clr)
        losses['tex_final'] = self.tex_loss(output_clr, gt_clr)
        losses['att'] = self.attention_loss(output_att)
        #self.loss_weights['tex'] = 0
        #self.loss_weights['tex_final'] = 0

        ## 3 train nerf loss

        points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps,
                                                               resolution=(const.img_res, const.img_res),
                                                               device=self.device, fov=fov_degree, ray_start=ray_start,
                                                               ray_end=ray_end)  # batch_size, pixels, num_steps, 1

        view_diff = input_batch['view_id'] - input_batch['target_view_id']

        # 1, img_size*img_size, num_steps, 3
        points_cam[:, :, :, 2] += cam_tz

        points_cam_source = self.rotate_points(points_cam, view_diff)
        if False:
            # import pdb; pdb.set_trace()
            ###
            num_ray = 1000
            ray_index = np.random.randint(0, img_size * img_size,  num_ray )
            sampled_points= points_cam_source[:, ray_index]
            #sampled_points_global = points_cam_global[:, ray_index]
            sampled_z_vals = z_vals[:, ray_index]
            sampled_rays_d_world = rays_d_cam[:, ray_index]
            gt_clr_nerf= target_img.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)[:, ray_index]



        if True:
            # import pdb; pdb.set_trace()
            ray_index = np.random.randint(0, pts_num, 1000)
            pts_world =input_batch['pts_world'][:, ray_index]

            pts_target = self.rotate_points(pts_world, input_batch['target_view_id'])
            points_proj = self.project_points(pts_target, cam_f, cam_c, cam_tz)
            h_grid = points_proj[:, :, 0].view(batch_size, points_proj.size(1), 1, 1)
            v_grid = points_proj[:, :, 1].view(batch_size, points_proj.size(1), 1, 1)
            grid_2d = torch.cat([h_grid, v_grid], dim=-1)
            gt_clr_nerf = F.grid_sample(input=target_img.to(self.device), grid=grid_2d.to(self.device),
                                        align_corners=False, mode='bilinear', padding_mode='border').permute(0, 2, 3,
                                                                                                             1).squeeze(
                2)  # b,5000,3

            ray_d_target = pts_target - cam_t
            ray_d_target = normalize_vecs(ray_d_target)
            z_vals_ = torch.linspace(const.ray_start, const.ray_end, const.num_steps, device=self.device).reshape(1, 1,
                                                                                                                  const.num_steps,
                                                                                                                  1).repeat(
                batch_size, pts_world.size(1), 1, 1)

            points = ray_d_target.unsqueeze(2).repeat(1, 1, const.num_steps, 1) * z_vals_
            # points = points.reshape(1, -1, 3)
            points = points + cam_t  # target view!!
            sampled_points = self.rotate_points(points, view_diff)  # source view!!
            #sampled_points_global = self.rotate_points(points, -input_batch['target_view_id'])
            num_ray = pts_world.size(1)
            sampled_z_vals = z_vals_
            sampled_rays_d_world = ray_d_target

            ###
            ray_index = np.random.randint(0, img_size * img_size, 1000)
            sampled_points_random = points_cam_source[:, ray_index]
            #sampled_points_global_random = points_cam_global[:, ray_index]
            sampled_z_vals_random = z_vals[:, ray_index]
            sampled_rays_d_world_random = rays_d_cam[:, ray_index]
            gt_clr_nerf_random = target_img.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)[:, ray_index]

            ##
            num_ray = num_ray + 1000
            sampled_points = torch.cat([sampled_points, sampled_points_random], 1)
            #sampled_points_global = torch.cat([sampled_points_global, sampled_points_global_random], 1)
            sampled_z_vals = torch.cat([sampled_z_vals, sampled_z_vals_random], 1)
            sampled_rays_d_world = torch.cat([sampled_rays_d_world, sampled_rays_d_world_random], 1)
            gt_clr_nerf = torch.cat([gt_clr_nerf, gt_clr_nerf_random], 1)




        sampled_points_proj = self.project_points(sampled_points, cam_f, cam_c, cam_tz)
        sampled_points = sampled_points.reshape(batch_size, -1, 3)
        sampled_points_proj = sampled_points_proj.reshape(batch_size, -1, 2)


        if const.hierarchical:
            with torch.no_grad():
                #import pdb; pdb.set_trace()
                _,_,_,_, occ= self.pamir_tex_net.forward( img, vol, sampled_points, sampled_points_proj)
                occ =occ.reshape(batch_size, num_ray, num_steps, -1)
                occ_diff = occ[:, :, 1:] - occ[:, :, :-1]
                start_index = occ_diff.argmax(dim=2)
                import pdb; pdb.set_trace()
                end_index = occ_diff.argmax(dim=2) + 1
                start_z_vals = torch.gather(sampled_z_vals, 2, start_index.unsqueeze(-1))
                end_z_vals = torch.gather(sampled_z_vals, 2, end_index.unsqueeze(-1))

                #import pdb; pdb.set_trace()

                z_vals_diff = end_z_vals - start_z_vals
                line_01= torch.linspace(0, 1, num_steps)[None,][None,].repeat(batch_size, num_ray, 1).cuda()
                fine_z_vals = start_z_vals.squeeze(-1) + z_vals_diff.squeeze(-1) *line_01 #batch, num_ray, 24


                #std=0.1
                #max_z_vals = end_z_vals
                #std_line = torch.linspace(-std / 2, std / 2, num_steps)[None,][None,].repeat(batch_size, num_ray, 1)
                #fine_z_vals = max_z_vals.squeeze(-1) + std_line.to(self.device)


                sampled_rays_d_world = sampled_rays_d_world.unsqueeze(-2).repeat(1, 1, num_steps, 1)
                fine_points = sampled_rays_d_world * fine_z_vals[..., None]
                fine_points[:, :, :, 2] += cam_tz
                fine_points = self.rotate_points(fine_points, view_diff)
                fine_points_proj = self.project_points(fine_points, cam_f, cam_c, cam_tz)

                all_points = torch.cat([sampled_points.reshape(batch_size, num_ray, num_steps, 3), fine_points], dim=2)
                all_points_proj = torch.cat([sampled_points_proj.reshape(batch_size, num_ray, num_steps, 2),fine_points_proj], dim=2)
                all_z_vals = torch.cat([sampled_z_vals, fine_z_vals.unsqueeze(-1)], dim=2)

                _, indices = torch.sort(all_z_vals, dim=2)
                all_z_vals = torch.gather(all_z_vals, 2, indices)
                all_points = torch.gather(all_points, 2, indices.expand(-1, -1, -1, 3)).reshape(batch_size, num_ray * num_steps * 2, 3)
                all_points_proj = torch.gather(all_points_proj, 2, indices.expand(-1, -1, -1, 2)).reshape(batch_size, num_ray * num_steps * 2, 2)

                sampled_points = all_points
                sampled_points_proj = all_points_proj
                sampled_z_vals = all_z_vals
                num_steps = num_steps*2

                #import pdb; pdb.set_trace()



        nerf_output_clr_, nerf_output_clr, nerf_output_att, nerf_smpl_feat, nerf_output_sigma = self.pamir_tex_net.forward(
            img, vol, sampled_points, sampled_points_proj)


        all_outputs = torch.cat([nerf_output_clr_, nerf_output_sigma], dim=-1)
        pixels_pred, _, _ = fancy_integration2(all_outputs.reshape(batch_size, num_ray, num_steps, -1),
                                               sampled_z_vals, device=self.device, white_back=True)

        all_outputs = torch.cat([nerf_output_clr, nerf_output_sigma], dim=-1)
        feature_pred, _, _ = fancy_integration2(all_outputs.reshape(batch_size, num_ray, num_steps, -1),
                                                sampled_z_vals, device=self.device, white_back=True)


        losses['nerf_tex'] = self.tex_loss(pixels_pred, gt_clr_nerf)
        losses['nerf_tex_final'] =self.tex_loss( feature_pred , gt_clr_nerf)
        #self.loss_weights['nerf_tex'] =0
        #self.loss_weights['nerf_tex_final'] =0


        if self.TrainGAN:
            ## GAN loss
            fake_score = self.pamir_tex_discriminator(pixels_high)
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
            fake_score_d = self.pamir_tex_discriminator(pixels_high.detach())
            real_score_d = self.pamir_tex_discriminator(F.interpolate(target_img, size=self.decoder_output_size))
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

    ## from
    def geo_loss(self, pred_ov, gt_ov):
        """Computes per-sample loss of the occupancy value"""
        if self.options.use_multistage_loss:
            loss = 0
            for o in pred_ov:
                loss += self.criterion_geo(o, gt_ov)
        else:
            loss = self.criterion_geo(pred_ov[-1], gt_ov)
        return loss
    def forward_gcmr(self, img):
        # GraphCMR forward
        batch_size = img.size()[0]
        img_ = self.img_norm(img)
        pred_vert_sub, pred_cam = self.graph_cnn(img_)
        pred_vert_sub = pred_vert_sub.transpose(1, 2)
        pred_vert = self.graph_mesh.upsample(pred_vert_sub)
        x = torch.cat(
            [pred_vert_sub, self.graph_mesh.ref_vertices[None, :, :].expand(batch_size, -1, -1)],
            dim=-1)
        pred_rotmat, pred_betas = self.smpl_param_regressor(x)
        pred_vert_tetsmpl = self.tet_smpl(pred_rotmat, pred_betas)
        pred_keypoints = self.smpl.get_joints(pred_vert)
        pred_keypoints_2d = orthographic_projection(pred_keypoints, pred_cam)
        return pred_cam, pred_rotmat, pred_betas, pred_vert_sub, \
               pred_vert, pred_vert_tetsmpl, pred_keypoints_2d

    def load_pretrained_gcmr(self, model_path):
        if os.path.isdir(model_path):
            tmp = glob.glob(os.path.join(model_path, 'gcmr*.pt'))
            assert len(tmp) == 1
            logging.info('Loading GraphCMR from ' + tmp[0])
            data = torch.load(tmp[0])
        else:
            data = torch.load(model_path)
        self.graph_cnn.load_state_dict(data['graph_cnn'])
        self.smpl_param_regressor.load_state_dict(data['smpl_param_regressor'])

    def forward_coordinate_conversion(self, pred_vert_tetsmpl, cam_f, cam_tz, cam_c,
                                      cam_r, cam_t, pred_cam, gt_trans):
        # calculates camera parameters
        with torch.no_grad():
            pred_smpl_joints = self.tet_smpl.get_smpl_joints(pred_vert_tetsmpl).detach()
            pred_root = pred_smpl_joints[:, 0:1, :]
            if gt_trans is not None:
                scale = pred_cam[:, 0:1] * cam_c * (cam_tz - gt_trans[:, 0, 2:3]) / cam_f
                trans_x = pred_cam[:, 1:2] * cam_c * (
                        cam_tz - gt_trans[:, 0, 2:3]) * pred_cam[:, 0:1] / cam_f
                trans_y = -pred_cam[:, 2:3] * cam_c * (
                        cam_tz - gt_trans[:, 0, 2:3]) * pred_cam[:, 0:1] / cam_f
                trans_z = gt_trans[:, 0, 2:3] + 2 * pred_root[:, 0, 2:3] * scale
            else:
                scale = pred_cam[:, 0:1] * cam_c * cam_tz / cam_f
                trans_x = pred_cam[:, 1:2] * cam_c * cam_tz * pred_cam[:, 0:1] / cam_f
                trans_y = -pred_cam[:, 2:3] * cam_c * cam_tz * pred_cam[:, 0:1] / cam_f
                trans_z = torch.zeros_like(trans_x)
            scale_ = torch.cat([scale, -scale, -scale], dim=-1).detach().view((-1, 1, 3))
            trans_ = torch.cat([trans_x, trans_y, trans_z], dim=-1).detach().view((-1, 1, 3))

        return scale_, trans_

    def forward_warp_gt_field(self, pred_vert_tetsmpl_gtshape_cam, gt_vert_cam,
                              pts, pts2smpl_idx, pts2smpl_wgt):
        with torch.no_grad():
            trans_gt2pred = pred_vert_tetsmpl_gtshape_cam - gt_vert_cam

            trans_z_pt_list = []
            for bi in range(self.options.batch_size):
                trans_pt_bi = (
                        trans_gt2pred[bi, pts2smpl_idx[bi, :, 0], 2] * pts2smpl_wgt[bi, :, 0] +
                        trans_gt2pred[bi, pts2smpl_idx[bi, :, 1], 2] * pts2smpl_wgt[bi, :, 1] +
                        trans_gt2pred[bi, pts2smpl_idx[bi, :, 2], 2] * pts2smpl_wgt[bi, :, 2] +
                        trans_gt2pred[bi, pts2smpl_idx[bi, :, 3], 2] * pts2smpl_wgt[bi, :, 3]
                )
                trans_z_pt_list.append(trans_pt_bi.unsqueeze(0))
            trans_z_pts = torch.cat(trans_z_pt_list, dim=0)
            # translate along z-axis to resolve depth inconsistency
            # pts[:, :, 2] += trans_z_pts
            pts[:, :, 2] += torch.tanh(trans_z_pts * 20) / 20
        return pts

    ## to
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