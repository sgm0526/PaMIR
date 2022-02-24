# Software License Agreement (BSD License)
#
# Copyright (c) 2019, Zerong Zheng (zzr18@mails.tsinghua.edu.cn)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the <organization> nor the
#  names of its contributors may be used to endorse or promote products
#  derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Data loader"""

from __future__ import division, print_function

import os
import glob
import math
import numpy as np
import scipy.spatial
import scipy.io as sio
import pickle as pkl
import json
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader
import constant
from .utils import load_data_list, generate_cam_Rt

from util.volume_rendering import *
import random

class TrainingImgDataset(Dataset):
    def __init__(self, dataset_dir,
                 img_h, img_w, training, testing_res,
                 view_num_per_item, point_num, load_pts2smpl_idx_wgt,
                 smpl_data_folder='./data'):
        super(TrainingImgDataset, self).__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.training = training
        self.testing_res = testing_res
        self.dataset_dir = dataset_dir
        self.view_num_per_item = view_num_per_item
        self.point_num = point_num
        self.load_pts2smpl_idx_wgt = load_pts2smpl_idx_wgt

        if self.training:
            self.data_list = load_data_list(dataset_dir, 'data_list_train.txt')
            self.len = len(self.data_list) * self.view_num_per_item
        else:
            self.data_list = load_data_list(dataset_dir, 'data_list_test.txt')
            self.model_2_viewindex = [138,155,195,73,303,225,240,333,136,197,222,272,291,298,147,38,194,275,348,40,1,13,325,273,186]
            self.model_2_targetviewindex = [249,56,349,291,240,218,243,49,298,162,166,344,133,77,35,232,197,256,288,68,184,174,15,193,198]
            self.len = len(self.data_list) #* self.view_num_per_item


        # load smpl model data for usage
        jmdata = np.load(os.path.join(smpl_data_folder, 'joint_model.npz'))
        self.J_dirs = jmdata['J_dirs']
        self.J_template = jmdata['J_template']

        # some default parameters for testing
        self.default_testing_cam_R = constant.cam_R
        self.default_testing_cam_t = constant.cam_t
        self.default_testing_cam_f = constant.cam_f

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        data_list = self.data_list


        if self.training:
            model_id = item // self.view_num_per_item
            view_id = item % self.view_num_per_item
        else:
            model_id = item
            view_id = self.model_2_viewindex[model_id]


        data_item = data_list[model_id]
        cam_f = self.default_testing_cam_f
        point_num = self.point_num

        if True :
            source_view_list = [view_id]
            #source_view_list = [view_id, (view_id+180)%360]

        source_img_list=[]
        source_mask_list = []
        pts_list=[]

        pts_r_list=[]
        pts_proj_list=[]
        pts_clr_list = []

        pts_occ_list = []
        pts_occ_proj_list = []
        pts_ov_list = []

        pose_list = []
        betas_list = []
        trans_list = []
        scale_list = []

        pose, betas, trans, scale = self.load_smpl_parameters(data_item)

        for i in source_view_list:
            img, mask = self.load_image(data_item, i)
            source_img_list.append(torch.from_numpy(img.transpose((2, 0, 1))).unsqueeze(0))
            source_mask_list.append(torch.from_numpy(mask).unsqueeze(0))
            cam_R, cam_t = self.load_cams(data_item, i)

            pts, pts_clr, all_pts, all_pts_clr = self.load_points(data_item,i, point_num)
            pts_list.append(torch.from_numpy(pts).unsqueeze(0))
            if not self.training:
                pts = all_pts
                pts_clr = all_pts_clr
            pts_r = self.rotate_points(pts, i)
            pts_proj = self.project_points(pts, cam_R, cam_t, cam_f)
            pts_r_list.append(torch.from_numpy(pts_r).unsqueeze(0))
            pts_proj_list.append(torch.from_numpy(pts_proj).unsqueeze(0))
            pts_clr_list.append(torch.from_numpy(pts_clr).unsqueeze(0))

            pts_ids, pts_occ, pts_ov = self.load_points_occ(data_item, point_num)
            pts_occ_r = self.rotate_points(pts_occ, i)
            pts_occ_proj = self.project_points(pts_occ, cam_R, cam_t, cam_f)
            pts_occ_list.append(torch.from_numpy(pts_occ_r ).unsqueeze(0))
            pts_occ_proj_list.append(torch.from_numpy(pts_occ_proj).unsqueeze(0))
            pts_ov_list.append(torch.from_numpy(pts_ov).unsqueeze(0))


            pose, betas, trans, scale = self.update_smpl_params(pose, betas, trans, scale, i)
            pose_list.append(torch.from_numpy(pose).unsqueeze(0))
            betas_list.append(torch.from_numpy(betas).unsqueeze(0))
            trans_list.append(torch.from_numpy(trans).unsqueeze(0))
            scale_list.append(torch.from_numpy(scale).unsqueeze(0))

        source_img_list= torch.cat(source_img_list, 0 )
        source_mask_list = torch.cat(source_mask_list, 0)
        pts_list= torch.cat(pts_list, 0)
        pts_r_list = torch.cat(pts_r_list, 0)
        pts_proj_list = torch.cat(pts_proj_list, 0)
        pts_clr_list = torch.cat(pts_clr_list, 0)[0]
        pts_occ_list = torch.cat(pts_occ_list, 0)
        pts_occ_proj_list = torch.cat(pts_occ_proj_list, 0)
        pts_ov_list = torch.cat(pts_ov_list, 0)[0]
        pose_list = torch.cat(pose_list, 0)[0]
        betas_list = torch.cat(betas_list, 0)[0]
        trans_list = torch.cat(trans_list, 0)[0]
        scale_list = torch.cat(scale_list, 0)[0]

        source_view_list_=None
        for i in source_view_list:
            if source_view_list_ is None:
                source_view_list_ = torch.tensor(i).unsqueeze(0)
            else:
                source_view_list_ = torch.cat([source_view_list_, torch.tensor(i).unsqueeze(0)], 0 )





        if self.training:
            target_view_list =list(range(0, self.view_num_per_item)) #0~359
            for i in source_view_list:
                target_view_list.remove(i)
            target_view_id = random.choice(target_view_list)

        else:
            target_view_id  = self.model_2_targetviewindex[model_id]

        if target_view_id == view_id:
            raise NotImplementedError()

        target_img , target_mask = self.load_image(data_item, target_view_id)

        ###


        return_dict = {
            'model_id': model_id,
            'view_id': source_view_list_,
            'data_item': data_item,
            'img': source_img_list,
            'mask': source_mask_list,
            'pts_world': pts_list,
            'pts':  pts_r_list,
            'pts_proj': pts_proj_list,
            'pts_clr': pts_clr_list,
            'pts_occ': pts_occ_list,
            'pts_occ_proj': pts_occ_proj_list,
            'pts_ov': pts_ov_list,
            'betas': betas_list,
            'pose': pose_list,
            'scale': scale_list,
            'trans': trans_list,
            'target_view_id': target_view_id,
            'target_img': torch.from_numpy(target_img .transpose((2, 0, 1))),
            'target_mask': torch.from_numpy(target_mask),


        }

        return return_dict

    def load_image(self, data_item, view_id):
        # img_fpath = os.path.join(
        #     self.dataset_dir, constant.dataset_image_subfolder, data_item, 'color/%04d.jpg' % view_id)
        img_fpath = os.path.join(
            self.dataset_dir, constant.dataset_image_subfolder.replace('image_data','image_data_nolight2'), data_item, 'color_re/%04d.jpg' % view_id)
        msk_fpath = os.path.join(
            self.dataset_dir, constant.dataset_image_subfolder, data_item, 'mask/%04d.png' % view_id)
        try:
            img = cv.imread(img_fpath).astype(np.uint8)
            msk = cv.imread(msk_fpath).astype(np.uint8)
        except:
            raise RuntimeError('Failed to load iamge: ' + img_fpath)

        #assert img.shape[0] == self.img_h and img.shape[1] == self.img_w ##
        img = np.float32(cv.cvtColor(img, cv.COLOR_RGB2BGR)) / 255.
        msk = np.float32(msk) / 255.
        if len(msk.shape) == 2:
            msk = np.expand_dims(msk, axis=-1)
        img = img * msk + (1 - msk)  # white background
        img_black = img * msk
        if not (img.shape[0] == self.img_h and img.shape[1] == self.img_w ):
            img = cv.resize(img, (self.img_w, self.img_h)) ##
            msk= cv.resize(msk, (self.img_w, self.img_h))  ##
        return img, msk

    def load_cams(self, data_item, view_id):
        dat_fpath = os.path.join(
            self.dataset_dir,  constant.dataset_image_subfolder, data_item,'meta/cam_data.mat')
        try:
            cams_data = sio.loadmat(dat_fpath, verify_compressed_data_integrity=False)
        except ValueError as e:
            print('Value error occurred when loading ' + dat_fpath)
            raise ValueError(str(e))
        cams_data = cams_data['cam'][0]
        cam_param = cams_data[view_id]
        cam_R, cam_t = generate_cam_Rt(
            center=cam_param['center'][0, 0], right=cam_param['right'][0, 0],
            up=cam_param['up'][0, 0], direction=cam_param['direction'][0, 0])
        cam_R = cam_R.astype(np.float32)
        cam_t = cam_t.astype(np.float32)
        return cam_R, cam_t

    def load_points(self, data_item, view_id, point_num):
        uvpos_fpath = os.path.join(
            self.dataset_dir, constant.dataset_image_subfolder, data_item, 'meta/uv_pos.exr')
        uvmsk_fpath = os.path.join(
            self.dataset_dir, constant.dataset_image_subfolder, data_item, 'meta/uv_mask.png')
        uvnml_fpath = os.path.join(
            self.dataset_dir, constant.dataset_image_subfolder, data_item, 'meta/uv_nml.png')
        #uvclr_fpath = os.path.join(
        #    self.dataset_dir, constant.dataset_image_subfolder, data_item, 'color_uv/%04d.png' % view_id)
        uvclr_fpath = os.path.join(
            self.dataset_dir, constant.dataset_image_subfolder.replace('image_data', 'image_data_nolight2'), data_item,
            'color_uv_re/%04d.png' % view_id)
        try:
            uv_mask = cv.imread(uvmsk_fpath)
            uv_mask = uv_mask[:, :, 0] != 0
            # UV render. each pixel is the color of the point.
            # [H, W, 3] 0 ~ 1 float
            uv_normal = cv.imread(uvnml_fpath)
            uv_normal = cv.cvtColor(uv_normal, cv.COLOR_BGR2RGB) / 255.0
            uv_normal = 2.0 * uv_normal - 1.0
            # Position render. each pixel is the xyz coordinates of the point
            uv_pos = cv.imread(uvpos_fpath, 2 | 4)[:, :, ::-1]

            uv_render = cv.imread(uvclr_fpath)
            uv_render = cv.cvtColor(uv_render, cv.COLOR_BGR2RGB) / 255.0
        except ValueError as e:
            print('Value error occurred when loading ' + uvclr_fpath)
            raise ValueError(str(e))

        ### In these few lines we flattern the masks, positions, and normals
        uv_mask = uv_mask.reshape((-1))
        uv_pos = uv_pos.reshape((-1, 3))
        uv_render = uv_render.reshape((-1, 3))
        uv_normal = uv_normal.reshape((-1, 3))

        surface_points = uv_pos[uv_mask]
        surface_colors = uv_render[uv_mask]
        surface_normal = uv_normal[uv_mask]

        all_points = surface_points
        all_points_clr = surface_colors

        sample_id = np.int32(np.random.rand(point_num) * len(surface_points))
        surface_points = surface_points[sample_id]
        surface_colors = surface_colors[sample_id]
        surface_normal = surface_normal[sample_id]

        surface_points += surface_normal * np.random.randn(point_num, 1) * 0.01

        return surface_points, surface_colors, all_points, all_points_clr

    def load_points_occ(self, data_item, point_num):
        dat_fpath = os.path.join(
            self.dataset_dir, constant.dataset_image_subfolder, data_item, 'sample/samples.mat')
        try:
            pts_data = sio.loadmat(dat_fpath, verify_compressed_data_integrity=False)
        except ValueError as e:
            print('Value error occurred when loading ' + dat_fpath)
            raise ValueError(str(e))

        pts_adp_idp = np.int32(np.random.rand(point_num//2) * len(pts_data['surface_points_inside']))
        pts_adp_idn = np.int32(np.random.rand(point_num//2) * len(pts_data['surface_points_outside']))
        pts_uni_idp = np.int32(np.random.rand(point_num//32) * len(pts_data['uniform_points_inside']))
        pts_uni_idn = np.int32(np.random.rand(point_num//32) * len(pts_data['uniform_points_outside']))

        pts_adp_p = pts_data['surface_points_inside'][pts_adp_idp]
        pts_adp_n = pts_data['surface_points_outside'][pts_adp_idn]
        pts_uni_p = pts_data['uniform_points_inside'][pts_uni_idp]
        pts_uni_n = pts_data['uniform_points_outside'][pts_uni_idn]

        pts = np.concatenate([pts_adp_p, pts_adp_n, pts_uni_p, pts_uni_n], axis=0)
        pts_ov = np.concatenate([
            np.ones([len(pts_adp_p), 1]), np.zeros([len(pts_adp_n), 1]),
            np.ones([len(pts_uni_p), 1]), np.zeros([len(pts_uni_n), 1]),
        ], axis=0)

        pts = pts.astype(np.float32)
        pts_ov = pts_ov.astype(np.float32)

        return (pts_adp_idp, pts_adp_idn, pts_uni_idp, pts_uni_idn), pts, pts_ov


    def load_points_occ_in(self, data_item, point_num):
        dat_fpath = os.path.join(
            self.dataset_dir, constant.dataset_image_subfolder, data_item, 'sample/samples.mat')
        try:
            pts_data = sio.loadmat(dat_fpath, verify_compressed_data_integrity=False)
        except ValueError as e:
            print('Value error occurred when loading ' + dat_fpath)
            raise ValueError(str(e))

        pts_adp_idp = np.int32(np.random.rand(point_num//2) * len(pts_data['surface_points_inside']))

        pts_uni_idp = np.int32(np.random.rand(point_num//32) * len(pts_data['uniform_points_inside']))


        pts_adp_p = pts_data['surface_points_inside'][pts_adp_idp]
        pts_uni_p = pts_data['uniform_points_inside'][pts_uni_idp]


        pts = np.concatenate([pts_adp_p, pts_uni_p], axis=0)
        pts_ov = np.concatenate([np.ones([len(pts_adp_p), 1]), np.ones([len(pts_uni_p), 1])], axis=0)

        pts = pts.astype(np.float32)
        pts_ov = pts_ov.astype(np.float32)

        return (pts_adp_idp, pts_uni_idp), pts, pts_ov

    def load_points_occ_out(self, data_item, point_num):
        dat_fpath = os.path.join(
            self.dataset_dir, constant.dataset_image_subfolder, data_item, 'sample/samples.mat')
        try:
            pts_data = sio.loadmat(dat_fpath, verify_compressed_data_integrity=False)
        except ValueError as e:
            print('Value error occurred when loading ' + dat_fpath)
            raise ValueError(str(e))


        pts_adp_idn = np.int32(np.random.rand(point_num//2) * len(pts_data['surface_points_outside']))
        pts_uni_idn = np.int32(np.random.rand(point_num//32) * len(pts_data['uniform_points_outside']))


        pts_adp_n = pts_data['surface_points_outside'][pts_adp_idn]
        pts_uni_n = pts_data['uniform_points_outside'][pts_uni_idn]

        pts = np.concatenate([pts_adp_n, pts_uni_n], axis=0)
        pts_ov = np.concatenate([np.zeros([len(pts_adp_n), 1]), np.zeros([len(pts_uni_n), 1]),
        ], axis=0)

        pts = pts.astype(np.float32)
        pts_ov = pts_ov.astype(np.float32)

        return (pts_adp_idn, pts_uni_idn), pts, pts_ov
    def load_sample2smpl_data(self, data_item, pts_ids):
        dat_fpath = os.path.join(
            self.dataset_dir, constant.dataset_image_subfolder, data_item, 'sample/sample2smpl.mat')
        try:
            data = sio.loadmat(dat_fpath, verify_compressed_data_integrity=False)
        except ValueError as e:
            print('Value error occurred when loading ' + dat_fpath)
            raise ValueError(str(e))

        idx0 = data['idx_surface_points_inside'][pts_ids[0]]
        idx1 = data['idx_surface_points_outside'][pts_ids[1]]
        idx2 = data['idx_uniform_points_inside'][pts_ids[2]]
        idx3 = data['idx_uniform_points_outside'][pts_ids[3]]
        idx = np.concatenate([idx0, idx1, idx2, idx3], axis=0).astype(np.long)
        dst0 = data['dist_surface_points_inside'][pts_ids[0]]
        dst1 = data['dist_surface_points_outside'][pts_ids[1]]
        dst2 = data['dist_uniform_points_inside'][pts_ids[2]]
        dst3 = data['dist_uniform_points_outside'][pts_ids[3]]
        dst = np.concatenate([dst0, dst1, dst2, dst3], axis=0).astype(np.float32)
        min_dst = np.min(dst, axis=1, keepdims=True)
        wgt = np.exp(-dst*dst / (2*min_dst*min_dst))
        wgt = wgt / np.sum(wgt, axis=1, keepdims=True)
        return idx, wgt

    def load_smpl_parameters(self, data_item):
        dat_fpath = os.path.join(
            self.dataset_dir, constant.dataset_mesh_subfolder, data_item, 'smpl/smpl_param.pkl')
        with open(dat_fpath, 'rb') as fp:
            data = pkl.load(fp)
            pose = np.float32(data['body_pose']).reshape((-1, ))
            betas = np.float32(data['betas']).reshape((-1,))
            trans = np.float32(data['global_body_translation']).reshape((1, -1))
            scale = np.float32(data['body_scale']).reshape((1, -1))
        return pose, betas, trans, scale

    def rotate_points(self, pts, view_id):
        # rotate points to current view
        angle = 2 * np.pi * view_id / self.view_num_per_item
        pts_rot = np.zeros_like(pts)
        pts_rot[:, 0] = pts[:, 0] * math.cos(angle) - pts[:, 2] * math.sin(angle)
        pts_rot[:, 1] = pts[:, 1]
        pts_rot[:, 2] = pts[:, 0] * math.sin(angle) + pts[:, 2] * math.cos(angle)
        return pts_rot.astype(np.float32)

    def project_points(self, pts, cam_R, cam_t, cam_f):
        pts_proj = np.dot(pts, cam_R.transpose()) + cam_t
        pts_proj[:, 0] = pts_proj[:, 0] * cam_f / pts_proj[:, 2] / (self.img_w / 2)
        pts_proj[:, 1] = pts_proj[:, 1] * cam_f / pts_proj[:, 2] / (self.img_h / 2)
        pts_proj = pts_proj[:, :2]
        return pts_proj.astype(np.float32)

    def update_smpl_params(self, pose, betas, trans, scale, view_id):
        # body shape and scale doesn't need to change
        betas_updated = np.copy(betas)
        scale_updated = np.copy(scale)

        # update body pose
        angle = 2 * np.pi * view_id / self.view_num_per_item
        delta_r = cv.Rodrigues(np.array([0, -angle, 0]))[0]
        root_rot = cv.Rodrigues(pose[:3])[0]
        root_rot_updated = np.matmul(delta_r, root_rot)
        pose_updated = np.copy(pose)
        pose_updated[:3] = np.squeeze(cv.Rodrigues(root_rot_updated)[0])

        # update body translation
        J = self.J_dirs.dot(betas) + self.J_template
        root = J[0]
        J_orig = np.expand_dims(root, axis=-1)
        J_new = np.dot(delta_r, np.expand_dims(root, axis=-1))
        J_orig, J_new = np.reshape(J_orig, (1, -1)), np.reshape(J_new, (1, -1))
        trans_updated = np.dot(delta_r, np.reshape(trans, (-1, 1)))
        trans_updated = np.reshape(trans_updated, (1, -1)) + (J_new - J_orig) * scale
        return np.float32(pose_updated), np.float32(betas_updated), \
               np.float32(trans_updated), np.float32(scale_updated)



class AllImgDataset(Dataset):
    def __init__(self, dataset_dir,
                 img_h, img_w, testing_res,
                 view_num_per_item, load_pts2smpl_idx_wgt,
                 smpl_data_folder='./data'):
        super(AllImgDataset, self).__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.testing_res = testing_res
        self.dataset_dir = dataset_dir
        self.view_num_per_item = view_num_per_item
        self.load_pts2smpl_idx_wgt = load_pts2smpl_idx_wgt


        self.data_list_all = load_data_list(dataset_dir, 'data_list_all.txt')
        #self.data_list = self.data_list_all[260:]
        self.data_list = self.data_list_all
        print(self.data_list)
        self.source_view_list = list(range(0, 360, 18))
        self.target_view_diff = [0, 90, 180, 270]
        self.len = len(self.data_list) * len(self.target_view_diff) * len(self.source_view_list)

        # load smpl model data for usage
        jmdata = np.load(os.path.join(smpl_data_folder, 'joint_model.npz'))
        self.J_dirs = jmdata['J_dirs']
        self.J_template = jmdata['J_template']

        # some default parameters for testing
        self.default_testing_cam_R = constant.cam_R
        self.default_testing_cam_t = constant.cam_t
        self.default_testing_cam_f = constant.cam_f

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        data_list = self.data_list_all


        # if self.training:
        #     model_id = item //con self.view_num_per_item
        #     view_id = item % self.view_num_per_item
        # else:
        #     model_id = item
        #     view_id = self.model_2_viewindex[model_id]


        # model_id = item // (len(self.target_view_diff) * len(self.source_view_list)) + 260
        model_id = item // (len(self.target_view_diff) * len(self.source_view_list))
        view_id = self.source_view_list[(item % (len(self.target_view_diff) * len(self.source_view_list))) // len(self.target_view_diff)]
        target_view_id_ind = (item % (len(self.target_view_diff) * len(self.source_view_list))) % len(self.target_view_diff)
        target_view_id = self.target_view_diff[target_view_id_ind]
        target_view_id = target_view_id + view_id
        if target_view_id >= 360:
            target_view_id = target_view_id - 360
        data_item = data_list[model_id]

        cam_f = self.default_testing_cam_f

        img, mask = self.load_image(data_item, view_id)
        cam_R, cam_t = self.load_cams(data_item, view_id)
        pts, pts_clr, all_pts, all_pts_clr = self.load_points(data_item, view_id, 1)

        pts = all_pts
        pts_clr = all_pts_clr

        target_img , target_mask = self.load_image(data_item, target_view_id)



        pts_r = self.rotate_points(pts, view_id)
        pts_proj = self.project_points(pts, cam_R, cam_t, cam_f)
        # pts_clr = pts_clr * alpha + beta
        pose, betas, trans, scale = self.load_smpl_parameters(data_item)
        pose, betas, trans, scale = self.update_smpl_params(pose, betas, trans, scale, view_id)

        return_dict = {
            'model_id': model_id,
            'view_id': view_id,
            'data_item': data_item,
            'img': torch.from_numpy(img.transpose((2, 0, 1))),
            'pts': torch.from_numpy(pts_r),
            'pts_proj': torch.from_numpy(pts_proj),
            'pts_clr': torch.from_numpy(pts_clr),
            # 'pts_clr': torch.from_numpy(pts_clr),
            # 'pts_clr_msk': torch.from_numpy(pts_clr_msk),
            'betas': torch.from_numpy(betas),
            'pose': torch.from_numpy(pose),
            'scale': torch.from_numpy(scale),
            'trans': torch.from_numpy(trans),
            'target_view_id': target_view_id,
            'target_img': torch.from_numpy(target_img .transpose((2, 0, 1))),
            'cam_r': torch.from_numpy(cam_R),
            'cam_t': torch.from_numpy(cam_t),
            'pts_world': torch.from_numpy(pts),
            'mask': torch.from_numpy(mask),
            'target_mask': torch.from_numpy(target_mask),


        }

        return return_dict

    def load_image(self, data_item, view_id):
        # img_fpath = os.path.join(
        #     self.dataset_dir, constant.dataset_image_subfolder, data_item, 'color/%04d.jpg' % view_id)
        img_fpath = os.path.join(
            self.dataset_dir, constant.dataset_image_subfolder.replace('image_data','image_data_nolight2'), data_item, 'color_re/%04d.jpg' % view_id)
        msk_fpath = os.path.join(
            self.dataset_dir, constant.dataset_image_subfolder, data_item, 'mask/%04d.png' % view_id)
        try:
            img = cv.imread(img_fpath).astype(np.uint8)
            msk = cv.imread(msk_fpath).astype(np.uint8)
        except:
            raise RuntimeError('Failed to load iamge: ' + img_fpath)

        #assert img.shape[0] == self.img_h and img.shape[1] == self.img_w ##
        img = np.float32(cv.cvtColor(img, cv.COLOR_RGB2BGR)) / 255.
        msk = np.float32(msk) / 255.
        if len(msk.shape) == 2:
            msk = np.expand_dims(msk, axis=-1)
        img = img * msk + (1 - msk)  # white background
        img_black = img * msk
        if not (img.shape[0] == self.img_h and img.shape[1] == self.img_w ):
            img = cv.resize(img, (self.img_w, self.img_h)) ##
            msk= cv.resize(msk, (self.img_w, self.img_h))  ##
        return img, msk

    def load_cams(self, data_item, view_id):
        dat_fpath = os.path.join(
            self.dataset_dir,  constant.dataset_image_subfolder, data_item,'meta/cam_data.mat')
        try:
            cams_data = sio.loadmat(dat_fpath, verify_compressed_data_integrity=False)
        except ValueError as e:
            print('Value error occurred when loading ' + dat_fpath)
            raise ValueError(str(e))
        cams_data = cams_data['cam'][0]
        cam_param = cams_data[view_id]
        cam_R, cam_t = generate_cam_Rt(
            center=cam_param['center'][0, 0], right=cam_param['right'][0, 0],
            up=cam_param['up'][0, 0], direction=cam_param['direction'][0, 0])
        cam_R = cam_R.astype(np.float32)
        cam_t = cam_t.astype(np.float32)
        return cam_R, cam_t

    def load_points(self, data_item, view_id, point_num):
        uvpos_fpath = os.path.join(
            self.dataset_dir, constant.dataset_image_subfolder, data_item, 'meta/uv_pos.exr')
        uvmsk_fpath = os.path.join(
            self.dataset_dir, constant.dataset_image_subfolder, data_item, 'meta/uv_mask.png')
        uvnml_fpath = os.path.join(
            self.dataset_dir, constant.dataset_image_subfolder, data_item, 'meta/uv_nml.png')
        #uvclr_fpath = os.path.join(
        #    self.dataset_dir, constant.dataset_image_subfolder, data_item, 'color_uv/%04d.png' % view_id)
        uvclr_fpath = os.path.join(
            self.dataset_dir, constant.dataset_image_subfolder.replace('image_data', 'image_data_nolight2'), data_item,
            'color_uv_re/%04d.png' % view_id)
        try:
            uv_mask = cv.imread(uvmsk_fpath)
            uv_mask = uv_mask[:, :, 0] != 0
            # UV render. each pixel is the color of the point.
            # [H, W, 3] 0 ~ 1 float
            uv_normal = cv.imread(uvnml_fpath)
            uv_normal = cv.cvtColor(uv_normal, cv.COLOR_BGR2RGB) / 255.0
            uv_normal = 2.0 * uv_normal - 1.0
            # Position render. each pixel is the xyz coordinates of the point
            uv_pos = cv.imread(uvpos_fpath, 2 | 4)[:, :, ::-1]

            uv_render = cv.imread(uvclr_fpath)
            uv_render = cv.cvtColor(uv_render, cv.COLOR_BGR2RGB) / 255.0
        except ValueError as e:
            print('Value error occurred when loading ' + uvclr_fpath)
            raise ValueError(str(e))

        ### In these few lines we flattern the masks, positions, and normals
        uv_mask = uv_mask.reshape((-1))
        uv_pos = uv_pos.reshape((-1, 3))
        uv_render = uv_render.reshape((-1, 3))
        uv_normal = uv_normal.reshape((-1, 3))

        surface_points = uv_pos[uv_mask]
        surface_colors = uv_render[uv_mask]
        surface_normal = uv_normal[uv_mask]

        all_points = surface_points
        all_points_clr = surface_colors

        sample_id = np.int32(np.random.rand(point_num) * len(surface_points))
        surface_points = surface_points[sample_id]
        surface_colors = surface_colors[sample_id]
        surface_normal = surface_normal[sample_id]

        surface_points += surface_normal * np.random.randn(point_num, 1) * 0.01

        return surface_points, surface_colors, all_points, all_points_clr

    def load_smpl_parameters(self, data_item):
        dat_fpath = os.path.join(
            self.dataset_dir, constant.dataset_mesh_subfolder, data_item, 'smpl/smpl_param.pkl')
        with open(dat_fpath, 'rb') as fp:
            data = pkl.load(fp)
            pose = np.float32(data['body_pose']).reshape((-1, ))
            betas = np.float32(data['betas']).reshape((-1,))
            trans = np.float32(data['global_body_translation']).reshape((1, -1))
            scale = np.float32(data['body_scale']).reshape((1, -1))
        return pose, betas, trans, scale

    def rotate_points(self, pts, view_id):
        # rotate points to current view
        angle = 2 * np.pi * view_id / self.view_num_per_item
        pts_rot = np.zeros_like(pts)
        pts_rot[:, 0] = pts[:, 0] * math.cos(angle) - pts[:, 2] * math.sin(angle)
        pts_rot[:, 1] = pts[:, 1]
        pts_rot[:, 2] = pts[:, 0] * math.sin(angle) + pts[:, 2] * math.cos(angle)
        return pts_rot.astype(np.float32)

    def project_points(self, pts, cam_R, cam_t, cam_f):
        pts_proj = np.dot(pts, cam_R.transpose()) + cam_t
        pts_proj[:, 0] = pts_proj[:, 0] * cam_f / pts_proj[:, 2] / (self.img_w / 2)
        pts_proj[:, 1] = pts_proj[:, 1] * cam_f / pts_proj[:, 2] / (self.img_h / 2)
        pts_proj = pts_proj[:, :2]
        return pts_proj.astype(np.float32)

    def update_smpl_params(self, pose, betas, trans, scale, view_id):
        # body shape and scale doesn't need to change
        betas_updated = np.copy(betas)
        scale_updated = np.copy(scale)

        # update body pose
        angle = 2 * np.pi * view_id / self.view_num_per_item
        delta_r = cv.Rodrigues(np.array([0, -angle, 0]))[0]
        root_rot = cv.Rodrigues(pose[:3])[0]
        root_rot_updated = np.matmul(delta_r, root_rot)
        pose_updated = np.copy(pose)
        pose_updated[:3] = np.squeeze(cv.Rodrigues(root_rot_updated)[0])

        # update body translation
        J = self.J_dirs.dot(betas) + self.J_template
        root = J[0]
        J_orig = np.expand_dims(root, axis=-1)
        J_new = np.dot(delta_r, np.expand_dims(root, axis=-1))
        J_orig, J_new = np.reshape(J_orig, (1, -1)), np.reshape(J_new, (1, -1))
        trans_updated = np.dot(delta_r, np.reshape(trans, (-1, 1)))
        trans_updated = np.reshape(trans_updated, (1, -1)) + (J_new - J_orig) * scale
        return np.float32(pose_updated), np.float32(betas_updated), \
               np.float32(trans_updated), np.float32(scale_updated)


def worker_init_fn(worker_id):  # set numpy's random seed
    seed = torch.initial_seed()
    seed = seed % (2 ** 32)
    np.random.seed(seed + worker_id)


class TrainingImgLoader(DataLoader):
    def __init__(self, dataset_dir, img_h, img_w, data_lists_and_repetition_factor,
                 training=True, testing_res=512,
                 view_num_per_item=60, point_num=5000,
                 load_pts2smpl_idx_wgt=False, batch_size=4, num_workers=8):
        self.dataset = TrainingImgDataset(
            dataset_dir=dataset_dir, img_h=img_h, img_w=img_w,
            data_lists_and_repetition_factor=data_lists_and_repetition_factor,
            training=training, testing_res=testing_res,
            view_num_per_item=view_num_per_item, point_num=point_num,
            load_pts2smpl_idx_wgt=load_pts2smpl_idx_wgt)
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.training = training
        self.testing_res = testing_res
        self.dataset_dir = dataset_dir
        self.view_num_per_item = view_num_per_item
        self.point_num = point_num
        super(TrainingImgLoader, self).__init__(
            self.dataset, batch_size=batch_size, shuffle=training, num_workers=num_workers,
            worker_init_fn=worker_init_fn, drop_last=True)


