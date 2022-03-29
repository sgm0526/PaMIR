from __future__ import division, print_function

import os
import torch
import pickle as pkl
from tqdm import tqdm

from util import util
from util import obj_io
from torch.nn import functional as F
import cv2 as cv

from Pytorch_metrics import metrics
# from TrainingDataPreparation.main_render_final import render_mesh
from TrainingDataPreparation.ObjIO import load_obj_data
import trimesh
import numpy as np
from torch.utils.data import DataLoader
import constant as const
from skimage import measure
import numpy as np
import mrcfile
from torchvision.utils import save_image
import open3d
from os import listdir
from OSNet.encoder import OsNetEncoder

from evaluator import Evaluator
from evaluator_tex_pamir import EvaluatorTex
from evaluator_tex import EvaluatorTex as EvaluatorTex_single
from evaluator_tex_multi import EvaluatorTex as EvaluatorTex_multi

from dataloader.dataloader_testing import TestingImgLoader
from dataloader.dataloader_tex import TrainingImgDataset, TrainingImgDataset_deephuman
from dataloader.dataloader_tex_multi import TrainingImgDataset as TrainingImgDataset_multi



def get_surface_dist(tgt_mesh , src_mesh): #tgt_meshname , meshname):
    num_sample = 10000

    #tgt_mesh = trimesh.load(tgt_meshname)
    #src_mesh= trimesh.load(meshname)

    src_surf_pts, _ = trimesh.sample.sample_surface(src_mesh, num_sample)
    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt_mesh, src_surf_pts)

    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    src_tgt_dist = src_tgt_dist.mean()
    return src_tgt_dist


def get_chamfer_dist(tgt_mesh , src_mesh): #tgt_meshname , meshname):
    num_sample = 10000

    #tgt_mesh = trimesh.load(tgt_meshname)
    #src_mesh = trimesh.load(meshname)

    src_surf_pts, _ = trimesh.sample.sample_surface(src_mesh, num_sample)
    tgt_surf_pts, _ = trimesh.sample.sample_surface(tgt_mesh, num_sample)

    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt_mesh, src_surf_pts)
    _, tgt_src_dist, _ = trimesh.proximity.closest_point(src_mesh, tgt_surf_pts)

    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    tgt_src_dist[np.isnan(tgt_src_dist)] = 0

    src_tgt_dist = src_tgt_dist.mean()
    tgt_src_dist = tgt_src_dist.mean()

    chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2

    return chamfer_dist


reid_encoder = OsNetEncoder(
        input_width=1,
        input_height=1,
        weight_filepath="./OSNet/weights/osnet_ibn_x1_0_imagenet.pth",
        batch_size=32,
        num_classes=2022,
        patch_height=256,
        patch_width=128,
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
        GPU=True)


def validation_pifu(pifu_dir):


    device = torch.device("cuda")
    evaluater = EvaluatorTex_single(device, pretrained_checkpoint_pamir, pretrained_checkpoint_pamirtex)
    val_ds = TrainingImgDataset(
        '/home/nas1_temp/dataset/Thuman', img_h=const.img_res, img_w=const.img_res,
        training=False, testing_res=256,
        view_num_per_item=360,
        point_num=5000,
        load_pts2smpl_idx_wgt=True,
        smpl_data_folder='./data')

    val_data_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8,
                                 worker_init_fn=None, drop_last=False)


    p2s_list=[]
    chamfer_list=[]
    psnr_list = []
    ssim_list = []
    lpips_list = []

    out_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/validation_' + pifu_dir.split('/')[-1]
    os.makedirs(out_dir, exist_ok=True)

    for step_val, batch in enumerate(tqdm(val_data_loader, desc='Testing', total=len(val_data_loader), initial=0)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


        # model_id = str(501 + batch['model_id'].item()).zfill(4)
        model_id = (str(501 + batch['model_id'].item()).zfill(4) + '_' + str(batch['view_id'].item()).zfill(4))

        print(model_id)

        vol_res = 256

        #mesh_fname =



        mesh = trimesh.load(mesh_fname)

        vertices1 = evaluater.rotate_points(torch.from_numpy(mesh['v']).cuda().unsqueeze(0), -batch['view_id'])
        mesh_fname = os.path.join(out_dir, model_id + '_sigma_mesh_gtview.obj')
        obj_io.save_obj_data({'v': vertices1[0].squeeze().detach().cpu().numpy(),
                              'f': mesh['f']},
                             mesh_fname)


        # save_image
        image_fname = os.path.join(out_dir, model_id + '_src_image.png')
        save_image(batch['img'],  image_fname )

        #measure dist
        mesh_fname = os.path.join(out_dir, model_id + '_sigma_mesh_gtview.obj')

        model_number = str(501 + batch['model_id'].item()).zfill(4)
        tgt_meshname = f'/home/nas1_temp/dataset/Thuman/mesh_data/{model_number}/{model_number}.obj'
        tgt_mesh = trimesh.load(tgt_meshname)
        src_mesh = trimesh.load(mesh_fname)
        #import pdb; pdb.set_trace()
        tgt_mesh  = trimesh.Trimesh.simplify_quadratic_decimation(tgt_mesh, 100000)

        p2s_dist = get_surface_dist(tgt_mesh, src_mesh)
        chamfer_dist= get_chamfer_dist(tgt_mesh, src_mesh)
        p2s_list.append(p2s_dist)
        chamfer_list.append(chamfer_dist)

        print('p2s:', p2s_dist)
        print('chamfer', chamfer_dist)

        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("model id: %s \n" % model_id)
        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("p2s: %f \n" % p2s_dist)
        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("chamfer: %f \n" % chamfer_dist)


    print('Testing Done. ')
    print('p2s mean:',np.mean(p2s_list))
    print('chamfer mean:', np.mean(chamfer_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("p2s mean: %f \n" % np.mean(p2s_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("chamfer mean: %f \n" % np.mean(chamfer_list))

    print('psnr mean:', np.mean(psnr_list))
    print('ssim mean:', np.mean(ssim_list))
    print('lpips mean:', np.mean(lpips_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("'psnr mean: %f \n" % np.mean(psnr_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("ssim mean: %f \n" %  np.mean(ssim_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("lpips mean: %f \n" % np.mean(lpips_list))



def validation_texture_pifu(pifu_dir):


    device = torch.device("cuda")
    evaluater = EvaluatorTex_single(device, pretrained_checkpoint_pamir, pretrained_checkpoint_pamirtex)


    val_ds = TrainingImgDataset(
        '/home/nas1_temp/dataset/Thuman', img_h=const.img_res, img_w=const.img_res,
        training=False, testing_res=256,
        view_num_per_item=360,
        point_num=5000,
        load_pts2smpl_idx_wgt=True,
        smpl_data_folder='./data')



    val_data_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8,
                                 worker_init_fn=None, drop_last=False)



    out_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/validation_vgglpips_texturedgtmesh__' + '_'.join([pretrained_checkpoint_pamirtex.split('/')[-3], pretrained_checkpoint_pamirtex.split('/')[-1][:-3]])
    os.makedirs(out_dir, exist_ok=True)
    psnr_list = []
    ssim_list = []
    lpips_list = []
    reid_list = []


    for step_val, batch in enumerate(tqdm(val_data_loader, desc='Testing', total=len(val_data_loader), initial=0)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        model_id = (str(501 + batch['model_id'].item()).zfill(4) + '_' + str(batch['view_id'].item()).zfill(4))
        print(model_id)
        model_number = str(501 + batch['model_id'].item()).zfill(4)



        ## render using vertex color
        target_viewid = -(batch['view_id'].item()+180)
        rendered_img = render_mesh(mesh_fname, render_angle=target_viewid )
        rendered_img = torch.from_numpy(rendered_img).permute(2,0,1).unsqueeze(0 )
        # save_image
        image_fname = os.path.join(out_dir, model_id + '_rendered_image.png')
        save_image(rendered_img, image_fname)

        ##
        if True:

            target_viewid =str((batch['view_id'].item()+180)%360).zfill(4)
            tgt_imagename = f'/home/nas1_temp/dataset/Thuman/image_data_vc/{model_number}/color/{target_viewid}.jpg'
            gt_img = cv.imread(tgt_imagename).astype(np.uint8)
            gt_img = np.float32(cv.cvtColor( gt_img , cv.COLOR_RGB2BGR)) / 255.
            gt_img= torch.from_numpy( gt_img .transpose((2, 0, 1))).unsqueeze(0).cuda()
        else:
            gt_img= render_mesh(tgt_meshname, render_angle=target_viewid  )
            gt_img = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0)




        image_fname = os.path.join(out_dir, model_id + '_rendered_gt_image.png')
        save_image(gt_img, image_fname)


        ## measure metrics
        psnr = metrics.PSNR()(rendered_img.cuda(), gt_img.cuda())
        ssim = metrics.SSIM()(rendered_img.cuda(), gt_img.cuda())
        lpips = metrics.LPIPS(True)(rendered_img.cuda(), gt_img.cuda())
        psnr_list.append(psnr.item())
        ssim_list.append(ssim.item())
        lpips_list.append(lpips.item())

        ## measure reID

        rendered_feat = reid_encoder.get_features(rendered_img.cuda())
        gt_target_feat = reid_encoder.get_features(gt_img.cuda())
        gt_source_feat = reid_encoder.get_features(batch['img'].cuda()*batch['mask'].permute(0,3,1,2).cuda())

        #reid_dist = reid_encoder.euclidean_squared_distance(gt_source_feat[0].unsqueeze(0), torch.cat((gt_target_feat[0][None,], rendered_feat[0][None,]),dim=0))
        reid_dist = reid_encoder.cosine_distance(gt_source_feat[0].unsqueeze(0), torch.cat(
            (gt_target_feat[0][None,], rendered_feat[0][None,]), dim=0))
        reid_list.append(reid_dist)

        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("model id: %s \n" % model_id)
        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("psnr: %f \n" % psnr)
        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("ssim : %f \n" % ssim)
        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("lpips : %f \n" % lpips)
        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("reid : %f , reid_gt : %f \n" % (reid_dist[:,1], reid_dist[:,0]))


        ## meshlab/blender capture
        if True:
            check = 1


    print('psnr mean:', np.mean(psnr_list))
    print('ssim mean:', np.mean(ssim_list))
    print('lpips mean:', np.mean(lpips_list))
    reid_mean = torch.vstack(reid_list).mean(0)

    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("'psnr mean: %f \n" % np.mean(psnr_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("ssim mean: %f \n" % np.mean(ssim_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("lpips mean: %f \n" % np.mean(lpips_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("reid mean : %f , reid_gt mean : %f \n" % (reid_mean[1].item(), reid_mean[0].item()))



def inference_pamir(test_img_dir, pretrained_checkpoint_pamir,
                      pretrained_checkpoint_pamirtex, iternum=100):


    smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = \
        util.read_smpl_constants('./data')

    out_dir = os.path.join(test_img_dir, 'outputs_pamir1')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda")
    loader = TestingImgLoader(test_img_dir, 512, 512, white_bg=True)

    evaluater_tex = EvaluatorTex(device, pretrained_checkpoint_pamir, pretrained_checkpoint_pamirtex)
    evaluator = Evaluator(device, pretrained_checkpoint_pamir, './results/gcmr_pretrained/gcmr_2020_12_10-21_03_12.pt')


    for step, batch in enumerate(tqdm(loader, desc='Testing', total=len(loader), initial=0)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        print(batch['img_dir'])
        img_dir = batch['img_dir'][0]
        model_id = os.path.split(img_dir)[1][:-4]

        vol_res = 256

        if False:
            surface_render_pred, surface_render_alpha = evaluater_tex.test_surface_rendering(batch['img'], batch['betas'],
                                                                                         batch['pose'], batch['scale'],
                                                                                         batch['trans'],
                                                                                         torch.ones(batch['img'].shape[
                                                                                                        0]).cuda() * 249)

            volume_render_pred, volume_render_alpha = evaluater_tex.test_nerf_target(batch['img'], batch['betas'],
                                                                                 batch['pose'], batch['scale'],
                                                                                 batch['trans'],
                                                                                 torch.ones(batch['img'].shape[
                                                                                                0]).cuda() * 249)

            image_fname = os.path.join(out_dir, model_id + '_surface_rendered_image.png')
            save_image(surface_render_pred, image_fname)
            image_fname = os.path.join(out_dir, model_id + '_volume_rendered_image.png')
            save_image(volume_render_pred, image_fname)

        ##gcmr_inference
        pred_betas, pred_rotmat, scale, trans, pred_smpl =evaluator.test_gcmr(batch['img'])
        pred_smpl = scale * pred_smpl + trans

        smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = \
            util.read_smpl_constants('./data')

        init_smpl_fname = os.path.join(out_dir, model_id + '_init_smpl.obj')
        obj_io.save_obj_data({'v': pred_smpl.squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                             init_smpl_fname)

        optm_thetas, optm_betas, optm_smpl = evaluator.optm_smpl_param_wokp(
            batch['img'], pred_betas, pred_rotmat, scale, trans, iter_num=iternum)

        optm_smpl_fname = os.path.join(out_dir, model_id + '_optm_smpl.obj')
        obj_io.save_obj_data({'v': optm_smpl.squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                             optm_smpl_fname)

        ##optimization end

        betas = optm_betas
        pose = optm_thetas


        mesh = evaluator.test_pifu(batch['img'], vol_res, betas, pose, scale, trans)

        # save .obj
        mesh_fname = os.path.join(out_dir, model_id + '_sigma_mesh.obj')
        obj_io.save_obj_data(mesh, mesh_fname)

        mesh_v, mesh_f = mesh['v'].astype(np.float32), mesh['f'].astype(np.int32)
        mesh_v = torch.from_numpy(mesh_v).cuda().unsqueeze(0)
        mesh_f = torch.from_numpy(mesh_f).cuda().unsqueeze(0)

        mesh_color = evaluater_tex.test_tex_pifu(batch['img'], mesh_v, betas, pose, scale, trans)

        mesh_fname = mesh_fname.replace('.obj', '_tex.obj')

        obj_io.save_obj_data({'v': mesh_v[0].squeeze().detach().cpu().numpy(),
                              'f': mesh_f[0].squeeze().detach().cpu().numpy(),
                              'vc': mesh_color.squeeze()},
                             mesh_fname)



    print('Testing Done. ')

def inference(test_img_dir, pretrained_checkpoint_pamir,
                      pretrained_checkpoint_pamirtex, iternum=100):


    smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = \
        util.read_smpl_constants('./data')

    out_dir = os.path.join(test_img_dir, 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda")
    loader = TestingImgLoader(test_img_dir, 512, 512, white_bg=True)

    evaluater = EvaluatorTex_single(device, pretrained_checkpoint_pamir, pretrained_checkpoint_pamirtex)
    evaluator_pretrained = Evaluator(device, pretrained_checkpoint_pamir,
                                     './results/gcmr_pretrained/gcmr_2020_12_10-21_03_12.pt')


    for step, batch in enumerate(tqdm(loader, desc='Testing', total=len(loader), initial=0)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        print(batch['img_dir'])
        img_dir = batch['img_dir'][0]
        model_id = os.path.split(img_dir)[1][:-4]

        vol_res = 256

        if False:
            surface_render_pred, surface_render_alpha = evaluater.test_surface_rendering(batch['img'], batch['betas'],
                                                                                         batch['pose'], batch['scale'],
                                                                                         batch['trans'],
                                                                                         torch.ones(batch['img'].shape[
                                                                                                        0]).cuda() * 249)

            volume_render_pred, volume_render_alpha = evaluater.test_nerf_target(batch['img'], batch['betas'],
                                                                                 batch['pose'], batch['scale'],
                                                                                 batch['trans'],
                                                                                 torch.ones(batch['img'].shape[
                                                                                                0]).cuda() * 249)

            image_fname = os.path.join(out_dir, model_id + '_surface_rendered_image.png')
            save_image(surface_render_pred, image_fname)
            image_fname = os.path.join(out_dir, model_id + '_volume_rendered_image.png')
            save_image(volume_render_pred, image_fname)

        ##gcmr_inference
        out_dir_smpl = os.path.join(out_dir, 'smpl_optm')
        os.makedirs(out_dir_smpl, exist_ok=True)

        pred_betas, pred_rotmat, scale, trans, pred_smpl = evaluator_pretrained.test_gcmr(batch['img'])
        pred_smpl = scale * pred_smpl + trans

        smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = \
            util.read_smpl_constants('./data')

        init_smpl_fname = os.path.join(out_dir_smpl, model_id + '_init_smpl.obj')
        obj_io.save_obj_data({'v': pred_smpl.squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                             init_smpl_fname)

        optm_thetas, optm_betas, optm_smpl, nerf_image_before, nerf_image = evaluater.optm_smpl_param(
            batch['img'], batch['mask'], pred_betas, pred_rotmat, scale, trans, iter_num=iternum)

        optm_smpl_fname = os.path.join(out_dir_smpl, model_id + '_optm_smpl.obj')
        obj_io.save_obj_data({'v': optm_smpl.squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                             optm_smpl_fname)

        ##optimization end
        # save_image
        image_fname = os.path.join(out_dir_smpl, model_id + '_nerf_image_before.png')
        save_image(nerf_image_before, image_fname)
        image_fname = os.path.join(out_dir_smpl, model_id + '_nerf_image_after.png')
        save_image(nerf_image, image_fname)
        betas = optm_betas
        pose = optm_thetas
        torch.save(betas.cpu(),  os.path.join(out_dir_smpl, model_id+'_betas.pth'))
        torch.save(pose.cpu(), os.path.join(out_dir_smpl, model_id + '_pose.pth'))
        torch.save(scale.cpu(), os.path.join(out_dir_smpl, model_id + '_scale.pth'))
        torch.save(trans.cpu(), os.path.join(out_dir_smpl, model_id + '_trans.pth'))
        # smpl_param_name = os.path.join(out_dir, 'results', img_fname[:-4] + '_smplparams.pkl')
        # with open(smpl_param_name, 'wb') as fp:
        #     pkl.dump({'betas': optm_betas.squeeze().detach().cpu().numpy(),
        #               'body_pose': optm_thetas.squeeze().detach().cpu().numpy(),
        #               'init_betas': pred_betas.squeeze().detach().cpu().numpy(),
        #               'init_body_pose': pred_rotmat.squeeze().detach().cpu().numpy(),
        #               'body_scale': scale.squeeze().detach().cpu().numpy(),
        #               'global_body_translation': trans.squeeze().detach().cpu().numpy()},
        #              fp)

        #for stage2
        if True:
            out_dir_stage1 = os.path.join(test_img_dir, 'output_stage1')
            nerf_color, weight_sum = evaluater.test_nerf_target(batch['img'], betas,
                                                                                   pose, scale,
                                                                                   trans,
                                                                                   torch.Tensor([-180]).cuda(),
                                                                                   return_flow_feature=True)


            # vol = nerf_color_warped[:, :128].numpy()[0]
            flow = nerf_color[:, :2]
            nerf_pts_tex = nerf_color[:, 2:5]
            # nerf_attention = nerf_color_warped[:, -1:]
            warped_image = F.grid_sample(batch['img'].cpu(), flow.permute(0, 2, 3, 1))

            flow_path = os.path.join(out_dir_stage1, model_id, 'flow')
            feature_path = os.path.join(out_dir_stage1, model_id, 'feature')
            warped_image_path = os.path.join(out_dir_stage1, model_id, 'warped_image')
            pred_image_path = os.path.join(out_dir_stage1, model_id, 'pred_image')
            attention_path = os.path.join(out_dir_stage1, model_id, 'attention')
            weightsum_path = os.path.join(out_dir_stage1, model_id, 'weight_sum')

            os.makedirs(flow_path, exist_ok=True)
            # os.makedirs(feature_path +'/32', exist_ok=True)
            # os.makedirs(feature_path + '/64', exist_ok=True)
            # os.makedirs(feature_path + '/128', exist_ok=True)
            os.makedirs(pred_image_path, exist_ok=True)
            os.makedirs(warped_image_path, exist_ok=True)
            # os.makedirs(attention_path, exist_ok=True)
            os.makedirs(weightsum_path, exist_ok=True)
            file_name = str(0).zfill(4) + '_' + str(180).zfill(4)
            save_image(torch.cat([(flow / 2 + 0.5), torch.zeros((flow.size(0), 1, flow.size(2), flow.size(3)))], dim=1),
                       os.path.join(flow_path, file_name + '.png'))
            save_image(warped_image, os.path.join(warped_image_path, file_name + '.png'))
            # save_image(nerf_attention, os.path.join(attention_path, file_name + '.png'))
            save_image(nerf_pts_tex, os.path.join(pred_image_path, file_name + '.png'))
            save_image(weight_sum, os.path.join(weightsum_path, file_name + '.png'))
            # if const.down_scale == 2:
            #     np.save(os.path.join(feature_path, '128', file_name + '.npy'), vol[:, ::2, ::2])
            # elif const.down_scale == 1:
            #     np.save(os.path.join(feature_path, '128', file_name + '.npy'), vol[:, ::4, ::4])
            # else:
            #     raise NotImplementedError()


        if True:
            mesh = evaluater.test_pifu(batch['img'], vol_res, betas, pose, scale, trans)
        else:
            vol_res = 128
            nerf_sigma = evaluater.test_nerf_target_sigma(batch['img'], betas, pose, scale, trans, vol_res=vol_res)
            thresh = 0.5
            vertices, simplices, normals, _ = measure.marching_cubes_lewiner(np.array(nerf_sigma), thresh)
            mesh = dict()
            mesh['v'] = vertices / vol_res - 0.5
            mesh['f'] = simplices[:, (1, 0, 2)]
            mesh['vn'] = normals

            # save .mrc
            with mrcfile.new_mmap(os.path.join(out_dir, model_id + '_sigma_mesh.mrc'), overwrite=True,
                                  shape=nerf_sigma.shape, mrc_mode=2) as mrc:
                mrc.data[:] = nerf_sigma

        # save .obj
        mesh_fname = os.path.join(out_dir, model_id + '_sigma_mesh.obj')
        obj_io.save_obj_data(mesh, mesh_fname)

        mesh_v, mesh_f = mesh['v'].astype(np.float32), mesh['f'].astype(np.int32)
        mesh_v = torch.from_numpy(mesh_v).cuda().unsqueeze(0)
        mesh_f = torch.from_numpy(mesh_f).cuda().unsqueeze(0)

        mesh_color = evaluater.test_tex_pifu(batch['img'], mesh_v, betas, pose, scale, trans)

        mesh_fname = mesh_fname.replace('.obj', '_tex.obj')

        obj_io.save_obj_data({'v': mesh_v[0].squeeze().detach().cpu().numpy(),
                              'f': mesh_f[0].squeeze().detach().cpu().numpy(),
                              'vc': mesh_color.squeeze()},
                             mesh_fname)



    print('Testing Done. ')



def validation_pamir(pretrained_checkpoint_pamir,
                      pretrained_checkpoint_pamirtex, iternum=100):


    device = torch.device("cuda")
    evaluater_tex = EvaluatorTex(device, pretrained_checkpoint_pamir, pretrained_checkpoint_pamirtex)
    evaluator = Evaluator(device, pretrained_checkpoint_pamir, './results/gcmr_pretrained/gcmr_2020_12_10-21_03_12.pt')
    val_ds = TrainingImgDataset(
        '/home/nas1_temp/dataset/Thuman', img_h=const.img_res, img_w=const.img_res,
        training=False, testing_res=256,
        view_num_per_item=360,
        point_num=5000,
        load_pts2smpl_idx_wgt=True,
        smpl_data_folder='./data')

    val_data_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8,
                                 worker_init_fn=None, drop_last=False)

    p2s_list=[]
    chamfer_list=[]
    psnr_list = []
    ssim_list = []
    lpips_list = []

    for step_val, batch in enumerate(tqdm(val_data_loader, desc='Testing', total=len(val_data_loader), initial=0)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        out_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/temp_validation_256gcmroptkp_gttrans__pamir_geometry_gtsmpl_epoch30_trainset_hg2_2022_02_25_11_28_01/'
        #out_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/validation_256gtsmpl__pamir_geometry_gtsmpl_epoch30_trainset_hg2_2022_02_25_11_28_01/'


        os.makedirs(out_dir, exist_ok=True)
        #model_id = str(501 + batch['model_id'].item()).zfill(4)
        model_id = (str(501 + batch['model_id'].item()) + '_' + str(batch['view_id'].item())).zfill(4)

        print(model_id)

        vol_res = 256

        if False:
            targetview = 180
            out_dir = f'/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/validation_render{targetview}__pamir_geometry_gtsmpl_epoch30_trainset_hg2_2022_02_25_11_28_01/'
            os.makedirs(out_dir, exist_ok=True)
            surface_render_pred, surface_render_alpha = evaluater_tex.test_surface_rendering(batch['img'],
                                                                                             batch['betas'],
                                                                                             batch['pose'],
                                                                                             batch['scale'],
                                                                                             batch['trans'],
                                                                                             torch.ones(
                                                                                                 batch['img'].shape[
                                                                                                     0]).cuda() * targetview)

            volume_render_pred, volume_render_alpha =evaluater_tex.test_nerf_target(batch['img'], batch['betas'],
                                                                                 batch['pose'], batch['scale'],
                                                                                 batch['trans'],
                                                                                 torch.ones(batch['img'].shape[
                                                                                                0]).cuda() * targetview)

            psnr = metrics.PSNR()(surface_render_alpha.cuda(), batch['target_img'])
            ssim = metrics.SSIM()(surface_render_alpha.cuda(), batch['target_img'])
            lpips = metrics.LPIPS(True)(surface_render_alpha.cuda(), batch['target_img'])
            psnr_list.append(psnr.item())
            ssim_list.append(ssim.item())
            lpips_list.append(lpips.item())

            with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
                f.write("model id: %s \n" % model_id)
            with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
                f.write("psnr: %f \n" % psnr)
            with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
                f.write("ssim : %f \n" % ssim)
            with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
                f.write("lpips : %f \n" % lpips)

            image_fname = os.path.join(out_dir, model_id + '_surface_rendered_image.png')
            save_image(surface_render_alpha, image_fname)
            image_fname = os.path.join(out_dir, model_id + '_volume_rendered_image.png')
            save_image(volume_render_alpha, image_fname)
            continue



        use_gcmr= False
        if use_gcmr :
            out_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/smpl_pamiroptm_4v_pamir'
            os.makedirs(out_dir, exist_ok=True)
            # pred_betas, pred_rotmat, scale, trans, pred_smpl = evaluator_pretrained.test_gcmr(batch['img'])
            # pred_smpl = scale * pred_smpl + trans

            ##
            pred_betas, pred_rotmat, _, _, pred_vert_tetsmpl, pred_cam = evaluator.test_gcmr(batch['img'], return_predcam=True)

            gt_trans= batch['trans']
            cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c

            with torch.no_grad():
                pred_smpl_joints = evaluator.tet_smpl.get_smpl_joints(pred_vert_tetsmpl).detach()
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

            scale= scale_
            trans= trans_
            pred_smpl = scale * pred_vert_tetsmpl[:, :6890] + trans
            ##




            smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = \
                util.read_smpl_constants('./data')
            init_smpl_fname = os.path.join(out_dir, model_id+ '_init_smpl.obj')
            obj_io.save_obj_data({'v': pred_smpl.squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                                 init_smpl_fname)

            #optm_thetas, optm_betas, optm_smpl = evaluator.optm_smpl_param_wokp(
            #    batch['img'], pred_betas, pred_rotmat, scale, trans, iter_num=iternum) ##not yet
            #optm_thetas, optm_betas, optm_smpl = evaluator.optm_smpl_param_mask(
            #    batch['img'], batch['mask'], pred_betas, pred_rotmat, scale, trans, iter_num=iternum)  ##not yet
            optm_thetas, optm_betas, optm_smpl = evaluator.optm_smpl_param(
                batch['img'],  batch['keypoints'],pred_betas, pred_rotmat, scale, trans, iter_num=iternum)  ##not yet



            optm_smpl_fname = os.path.join(out_dir, model_id+'_optm_smpl.obj')
            obj_io.save_obj_data({'v': optm_smpl.squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                                optm_smpl_fname)


            betas =optm_betas
            pose = optm_thetas
            torch.save(betas.cpu(), os.path.join(out_dir, model_id + '_betas.pth'))
            torch.save(pose.cpu(), os.path.join(out_dir, model_id + '_pose.pth'))
            torch.save(scale.cpu(), os.path.join(out_dir, model_id + '_scale.pth'))
            torch.save(trans.cpu(), os.path.join(out_dir, model_id + '_trans.pth'))
            continue

        else:
            betas= batch['betas']
            pose = batch['pose']
            scale = batch['scale']
            trans = batch['trans']
            betas = torch.load(os.path.join(
                '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/smpl_pamiroptm_4v_pamir',
                f'{model_id}_betas.pth')).cuda()
            pose = torch.load(os.path.join(
                '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/smpl_pamiroptm_4v_pamir',
                f'{model_id}_pose.pth')).cuda()
            scale = torch.load(os.path.join(
                '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/smpl_pamiroptm_4v_pamir',
                f'{model_id}_scale.pth')).cuda()
            trans = torch.load(os.path.join(
                '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/smpl_pamiroptm_4v_pamir',
                f'{model_id}_trans.pth')).cuda()

        mesh = evaluator.test_pifu(batch['img'], vol_res, betas, pose, scale, trans)
        # save .obj
        mesh_fname = os.path.join(out_dir, model_id + '_sigma_mesh.obj')
        obj_io.save_obj_data(mesh, mesh_fname)

        mesh_v, mesh_f = mesh['v'].astype(np.float32), mesh['f'].astype(np.int32)
        mesh_v = torch.from_numpy(mesh_v).cuda().unsqueeze(0)
        mesh_f = torch.from_numpy(mesh_f).cuda().unsqueeze(0)

        mesh_color = evaluater_tex.test_tex_pifu(batch['img'], mesh_v, betas, pose, scale, trans)
        mesh_fname = mesh_fname.replace('.obj', '_tex.obj')

        obj_io.save_obj_data({'v': mesh_v[0].squeeze().detach().cpu().numpy(),
                              'f': mesh_f[0].squeeze().detach().cpu().numpy(),
                              'vc': mesh_color.squeeze()},
                             mesh_fname)

        ## rotate to gt view
        # import pdb; pdb.set_trace()
        vertices1 = evaluater_tex.rotate_points(torch.from_numpy(mesh['v']).cuda().unsqueeze(0), -batch['view_id'])
        mesh_fname = os.path.join(out_dir, model_id + '_sigma_mesh_gtview.obj')
        obj_io.save_obj_data({'v': vertices1[0].squeeze().detach().cpu().numpy(),
                              'f': mesh['f']},
                             mesh_fname)





        # save_image
        image_fname = os.path.join(out_dir, model_id + '_src_image.png')
        save_image(batch['img'],  image_fname )

        #measure dist
        mesh_fname = os.path.join(out_dir, model_id + '_sigma_mesh_gtview.obj')
        model_number = str(501 + batch['model_id'].item()).zfill(4)
        tgt_meshname = f'/home/nas1_temp/dataset/Thuman/mesh_data/{model_number}/{model_number}.obj'
        tgt_mesh = trimesh.load(tgt_meshname)
        src_mesh = trimesh.load(mesh_fname)
        tgt_mesh  = trimesh.Trimesh.simplify_quadratic_decimation(tgt_mesh, 100000)

        p2s_dist = get_surface_dist(tgt_mesh, src_mesh)
        chamfer_dist= get_chamfer_dist(tgt_mesh, src_mesh)
        p2s_list.append(p2s_dist)
        chamfer_list.append(chamfer_dist)

        print('p2s:', p2s_dist)
        print('chamfer', chamfer_dist)

        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("model id: %s \n" % model_id)
        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("p2s: %f \n" % p2s_dist)
        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("chamfer: %f \n" % chamfer_dist)



    print('Testing Done. ')
    print('p2s mean:',np.mean(p2s_list))
    print('chamfer mean:', np.mean(chamfer_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("p2s mean: %f \n" % np.mean(p2s_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("chamfer mean: %f \n" % np.mean(chamfer_list))

    print('psnr mean:', np.mean(psnr_list))
    print('ssim mean:', np.mean(ssim_list))
    print('lpips mean:', np.mean(lpips_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("'psnr mean: %f \n" % np.mean(psnr_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("ssim mean: %f \n" %  np.mean(ssim_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("lpips mean: %f \n" % np.mean(lpips_list))

def validation(pretrained_checkpoint_pamir,
                      pretrained_checkpoint_pamirtex, iternum=50):


    device = torch.device("cuda")
    evaluater = EvaluatorTex_single(device, pretrained_checkpoint_pamir, pretrained_checkpoint_pamirtex)
    evaluater_pretrained = Evaluator(device, pretrained_checkpoint_pamir, './results/gcmr_pretrained/gcmr_2020_12_10-21_03_12.pt')

    measure_deephuman=False

    if measure_deephuman:
        val_ds = TrainingImgDataset_deephuman(
            '/home/nas1_temp/dataset/Deephuman_norot', img_h=const.img_res, img_w=const.img_res,
            training=False, testing_res=256,
            view_num_per_item=360,
            point_num=5000,
            load_pts2smpl_idx_wgt=True,
            smpl_data_folder='./data')
    else:
        val_ds = TrainingImgDataset(
            '/home/nas1_temp/dataset/Thuman', img_h=const.img_res, img_w=const.img_res,
            training=False, testing_res=256,
            view_num_per_item=360,
            point_num=5000,
            load_pts2smpl_idx_wgt=True,
            smpl_data_folder='./data')



    val_data_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8,
                                 worker_init_fn=None, drop_last=False)



    p2s_list=[]
    chamfer_list=[]
    psnr_list = []
    ssim_list = []
    lpips_list = []

    #out_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/validation_256gcmroptmask50_gttrans__' + '_'.join([pretrained_checkpoint_pamirtex.split('/')[-3], pretrained_checkpoint_pamirtex.split('/')[-1][:-3]])
    out_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/temp_validation_256gtsmpl__' + '_'.join(
        [pretrained_checkpoint_pamirtex.split('/')[-3], pretrained_checkpoint_pamirtex.split('/')[-1][:-3]])
    os.makedirs(out_dir, exist_ok=True)

    for step_val, batch in enumerate(tqdm(val_data_loader, desc='Testing', total=len(val_data_loader), initial=0)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}



        # model_id = str(501 + batch['model_id'].item()).zfill(4)
        if measure_deephuman:
            model_id = (str(batch['model_id'].item()).zfill(4) + '_' + str(batch['view_id'].item()).zfill(4))
        else:
            model_id = (str(501 + batch['model_id'].item()).zfill(4) + '_' + str(batch['view_id'].item()).zfill(4))

        print(model_id)

        vol_res = 256

        if False:

            targetview=180
            out_dir = f'/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/validation_render{targetview}__pamir_nerf_0222_48_03_rayontarget_rayonpts_occ_attloss_inout_24hie_2022_02_25_01_56_52/'
            os.makedirs(out_dir, exist_ok=True)
            surface_render_pred, surface_render_alpha= evaluater.test_surface_rendering(batch['img'], batch['betas'], batch['pose'], batch['scale'], batch['trans'],
                                                              torch.ones(batch['img'].shape[0]).cuda() * targetview)

            volume_render_pred, volume_render_alpha = evaluater.test_nerf_target(batch['img'], batch['betas'],
                                                                                         batch['pose'], batch['scale'],
                                                                                         batch['trans'],
                                                                                         torch.ones(batch['img'].shape[
                                                                                                        0]).cuda() *targetview)

            psnr=  metrics.PSNR()(surface_render_alpha.cuda(), batch['target_img'])
            ssim = metrics.SSIM()(surface_render_alpha.cuda(), batch['target_img'])
            lpips = metrics.LPIPS(True)(surface_render_alpha.cuda(), batch['target_img'])
            psnr_list.append(psnr.item())
            ssim_list.append(ssim.item())
            lpips_list.append(lpips.item())


            with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
                f.write("model id: %s \n" % model_id)
            with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
                f.write("psnr: %f \n" % psnr)
            with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
                f.write("ssim : %f \n" % ssim )
            with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
                f.write("lpips : %f \n" % lpips )

            image_fname = os.path.join(out_dir, model_id + '_surface_rendered_image.png')
            save_image(surface_render_alpha, image_fname)
            image_fname = os.path.join(out_dir, model_id + '_volume_rendered_image.png')
            save_image(volume_render_alpha, image_fname)
            continue


        use_gcmr=False
        if use_gcmr :
            out_dir_smpl = os.path.join(out_dir, 'smpl_optm')
            os.makedirs(out_dir_smpl, exist_ok=True)
            #pred_betas, pred_rotmat, scale, trans, pred_smpl = evaluator_pretrained.test_gcmr(batch['img'])
            #pred_smpl = scale * pred_smpl + trans

            ##
            pred_betas, pred_rotmat,scale,trans, pred_vert_tetsmpl, pred_cam =evaluater_pretrained.test_gcmr(batch['img'],
                                                                                             return_predcam=True)


            gt_trans = batch['trans']
            cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c

            with torch.no_grad():
                pred_smpl_joints = evaluater.tet_smpl.get_smpl_joints(pred_vert_tetsmpl).detach()
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

            scale = scale_
            trans = trans_
            pred_smpl = scale * pred_vert_tetsmpl[:, :6890] + trans
            ##

            ##optimization with nerf
            smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = \
                util.read_smpl_constants('./data')

            init_smpl_fname = os.path.join(out_dir_smpl, model_id+ '_init_smpl.obj')
            obj_io.save_obj_data({'v': pred_smpl.squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                                 init_smpl_fname)



            #optm_thetas, optm_betas, optm_smpl, nerf_image_before, nerf_image = evaluater.optm_smpl_param_pamirwokp(
            #    batch['img'], batch['mask'], pred_betas, pred_rotmat, scale, trans, iter_num=iternum)

            optm_thetas, optm_betas, optm_smpl , nerf_image_before, nerf_image = evaluater.optm_smpl_param(
                batch['img'], batch['mask'], pred_betas, pred_rotmat, scale, trans, iter_num=iternum)  ##not yet

            #optm_thetas, optm_betas, optm_smpl, nerf_image_before, nerf_image = evaluater.optm_smpl_param_pamir(
            #    batch['img'], batch['keypoints'], pred_betas, pred_rotmat, scale, trans, iter_num=iternum)  ##not yet
            #optm_thetas, optm_betas, optm_smpl, nerf_image_before, nerf_image = evaluater.optm_smpl_param_kp_mask(
            #    batch['img'], batch['mask'],batch['keypoints'], pred_betas, pred_rotmat, scale, trans, iter_num=iternum)  ##not yet

            optm_smpl_fname = os.path.join(out_dir_smpl, model_id+'_optm_smpl.obj')
            obj_io.save_obj_data({'v': optm_smpl.squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                                optm_smpl_fname)

            ##optimization end
            #save_image
            image_fname = os.path.join(out_dir_smpl, model_id + '_nerf_image_before.png')
            save_image(nerf_image_before, image_fname)
            image_fname = os.path.join(out_dir_smpl, model_id + '_nerf_image_after.png')
            save_image(nerf_image, image_fname)

            betas =optm_betas
            pose = optm_thetas

            torch.save(betas.cpu(),  os.path.join(out_dir_smpl, model_id+'_betas.pth'))
            torch.save(pose.cpu(), os.path.join(out_dir_smpl, model_id + '_pose.pth'))
            torch.save(scale.cpu(), os.path.join(out_dir_smpl, model_id + '_scale.pth'))
            torch.save(trans.cpu(), os.path.join(out_dir_smpl, model_id + '_trans.pth'))
            #continue

        else:
            betas= batch['betas']
            pose = batch['pose']
            scale = batch['scale']
            trans = batch['trans']

            #import pdb; pdb.set_trace()
            # betas = torch.load(os.path.join(
            #     '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/smpl_kpmaskoptm_4v',
            #     f'{model_id}_betas.pth')).cuda()
            # pose = torch.load(os.path.join(
            #     '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/smpl_kpmaskoptm_4v',
            #     f'{model_id}_pose.pth')).cuda()
            # scale = torch.load(os.path.join(
            #     '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/smpl_kpmaskoptm_4v',
            #     f'{model_id}_scale.pth')).cuda()
            # trans = torch.load(os.path.join(
            #     '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/smpl_kpmaskoptm_4v',
            #     f'{model_id}_trans.pth')).cuda()

        #for stage2
        if True:
            out_dir_stage1 = os.path.join(out_dir, 'output_stage1')

            nerf_color, weight_sum = evaluater.test_nerf_target(batch['img'], betas,
                                                                                   pose, scale,
                                                                                   trans,
                                                                                   batch["view_id"] - batch[
                                                                                       'target_view_id'],
                                                                                   return_flow_feature=True)

            # vol = nerf_color_warped[:, :128].numpy()[0]
            flow = nerf_color[:, :2]
            nerf_pts_tex = nerf_color[:, 2:5]
            # nerf_attention = nerf_color_warped[:, -1:]
            warped_image = F.grid_sample(batch['img'].cpu(), flow.permute(0, 2, 3, 1))




            # flow_path = os.path.join(out_dir_stage1, str(batch['model_id'].item()).zfill(4), 'flow')
            # feature_path = os.path.join(out_dir_stage1, str(batch['model_id'].item()).zfill(4), 'feature')
            # warped_image_path = os.path.join(out_dir_stage1, str(batch['model_id'].item()).zfill(4), 'warped_image')
            # pred_image_path = os.path.join(out_dir_stage1, str(batch['model_id'].item()).zfill(4), 'pred_image')
            # attention_path = os.path.join(out_dir_stage1, str(batch['model_id'].item()).zfill(4), 'attention')
            # weightsum_path = os.path.join(out_dir_stage1, str(batch['model_id'].item()).zfill(4), 'weight_sum')
            flow_path = os.path.join(out_dir_stage1, str(batch['model_id'].item() + 501).zfill(4), 'flow')
            feature_path = os.path.join(out_dir_stage1, str(batch['model_id'].item() + 501).zfill(4), 'feature')
            warped_image_path = os.path.join(out_dir_stage1, str(batch['model_id'].item() + 501).zfill(4), 'warped_image')
            pred_image_path = os.path.join(out_dir_stage1, str(batch['model_id'].item() + 501).zfill(4), 'pred_image')
            attention_path = os.path.join(out_dir_stage1, str(batch['model_id'].item() + 501).zfill(4), 'attention')
            weightsum_path = os.path.join(out_dir_stage1, str(batch['model_id'].item() + 501).zfill(4), 'weight_sum')
            os.makedirs(flow_path, exist_ok=True)
            # os.makedirs(feature_path +'/32', exist_ok=True)
            # os.makedirs(feature_path + '/64', exist_ok=True)
            # os.makedirs(feature_path + '/128', exist_ok=True)
            os.makedirs(pred_image_path, exist_ok=True)
            os.makedirs(warped_image_path, exist_ok=True)
            # os.makedirs(attention_path, exist_ok=True)
            os.makedirs(weightsum_path, exist_ok=True)
            file_name = str(batch["view_id"].item()).zfill(4) + '_' + str(batch["target_view_id"].item()).zfill(4)
            save_image(torch.cat([(flow / 2 + 0.5), torch.zeros((flow.size(0), 1, flow.size(2), flow.size(3)))], dim=1),
                       os.path.join(flow_path, file_name + '.png'))
            save_image(warped_image, os.path.join(warped_image_path, file_name + '.png'))
            # save_image(nerf_attention, os.path.join(attention_path, file_name + '.png'))
            save_image(nerf_pts_tex, os.path.join(pred_image_path, file_name + '.png'))
            save_image(weight_sum, os.path.join(weightsum_path, file_name + '.png'))
            # if const.down_scale == 2:
            #     np.save(os.path.join(feature_path, '128', file_name + '.npy'), vol[:, ::2, ::2])
            # elif const.down_scale == 1:
            #     np.save(os.path.join(feature_path, '128', file_name + '.npy'), vol[:, ::4, ::4])
            # else:
            #     raise NotImplementedError()
            # np.save(os.path.join(feature_path, '64', file_name + '.npy'), vol[:, ::4, ::4])
            # np.save(os.path.join(feature_path, '32', file_name + '.npy'), vol[:, ::8, ::8])





        if True:
            mesh = evaluater.test_pifu(batch['img'], vol_res, betas, pose, scale, trans)
        else:
            vol_res=128
            nerf_sigma = evaluater.test_nerf_target_sigma(batch['img'],betas, pose, scale, trans, vol_res=vol_res)
            thresh = 0.5
            vertices, simplices, normals, _ = measure.marching_cubes_lewiner(np.array(nerf_sigma), thresh)
            mesh = dict()
            mesh['v'] = vertices / vol_res - 0.5
            mesh['f'] = simplices[:, (1, 0, 2)]
            mesh['vn'] = normals

            # save .mrc
            with mrcfile.new_mmap(os.path.join(out_dir, model_id + '_sigma_mesh.mrc'), overwrite=True,
                                  shape=nerf_sigma.shape, mrc_mode=2) as mrc:
                mrc.data[:] = nerf_sigma

        # save .obj
        mesh_fname = os.path.join(out_dir, model_id + '_sigma_mesh.obj')
        obj_io.save_obj_data(mesh, mesh_fname)

        mesh_v, mesh_f = mesh['v'].astype(np.float32), mesh['f'].astype(np.int32)
        mesh_v = torch.from_numpy(mesh_v).cuda().unsqueeze(0)
        mesh_f = torch.from_numpy(mesh_f).cuda().unsqueeze(0)

        mesh_color = evaluater.test_tex_pifu(batch['img'], mesh_v, betas, pose, scale, trans)

        mesh_fname = mesh_fname.replace('.obj', '_tex.obj')

        obj_io.save_obj_data({'v': mesh_v[0].squeeze().detach().cpu().numpy(),
                              'f': mesh_f[0].squeeze().detach().cpu().numpy(),
                              'vc': mesh_color.squeeze()},
                             mesh_fname)

        ## rotate to gt view
        # import pdb; pdb.set_trace()


        vertices1 = evaluater.rotate_points(torch.from_numpy(mesh['v']).cuda().unsqueeze(0), -batch['view_id'])
        mesh_fname = os.path.join(out_dir, model_id + '_sigma_mesh_gtview.obj')
        obj_io.save_obj_data({'v': vertices1[0].squeeze().detach().cpu().numpy(),
                              'f': mesh['f']},
                             mesh_fname)


        # save_image
        image_fname = os.path.join(out_dir, model_id + '_src_image.png')
        save_image(batch['img'],  image_fname )

        #measure dist
        mesh_fname = os.path.join(out_dir, model_id + '_sigma_mesh_gtview.obj')

        if measure_deephuman:
            model_number = str( batch['model_id'].item()).zfill(4)
            tgt_meshname =f'/home/nas1_temp/dataset/Deephuman_norot/image_data/{model_number}/mesh_after.obj'
        else:
            model_number = str(501 + batch['model_id'].item()).zfill(4)
            tgt_meshname = f'/home/nas1_temp/dataset/Thuman/mesh_data/{model_number}/{model_number}.obj'
        tgt_mesh = trimesh.load(tgt_meshname)
        src_mesh = trimesh.load(mesh_fname)
        #import pdb; pdb.set_trace()
        tgt_mesh  = trimesh.Trimesh.simplify_quadratic_decimation(tgt_mesh, 100000)

        p2s_dist = get_surface_dist(tgt_mesh, src_mesh)
        chamfer_dist= get_chamfer_dist(tgt_mesh, src_mesh)
        p2s_list.append(p2s_dist)
        chamfer_list.append(chamfer_dist)

        print('p2s:', p2s_dist)
        print('chamfer', chamfer_dist)

        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("model id: %s \n" % model_id)
        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("p2s: %f \n" % p2s_dist)
        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("chamfer: %f \n" % chamfer_dist)


    print('Testing Done. ')
    print('p2s mean:',np.mean(p2s_list))
    print('chamfer mean:', np.mean(chamfer_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("p2s mean: %f \n" % np.mean(p2s_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("chamfer mean: %f \n" % np.mean(chamfer_list))

    print('psnr mean:', np.mean(psnr_list))
    print('ssim mean:', np.mean(ssim_list))
    print('lpips mean:', np.mean(lpips_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("'psnr mean: %f \n" % np.mean(psnr_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("ssim mean: %f \n" %  np.mean(ssim_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("lpips mean: %f \n" % np.mean(lpips_list))


def validation_texture(pretrained_checkpoint_pamir,
                      pretrained_checkpoint_pamirtex):


    device = torch.device("cuda")
    evaluater = EvaluatorTex_single(device, pretrained_checkpoint_pamir, pretrained_checkpoint_pamirtex)


    val_ds = TrainingImgDataset(
        '/home/nas1_temp/dataset/Thuman', img_h=const.img_res, img_w=const.img_res,
        training=False, testing_res=256,
        view_num_per_item=360,
        point_num=5000,
        load_pts2smpl_idx_wgt=True,
        smpl_data_folder='./data')



    val_data_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8,
                                 worker_init_fn=None, drop_last=False)



    out_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/validation_vgglpips_texturedgtmesh__' + '_'.join([pretrained_checkpoint_pamirtex.split('/')[-3], pretrained_checkpoint_pamirtex.split('/')[-1][:-3]])
    os.makedirs(out_dir, exist_ok=True)
    psnr_list = []
    ssim_list = []
    lpips_list = []
    reid_list = []


    for step_val, batch in enumerate(tqdm(val_data_loader, desc='Testing', total=len(val_data_loader), initial=0)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        model_id = (str(501 + batch['model_id'].item()).zfill(4) + '_' + str(batch['view_id'].item()).zfill(4))
        print(model_id)
        model_number = str(501 + batch['model_id'].item()).zfill(4)
        tgt_meshname = f'/home/nas1_temp/dataset/Thuman/mesh_data_vc/{model_number}/{model_number}.obj'


        tgt_mesh = trimesh.load(tgt_meshname)
        #tgt_mesh = load_obj_data(tgt_meshname)

        mesh_v = tgt_mesh.vertices.astype(np.float32)
        #mesh_v = tgt_mesh['v'].astype(np.float32)
        mesh_v = torch.from_numpy(mesh_v).cuda().unsqueeze(0)

        mesh_f = tgt_mesh.faces.astype(np.int32)
        #mesh_f = tgt_mesh['f'].astype(np.int32)
        mesh_f = torch.from_numpy(mesh_f).cuda().unsqueeze(0)

        v_cam = evaluater.rotate_points(mesh_v, batch['view_id'])

        mesh_color = evaluater.test_tex_pifu(batch['img'], v_cam, batch['betas'],batch['pose'], batch['scale'],batch['trans'])



        mesh_fname = os.path.join(out_dir, model_id + '_gt_textured.obj')

        obj_io.save_obj_data({'v': mesh_v[0].squeeze().detach().cpu().numpy(),
                              'f': mesh_f[0].squeeze().detach().cpu().numpy(),
                              'vc': mesh_color.squeeze()},
                             mesh_fname)


        ## render using vertex color
        target_viewid = -(batch['view_id'].item()+180)
        rendered_img = render_mesh(mesh_fname, render_angle=target_viewid )
        rendered_img = torch.from_numpy(rendered_img).permute(2,0,1).unsqueeze(0 )
        # save_image
        image_fname = os.path.join(out_dir, model_id + '_rendered_image.png')
        save_image(rendered_img, image_fname)

        ##
        if True:

            target_viewid =str((batch['view_id'].item()+180)%360).zfill(4)
            tgt_imagename = f'/home/nas1_temp/dataset/Thuman/image_data_vc/{model_number}/color/{target_viewid}.jpg'
            gt_img = cv.imread(tgt_imagename).astype(np.uint8)
            gt_img = np.float32(cv.cvtColor( gt_img , cv.COLOR_RGB2BGR)) / 255.
            gt_img= torch.from_numpy( gt_img .transpose((2, 0, 1))).unsqueeze(0).cuda()
        else:
            gt_img= render_mesh(tgt_meshname, render_angle=target_viewid  )
            gt_img = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0)




        image_fname = os.path.join(out_dir, model_id + '_rendered_gt_image.png')
        save_image(gt_img, image_fname)


        ## measure metrics
        psnr = metrics.PSNR()(rendered_img.cuda(), gt_img.cuda())
        ssim = metrics.SSIM()(rendered_img.cuda(), gt_img.cuda())
        lpips = metrics.LPIPS(True)(rendered_img.cuda(), gt_img.cuda())
        psnr_list.append(psnr.item())
        ssim_list.append(ssim.item())
        lpips_list.append(lpips.item())

        ## measure reID

        rendered_feat = reid_encoder.get_features(rendered_img.cuda())
        gt_target_feat = reid_encoder.get_features(gt_img.cuda())
        gt_source_feat = reid_encoder.get_features(batch['img'].cuda()*batch['mask'].permute(0,3,1,2).cuda())

        #reid_dist = reid_encoder.euclidean_squared_distance(gt_source_feat[0].unsqueeze(0), torch.cat((gt_target_feat[0][None,], rendered_feat[0][None,]),dim=0))
        reid_dist = reid_encoder.cosine_distance(gt_source_feat[0].unsqueeze(0), torch.cat(
            (gt_target_feat[0][None,], rendered_feat[0][None,]), dim=0))
        reid_list.append(reid_dist)

        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("model id: %s \n" % model_id)
        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("psnr: %f \n" % psnr)
        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("ssim : %f \n" % ssim)
        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("lpips : %f \n" % lpips)
        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("reid : %f , reid_gt : %f \n" % (reid_dist[:,1], reid_dist[:,0]))


        ## meshlab/blender capture
        if True:
            check = 1


    print('psnr mean:', np.mean(psnr_list))
    print('ssim mean:', np.mean(ssim_list))
    print('lpips mean:', np.mean(lpips_list))
    reid_mean = torch.vstack(reid_list).mean(0)

    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("'psnr mean: %f \n" % np.mean(psnr_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("ssim mean: %f \n" % np.mean(ssim_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("lpips mean: %f \n" % np.mean(lpips_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("reid mean : %f , reid_gt mean : %f \n" % (reid_mean[1].item(), reid_mean[0].item()))


def inference_multi(test_img_dir,pretrained_checkpoint_pamir,
                      pretrained_checkpoint_pamirtex, iternum=100):

    device = torch.device("cuda")
    evaluater = EvaluatorTex_multi(device, pretrained_checkpoint_pamir, pretrained_checkpoint_pamirtex)
    evaluator_pretrained = Evaluator(device, pretrained_checkpoint_pamir, './results/gcmr_pretrained/gcmr_2020_12_10-21_03_12.pt')
    out_dir = os.path.join(test_img_dir, 'stage3_outputs')
    os.makedirs(out_dir, exist_ok=True)

    #import pdb; pdb.set_trace()
    folder_list= listdir(os.path.join(test_img_dir, 'stage2_outputs'))
    for i in folder_list:
        model_id= i
        img_fpath1 = os.path.join(test_img_dir,'stage2_outputs', i, '0000.png')
        img1 = cv.imread(img_fpath1).astype(np.uint8)
        img1 = np.float32(cv.cvtColor(img1, cv.COLOR_RGB2BGR)) / 255.
        img1 = torch.from_numpy(img1.transpose((2, 0, 1))).unsqueeze(0).cuda()
        img_fpath2 = os.path.join(test_img_dir, 'stage2_outputs', i, '0000_0180.png')
        img2 = cv.imread(img_fpath2).astype(np.uint8)
        img2 = np.float32(cv.cvtColor(img2, cv.COLOR_RGB2BGR)) / 255.
        img2 = torch.from_numpy(img2.transpose((2, 0, 1))).unsqueeze(0).cuda()
        img_pair = torch.stack([img1, img2], 1)
        mask1_fpath1 =  os.path.join(test_img_dir,'image', i)+'_mask.png'
        msk = cv.imread(mask1_fpath1, cv.IMREAD_GRAYSCALE).astype(np.uint8)
        msk = np.float32(msk) / 255
        msk = np.reshape(msk, [img1.size(2), img1.size(2), 1])
        #import pdb; pdb.set_trace()
        mask1 = torch.from_numpy(msk).unsqueeze(0).cuda()

        view_id = torch.cat([torch.ones(img1.shape[0], 1).cuda() * 0, torch.ones(img1.shape[0],1 ).cuda() * 180],1)

        vol_res = 256

        pred_betas, pred_rotmat, scale, trans, pred_smpl = evaluator_pretrained.test_gcmr(img_pair[:,0])
        pred_smpl = scale * pred_smpl + trans

        ##optimization with nerf

        smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = \
            util.read_smpl_constants('./data')

        init_smpl_fname = os.path.join(out_dir, model_id + '_init_smpl.obj')
        obj_io.save_obj_data({'v': pred_smpl.squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                             init_smpl_fname)

        optm_thetas, optm_betas, optm_smpl = evaluater.optm_smpl_param(
            img_pair[:, 0],  mask1 , pred_betas, pred_rotmat, scale[:, 0], trans[:, 0],
            iter_num=iternum)

        optm_smpl_fname = os.path.join(out_dir, model_id + '_optm_smpl.obj')
        obj_io.save_obj_data({'v': optm_smpl.squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                             optm_smpl_fname)

        ##optimization end
        betas = optm_betas
        pose = optm_thetas


        if True:
            mesh = evaluater.test_pifu(img_pair, view_id , vol_res, betas, pose, scale, trans)
            #mesh = evaluater.test_pifu(torch.stack([img1, img1], 1),
            #                           torch.cat([batch['view_id'][:, 0:1], batch['view_id'][:, 0:1]], -1), vol_res, betas,
            #                           pose, scale, trans)
        else:
            vol_res=128
            nerf_sigma = evaluater.test_nerf_target_sigma(img_pair,view_id ,betas, pose, scale, trans, vol_res=vol_res)
            thresh = 0.5
            vertices, simplices, normals, _ = measure.marching_cubes_lewiner(np.array(nerf_sigma), thresh)
            mesh = dict()
            mesh['v'] = vertices / vol_res - 0.5
            mesh['f'] = simplices[:, (1, 0, 2)]
            mesh['vn'] = normals

            # save .mrc
            with mrcfile.new_mmap(os.path.join(out_dir, model_id + '_sigma_mesh.mrc'), overwrite=True,
                                  shape=nerf_sigma.shape, mrc_mode=2) as mrc:
                mrc.data[:] = nerf_sigma

        # save .obj
        mesh_fname = os.path.join(out_dir, model_id + '_sigma_mesh.obj')
        obj_io.save_obj_data(mesh, mesh_fname)

        mesh_v, mesh_f = mesh['v'].astype(np.float32), mesh['f'].astype(np.int32)
        mesh_v = torch.from_numpy(mesh_v).cuda().unsqueeze(0)
        mesh_f = torch.from_numpy(mesh_f).cuda().unsqueeze(0)

        mesh_color = evaluater.test_tex_pifu(img_pair,view_id , mesh_v, betas, pose, scale, trans)

        mesh_fname = mesh_fname.replace('.obj', '_tex.obj')

        obj_io.save_obj_data({'v': mesh_v[0].squeeze().detach().cpu().numpy(),
                              'f': mesh_f[0].squeeze().detach().cpu().numpy(),
                              'vc': mesh_color.squeeze()},
                             mesh_fname)







def validation_multi(pretrained_checkpoint_pamir,
                      pretrained_checkpoint_pamirtex, iternum=100):

    device = torch.device("cuda")
    evaluater = EvaluatorTex_multi(device, pretrained_checkpoint_pamir, pretrained_checkpoint_pamirtex)
    evaluater_pretrained = Evaluator(device, pretrained_checkpoint_pamir, './results/gcmr_pretrained/gcmr_2020_12_10-21_03_12.pt')

    val_ds = TrainingImgDataset_multi(
        '/home/nas1_temp/dataset/Thuman', img_h=const.img_res, img_w=const.img_res,
        training=False, testing_res=256,
        view_num_per_item=360,
        point_num=5000,
        load_pts2smpl_idx_wgt=True,
        smpl_data_folder='./data')

    val_data_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8,
                                 worker_init_fn=None, drop_last=False)

    p2s_list=[]
    chamfer_list=[]
    psnr_list = []
    ssim_list = []
    lpips_list = []

    #/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/validation_256gcmroptmask50_gttrans__pamir_nerf_0302_48_03_rayontarget_rayonpts_occ_attloss_inout_24hie_2022_03_06_05_54_57/output_stage2/0303_novolfeat_onlyback_b1_oridata/epoch_200
    out_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/validation_256gcmroptmask50_gttrans__' + '_'.join(
        [pretrained_checkpoint_pamirtex.split('/')[-3], pretrained_checkpoint_pamirtex.split('/')[-1][:-3]])
    os.makedirs(out_dir, exist_ok=True)
    stage1_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/validation_256gcmroptmask50_gttrans__pamir_nerf_0302_48_03_rayontarget_rayonpts_occ_attloss_inout_24hie_2022_03_06_05_54_57'
    stage2_dir = stage1_dir+'/output_stage2/0303_novolfeat_onlyback_b1_oridata/epoch_200'

    for step_val, batch in enumerate(tqdm(val_data_loader, desc='Testing', total=len(val_data_loader), initial=0)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        model_num = str(501 + batch['model_id'].item()).zfill(4)
        model_id = (str(501 + batch['model_id'].item()).zfill(4) + '_' + str(batch['view_id'][:,0].item()).zfill(4))
        print(model_id)

        view_id1 = str(batch['view_id'][:, 0].item()).zfill(4)
        view_id2 = str(batch['view_id'][:, 1].item()).zfill(4)

        img_fpath1 = stage2_dir+  f'/{model_num}/{view_id1}.png'#f'/home/nas1_temp/dataset/Thuman/output_stage2/0303_novolfeat_onlyback_b1_oridata/epoch_33/{model_num}/{view_id1}.png'
        img1 = cv.imread(img_fpath1).astype(np.uint8)
        img1 = np.float32(cv.cvtColor(img1, cv.COLOR_RGB2BGR)) / 255.
        img1 = torch.from_numpy(img1.transpose((2, 0, 1))).unsqueeze(0).cuda()
        img_fpath2 =stage2_dir+  f'/{model_num}/{view_id1}_{view_id2}.png'# f'/home/nas1_temp/dataset/Thuman/output_stage2/0303_novolfeat_onlyback_b1_oridata/epoch_33/{model_num}/{view_id1}_{view_id2}.png'
        img2 = cv.imread(img_fpath2).astype(np.uint8)
        img2 = np.float32(cv.cvtColor(img2, cv.COLOR_RGB2BGR)) / 255.
        img2 = torch.from_numpy(img2.transpose((2, 0, 1))).unsqueeze(0).cuda()
        img_pair = torch.stack([img1, img2], 1)




        vol_res = 256
        if False:

            targetview=180
            out_dir = f'/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/validation_render{targetview}__pamir_nerf_0227_24hie0.5_03_occ_2v_alpha_concat_2022_03_02_06_24_00/'
            os.makedirs(out_dir, exist_ok=True)
            surface_render_pred, surface_render_alpha= evaluater.test_surface_rendering(img_pair, batch['betas'], batch['pose'], batch['scale'], batch['trans'],
                                                              batch['view_id'], batch['target_view_id'] )

            volume_render_pred, volume_render_alpha = evaluater.test_nerf_target(img_pair, batch['betas'],
                                                                                         batch['pose'], batch['scale'],
                                                                                         batch['trans'],
                                                                                         batch['view_id'], batch['target_view_id'])

            psnr=  metrics.PSNR()(surface_render_alpha.cuda(), batch['target_img'])
            ssim = metrics.SSIM()(surface_render_alpha.cuda(), batch['target_img'])
            lpips = metrics.LPIPS(True)(surface_render_alpha.cuda(), batch['target_img'])
            psnr_list.append(psnr.item())
            ssim_list.append(ssim.item())
            lpips_list.append(lpips.item())


            with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
                f.write("model id: %s \n" % model_id)
            with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
                f.write("psnr: %f \n" % psnr)
            with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
                f.write("ssim : %f \n" % ssim )
            with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
                f.write("lpips : %f \n" % lpips )

            image_fname = os.path.join(out_dir, model_id + '_surface_rendered_image.png')
            save_image(surface_render_alpha, image_fname)
            image_fname = os.path.join(out_dir, model_id + '_volume_rendered_image.png')
            save_image(volume_render_alpha, image_fname)
            continue



        use_gcmr= False
        if use_gcmr :
            #pred_betas, pred_rotmat, scale, trans, pred_smpl = evaluator_pretrained.test_gcmr(batch['img'])
            #pred_smpl = scale * pred_smpl + trans

            ##
            pred_betas, pred_rotmat, _, _, pred_vert_tetsmpl, pred_cam =evaluater_pretrained.test_gcmr(batch['img'][:,0],
                                                                                             return_predcam=True)

            gt_trans = batch['trans']
            cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c

            with torch.no_grad():
                pred_smpl_joints = evaluater.tet_smpl.get_smpl_joints(pred_vert_tetsmpl).detach()
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

            scale = scale_
            trans = trans_
            pred_smpl = scale * pred_vert_tetsmpl[:, :6890] + trans
            ##

            ##optimization with nerf

            smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = \
                util.read_smpl_constants('./data')

            init_smpl_fname = os.path.join(out_dir, model_id+ '_init_smpl.obj')
            obj_io.save_obj_data({'v': pred_smpl.squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                                 init_smpl_fname)

            optm_thetas, optm_betas, optm_smpl = evaluater.optm_smpl_param(
                batch['img'][:,0], batch['mask'][:,0], pred_betas, pred_rotmat, scale[:,0], trans[:,0], iter_num=iternum)


            optm_smpl_fname = os.path.join(out_dir, model_id+'_optm_smpl.obj')
            obj_io.save_obj_data({'v': optm_smpl.squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                                optm_smpl_fname)

            ##optimization end
            betas =optm_betas
            pose = optm_thetas

        else:
            #betas= batch['betas']
            #pose = batch['pose']
            #scale = batch['scale']
            #trans = batch['trans']

            betas = torch.load( os.path.join(stage1_dir, 'smpl_optm', f'{model_id}_betas.pth')).cuda()#torch.load(os.path.join('/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/smpl_maskoptimization_4v', f'{model_id}_betas.pth')).cuda()
            pose = torch.load( os.path.join(stage1_dir, 'smpl_optm', f'{model_id}_pose.pth')).cuda()#torch.load(os.path.join('/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/smpl_maskoptimization_4v', f'{model_id}_pose.pth')).cuda()
            scale = torch.load( os.path.join(stage1_dir, 'smpl_optm', f'{model_id}_scale.pth')).cuda()# torch.load(os.path.join('/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/smpl_maskoptimization_4v', f'{model_id}_scale.pth')).cuda()
            trans = torch.load( os.path.join(stage1_dir, 'smpl_optm', f'{model_id}_trans.pth')).cuda()# torch.load(os.path.join('/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/smpl_maskoptimization_4v', f'{model_id}_trans.pth')).cuda()


        if True:
            mesh = evaluater.test_pifu(img_pair, batch['view_id'], vol_res, betas, pose, scale, trans)
            #mesh = evaluater.test_pifu(torch.stack([img1, img1], 1),
            #                           torch.cat([batch['view_id'][:, 0:1], batch['view_id'][:, 0:1]], -1), vol_res, betas,
            #                           pose, scale, trans)
        else:
            vol_res=128
            nerf_sigma = evaluater.test_nerf_target_sigma(img_pair,batch['view_id'],betas, pose, scale, trans, vol_res=vol_res)
            thresh = 0.5
            vertices, simplices, normals, _ = measure.marching_cubes_lewiner(np.array(nerf_sigma), thresh)
            mesh = dict()
            mesh['v'] = vertices / vol_res - 0.5
            mesh['f'] = simplices[:, (1, 0, 2)]
            mesh['vn'] = normals

            # save .mrc
            with mrcfile.new_mmap(os.path.join(out_dir, model_id + '_sigma_mesh.mrc'), overwrite=True,
                                  shape=nerf_sigma.shape, mrc_mode=2) as mrc:
                mrc.data[:] = nerf_sigma

        # save .obj
        mesh_fname = os.path.join(out_dir, model_id + '_sigma_mesh.obj')
        obj_io.save_obj_data(mesh, mesh_fname)

        mesh_v, mesh_f = mesh['v'].astype(np.float32), mesh['f'].astype(np.int32)
        mesh_v = torch.from_numpy(mesh_v).cuda().unsqueeze(0)
        mesh_f = torch.from_numpy(mesh_f).cuda().unsqueeze(0)

        mesh_color = evaluater.test_tex_pifu(img_pair, batch['view_id'], mesh_v, betas, pose, scale, trans)

        mesh_fname = mesh_fname.replace('.obj', '_tex.obj')

        obj_io.save_obj_data({'v': mesh_v[0].squeeze().detach().cpu().numpy(),
                              'f': mesh_f[0].squeeze().detach().cpu().numpy(),
                              'vc': mesh_color.squeeze()},
                             mesh_fname)

        ## rotate to gt view
        # import pdb; pdb.set_trace()
        vertices1 = evaluater.rotate_points(torch.from_numpy(mesh['v']).cuda().unsqueeze(0), -batch['view_id'][:,0])
        mesh_fname = os.path.join(out_dir, model_id + '_sigma_mesh_gtview.obj')
        obj_io.save_obj_data({'v': vertices1[0].squeeze().detach().cpu().numpy(),
                              'f': mesh['f']},
                             mesh_fname)


        # save_image
        image_fname = os.path.join(out_dir, model_id + '_src_image.png')
        save_image(img_pair.reshape(-1, 3, batch['img'].size(3), batch['img'].size(3)), image_fname)

        #measure dist
        mesh_fname = os.path.join(out_dir, model_id + '_sigma_mesh_gtview.obj')

        tgt_meshname = f'/home/nas1_temp/dataset/Thuman/mesh_data/{model_num}/{model_num}.obj'
        tgt_mesh = trimesh.load(tgt_meshname)
        src_mesh = trimesh.load(mesh_fname)
        tgt_mesh  = trimesh.Trimesh.simplify_quadratic_decimation(tgt_mesh, 100000)

        p2s_dist = get_surface_dist(tgt_mesh, src_mesh)
        chamfer_dist= get_chamfer_dist(tgt_mesh, src_mesh)
        p2s_list.append(p2s_dist)
        chamfer_list.append(chamfer_dist)

        print('p2s:', p2s_dist)
        print('chamfer', chamfer_dist)

        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("model id: %s \n" % model_id)
        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("p2s: %f \n" % p2s_dist)
        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("chamfer: %f \n" % chamfer_dist)


    print('Testing Done. ')
    print('p2s mean:',np.mean(p2s_list))
    print('chamfer mean:', np.mean(chamfer_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("p2s mean: %f \n" % np.mean(p2s_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("chamfer mean: %f \n" % np.mean(chamfer_list))
    print('psnr mean:', np.mean(psnr_list))
    print('ssim mean:', np.mean(ssim_list))
    print('lpips mean:', np.mean(lpips_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("'psnr mean: %f \n" % np.mean(psnr_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("ssim mean: %f \n" % np.mean(ssim_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("lpips mean: %f \n" % np.mean(lpips_list))


def validation_texture_multi(pretrained_checkpoint_pamir,
                      pretrained_checkpoint_pamirtex):


    device = torch.device("cuda")
    evaluater = EvaluatorTex_multi(device, pretrained_checkpoint_pamir, pretrained_checkpoint_pamirtex)

    val_ds = TrainingImgDataset_multi(
        '/home/nas1_temp/dataset/Thuman', img_h=const.img_res, img_w=const.img_res,
        training=False, testing_res=256,
        view_num_per_item=360,
        point_num=5000,
        load_pts2smpl_idx_wgt=True,
        smpl_data_folder='./data')

    val_data_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8,
                                 worker_init_fn=None, drop_last=False)



    out_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/validation_vgglpips_texturedgtmesh__' + '_'.join([pretrained_checkpoint_pamirtex.split('/')[-3], pretrained_checkpoint_pamirtex.split('/')[-1][:-3]])
    os.makedirs(out_dir, exist_ok=True)
    stage1_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/validation_256gtsmpl__pamir_nerf_0302_48_03_rayontarget_rayonpts_occ_attloss_inout_24hie_2022_03_06_05_54_57'
    stage2_dir = stage1_dir + '/output_stage2/0303_novolfeat_onlyback_b1_oridata/epoch_200'



    psnr_list = []
    ssim_list = []
    lpips_list = []
    reid_list=[]
    for step_val, batch in enumerate(tqdm(val_data_loader, desc='Testing', total=len(val_data_loader), initial=0)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        model_id = (str(501 + batch['model_id'].item()).zfill(4) + '_' + str(batch['view_id'][:,0].item()).zfill(4))
        print(model_id)
        model_number = str(501 + batch['model_id'].item()).zfill(4)
        tgt_meshname = f'/home/nas1_temp/dataset/Thuman/mesh_data_vc/{model_number}/{model_number}.obj'


        tgt_mesh = trimesh.load(tgt_meshname)


        mesh_v = tgt_mesh.vertices.astype(np.float32)
        mesh_v = torch.from_numpy(mesh_v).cuda().unsqueeze(0)

        mesh_f = tgt_mesh.faces.astype(np.int32)
        mesh_f = torch.from_numpy(mesh_f).cuda().unsqueeze(0)

        v_cam = evaluater.rotate_points(mesh_v, batch['view_id'][:,0])


        ###
        view_id1 = str(batch['view_id'][:, 0].item()).zfill(4)
        view_id2 = str(batch['view_id'][:, 1].item()).zfill(4)

        img_fpath1 = stage2_dir + f'/{model_number}/{view_id1}.png'  # f'/home/nas1_temp/dataset/Thuman/output_stage2/0303_novolfeat_onlyback_b1_oridata/epoch_33/{model_num}/{view_id1}.png'
        img1 = cv.imread(img_fpath1).astype(np.uint8)
        img1 = np.float32(cv.cvtColor(img1, cv.COLOR_RGB2BGR)) / 255.
        img1 = torch.from_numpy(img1.transpose((2, 0, 1))).unsqueeze(0).cuda()
        img_fpath2 = stage2_dir + f'/{model_number}/{view_id1}_{view_id2}.png'  # f'/home/nas1_temp/dataset/Thuman/output_stage2/0303_novolfeat_onlyback_b1_oridata/epoch_33/{model_num}/{view_id1}_{view_id2}.png'
        img2 = cv.imread(img_fpath2).astype(np.uint8)
        img2 = np.float32(cv.cvtColor(img2, cv.COLOR_RGB2BGR)) / 255.
        img2 = torch.from_numpy(img2.transpose((2, 0, 1))).unsqueeze(0).cuda()
        img_pair = torch.stack([img1, img2], 1)
        ###


        mesh_color = evaluater.test_tex_pifu(img_pair, batch['view_id'], v_cam, batch['betas'],batch['pose'], batch['scale'],batch['trans'])



        mesh_fname = os.path.join(out_dir, model_id + '_gt_textured.obj')

        obj_io.save_obj_data({'v': mesh_v[0].squeeze().detach().cpu().numpy(),
                              'f': mesh_f[0].squeeze().detach().cpu().numpy(),
                              'vc': mesh_color.squeeze()},
                             mesh_fname)


        ## render using vertex color
        target_viewid = -(batch['view_id'][:,0].item()+180)
        rendered_img = render_mesh(mesh_fname, render_angle=target_viewid )
        rendered_img = torch.from_numpy(rendered_img).permute(2,0,1).unsqueeze(0 )
        # save_image
        image_fname = os.path.join(out_dir, model_id + '_rendered_image.png')
        save_image(rendered_img, image_fname)

        ##
        if True:

            target_viewid =str((batch['view_id'][:,0].item()+180)%360).zfill(4)
            tgt_imagename = f'/home/nas1_temp/dataset/Thuman/image_data_vc/{model_number}/color/{target_viewid}.jpg'
            gt_img = cv.imread(tgt_imagename).astype(np.uint8)
            gt_img = np.float32(cv.cvtColor( gt_img , cv.COLOR_RGB2BGR)) / 255.
            gt_img= torch.from_numpy( gt_img .transpose((2, 0, 1))).unsqueeze(0).cuda()
        else:
            gt_img= render_mesh(tgt_meshname, render_angle=target_viewid  )
            gt_img = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0)




        image_fname = os.path.join(out_dir, model_id + '_rendered_gt_image.png')
        save_image(gt_img, image_fname)


        ## measure metrics

        psnr = metrics.PSNR()(rendered_img .cuda(), gt_img.cuda())
        ssim = metrics.SSIM()(rendered_img .cuda(), gt_img.cuda())
        lpips = metrics.LPIPS(True)(rendered_img.cuda(), gt_img.cuda())
        psnr_list.append(psnr.item())
        ssim_list.append(ssim.item())
        lpips_list.append(lpips.item())

        ##
        rendered_feat = reid_encoder.get_features(rendered_img.cuda())
        gt_target_feat = reid_encoder.get_features(gt_img.cuda())
        gt_source_feat = reid_encoder.get_features(batch['img'][:,0].cuda() * batch['mask'][:,0].permute(0, 3, 1, 2).cuda())

        # reid_dist = reid_encoder.euclidean_squared_distance(gt_source_feat[0].unsqueeze(0), torch.cat((gt_target_feat[0][None,], rendered_feat[0][None,]),dim=0))
        reid_dist = reid_encoder.cosine_distance(gt_source_feat[0].unsqueeze(0), torch.cat(
            (gt_target_feat[0][None,], rendered_feat[0][None,]), dim=0))
        reid_list.append(reid_dist)

        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("model id: %s \n" % model_id)
        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("psnr: %f \n" % psnr)
        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("ssim : %f \n" % ssim)
        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("lpips : %f \n" % lpips)
        with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
            f.write("reid : %f , reid_gt : %f \n" % (reid_dist[:, 1], reid_dist[:, 0]))

        ## meshlab/blender capture
        if True:
            check = 1

    print('psnr mean:', np.mean(psnr_list))
    print('ssim mean:', np.mean(ssim_list))
    print('lpips mean:', np.mean(lpips_list))
    reid_mean = torch.vstack(reid_list).mean(0)

    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("'psnr mean: %f \n" % np.mean(psnr_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("ssim mean: %f \n" % np.mean(ssim_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("lpips mean: %f \n" % np.mean(lpips_list))
    with open(os.path.join(out_dir, 'validation_result.txt'), 'a') as f:
        f.write("reid mean : %f , reid_gt mean : %f \n" % (reid_mean[1].item(), reid_mean[0].item()))


def main_test_flow_feature(out_dir, pretrained_checkpoint_pamir,
                      pretrained_checkpoint_pamirtex):
    from dataloader.dataloader_tex import AllImgDataset
    dataset = AllImgDataset(
        '/home/nas1_temp/dataset/tt_dataset', img_h=512, img_w=512,
        testing_res=256,
        view_num_per_item=360,
        load_pts2smpl_idx_wgt=True,
        smpl_data_folder='./data')

    # val_ds = TrainingImgDataset(
    #     '/home/nas1_temp/dataset/Thuman', img_h=const.img_res, img_w=const.img_res,
    #     training=False, testing_res=256,
    #     view_num_per_item=360,
    #     point_num=5000,
    #     load_pts2smpl_idx_wgt=True,
    #     smpl_data_folder='./data')


    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8,
                                 worker_init_fn=None, drop_last=False)
    # data_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8,
    #                              worker_init_fn=None, drop_last=False)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'results'), exist_ok=True)

    device = torch.device("cuda")

    evaluater = EvaluatorTex_single(device, pretrained_checkpoint_pamir, pretrained_checkpoint_pamirtex)
    for step, batch in enumerate(tqdm(data_loader, desc='Testing', total=len(data_loader), initial=0)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # if not ('betas' in batch and 'pose' in batch):
        #     raise FileNotFoundError('Cannot found SMPL parameters! You need to run PaMIR-geometry first!')
        # if not ('mesh_vert' in batch and 'mesh_face' in batch):
        #     raise FileNotFoundError('Cannot found the mesh for texturing! You need to run PaMIR-geometry first!')

        # if True:
        #     load_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/smpl_maskoptimization'
        #     model_id = str(batch['model_id'].item()+501).zfill(4)
        #     batch['betas'] = torch.load(os.path.join(load_dir, f'{model_id}_betas.pth')).cuda()
        #     batch['pose'] = torch.load(os.path.join(load_dir, f'{model_id}_pose.pth')).cuda()
        #     batch['scale'] = torch.load(os.path.join(load_dir, f'{model_id}_scale.pth')).cuda()
        #     batch['trans'] = torch.load(os.path.join(load_dir, f'{model_id}_trans.pth')).cuda()

        nerf_color, weight_sum = evaluater.test_nerf_target(batch['img'], batch['betas'],
                                         batch['pose'], batch['scale'], batch['trans'],batch["view_id"] - batch['target_view_id'], return_flow_feature=True)

        # vol = nerf_color_warped[:, :128].numpy()[0]
        flow = nerf_color[:, :2]
        nerf_pts_tex = nerf_color[:, 2:5]
        # nerf_attention= nerf_color_warped[:, -1:]
        warped_image = F.grid_sample(batch['img'].cpu(), flow.permute(0, 2, 3, 1))


        str(batch['model_id'].item()).zfill(4)
        flow_path = os.path.join(out_dir, str(batch['model_id'].item()).zfill(4), 'flow')
        feature_path = os.path.join(out_dir, str(batch['model_id'].item()).zfill(4), 'feature')
        warped_image_path = os.path.join(out_dir, str(batch['model_id'].item()).zfill(4), 'warped_image')
        pred_image_path = os.path.join(out_dir, str(batch['model_id'].item()).zfill(4), 'pred_image')
        attention_path = os.path.join(out_dir, str(batch['model_id'].item()).zfill(4), 'attention')
        weightsum_path = os.path.join(out_dir, str(batch['model_id'].item()).zfill(4), 'weight_sum')
        # flow_path = os.path.join(out_dir, str(batch['model_id'].item() + 501).zfill(4), 'flow')
        # feature_path = os.path.join(out_dir, str(batch['model_id'].item() + 501).zfill(4), 'feature')
        # warped_image_path = os.path.join(out_dir, str(batch['model_id'].item() + 501).zfill(4), 'warped_image')
        # pred_image_path = os.path.join(out_dir, str(batch['model_id'].item() + 501).zfill(4), 'pred_image')
        # attention_path = os.path.join(out_dir, str(batch['model_id'].item() + 501).zfill(4), 'attention')
        # weightsum_path = os.path.join(out_dir, str(batch['model_id'].item() + 501).zfill(4), 'weight_sum')
        os.makedirs(flow_path, exist_ok=True)
        # os.makedirs(feature_path +'/32', exist_ok=True)
        # os.makedirs(feature_path + '/64', exist_ok=True)
        # os.makedirs(feature_path + '/128', exist_ok=True)
        os.makedirs(pred_image_path, exist_ok=True)
        os.makedirs(warped_image_path, exist_ok=True)
        # os.makedirs(attention_path, exist_ok=True)
        os.makedirs(weightsum_path, exist_ok=True)
        file_name = str(batch["view_id"].item()).zfill(4) + '_' + str(batch["target_view_id"].item()).zfill(4)
        save_image(torch.cat([(flow/2 + 0.5), torch.zeros((flow.size(0), 1, flow.size(2), flow.size(3)))],dim=1), os.path.join(flow_path, file_name + '.png'))
        save_image(warped_image, os.path.join(warped_image_path, file_name + '.png'))
        # save_image(nerf_attention, os.path.join(attention_path, file_name + '.png'))
        save_image(nerf_pts_tex, os.path.join(pred_image_path, file_name + '.png'))
        save_image(weight_sum, os.path.join(weightsum_path, file_name + '.png'))
        # np.save(os.path.join(feature_path, '128', file_name + '.npy'), vol[:, ::2, ::2])
        # np.save(os.path.join(feature_path, '64', file_name + '.npy'), vol[:, ::4, ::4])
        # np.save(os.path.join(feature_path, '32', file_name + '.npy'), vol[:, ::8, ::8])

    import pdb; pdb.set_trace()
    print('Testing Done. ')


if __name__ == '__main__':

    pifu_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_texture_epoch200_trainset/checkpoints/'
    #validation_pifu(pifu_dir )



    #geometry_model_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_geometry/checkpoints/latest.pt'
    #texture_model_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_texture/checkpoints/latest.pt'
    geometry_model_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_geometry_gtsmpl_epoch30_trainset_hg2/checkpoints/2022_02_25_11_28_01.pt'
    texture_model_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_texture_epoch200_trainset/checkpoints/2022_03_03_15_14_56.pt'
    #validation_pamir(geometry_model_dir, texture_model_dir)
    #inference_pamir('/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/test_data_check', geometry_model_dir, texture_model_dir)


    texture_model_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_nerf_0302_48_03_rayontarget_rayonpts_occ_attloss_inout_24hie/checkpoints/2022_03_06_05_54_57.pt'
    #validation_texture(geometry_model_dir, texture_model_dir)
    # validation(geometry_model_dir, texture_model_dir)
    #inference('/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/test_data_check', geometry_model_dir, texture_model_dir)

    texture_model_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_nerf_0302_24hie_03_occ_2v_alpha_concat/checkpoints/2022_03_06_01_07_09.pt'
    #validation_multi(geometry_model_dir , texture_model_dir)
    #inference_multi('/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/test_data_check', geometry_model_dir,texture_model_dir)
    #validation_texture_multi(geometry_model_dir, texture_model_dir)


    main_test_flow_feature(
       '/home/nas1_temp/dataset/Thuman/output_stage1/0329_test',
       pretrained_checkpoint_pamir='/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_geometry/checkpoints/latest.pt',
       pretrained_checkpoint_pamirtex='/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/0328_1_tt_nerf_24hie_03_rayontarget_attloss_occinout/checkpoints/latest.pt')

