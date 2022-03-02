from __future__ import division, print_function

import os
import torch
import pickle as pkl
from tqdm import tqdm

from util import util
from util import obj_io
from torch.nn import functional as F

def main_test_with_gt_smpl(test_img_dir, out_dir, pretrained_checkpoint, pretrained_gcmr_checkpoint):
    from evaluator import Evaluator
    from dataloader.dataloader_testing import TestingImgLoader

    os.makedirs(out_dir, exist_ok=True)
    os.system('cp -r %s/*.* %s/' % (test_img_dir, out_dir))
    os.makedirs(os.path.join(out_dir, 'results'), exist_ok=True)


    device = torch.device("cuda")
    loader = TestingImgLoader(out_dir, 512, 512)
    evaluator = Evaluator(device, pretrained_checkpoint, pretrained_gcmr_checkpoint)
    for step, batch in enumerate(tqdm(loader, desc='Testing', total=len(loader), initial=0)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        mesh = evaluator.test_pifu(batch['img'],256, batch['betas'], batch['pose'], batch['scale'],
                                   batch['trans'])
        img_dir = batch['img_dir'][0]
        img_fname = os.path.split(img_dir)[1]
        mesh_fname = os.path.join(out_dir, 'results', img_fname[:-4] + '.obj')
        obj_io.save_obj_data(mesh, mesh_fname)
    print('Testing Done. ')


def main_test_wo_gt_smpl_with_optm(test_img_dir, out_dir, pretrained_checkpoint, pretrained_gcmr_checkpoint,
                                   iternum=50):
    from evaluator import Evaluator
    from dataloader.dataloader_testing import TestingImgLoader

    smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = \
        util.read_smpl_constants('./data')

    os.makedirs(out_dir, exist_ok=True)
    os.system('cp -r %s/*.* %s/' % (test_img_dir, out_dir))
    os.makedirs(os.path.join(out_dir, 'results'), exist_ok=True)

    device = torch.device("cuda")
    loader = TestingImgLoader(out_dir, 512, 512, white_bg=True)
    evaluator = Evaluator(device, pretrained_checkpoint, pretrained_gcmr_checkpoint)

    for step, batch in enumerate(tqdm(loader, desc='Testing', total=len(loader), initial=0)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        print(batch['img_dir'])
        pred_betas, pred_rotmat, scale, trans, pred_smpl = evaluator.test_gcmr(batch['img'])
        optm_thetas, optm_betas, optm_smpl = evaluator.optm_smpl_param(
            batch['img'], batch['keypoints'], pred_betas, pred_rotmat, scale, trans, iternum)
        optm_betas = optm_betas.detach()
        optm_thetas = optm_thetas.detach()
        scale, trans = scale.detach(), trans.detach()
        mesh = evaluator.test_pifu(batch['img'], 256, optm_betas, optm_thetas, scale, trans)
        img_dir = batch['img_dir'][0]
        img_fname = os.path.split(img_dir)[1]
        mesh_fname = os.path.join(out_dir, 'results', img_fname[:-4] + '.obj')
        init_smpl_fname = os.path.join(out_dir, 'results', img_fname[:-4] + '_init_smpl.obj')
        optm_smpl_fname = os.path.join(out_dir, 'results', img_fname[:-4] + '_optm_smpl.obj')
        obj_io.save_obj_data(mesh, mesh_fname)
        obj_io.save_obj_data({'v': pred_smpl.squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                             init_smpl_fname)
        obj_io.save_obj_data({'v': optm_smpl.squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                             optm_smpl_fname)
        smpl_param_name = os.path.join(out_dir, 'results', img_fname[:-4] + '_smplparams.pkl')
        with open(smpl_param_name, 'wb') as fp:
            pkl.dump({'betas': optm_betas.squeeze().detach().cpu().numpy(),
                      'body_pose': optm_thetas.squeeze().detach().cpu().numpy(),
                      'init_betas': pred_betas.squeeze().detach().cpu().numpy(),
                      'init_body_pose': pred_rotmat.squeeze().detach().cpu().numpy(),
                      'body_scale': scale.squeeze().detach().cpu().numpy(),
                      'global_body_translation': trans.squeeze().detach().cpu().numpy()},
                     fp)
        # os.system('cp %s %s.original' % (mesh_fname, mesh_fname))
        # os.system('%s %s %s' % (REMESH_BIN, mesh_fname, mesh_fname))
        # os.system('%s %s %s' % (ISOLATION_REMOVAL_BIN, mesh_fname, mesh_fname))
    print('Testing Done. ')

from torchvision.utils import save_image

def main_test_texture(test_img_dir, out_dir, pretrained_checkpoint_pamir,
                      pretrained_checkpoint_pamirtex):
    from evaluator_tex import EvaluatorTex
    from dataloader.dataloader_testing import TestingImgLoader

    os.makedirs(out_dir, exist_ok=True)
    os.system('cp -r %s/*.* %s/' % (test_img_dir, out_dir))
    os.makedirs(os.path.join(out_dir, 'results'), exist_ok=True)

    device = torch.device("cuda")
    loader = TestingImgLoader(out_dir, 512, 512, white_bg=True)
    evaluater = EvaluatorTex(device, pretrained_checkpoint_pamir, pretrained_checkpoint_pamirtex)
    for step, batch in enumerate(tqdm(loader, desc='Testing', total=len(loader), initial=0)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if not ('betas' in batch and 'pose' in batch):
            raise FileNotFoundError('Cannot found SMPL parameters! You need to run PaMIR-geometry first!')
        if not ('mesh_vert' in batch and 'mesh_face' in batch):
            raise FileNotFoundError('Cannot found the mesh for texturing! You need to run PaMIR-geometry first!')

        # for i in [0]:#,1,2,3]:
        #     mesh_color = evaluater.test_tex_featurenerf(batch['img'], batch['mesh_vert'], batch['betas'],
        #                                             batch['pose'], batch['scale'], batch['trans'], i)
        #     img_dir = batch['img_dir'][0]
        #     img_fname = os.path.split(img_dir)[1]
        #     mesh_fname = os.path.join(out_dir, 'results', img_fname[:-4] + f'_tex_one{i}.obj')
        #     obj_io.save_obj_data({'v': batch['mesh_vert'][0].squeeze().detach().cpu().numpy(),
        #                           'f': batch['mesh_face'][0].squeeze().detach().cpu().numpy(),
        #                           'vc': mesh_color.squeeze()},
        #                          mesh_fname)
        #
        #
        # import pdb; pdb.set_trace()


        for i in range(0, 370, 10 ):


            nerf_color, nerf_color_wapred = evaluater.test_nerf_target(batch['img'], batch['betas'],
                                             batch['pose'], batch['scale'], batch['trans'],  torch.ones(batch['img'].shape[0]).to(device)*i)

            save_image(nerf_color, f'./0216_nerf_source_occth0.5{str(i).zfill(3)}.png')
            #save_image(nerf_color_wapred, f'./0216_nerfwapred_source_{str(i).zfill(3)}.png')

        import pdb;
        pdb.set_trace()




        mesh_color = evaluater.test_tex_pifu(batch['img'], batch['mesh_vert'], batch['betas'],
                                             batch['pose'], batch['scale'], batch['trans'])

        img_dir = batch['img_dir'][0]
        img_fname = os.path.split(img_dir)[1]
        mesh_fname = os.path.join(out_dir, 'results', img_fname[:-4] + '_tex.obj')
        obj_io.save_obj_data({'v': batch['mesh_vert'][0].squeeze().detach().cpu().numpy(),
                              'f': batch['mesh_face'][0].squeeze().detach().cpu().numpy(),
                              'vc': mesh_color.squeeze()},
                             mesh_fname)
    print('Testing Done. ')


def main_test_flow_feature(out_dir, pretrained_checkpoint_pamir,
                      pretrained_checkpoint_pamirtex):
    from evaluator_tex import EvaluatorTex
    from dataloader.dataloader_tex import AllImgDataset, TrainingImgDataset
    # dataset = AllImgDataset(
    #     '/home/nas1_temp/dataset/Thuman', img_h=512, img_w=512,
    #     testing_res=256,
    #     view_num_per_item=360,
    #     load_pts2smpl_idx_wgt=True,
    #     smpl_data_folder='./data')

    val_ds = TrainingImgDataset(
        '/home/nas1_temp/dataset/Thuman', img_h=const.img_res, img_w=const.img_res,
        training=False, testing_res=256,
        view_num_per_item=360,
        point_num=5000,
        load_pts2smpl_idx_wgt=True,
        smpl_data_folder='./data')


    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8,
    #                              worker_init_fn=None, drop_last=False)
    data_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8,
                                 worker_init_fn=None, drop_last=False)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'results'), exist_ok=True)

    device = torch.device("cuda")

    evaluater = EvaluatorTex(device, pretrained_checkpoint_pamir, pretrained_checkpoint_pamirtex)
    for step, batch in enumerate(tqdm(data_loader, desc='Testing', total=len(data_loader), initial=0)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # if not ('betas' in batch and 'pose' in batch):
        #     raise FileNotFoundError('Cannot found SMPL parameters! You need to run PaMIR-geometry first!')
        # if not ('mesh_vert' in batch and 'mesh_face' in batch):
        #     raise FileNotFoundError('Cannot found the mesh for texturing! You need to run PaMIR-geometry first!')

        if False:
            print('view_id', batch['view_id'])
            print('target_view_id', batch['target_view_id'])
            continue


        if True:
            load_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/smpl_maskoptimization'
            model_id = str(batch['model_id'].item()+501).zfill(4)
            batch['betas'] = torch.load(os.path.join(load_dir, f'{model_id}_betas.pth')).cuda()
            batch['pose'] = torch.load(os.path.join(load_dir, f'{model_id}_pose.pth')).cuda()
            batch['scale'] = torch.load(os.path.join(load_dir, f'{model_id}_scale.pth')).cuda()
            batch['trans'] = torch.load(os.path.join(load_dir, f'{model_id}_trans.pth')).cuda()

        nerf_color, nerf_color_warped, weight_sum = evaluater.test_nerf_target(batch['img'], batch['betas'],
                                         batch['pose'], batch['scale'], batch['trans'],batch["view_id"] - batch['target_view_id'], return_flow_feature=True)

        vol = nerf_color_warped[:, :128].numpy()[0]
        flow = nerf_color[:, :2]
        nerf_pts_tex = nerf_color[:, 2:5]
        nerf_attention= nerf_color_warped[:, -1:]
        warped_image = F.grid_sample(batch['img'].cpu(), flow.permute(0, 2, 3, 1))


        str(batch['model_id'].item()).zfill(4)
        # flow_path = os.path.join(out_dir, str(batch['model_id'].item()).zfill(4), 'flow')
        # feature_path = os.path.join(out_dir, str(batch['model_id'].item()).zfill(4), 'feature')
        # warped_image_path = os.path.join(out_dir, str(batch['model_id'].item()).zfill(4), 'warped_image')
        # pred_image_path = os.path.join(out_dir, str(batch['model_id'].item()).zfill(4), 'pred_image')
        # attention_path = os.path.join(out_dir, str(batch['model_id'].item()).zfill(4), 'attention')
        # weightsum_path = os.path.join(out_dir, str(batch['model_id'].item()).zfill(4), 'weight_sum')
        flow_path = os.path.join(out_dir, str(batch['model_id'].item() + 501).zfill(4), 'flow')
        feature_path = os.path.join(out_dir, str(batch['model_id'].item() + 501).zfill(4), 'feature')
        warped_image_path = os.path.join(out_dir, str(batch['model_id'].item() + 501).zfill(4), 'warped_image')
        pred_image_path = os.path.join(out_dir, str(batch['model_id'].item() + 501).zfill(4), 'pred_image')
        attention_path = os.path.join(out_dir, str(batch['model_id'].item() + 501).zfill(4), 'attention')
        weightsum_path = os.path.join(out_dir, str(batch['model_id'].item() + 501).zfill(4), 'weight_sum')
        os.makedirs(flow_path, exist_ok=True)
        # os.makedirs(feature_path +'/32', exist_ok=True)
        # os.makedirs(feature_path + '/64', exist_ok=True)
        os.makedirs(feature_path + '/128', exist_ok=True)
        os.makedirs(pred_image_path, exist_ok=True)
        os.makedirs(warped_image_path, exist_ok=True)
        os.makedirs(attention_path, exist_ok=True)
        os.makedirs(weightsum_path, exist_ok=True)
        file_name = str(batch["view_id"].item()).zfill(4) + '_' + str(batch["target_view_id"].item()).zfill(4)
        save_image(torch.cat([(flow/2 + 0.5), torch.zeros((flow.size(0), 1, flow.size(2), flow.size(3)))],dim=1), os.path.join(flow_path, file_name + '.png'))
        save_image(warped_image, os.path.join(warped_image_path, file_name + '.png'))
        save_image(nerf_attention, os.path.join(attention_path, file_name + '.png'))
        save_image(nerf_pts_tex, os.path.join(pred_image_path, file_name + '.png'))
        save_image(weight_sum, os.path.join(weightsum_path, file_name + '.png'))
        np.save(os.path.join(feature_path, '128', file_name + '.npy'), vol[:, ::2, ::2])
        # np.save(os.path.join(feature_path, '64', file_name + '.npy'), vol[:, ::4, ::4])
        # np.save(os.path.join(feature_path, '32', file_name + '.npy'), vol[:, ::8, ::8])

    import pdb; pdb.set_trace()
    print('Testing Done. ')
def main_test_sigma(test_img_dir, out_dir, pretrained_checkpoint_pamir,
                      pretrained_checkpoint_pamirtex):
    from evaluator_tex import EvaluatorTex
    from dataloader.dataloader_testing import TestingImgLoader

    os.makedirs(out_dir, exist_ok=True)
    os.system('cp -r %s/*.* %s/' % (test_img_dir, out_dir))
    os.makedirs(os.path.join(out_dir, 'results'), exist_ok=True)

    device = torch.device("cuda")
    loader = TestingImgLoader(out_dir, 512, 512, white_bg=True)
    evaluater = EvaluatorTex(device, pretrained_checkpoint_pamir, pretrained_checkpoint_pamirtex)
    for step, batch in enumerate(tqdm(loader, desc='Testing', total=len(loader), initial=0)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if not ('betas' in batch and 'pose' in batch):
            raise FileNotFoundError('Cannot found SMPL parameters! You need to run PaMIR-geometry first!')
        if not ('mesh_vert' in batch and 'mesh_face' in batch):
            raise FileNotFoundError('Cannot found the mesh for texturing! You need to run PaMIR-geometry first!')

        vol_res = 128

        nerf_sigma = evaluater.test_nerf_target_sigma(batch['img'], batch['betas'],
                                                      batch['pose'], batch['scale'], batch['trans'], vol_res = vol_res)

        import mrcfile
        from skimage import measure
        import numpy as np
        img_dir = batch['img_dir'][0]
        img_fname = os.path.split(img_dir)[1]


        mesh_fname = os.path.join(out_dir, 'results', img_fname[:-4] + '_sigma_mesh_nonerf.obj')

        with mrcfile.new_mmap(os.path.join(out_dir, 'results',  img_fname[:-4] + '_sigma_mesh_nonerf.mrc'), overwrite=True, shape=nerf_sigma.shape, mrc_mode=2) as mrc:
            mrc.data[:] = nerf_sigma

        thresh = 0.95
        vertices, simplices, normals, _ = measure.marching_cubes_lewiner(np.array(nerf_sigma ), thresh)
        mesh = dict()
        mesh['v'] = vertices /  vol_res - 0.5
        mesh['f'] = simplices[:, (1, 0, 2)]
        mesh['vn'] = normals

        obj_io.save_obj_data(mesh, mesh_fname)

        mesh = obj_io.load_obj_data(mesh_fname)
        mesh_v, mesh_f = mesh['v'].astype(np.float32), mesh['f'].astype(np.int32)
        mesh_v = torch.from_numpy(mesh_v).cuda().unsqueeze(0)
        mesh_f = torch.from_numpy(mesh_f).cuda().unsqueeze(0)


        mesh_color = evaluater.test_tex_pifu(batch['img'], mesh_v, batch['betas'],
                                             batch['pose'], batch['scale'], batch['trans'])
        mesh_fname = mesh_fname.replace('.obj', '_tex.obj')


        obj_io.save_obj_data({'v': mesh_v[0].squeeze().detach().cpu().numpy(),
                              'f': mesh_f[0].squeeze().detach().cpu().numpy(),
                              'vc': mesh_color.squeeze()},
                             mesh_fname)
    print('Testing Done. ')

import trimesh
import numpy as np
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


from torch.utils.data import DataLoader
import constant as const
from skimage import measure
import numpy as np
import mrcfile
from torchvision.utils import save_image
import open3d
def validation_pamir(pretrained_checkpoint_pamir,
                      pretrained_checkpoint_pamirtex, iternum=100):

    from evaluator_tex_pamir import EvaluatorTex as EvaluatorTex
    from evaluator import Evaluator
    from dataloader.dataloader_tex import TrainingImgDataset
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

    for step_val, batch in enumerate(tqdm(val_data_loader, desc='Testing', total=len(val_data_loader), initial=0)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        out_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/validation_0302_pamir_256gcmroptmask_gttrans__pamir_geometry_gtsmpl_epoch30_trainset_hg2/'
        os.makedirs(out_dir, exist_ok=True)
        model_id = str(501 + batch['model_id'].item()).zfill(4)
        print(model_id)

        vol_res = 256

        if False:
            surface_render_pred, surface_render_alpha = evaluater_tex.test_surface_rendering(batch['img'],
                                                                                             batch['betas'],
                                                                                             batch['pose'],
                                                                                             batch['scale'],
                                                                                             batch['trans'],
                                                                                             torch.ones(
                                                                                                 batch['img'].shape[
                                                                                                     0]).cuda() * 249)
            image_fname = os.path.join(out_dir, model_id + '_surface_rendered_image.png')
            save_image(surface_render_pred,  image_fname)



        use_gcmr= True
        if use_gcmr :
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
            optm_thetas, optm_betas, optm_smpl = evaluator.optm_smpl_param_mask(
                batch['img'], batch['mask'], pred_betas, pred_rotmat, scale, trans, iter_num=iternum)  ##not yet



            optm_smpl_fname = os.path.join(out_dir, model_id+'_optm_smpl.obj')
            obj_io.save_obj_data({'v': optm_smpl.squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                                optm_smpl_fname)


            betas =optm_betas
            pose = optm_thetas

        else:
            betas= batch['betas']
            pose = batch['pose']
            scale = batch['scale']
            trans = batch['trans']

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
        tgt_meshname = f'/home/nas1_temp/dataset/Thuman/mesh_data/{model_id}/{model_id}.obj'
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

def validation(pretrained_checkpoint_pamir,
                      pretrained_checkpoint_pamirtex, iternum=100):
    from evaluator_tex import EvaluatorTex

    from dataloader.dataloader_tex import TrainingImgDataset
    device = torch.device("cuda")
    evaluater = EvaluatorTex(device, pretrained_checkpoint_pamir, pretrained_checkpoint_pamirtex)

    from evaluator import Evaluator
    evaluater_pretrained = Evaluator(device, pretrained_checkpoint_pamir, './results/gcmr_pretrained/gcmr_2020_12_10-21_03_12.pt')

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

    for step_val, batch in enumerate(tqdm(val_data_loader, desc='Testing', total=len(val_data_loader), initial=0)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        out_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/validation_0302_ours_256gcmroptmask_gttrans__pamir_nerf_0222_48_03_rayontarget_rayonpts_occ_attloss_inout_24hie/'
        os.makedirs(out_dir, exist_ok=True)
        model_id = str(501 + batch['model_id'].item()).zfill(4)
        print(model_id)

        vol_res = 256

        if False:
            surface_render_pred, surface_render_alpha= evaluater.test_surface_rendering(batch['img'], batch['betas'], batch['pose'], batch['scale'], batch['trans'],
                                                              torch.ones(batch['img'].shape[0]).cuda() * 249)

            volume_render_pred, volume_render_alpha = evaluater.test_nerf_target(batch['img'], batch['betas'],
                                                                                         batch['pose'], batch['scale'],
                                                                                         batch['trans'],
                                                                                         torch.ones(batch['img'].shape[
                                                                                                        0]).cuda() *249)

            image_fname = os.path.join(out_dir, model_id + '_surface_rendered_image.png')
            save_image(surface_render_pred, image_fname)
            image_fname = os.path.join(out_dir, model_id + '_volume_rendered_image.png')
            save_image(volume_render_pred, image_fname)


        use_gcmr= True
        if use_gcmr :
            #pred_betas, pred_rotmat, scale, trans, pred_smpl = evaluator_pretrained.test_gcmr(batch['img'])
            #pred_smpl = scale * pred_smpl + trans

            ##
            pred_betas, pred_rotmat, _, _, pred_vert_tetsmpl, pred_cam =evaluater_pretrained.test_gcmr(batch['img'],
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

            optm_thetas, optm_betas, optm_smpl, nerf_image_before, nerf_image = evaluater.optm_smpl_param(
                batch['img'], batch['mask'], pred_betas, pred_rotmat, scale, trans, iter_num=iternum)

            optm_smpl_fname = os.path.join(out_dir, model_id+'_optm_smpl.obj')
            obj_io.save_obj_data({'v': optm_smpl.squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                                optm_smpl_fname)

            ##optimization end
            #save_image
            image_fname = os.path.join(out_dir, model_id + '_nerf_image_before.png')
            save_image(nerf_image_before, image_fname)
            image_fname = os.path.join(out_dir, model_id + '_nerf_image_after.png')
            save_image(nerf_image, image_fname)

            betas =optm_betas
            pose = optm_thetas

        else:
            betas= batch['betas']
            pose = batch['pose']
            scale = batch['scale']
            trans = batch['trans']

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

        tgt_meshname = f'/home/nas1_temp/dataset/Thuman/mesh_data/{model_id}/{model_id}.obj'
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



if __name__ == '__main__':
    iternum=50
    input_image_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/test_thuman_0525_gtsmpl/'
    output_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/test_thuman_0525_gtsmpl/'
    # input_image_dir = './results/test_data_real/'
    # output_dir = './results/test_data_real/'
    # input_image_dir = './results/test_data_rendered/'
    # output_dir = './results/test_data_rendered/'
    #geometry_model_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_geometry/checkpoints/latest.pt'
    geometry_model_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_geometry_gtsmpl_epoch30_trainset_hg2/checkpoints/2022_02_25_09_59_41.pt'
    texture_model_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_texture/checkpoints/latest.pt'


    validation_pamir(geometry_model_dir, texture_model_dir)


    #texture_model_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_nerf_0225_48_03_rayontarget_rayonpts_occ_attloss_inout_24hiefirstbin_hg/checkpoints/latest.pt'
    #texture_model_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_nerf_0228_24hiesurface_03_occ_inout_hg/checkpoints/latest.pt'
    texture_model_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_nerf_0222_48_03_rayontarget_rayonpts_occ_attloss_inout_24hie/checkpoints/2022_02_24_19_21_39.pt'

    #validation(geometry_model_dir, texture_model_dir)


    # #! NOTE: We recommend using this when accurate SMPL estimation is available (e.g., through external optimization / annotation)
    # main_test_with_gt_smpl(input_image_dir,
    #                        output_dir,
    #                        pretrained_checkpoint= geometry_model_dir ,
    #                        pretrained_gcmr_checkpoint='./results/gcmr_pretrained/gcmr_2020_12_10-21_03_12.pt')

    #! Otherwise, use this function to predict and optimize a SMPL model for the input image
    # if not os.path.exists(output_dir):
    #     main_test_wo_gt_smpl_with_optm(input_image_dir,
    #                                output_dir,
    #                                pretrained_checkpoint='/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_geometry/checkpoints/latest.pt',
    #                                pretrained_gcmr_checkpoint='/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/gcmr_pretrained/gcmr_2020_12_10-21_03_12.pt')


    # main_test_texture(output_dir,
    #                   output_dir,
    #                   pretrained_checkpoint_pamir= geometry_model_dir ,
    #                   pretrained_checkpoint_pamirtex=texture_model_dir)

    # main_test_sigma(output_dir,
    #                   output_dir,
    #                   pretrained_checkpoint_pamir= geometry_model_dir ,
    #                   pretrained_checkpoint_pamirtex=texture_model_dir)

    #main_test_flow_feature(
    #    '/home/nas1_temp/dataset/Thuman/output_stage1/pamir_nerf_0222_48_03_rayontarget_rayonpts_occ_attloss_inout_24hie_val_GCMR',
    #    pretrained_checkpoint_pamir='/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_geometry/checkpoints/latest.pt',
    #    pretrained_checkpoint_pamirtex='/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_nerf_0222_48_03_rayontarget_rayonpts_occ_attloss_inout_24hie/checkpoints/0223_checkpoint.pt')