from __future__ import division, print_function

import os
import torch
import pickle as pkl
from tqdm import tqdm

from util import util
from util import obj_io
from torch.nn import functional as F

from Pytorch_metrics import metrics

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
    from dataloader.dataloader_tex import AllImgDataset
    dataset = AllImgDataset(
        '/home/nas1_temp/dataset/Thuman', img_h=512, img_w=512,
        testing_res=256,
        view_num_per_item=360,
        load_pts2smpl_idx_wgt=True,
        smpl_data_folder='./data')


    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8,
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
        nerf_color, nerf_color_warped = evaluater.test_nerf_target(batch['img'], batch['betas'],
                                         batch['pose'], batch['scale'], batch['trans'],batch["view_id"] - batch['target_view_id'], return_flow_feature=True)
        import pdb; pdb.set_trace()
        vol = nerf_color_warped[:, :32].numpy()[0]
        warped_image = F.grid_sample(batch['img'].cpu(), nerf_color.permute(0, 2, 3, 1))


        str(batch['model_id'].item()).zfill(4)
        flow_path = os.path.join(out_dir, str(batch['model_id'].item()).zfill(4), 'flow')
        feature_path = os.path.join(out_dir, str(batch['model_id'].item()).zfill(4), 'feature')
        image_path = os.path.join(out_dir, str(batch['model_id'].item()).zfill(4), 'image')
        os.makedirs(flow_path, exist_ok=True)
        os.makedirs(feature_path +'/32', exist_ok=True)
        os.makedirs(feature_path + '/64', exist_ok=True)
        os.makedirs(feature_path + '/128', exist_ok=True)
        os.makedirs(image_path, exist_ok=True)
        file_name = str(batch["view_id"].item()).zfill(4) + '_' + str(batch["target_view_id"].item()).zfill(4)
        save_image(torch.cat([(nerf_color/2 + 0.5), torch.zeros((nerf_color.size(0), 1, nerf_color.size(2), nerf_color.size(3)))],dim=1), os.path.join(flow_path, file_name + '.png'))
        save_image(warped_image, os.path.join(image_path, file_name + '.png'))
        np.save(os.path.join(feature_path, '128', file_name + '.npy'), vol[:, ::2, ::2])
        np.save(os.path.join(feature_path, '64', file_name + '.npy'), vol[:, ::4, ::4])
        np.save(os.path.join(feature_path, '32', file_name + '.npy'), vol[:, ::8, ::8])


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
import cv2 as cv
from os import listdir
def inference(test_img_dir,pretrained_checkpoint_pamir,
                      pretrained_checkpoint_pamirtex, iternum=100):
    #/home/nas1_temp/dataset/deepfashion/our_test
    from evaluator_tex import EvaluatorTex
    from evaluator import Evaluator


    device = torch.device("cuda")
    evaluater = EvaluatorTex(device, pretrained_checkpoint_pamir, pretrained_checkpoint_pamirtex)
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
    psnr_list = []
    ssim_list = []
    lpips_list = []

    for step_val, batch in enumerate(tqdm(val_data_loader, desc='Testing', total=len(val_data_loader), initial=0)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        #out_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/smpl_maskoptimization_4v/'
        out_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/validation_256gcmroptmask_gttrans_pamir_nerf_0227_24hie0.5_03_occ_2v_alpha_concat_2022_03_02_06_24_00/'
        os.makedirs(out_dir, exist_ok=True)
        model_num = str(501 + batch['model_id'].item()).zfill(4)
        model_id = (str(501 + batch['model_id'].item()) + '_' + str(batch['view_id'][:,0].item())).zfill(4)
        print(model_id)

        view_id1 = str(batch['view_id'][:, 0].item()).zfill(4)
        view_id2 = str(batch['view_id'][:, 1].item()).zfill(4)

        img_fpath1 = f'/home/nas1_temp/dataset/Thuman/output_stage2/0303_novolfeat_onlyback_b1_oridata/epoch_33/{model_num}/{view_id1}.png'
        img1 = cv.imread(img_fpath1).astype(np.uint8)
        img1 = np.float32(cv.cvtColor(img1, cv.COLOR_RGB2BGR)) / 255.
        img1 = torch.from_numpy(img1.transpose((2, 0, 1))).unsqueeze(0).cuda()
        img_fpath2 = f'/home/nas1_temp/dataset/Thuman/output_stage2/0303_novolfeat_onlyback_b1_oridata/epoch_33/{model_num}/{view_id1}_{view_id2}.png'
        img2 = cv.imread(img_fpath2).astype(np.uint8)
        img2 = np.float32(cv.cvtColor(img2, cv.COLOR_RGB2BGR)) / 255.
        img2 = torch.from_numpy(img2.transpose((2, 0, 1))).unsqueeze(0).cuda()
        img_pair = torch.stack([img1, img2], 1)


        vol_res = 256
        if True:

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
            betas = torch.load(os.path.join('/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/smpl_maskoptimization_4v', f'{model_id}_betas.pth')).cuda()
            pose =torch.load(os.path.join('/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/smpl_maskoptimization_4v', f'{model_id}_pose.pth')).cuda()
            scale =  torch.load(os.path.join('/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/smpl_maskoptimization_4v', f'{model_id}_scale.pth')).cuda()
            trans =  torch.load(os.path.join('/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/smpl_maskoptimization_4v', f'{model_id}_trans.pth')).cuda()

        # torch.save(betas.cpu(),  os.path.join(out_dir, model_id+'_betas.pth'))
        # torch.save(pose.cpu(), os.path.join(out_dir, model_id + '_pose.pth'))
        # torch.save(scale.cpu(), os.path.join(out_dir, model_id + '_scale.pth'))
        # torch.save(trans.cpu(), os.path.join(out_dir, model_id + '_trans.pth'))
        # continue

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


if __name__ == '__main__':
    iternum=50
    input_image_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/test_thuman_0525_gtsmpl/'
    output_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/test_thuman_0525_gtsmpl/'
    # input_image_dir = './results/test_data_real/'
    # output_dir = './results/test_data_real/'
    # input_image_dir = './results/test_data_rendered/'
    # output_dir = './results/test_data_rendered/'

    #geometry_model_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_geometry/checkpoints/latest.pt'
    geometry_model_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_geometry_gtsmpl_epoch30/checkpoints/latest.pt'


    texture_model_dir = '/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_nerf_0227_24hie0.5_03_occ_2v_alpha_concat/checkpoints/2022_03_02_06_24_00.pt'

    validation(geometry_model_dir , texture_model_dir)
    #inference('/home/nas1_temp/dataset/deepfashion/our_test',geometry_model_dir, texture_model_dir)


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

    # main_test_flow_feature(
    #     '/home/nas1_temp/dataset/Thuman/output_stage1/nerf_flowvr_0216data_maskloss_',
    #     pretrained_checkpoint_pamir='/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_geometry/checkpoints/latest.pt',
    #     pretrained_checkpoint_pamirtex='/home/nas3_userJ/shimgyumin/fasker/research/pamir/networks/results/pamir_nerf_0222_48_03_rayontarget_rayonpts_occ_attloss_inout_24hie/checkpoints/0223_checkpoint.pt')