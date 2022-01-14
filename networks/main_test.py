from __future__ import division, print_function

import os
import torch
import pickle as pkl
from tqdm import tqdm

from util import util
from util import obj_io

from torchvision.utils import save_image

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
        mesh = evaluator.test_pifu(batch['img'], 256, batch['betas'], batch['pose'], batch['scale'],
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

def xyz_to_uv_mapping(xyz_dir, evaluater, batch):
    asd = torch.load(xyz_dir)
    colors = evaluater.test_tex_pifu(batch['img'], asd[None,], batch['betas'], batch['pose'], batch['scale'],
                                     batch['trans'])
    colors_1 = colors.reshape(256, 256, 3)
    colors_2 = torch.Tensor(colors_1)
    uv_img = colors_2.permute(2, 0, 1)
    return uv_img

def xyz_to_uv_mapping_2(evaluater, batch):
    mapper = evaluater.mapper
    mesh_vert = batch['mesh_vert'][0]
    f_global = mapper.encoding_global(mesh_vert)
    uv = mapper.encoding(mesh_vert, f_global)

    uv_space = torch.meshgrid(torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256))
    uv_space = torch.stack(uv_space, dim=-1)
    uv_space_lin = uv_space.reshape(-1, 2)
    with torch.no_grad():
        xyz_hat_in = mapper.decoding(uv_space_lin.cuda(), f_global)

    colors = evaluater.test_tex_pifu(batch['img'], xyz_hat_in[None,], batch['betas'], batch['pose'], batch['scale'],
                                     batch['trans'])
    colors_1 = colors.reshape(256, 256, 3)
    colors_2 = torch.Tensor(colors_1)
    uv_img = colors_2.permute(2, 0, 1)
    return uv_img, torch.cat([uv[:,1:2], uv[:,0:1]], dim=1)

def xyz_to_uv_mapping_3(evaluater, batch):
    img_res = 8192
    mapper = evaluater.mapper
    mesh_vert = batch['mesh_vert'][0]
    # f_global = mapper.encoding_global(mesh_vert)
    uv = mapper.encoding(mesh_vert)
    xyz_hat = mapper.decoding(uv)
    import pdb; pdb.set_trace()
    uv_space = torch.meshgrid(torch.linspace(0, 1, img_res), torch.linspace(0, 1, img_res))
    uv_space = torch.stack(uv_space, dim=-1)
    uv_space_lin = uv_space.reshape(-1, 2)
    with torch.no_grad():
        xyz_hat_in = mapper.decoding(uv_space_lin.cuda())

    colors = evaluater.test_tex_pifu(batch['img'], xyz_hat_in[None,], batch['betas'], batch['pose'], batch['scale'],
                                     batch['trans'])
    colors_1 = colors.reshape(img_res, img_res, 3)
    colors_2 = torch.Tensor(colors_1)
    uv_img = colors_2.permute(2, 0, 1)
    return uv_img, torch.cat([uv[:,1:2], 1 - uv[:,0:1]], dim=1), xyz_hat

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


        # uv_img, uv_mapping = xyz_to_uv_mapping('./mesh_vert_hat.pt', evaluater, batch)
        evaluater.mapper_load('./mapper_weight.pt')
        evaluater.mapper.eval()
        uv_img, uv_mapping, xyz_hat = xyz_to_uv_mapping_3(evaluater, batch)
        # uv_mapping = uv_mapping / 2 + 0.5

        # save_image(uv_img, './uv_made_4.png')
        import pdb;
        pdb.set_trace()
        mesh_color_hat = evaluater.test_tex_pifu(batch['img'], xyz_hat.unsqueeze(0), batch['betas'],
                                             batch['pose'], batch['scale'], batch['trans'])

        img_dir = batch['img_dir'][0]
        img_fname = os.path.split(img_dir)[1]
        mesh_fname = os.path.join(out_dir, 'results', img_fname[:-4] + '_tex.obj')
        uv_fname = mesh_fname.replace('.obj', '_uv.png')
        save_image(uv_img, uv_fname)

        obj_io.save_obj_data({'v': batch['mesh_vert'][0].squeeze().detach().cpu().numpy(),
                              'f': batch['mesh_face'][0].squeeze().detach().cpu().numpy(),
                              'vc': mesh_color_hat.squeeze()},
                             mesh_fname)
        obj_io.save_obj_data_with_mat({'v': batch['mesh_vert'][0].squeeze().detach().cpu().numpy(),
                              'f': batch['mesh_face'][0].squeeze().detach().cpu().numpy(),
                              'ft': batch['mesh_face'][0].squeeze().detach().cpu().numpy(),
                              'fn': batch['mesh_face'][0].squeeze().detach().cpu().numpy(),
                              'vt': uv_mapping.detach().cpu().numpy()},
                             mesh_fname.replace('.obj', '_vt.obj'),
                                      uv_fname)
    print('Testing Done. ')


if __name__ == '__main__':
    iternum=1
    input_image_dir = './results/test_data/'
    output_dir = './results/test_data/'
    # input_image_dir = './results/test_data_real/'
    # output_dir = './results/test_data_real/'
    # input_image_dir = './results/test_data_rendered/'
    # output_dir = './results/test_data_rendered/'

    #! NOTE: We recommend using this when accurate SMPL estimation is available (e.g., through external optimization / annotation)
    # main_test_with_gt_smpl(input_image_dir,
    #                        output_dir,
    #                        pretrained_checkpoint='./results/pamir_geometry/checkpoints/latest.pt',
    #                        pretrained_gcmr_checkpoint='./results/gcmr_pretrained/gcmr_2020_12_10-21_03_12.pt')

    #! Otherwise, use this function to predict and optimize a SMPL model for the input image
    # main_test_wo_gt_smpl_with_optm(input_image_dir,
    #                                output_dir,
    #                                pretrained_checkpoint='./results/pamir_geometry/checkpoints/latest.pt',
    #                                pretrained_gcmr_checkpoint='./results/gcmr_pretrained/gcmr_2020_12_10-21_03_12.pt',
    #                                iternum=iternum)

    main_test_texture(output_dir,
                      output_dir,
                      pretrained_checkpoint_pamir='./results/pamir_geometry/checkpoints/latest.pt',
                      pretrained_checkpoint_pamirtex='./results/pamir_texture/checkpoints/latest.pt')
