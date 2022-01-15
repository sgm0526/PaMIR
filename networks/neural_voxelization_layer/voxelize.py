from __future__ import division, print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function

from torch.utils.cpp_extension import load
if os.path.exists('./cuda/voxelize_cuda.cu') and os.path.exists('./cuda/voxelize_cuda.cpp'):
    voxelize_cuda = load(
        name='voxelize_cuda',
        sources=['./cuda/voxelize_cuda.cpp', './cuda/voxelize_cuda.cu'],
        extra_ldflags=['-L/usr/local/cuda/targets/x86_64-linux/lib'],
        verbose=True)
elif os.path.exists('./neural_voxelization_layer/cuda/voxelize_cuda.cu') and \
    os.path.exists('./neural_voxelization_layer/cuda/voxelize_cuda.cpp'):
    voxelize_cuda = load(
        name='voxelize_cuda',
        sources=['./neural_voxelization_layer/cuda/voxelize_cuda.cpp',
                 './neural_voxelization_layer/cuda/voxelize_cuda.cu'],
        extra_ldflags=['-L/usr/local/cuda/targets/x86_64-linux/lib'],
        verbose=True)
else:
    raise IOError('Cannot find cuda/voxelize_cuda.cu and ./cuda/voxelize_cuda.cpp')


class VoxelizationFunction(Function):
    """
    Definition of differentiable voxelization function
    Currently implemented only for cuda Tensors
    """
    @staticmethod
    def forward(
        ctx, smpl_vertices, smpl_face_center, smpl_face_normal, 
        smpl_vertex_code, smpl_face_code, smpl_tetrahedrons,
        volume_res, sigma, smooth_kernel_size):
        """
        forward pass
        Output format: (batch_size, z_dims, y_dims, x_dims, channel_num) 
        """
        assert(smpl_vertices.size()[1] == smpl_vertex_code.size()[1])
        assert(smpl_face_center.size()[1] == smpl_face_normal.size()[1])
        assert(smpl_face_center.size()[1] == smpl_face_code.size()[1])
        ctx.batch_size = smpl_vertices.size()[0]
        ctx.volume_res = volume_res
        ctx.sigma = sigma
        ctx.smooth_kernel_size = smooth_kernel_size
        ctx.smpl_vertex_num = smpl_vertices.size()[1]
        ctx.device = smpl_vertices.device

        smpl_vertices = smpl_vertices.contiguous()
        smpl_face_center = smpl_face_center.contiguous()
        smpl_face_normal = smpl_face_normal.contiguous()
        smpl_vertex_code = smpl_vertex_code.contiguous()
        smpl_face_code = smpl_face_code.contiguous()
        smpl_tetrahedrons = smpl_tetrahedrons.contiguous()

        occ_volume = torch.cuda.FloatTensor(ctx.batch_size, ctx.volume_res, ctx.volume_res, ctx.volume_res).fill_(0.0)
        semantic_volume = torch.cuda.FloatTensor(ctx.batch_size, ctx.volume_res, ctx.volume_res, ctx.volume_res, 3).fill_(0.0)
        weight_sum_volume = torch.cuda.FloatTensor(ctx.batch_size, ctx.volume_res, ctx.volume_res, ctx.volume_res).fill_(1e-3)
        
        occ_volume, semantic_volume, weight_sum_volume = voxelize_cuda.forward_semantic_voxelization(
            smpl_vertices, smpl_vertex_code, smpl_tetrahedrons,
             occ_volume, semantic_volume, weight_sum_volume, sigma)

        def generate_point_grids(vol_res):
            x_coords = np.array(range(0, vol_res), dtype=np.float32)
            y_coords = np.array(range(0, vol_res), dtype=np.float32)
            z_coords = np.array(range(0, vol_res), dtype=np.float32)

            yv, xv, zv = np.meshgrid(x_coords, y_coords, z_coords)
            xv = np.reshape(xv, (vol_res, vol_res, vol_res, 1))
            yv = np.reshape(yv, (vol_res, vol_res, vol_res, 1))
            zv = np.reshape(zv, (vol_res, vol_res, vol_res, 1))
            xv = xv / vol_res - 0.5 + 0.5 / vol_res
            yv = yv / vol_res - 0.5 + 0.5 / vol_res
            zv = zv / vol_res - 0.5 + 0.5 / vol_res

            pts = np.concatenate([zv, yv, xv], axis=-1) #zyx
            pts = np.float32(pts)
            pts *= 2.0

            return pts

        #grid_3d = generate_point_grids(ctx.volume_res)
        #grid_3d = torch.from_numpy(grid_3d).unsqueeze(0).repeat(ctx.batch_size, 1, 1, 1, 1).cuda()
        #volume = torch.cat([occ_volume.unsqueeze(-1), grid_3d], -1)
        volume = semantic_volume

        return volume


class Voxelization(nn.Module):
    """
    Wrapper around the autograd function VoxelizationFunction
    """
    def __init__(self, smpl_vertex_code, smpl_face_code,
                 smpl_face_indices, smpl_tetraderon_indices,  
                 volume_res, sigma, smooth_kernel_size, batch_size):
        super(Voxelization, self).__init__()
        assert (len(smpl_face_indices.shape) == 2)
        assert (len(smpl_tetraderon_indices.shape) == 2)
        assert (smpl_face_indices.shape[1] == 3)
        assert (smpl_tetraderon_indices.shape[1] == 4)

        self.volume_res = volume_res
        self.sigma = sigma
        self.smooth_kernel_size = smooth_kernel_size
        self.batch_size = batch_size
        smpl_vertex_code_batch = np.tile(smpl_vertex_code, (batch_size, 1, 1))
        smpl_face_code_batch = np.tile(smpl_face_code, (batch_size, 1, 1))
        smpl_face_indices_batch = np.tile(smpl_face_indices, (batch_size, 1, 1))
        smpl_tetraderon_indices_batch = np.tile(smpl_tetraderon_indices, (batch_size, 1, 1))
        smpl_vertex_code_batch = torch.from_numpy(smpl_vertex_code_batch).contiguous()
        smpl_face_code_batch = torch.from_numpy(smpl_face_code_batch).contiguous()
        smpl_face_indices_batch = torch.from_numpy(smpl_face_indices_batch).contiguous()
        smpl_tetraderon_indices_batch = torch.from_numpy(smpl_tetraderon_indices_batch).contiguous()
        self.register_buffer('smpl_vertex_code_batch', smpl_vertex_code_batch)
        self.register_buffer('smpl_face_code_batch', smpl_face_code_batch)
        self.register_buffer('smpl_face_indices_batch', smpl_face_indices_batch)
        self.register_buffer('smpl_tetraderon_indices_batch', smpl_tetraderon_indices_batch)

    def forward(self, smpl_vertices):
        """
        Generate semantic volumes from SMPL vertices
        """
        assert(smpl_vertices.size()[0] == self.batch_size)
        self.check_input(smpl_vertices)
        smpl_faces = self.vertices_to_faces(smpl_vertices)
        smpl_tetrahedrons = self.vertices_to_tetrahedrons(smpl_vertices)
        smpl_face_center = self.calc_face_centers(smpl_faces)
        smpl_face_normal = self.calc_face_normals(smpl_faces)
        smpl_surface_vertex_num = self.smpl_vertex_code_batch.size()[1]
        smpl_vertices_surface = smpl_vertices[:, :smpl_surface_vertex_num, :]
        vol = VoxelizationFunction.apply(
            smpl_vertices_surface, smpl_face_center, smpl_face_normal, self.smpl_vertex_code_batch, 
            self.smpl_face_code_batch, smpl_tetrahedrons, 
            self.volume_res, self.sigma, self.smooth_kernel_size)
        return vol.permute((0, 4, 1, 2, 3)) # (bzyxc --> bcdhw)

    def vertices_to_faces(self, vertices):
        assert (vertices.ndimension() == 3)
        bs, nv = vertices.shape[:2]
        device = vertices.device
        face = self.smpl_face_indices_batch + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
        vertices_ = vertices.reshape((bs * nv, 3))
        return vertices_[face.long()]

    def vertices_to_tetrahedrons(self, vertices):
        assert (vertices.ndimension() == 3)
        bs, nv = vertices.shape[:2]
        device = vertices.device
        tets = self.smpl_tetraderon_indices_batch + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
        vertices_ = vertices.reshape((bs * nv, 3))
        return vertices_[tets.long()]

    def calc_face_centers(self, face_verts):
        assert len(face_verts.shape) == 4
        assert face_verts.shape[2] == 3
        assert face_verts.shape[3] == 3
        bs, nf = face_verts.shape[:2]
        face_centers = (face_verts[:, :, 0, :] + face_verts[:, :, 1, :] + face_verts[:, :, 2, :]) / 3.0 
        face_centers = face_centers.reshape((bs, nf, 3))
        return face_centers
    
    def calc_face_normals(self, face_verts):
        assert len(face_verts.shape) == 4
        assert face_verts.shape[2] == 3
        assert face_verts.shape[3] == 3
        bs, nf = face_verts.shape[:2]
        face_verts = face_verts.reshape((bs * nf, 3, 3))
        v10 = face_verts[:, 0] - face_verts[:, 1]
        v12 = face_verts[:, 2] - face_verts[:, 1]
        normals = F.normalize(torch.cross(v10, v12), eps=1e-5)
        normals = normals.reshape((bs, nf, 3))
        return normals

    def check_input(self, x):
        if x.device == 'cpu':
            raise TypeError('Voxelization module supports only cuda tensors')
        if x.type() != 'torch.cuda.FloatTensor':
            raise TypeError('Voxelization module supports only float32 tensors')