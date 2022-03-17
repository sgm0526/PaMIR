from typing import List
import time

import torch
from torch.nn import functional as F

from .OSNet import osnet_ibn_x1_0
from .utils import load_pretrained_weights, compress_feature_vector
from .image_handler import normalize, resize, ndarray_to_tensor

import numpy as np


class OsNetEncoder:

    # Encoder constants
    PRETRAINED_MODEL = False
    LOSS = 'softmax'

    def __init__(self, input_width: int, input_height: int, weight_filepath: str, batch_size: str, num_classes: int, patch_height: int, patch_width: int, norm_mean: List[float], norm_std: List[float], GPU: bool):

        self._input_width = input_width
        self._input_height = input_height
        self._weight_filepath = weight_filepath
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.patch_size = (patch_height, patch_width)
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.GPU = GPU
        self._model = osnet_ibn_x1_0(
            num_classes=self.num_classes,
            loss=OsNetEncoder.LOSS,
            pretrained=OsNetEncoder.PRETRAINED_MODEL,
            use_gpu=self.GPU
        )
        self._model.eval()  # Set the torch model for evaluation
        self.weights_loaded = load_pretrained_weights(
            model=self._model,
            weight_path=self._weight_filepath
        )
        if self.GPU:
            self._model = self._model.cuda()

    def load_image(self, patch):
        ''' load image involves three processes: resizing, normalising and translating
        the np.ndarray into a torch.Tensor ready for GPU.

        :param patch: single detection patch, in np.ndarray format
        :return: resized and normalised single detection tensor
        '''

        if self.GPU:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # resized_patch = resize(
        #     img=patch,
        #     size=self.patch_size
        # )

        resized_patch = F.interpolate(patch[None,], size=self.patch_size)[0]
        # torch_tensor = ndarray_to_tensor(pic=resized_patch)

        normalized_tensor = normalize(
            tensor=resized_patch,
            mean=self.norm_mean,
            std=self.norm_std,
            inplace=False
        )

        # Transforms the normalised tensor to a cuda tensor or a cpu tensor wrt which device is available
        gpu_tensor = normalized_tensor.to(device)

        return gpu_tensor

    def get_features(self, image_patches):

        ''' Extract the 512 features associated to each detection
        :param image_patches: List[np.ndarray] of detections
        :return features: List[np.ndarray] of features associated to each detection
        '''
        features = list()

        for patch in image_patches:
            if patch is not None:
                initial_time = time.time()
                patch_gpu_tensor = self.load_image(patch)
                patch_features = self._model(patch_gpu_tensor)
                # Translating to np.ndarray avoids further issues with deepcopying torch.Tensors (in "tracker")
                feature = patch_features.cpu().detach()
                # numpy_features returns a list of a list of features, so we get the first entry as an action of flattening
                features.append(feature[0])
                # print(list(numpy_features[0])) # Debugging console out
                final_time = time.time()
                # print(f"[INFO] {numpy_features} appended")
                print(f"[PERFORMANCE] Features extracted: {1/(final_time-initial_time)} Hz")
            else:
                features.append(None)
                # print("[INFO] None appended")

        return features

    def compute_distance_matrix(self, input1, input2, metric='euclidean'):
        """A wrapper function for computing distance matrix.
        Args:
            input1 (torch.Tensor): 2-D feature matrix.
            input2 (torch.Tensor): 2-D feature matrix.
            metric (str, optional): "euclidean" or "cosine".
                Default is "euclidean".
        Returns:
            torch.Tensor: distance matrix.
        """

        # check input
        assert isinstance(input1, torch.Tensor)
        assert isinstance(input2, torch.Tensor)
        assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
            input1.dim()
        )
        assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
            input2.dim()
        )
        assert input1.size(1) == input2.size(1)

        if metric == 'euclidean':
            distmat = self.euclidean_squared_distance(input1, input2)
        elif metric == 'cosine':
            distmat = self.cosine_distance(input1, input2)
        else:
            raise ValueError(
                'Unknown distance metric: {}. '
                'Please choose either "euclidean" or "cosine"'.format(metric)
            )

        return distmat

    def euclidean_squared_distance(self, input1, input2):
        """Computes euclidean squared distance.
        Args:
            input1 (torch.Tensor): 2-D feature matrix.
            input2 (torch.Tensor): 2-D feature matrix.
        Returns:
            torch.Tensor: distance matrix.
        """
        m, n = input1.size(0), input2.size(0)
        mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
        mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat = mat1 + mat2
        distmat.addmm_(input1, input2.t(), beta=1, alpha=-2)
        return distmat

    def cosine_distance(self, input1, input2):
        """Computes cosine distance.
        Args:
            input1 (torch.Tensor): 2-D feature matrix.
            input2 (torch.Tensor): 2-D feature matrix.
        Returns:
            torch.Tensor: distance matrix.
        """
        input1_normed = F.normalize(input1, p=2, dim=1)
        input2_normed = F.normalize(input2, p=2, dim=1)
        distmat = 1 - torch.mm(input1_normed, input2_normed.t())
        return distmat