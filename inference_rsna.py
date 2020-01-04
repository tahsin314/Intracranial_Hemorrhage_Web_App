import numpy as np
import pydicom
import time
import torch
from torch import nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from apex import amp
import albumentations as A
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)
from albumentations.pytorch import ToTensor
import pretrainedmodels
import pickle
from apex import amp
import sys
import cv2
from scipy.misc import imresize
from PIL import Image
from skimage.transform import resize
from utils import get_dicom_meta, rescale_image, apply_window_policy, get_model

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
import cv2

from matplotlib import pyplot as plt

apex = False

class TestDataset(torch.utils.data.Dataset):

    def __init__(self, window_policy, path, transform=None):

        # self.transforms = factory.get_transforms(self.cfg)
        self.path = path
        self.transforms = transform
        
        # self.idx_list = idx_list
        self.window_policy = window_policy

    def __len__(self):
        return len(self.path)
        

    def __getitem__(self, item):
        
        dicom = pydicom.dcmread(self.path[item])
        dicom_data = get_dicom_meta(dicom)
        image = dicom.pixel_array
        image = rescale_image(image, dicom_data['RescaleSlope'], dicom_data['RescaleIntercept'])
        image = apply_window_policy(image, dicom_data, self.window_policy)
        image = resize(image, (512, 512))
        if self.transforms:
            image = self.transforms(image=image)['image']

        return np.swapaxes(image,0,2)


def pred(path, model, device = 'cuda'):
    ds = TestDataset(window_policy=2, path = [path])
    dl = DataLoader(ds)
    img = next(iter(dl))
    img = img.type(torch.FloatTensor).cuda()
    model.eval()
    pred = model(img)
    grad_cam_gen(model, img, path)
    return torch.sigmoid(pred).data.cpu().numpy()[0]

def grad_cam_gen(model, img, path, apex = False, device = 'cuda'):
    if apex:
        model, optim = amp.initialize(model, optim, opt_level='O1')
         
    configs = [dict(model_type='seresnext', arch=model, layer_name='layer4_2_se_module_fc2')]
    # with amp.disable_casts():
    for config in configs:
        config['arch'].to(device).eval()

    cams = [
    [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
        for config in configs]

    indices = {0: 'Epidural', 1: 'Intraparenchymal', 2: 'Intraventricular', 3: 'Subarachnoid', 4:'Subdural', 5:'any'}
        

    for _, gradcam_pp in cams:
        for cls_idx in range(6):
            mask_pp, _ = gradcam_pp(img, cls_idx)
            heatmap_pp, result_pp = visualize_cam(mask_pp, img)
            result_pp = result_pp.cpu().numpy()
            #convert image back to Height,Width,Channels
            result_pp = np.transpose(result_pp, (1,2,0))
            path = path.split('/')[-1].split('.')[0]
            plt.imsave('uploads/{}_grad_cam_{}.png'.format(path, indices[cls_idx]), np.transpose(result_pp, (1, 0, 2)))
            plt.show()  
