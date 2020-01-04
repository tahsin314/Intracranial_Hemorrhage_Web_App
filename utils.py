import numpy as np
import pydicom
import time
import torch
from torch import nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
import sys
import cv2
from scipy.misc import imresize
from PIL import Image
from skimage.transform import resize

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

def get_dicom_value(x, cast=int):
    if type(x) in [pydicom.multival.MultiValue, tuple]:
        return cast(x[0])
    else:
        return cast(x)


def cast(value):
    if type(value) is pydicom.valuerep.MultiValue:
        return tuple(value)
    return value


def get_dicom_raw(dicom):
    return {attr:cast(getattr(dicom,attr)) for attr in dir(dicom) if attr[0].isupper() and attr not in ['PixelData']}


def rescale_image(image, slope, intercept):
    return image * slope + intercept


def apply_window(image, center, width):
    image = image.copy()
    min_value = center - width // 2
    max_value = center + width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
    return image

def apply_window_tahsin(image, window_center, window_width, intercept, slope=1):
    img = image.copy()
    img = (img * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max

    return img



def get_dicom_meta(dicom):
    return {
        'PatientID': dicom.PatientID, # can be grouped (20-548)
        'StudyInstanceUID': dicom.StudyInstanceUID, # can be grouped (20-60)
        'SeriesInstanceUID': dicom.SeriesInstanceUID, # can be grouped (20-60)
        'WindowWidth': get_dicom_value(dicom.WindowWidth),
        'WindowCenter': get_dicom_value(dicom.WindowCenter),
        'RescaleIntercept': float(dicom.RescaleIntercept),
        'RescaleSlope': float(dicom.RescaleSlope), # all same (1.0)
    }



def apply_window_policy(image, row, policy):
    if policy == 1:
        image1 = apply_window(image, 40, 80) # brain
        image2 = apply_window(image, 80, 200) # subdural
        image3 = apply_window(image, row['WindowCenter'], row['WindowWidth'])
        image1 = (image1 - 0) / 80
        image2 = (image2 - (-20)) / 200
        image3 = (image3 - image3.min()) / (image3.max()-image3.min())
        image = np.array([
            image1 - image1.mean(),
            image2 - image2.mean(),
            image3 - image3.mean(),
        ]).transpose(1,2,0)

    elif policy == 2:
        image1 = apply_window(image, 40, 80) # brain
        image2 = apply_window(image, 80, 200) # subdural
        image3 = apply_window(image, 40, 380) # bone
        image1 = (image1 - 0) / 80
        image2 = (image2 - (-20)) / 200
        image3 = (image3 - (-150)) / 380
        image = np.array([
            image1 - image1.mean(),
            image2 - image2.mean(),
            image3 - image3.mean(),
        ]).transpose(1,2,0)
    else:
        raise

    return image

def get_transforms(cfg):
    def get_object(transform):
        if hasattr(A, transform.name):
            return getattr(A, transform.name)
        else:
            return eval(transform.name)
    transforms = [get_object(transform)(**transform.params) for transform in cfg.transforms]
    return A.Compose(transforms)

def get_model(model_name, n_output=6, pretrained='imagenet'):

    if model_name in ['resnext101_32x8d_wsl']:
        model = torch.hub.load('facebookresearch/WSL-Images', model_name)
        model.fc = torch.nn.Linear(2048, n_output)
        return model.cuda()

    try:
        model_func = pretrainedmodels.__dict__[model_name]
    except KeyError as e:
        model_func = eval(model_name)

    model = model_func(num_classes=1000, pretrained=pretrained)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Linear(
        model.last_linear.in_features,
        n_output,
    )
    return model.cuda()

def window_image(img, window_center, window_width, intercept, slope):
    """
    https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing
    """
    img = (img * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max

    return img


def get_first_of_dicom_field_as_int(x):
    """
    https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing
    """
    # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def get_windowing(data):
    """
    https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing
    """
    dicom_fields = [data[('0028', '1050')].value,  # window center
                    data[('0028', '1051')].value,  # window width
                    data[('0028', '1052')].value,  # intercept
                    data[('0028', '1053')].value]  # slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def convert_to_png(dcm_in, save_file_name, window_name=None):
    print(save_file_name)
    dcm = pydicom.dcmread(dcm_in)
    img = pydicom.read_file(dcm_in).pixel_array
    meta_data = get_dicom_meta(dcm)
    
    intercept = meta_data['RescaleIntercept']
    slope = meta_data['RescaleSlope']
    params = {'brain':{'center':40 , 'width':80}, 'subdural':{'center':80 , 'width':200}, 'bone':{'center':40 , 'width':380}}
    if window_name == None:
        window_center, window_width, intercept, slope = get_windowing(dcm)
        img = window_image(img, window_center, window_width, intercept, slope)
        cv2.imwrite(save_file_name.split('.')[0]+'.png', img)
    else:
        p = params[window_name]

        img = pydicom.read_file(dcm_in).pixel_array
        img = apply_window_tahsin(img, p['center'], p['width'], intercept, slope)
        cv2.imwrite(save_file_name.split('.')[0]+'_'+window_name+'.png', img)
