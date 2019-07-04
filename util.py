from sklearn.metrics import roc_auc_score
from scipy.ndimage.interpolation import zoom, rotate
from scipy.ndimage.filters import gaussian_filter 
import scipy
from skimage import exposure

import re
import torch
import torchvision.transforms as transforms
import numpy as np
import time
import os
from PIL import Image

from torch.utils.data import DataLoader
from read_data_challenge import ChestXrayDataSet

from model.model import Densenet121,Densenet169,ResNet50,ResNet101
from model.inceptionresnetv2 import inceptionresnetv2
from model.xception import xception

def get_dataloader(test_list,grayscale,img_size):
    '''
    get dataloader for final submit
    '''
    if not grayscale:
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        test_dataset = ChestXrayDataSet(image_list_file=test_list,
                                        transform=transforms.Compose([
                                            transforms.Resize([img_size,img_size]),
                                            transforms.ToTensor(),
                                            normalize,
                                        ]),
                                        gray_scale = None)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                                 shuffle=False, num_workers=4, pin_memory=True)
        return test_loader

    elif grayscale == 1:
        normalize_gray = transforms.Normalize([0.5],
                                             [0.5])
        test_dataset_gray = ChestXrayDataSet(image_list_file=test_list,
                                            transform=transforms.Compose([
                                            transforms.Resize([img_size,img_size]),
                                            transforms.ToTensor(),
                                            normalize_gray,
                                        ]),
                                        gray_scale = 1)
        test_loader_gray = DataLoader(dataset=test_dataset_gray, batch_size=1,
                                 shuffle=False, num_workers=4, pin_memory=True)
        return test_loader_gray

def get_trained_model(model_name,resume):
    '''
    for final submit
    '''
    def get_element(resume_path):
        gray = None
        image_size = 224
        ori_d121 = None
        # 
        if 'gray_scale' in resume_path:
            gray = 1
        #
        image_size = int(resume_path.split('img_size_')[1].split('_epoch')[0])
        #
        if 'ori_d121' in resume_path:
            ori_d121 = 1

        return [gray, image_size, ori_d121]

    N_CLASSES = 5
    if model_name.lower() == 'densenet121':
        model = Densenet121(pretrained = False, classes = N_CLASSES, gray = get_element(resume)[0])
        if os.path.isfile(resume):
            checkpoint = torch.load(resume)
            model = load_pretrained_original_densenet_state(model, checkpoint, N_CLASSES, resume.lower(), get_element(resume)[2])
        else:
            print('original d121 model wrong!')
            exit(0)
    elif model_name.lower() == 'densenet169':
        model = Densenet169(pretrained = False, classes = N_CLASSES, gray = get_element(resume)[0])
    elif model_name.lower() == 'resnet50':
        model = ResNet50(pretrained = False, classes = N_CLASSES, gray = get_element(resume)[0], img_size = get_element(resume)[1])
    elif model_name.lower() == 'resnet101':
        model = ResNet101(pretrained = False, classes = N_CLASSES, gray = get_element(resume)[0], img_size = get_element(resume)[1])
    elif model_name.lower() == 'inceptionresnetv2':
        model = inceptionresnetv2(pretrained = None, classes = N_CLASSES, gray = get_element(resume)[0], img_size = get_element(resume)[1])
    elif model_name.lower() == 'xception':
        model = xception(pretrained = None, classes = N_CLASSES, gray = get_element(resume)[0])

    #print(get_element(resume)[0])

    if os.path.isfile(resume) and model_name.lower() != 'densenet121':
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint)
    elif checkpoint:
        pass
    else:
        print('no ckpt found!')
        exit()

    return model, get_element(resume)[0], get_element(resume)[1]

def load_pretrained_original_densenet_state(model,ckpt,classes_num, model_name, ori_d121):
    '''
    for final inference and submit
    '''
    if ori_d121 == 1 and 'gray' not in model_name:
        # to load state
        # Code modified from torchvision densenet source for loading from pre .4 densenet weights.
        state_dict = ckpt
        remove_data_parallel = True # Change if you don't want to use nn.DataParallel(model)

        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            ori_key =  key
            key = key.replace('densenet121.','')#.replace('module.','')
            #print('key',key)
            match = pattern.match(key)
            new_key = match.group(1) + match.group(2) if match else key
            new_key = new_key[7:] if remove_data_parallel else new_key
            #print('new_key',new_key)
            if '.0.' in new_key:
                new_key = new_key.replace('0.','')
            state_dict[new_key] = state_dict[ori_key]
            # Delete old key only if modified.
            if match or remove_data_parallel:
                del state_dict[ori_key]
        
        state_dict['classifier.0.weight'] = state_dict['classifier.weight']
        state_dict['classifier.0.bias'] = state_dict['classifier.bias']
        del state_dict['classifier.weight'], state_dict['classifier.bias']
        model.load_state_dict(state_dict)
    else:
        # for gray scale
        state_dict = ckpt  
        model.load_state_dict(state_dict)


    return model


def load_pretrained_state_avg_fc(model,ckpt,classes_num, model_name, ori_d121):
    if ori_d121 == 1:
        '''
        to load original d121 model
        drop the state of fc layer
        model with 2 fc layers and one dropout layer
        '''
        model = torch.nn.DataParallel(model).cuda()
        # to load state
        # Code modified from torchvision densenet source for loading from pre .4 densenet weights.
        state_dict = ckpt['state_dict']
        remove_data_parallel = True # Change if you don't want to use nn.DataParallel(model)

        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            ori_key =  key
            key = key.replace('densenet121.','')#.replace('module.','')
            #print('key',key)
            match = pattern.match(key)
            new_key = match.group(1) + match.group(2) if match else key
            new_key = 'module.'+new_key[7:] if remove_data_parallel else new_key
            #print('new_key',new_key)
            if '.0.' in new_key:
                new_key = new_key.replace('0.','')
            state_dict[new_key] = state_dict[ori_key]
            # Delete old key only if modified.
            if match or remove_data_parallel: 
                del state_dict[ori_key]

        now_model_state = model.state_dict() 
        for k in now_model_state:
            if k in state_dict:
                now_model_state[k] = state_dict[k]

        model.load_state_dict(now_model_state)

    elif 'denstnet' in model_name:
        state_weight =  ckpt['classifier.2.weight']
        state_bias =  ckpt['classifier.2.bias']
        state_weight = torch.mean(state_weight,dim=0,keepdim=True).repeat([classes_num,1])
        state_bias = torch.mean(state_bias,dim=0,keepdim=True).repeat([classes_num])
        ckpt['classifier.2.weight'] = state_weight
        ckpt['classifier.2.bias'] = state_bias

        model.load_state_dict(ckpt)

    elif 'resnet50' in model_name or 'resnet101' in model_name:
        state_weight =  ckpt['fc.3.weight']
        state_bias =  ckpt['fc.3.bias']
        state_weight = torch.mean(state_weight,dim=0,keepdim=True).repeat([classes_num,1])
        state_bias = torch.mean(state_bias,dim=0,keepdim=True).repeat([classes_num])
        ckpt['fc.3.weight'] = state_weight
        ckpt['fc.3.bias'] = state_bias            

        model.load_state_dict(ckpt)

    elif 'ception' in model_name:
        state_weight =  ckpt['last_linear.weight']
        state_bias =  ckpt['last_linear.bias']
        state_weight = torch.mean(state_weight,dim=0,keepdim=True).repeat([classes_num,1])
        state_bias = torch.mean(state_bias,dim=0,keepdim=True).repeat([classes_num])
        ckpt['last_linear.weight'] = state_weight
        ckpt['last_linear.bias'] = state_bias
 
        model.load_state_dict(ckpt)

    return model

def compute_AUCs(gt, pred, N_CLASSES):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

def crop_center(img, cropx, cropy):
    y,x = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx,:]

def data_augmentation(image):
    # Input should be ONE image with shape: (L, W, CH)
    #print(image.shape)
    options = ["gaussian_smooth", "rotate", "randomzoom", "adjust_gamma"]  
    # Probabilities for each augmentation were arbitrarily assigned 
    which_option = np.random.choice(options)
    #print(which_option)
    if which_option == "gaussian_smooth": 
        sigma = np.random.uniform(0.2, 1.0)
        image = gaussian_filter(image, sigma)
    
    elif which_option == "randomzoom": 
        # Assumes image is square
        min_crop = int(image.shape[1]*0.85)
        max_crop = int(image.shape[1]*0.95)
        crop_size = np.random.randint(min_crop, max_crop)
        crop = crop_center(image, crop_size, crop_size)
        #crop = transforms.CenterCrop(crop_size) 
        if crop.shape[-1] == 1: crop = crop[:,:,0] # for grayscale
        image = scipy.misc.imresize(crop, image.shape) 
    
    elif which_option == "rotate":
        angle = np.random.uniform(-15, 15)
        image = rotate(image, angle, reshape=False)
    
    elif which_option == "adjust_gamma": 
        #image = image / 255. 
        image = exposure.adjust_gamma(image, np.random.uniform(0.75,1.25))
        #image = image * 255. 
    if len(image.shape) == 2: image = np.expand_dims(image, axis=2)
    
    return Image.fromarray(image.astype('uint8')).convert('RGB')


def state_fusion(models):

    models_num = len(models)

    n0_state = models.pop(0).state_dict()

    n1_state = models.pop(0).state_dict()
    n2_state = models.pop(0).state_dict()
    #print(len(models))
    #exit(0)
    for k in n0_state:
        print('0',n0_state[k])
        print('1',n1_state[k])
        print('2',n2_state[k])
    exit(0)
    for k in empty_net_state:
        empty_net_state[k] *= (1.0/float(models_num)) 

    while models:
        net = models.pop(0)
        for k in empty_net_state:
            print('before',empty_net_state[k])
            empty_net_state[k] += net.state_dict()[k]*(1.0/float(models_num))
            print('after',empty_net_state[k])
        del net
        time.sleep(5)
    
    exit(0)