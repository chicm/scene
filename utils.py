import settings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os, glob
import cv2
import random
import argparse
import bcolz
import pandas as pd
import random
from PIL import Image
#from inception import inception_v3
from vgg import vgg19_bn, vgg16_bn
#from inceptionresv2 import inceptionresnetv2

MODEL_DIR = settings.MODEL_DIR
C = settings.NUM_CLASSES

w_files_training = []

def get_acc_from_w_filename(filename):
    try:
        stracc = filename.split('_')[-2]
        return float(stracc)
    except:
        return 0.

def load_best_weights(model):
    w_files = glob.glob(os.path.join(MODEL_DIR, model.name) + '_*.pth')
    max_acc = 0
    best_file = None
    saved_epoch = -1
    for w_file in w_files:
        try:
            stracc = w_file.split('_')[-2]
            epoch = w_file.split('_')[-3]
            acc = float(stracc)
            if acc > max_acc:
                best_file = w_file
                max_acc = acc
                saved_epoch = int(epoch)
            w_files_training.append((acc, w_file))
        except:
            continue
    if max_acc > 0:
        print('loading weight: {}'.format(best_file))
        model.load_state_dict(torch.load(best_file))
    return saved_epoch

def save_weights(acc, model, epoch, max_num=2):
    f_name = '{}_{}_{:.5f}_.pth'.format(model.name, epoch, acc)
    w_file_path = os.path.join(MODEL_DIR, f_name)
    if len(w_files_training) < max_num:
        w_files_training.append((acc, w_file_path))
        torch.save(model.state_dict(), w_file_path)
        return
    min = 10.0
    index_min = -1
    for i, item in enumerate(w_files_training):
        val_acc, fp = item
        if min > val_acc:
            index_min = i
            min = val_acc
    #print(min)
    if acc > min:
        torch.save(model.state_dict(), w_file_path)
        try:
            os.remove(w_files_training[index_min][1])
        except:
            print('Failed to delete file: {}'.format(w_files_training[index_min][1]))
        w_files_training[index_min] = (acc, w_file_path)

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]

def load_weights_file(model, w_file):
    model.load_state_dict(torch.load(w_file))

def create_res18(load_weights=False, freeze=False):
    model_ft = models.resnet18(pretrained=True)
    if freeze:
        for param in model_ft.parameters():
            param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, C)) #, nn.Softmax())
    model_ft = model_ft.cuda()

    model_ft.name = 'res18'
    model_ft.batch_size = 256
    return model_ft

def create_res34(load_weights=False, freeze=False):
    model_ft = models.resnet34(pretrained=True)
    if freeze:
        for param in model_ft.parameters():
            param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, C)) #, nn.Softmax())
    model_ft = model_ft.cuda()

    model_ft.name = 'res34'
    model_ft.batch_size = 128
    return model_ft

def create_res50(load_weights=False, freeze=False):
    model_ft = models.resnet50(pretrained=True)
    if freeze:
        for param in model_ft.parameters():
            param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, C)) #, nn.Softmax())
    model_ft = model_ft.cuda()

    model_ft.name = 'res50'
    model_ft.batch_size = 32
    return model_ft

def create_res101(load_weights=False, freeze=False):
    model_ft = models.resnet101(pretrained=True)
    if freeze:
        for param in model_ft.parameters():
            param.requires_grad = False
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, C))
    model_ft = model_ft.cuda()

    model_ft.name = 'res101'
    model_ft.batch_size = 32
    return model_ft

def create_res152(load_weights=False, freeze=False):
    res152 = models.resnet152(pretrained=True)
    if freeze:
        for param in res152.parameters():
            param.requires_grad = False
    num_ftrs = res152.fc.in_features
    res152.fc = nn.Sequential(nn.Linear(num_ftrs, C))
    res152 = res152.cuda()

    res152.name = 'res152'
    return res152

def create_dense161(load_weights=False, freeze=False):
    desnet_ft = models.densenet161(pretrained=True)
    if freeze:
        for param in desnet_ft.parameters():
            param.requires_grad = False
    num_ftrs = desnet_ft.classifier.in_features
    desnet_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, C))
    desnet_ft = desnet_ft.cuda()

    desnet_ft.name = 'dense161'
    #desnet_ft.batch_size = 32
    return desnet_ft

def create_dense169(load_weights=False, freeze=False):
    desnet_ft = models.densenet169(pretrained=True)
    if freeze:
        for param in desnet_ft.parameters():
            param.requires_grad = False
    num_ftrs = desnet_ft.classifier.in_features
    desnet_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, C))
    desnet_ft = desnet_ft.cuda()

    desnet_ft.name = 'dense169'
    #desnet_ft.batch_size = 32
    return desnet_ft

def create_dense121(load_weights=False, freeze=False):
    desnet_ft = models.densenet121(pretrained=True)
    if freeze:
        for param in desnet_ft.parameters():
            param.requires_grad = False
    num_ftrs = desnet_ft.classifier.in_features
    desnet_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, C))
    desnet_ft = desnet_ft.cuda()

    desnet_ft.name = 'dense121'
    desnet_ft.batch_size = 32
    return desnet_ft

def create_dense201(load_weights=False, freeze=False):
    desnet_ft = models.densenet201(pretrained=True)
    if freeze:
        for param in desnet_ft.parameters():
            param.requires_grad = False
    num_ftrs = desnet_ft.classifier.in_features
    desnet_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, C))
    desnet_ft = desnet_ft.cuda()
 
    desnet_ft.name = 'dense201'
    #desnet_ft.batch_size = 32
    return desnet_ft

def create_vgg19bn(load_weights=False, freeze=False):
    vgg19_bn_ft = vgg19_bn(pretrained=True)
    if freeze:
        for param in vgg19_bn_ft.parameters():
            param.requires_grad = False
    #vgg19_bn_ft.classifier = nn.Linear(25088, 3)
    vgg19_bn_ft.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, C))

    vgg19_bn_ft = vgg19_bn_ft.cuda()

    vgg19_bn_ft.name = 'vgg19bn'
    vgg19_bn_ft.max_num = 1
    #vgg19_bn_ft.batch_size = 32
    return vgg19_bn_ft

def create_vgg16bn(load_weights=False, freeze=False):
    vgg16_bn_ft = vgg16_bn(pretrained=True)
    if freeze:
        for param in vgg16_bn_ft.parameters():
            param.requires_grad = False
    #vgg16_bn_ft.classifier = nn.Linear(25088, 3)
    vgg16_bn_ft.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, C))

    vgg16_bn_ft = vgg16_bn_ft.cuda()

    vgg16_bn_ft.name = 'vgg16bn'
    vgg16_bn_ft.max_num = 1
    #vgg16_bn_ft.batch_size = 32
    return vgg16_bn_ft

def create_inceptionv3(load_weights=False, freeze=False):
    incept_ft = models.inception_v3(pretrained=True)
    if freeze:
        for param in incept_ft.parameters():
            param.requires_grad = False
    num_ftrs = incept_ft.fc.in_features
    incept_ft.fc = nn.Sequential(nn.Linear(num_ftrs, C))
    incept_ft.aux_logits=False
    incept_ft = incept_ft.cuda()

    incept_ft.name = 'inceptionv3'
    incept_ft.batch_size = 32
    return incept_ft

def create_inceptionresv2(load_weights=False, freeze=False):
    model_ft = inceptionresnetv2(pretrained=True)
    num_ftrs = model_ft.classif.in_features
    model_ft.classif = nn.Sequential(nn.Linear(num_ftrs, C))
    model_ft = model_ft.cuda()

    model_ft.name = 'inceptionresv2'
    model_ft.batch_size = 8
    return model_ft

def create_model(model_name, freeze=False):
    create_func = 'create_' + model_name

    model = eval(create_func)(freeze=freeze)
    if not hasattr(model, 'batch_size'):
        model.batch_size = 16
    return model
