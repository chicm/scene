import settings
import os, cv2, glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
import random
#import transforms
import json

DATA_DIR = settings.DATA_DIR
TRAIN_IMG_DIR = settings.TRAIN_IMG_DIR
VAL_IMG_DIR = settings.VAL_IMG_DIR
TEST_IMG_DIR = settings.TEST_IMG_DIR

NUM_CLASSES = settings.NUM_CLASSES

def pil_load(img_path):
    with open(img_path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class SceneDataset(data.Dataset):
    def __init__(self, json_path, img_path, transform=None):
        if json_path is None:
            self.filenames = [fn.split('/')[-1] for fn in glob.glob(img_path + '/*.jpg')]
            self.has_label = False
        else:
            with open(json_path, 'r') as f:
                jdata = json.load(f)
            self.filenames = [jd['image_id'] for jd in jdata]
            self.labels = np.array([int(jd['label_id']) for jd in jdata])
            self.labels2 = np.array([[int(jd['label_id'])]*3 for jd in jdata])
            #self.labels = np.array([np.eye(NUM_CLASSES)[n] for n in self.labels], dtype=np.int)
            self.has_label = True

        self.num = len(self.filenames)
        self.transform = transform
        self.img_path = img_path
            
  
    def __getitem__(self, index):
        img = pil_load(self.img_path + '/' + self.filenames[index])
        if self.transform is not None:
            img = self.transform(img)

        if self.has_label:
            return img, self.labels[index], self.labels2[index], self.filenames[index]
        else:
            return img, self.filenames[index]

    def __len__(self):
        return self.num

def randomRotate(img):
    d = random.randint(0,4) * 90
    img2 = img.rotate(d, resample=Image.NEAREST)
    return img2


data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(256), 
        transforms.RandomSizedCrop(224),
        #transforms.Scale(224), 
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: randomRotate(x)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'trainv3': transforms.Compose([
        transforms.Scale(480), 
        transforms.RandomSizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: randomRotate(x)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'valid': transforms.Compose([
        transforms.Scale(256), 
        #transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validv3': transforms.Compose([
        transforms.Scale(480),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Scale(224),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'testv3': transforms.Compose([
        transforms.Scale(299),
        #transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

'''
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'valid']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'valid']}

dset_sizes = {x: len(dsets[x]) for x in ['train', 'valid']}
dset_classes = dsets['train'].classes
save_array(CLASSES_FILE, dset_classes)
'''

def get_train_loader(model, batch_size = 16, shuffle = True):
    if model.name.startswith('inception'):
        transkey = 'trainv3'
    else:
        transkey = 'train'
    if hasattr(model, 'batch_size'):
        batch_size = model.batch_size
    #train_v2.csv
    dset = SceneDataset(DATA_DIR+'/train/train.json', img_path=TRAIN_IMG_DIR, transform=data_transforms[transkey])
    dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    dloader.num = dset.num
    return dloader

def get_val_loader(model, batch_size = 16, shuffle = True):
    if model.name.startswith('inception'):
        transkey = 'validv3'
    else:
        transkey = 'valid'
    if hasattr(model, 'batch_size'):
        batch_size = model.batch_size
    #train_v2.csv
    dset = SceneDataset(DATA_DIR+'/val/val.json', img_path=VAL_IMG_DIR, transform=data_transforms[transkey])
    dloader = torch.utils.data.DataLoader(dset,  batch_size=batch_size, shuffle=shuffle, num_workers=4)
    dloader.num = dset.num
    return dloader

def get_test_loader(model, batch_size = 16, shuffle = False):
    if model.name.startswith('inception'):
        transkey = 'testv3'
    else:
        transkey = 'test'
    if hasattr(model, 'batch_size'):
        batch_size = model.batch_size

    dset = SceneDataset(None, img_path=TEST_IMG_DIR, transform=data_transforms[transkey])
    dloader = torch.utils.data.DataLoader(dset,  batch_size=batch_size, shuffle=shuffle, num_workers=4)
    dloader.num = dset.num
    return dloader

def get_tta_loader(model, batch_size = 16, shuffle = False):
    if model.name.startswith('inception'):
        transkey = 'trainv3'
    else:
        transkey = 'train'
    if hasattr(model, 'batch_size'):
        batch_size = model.batch_size

    dset = SceneDataset(None, img_path=TEST_IMG_DIR, transform=data_transforms[transkey])
    dloader = torch.utils.data.DataLoader(dset,  batch_size=batch_size, shuffle=shuffle, num_workers=4)
    dloader.num = dset.num
    return dloader

from utils import create_model

if __name__ == '__main__':
    model = create_model('res50')
    model.batch_size=4
    loader = get_train_loader(model)
    print(loader.num)
    for i, data in enumerate(loader):
        img, label, label2, fn = data
        if i > 0:
            break
        print(fn)
        #print(label)
        #print(label2)
        
    for i, data in enumerate(loader):
        img, label, label2, fn = data
        if i > 0:
            break
        print(fn)
        #print(label)
        #print(label2)
        

    loader = get_val_loader(model, shuffle = False)
    print(loader.num)
    for i, data in enumerate(loader):
        img, label, label2, fn = data
        if i > 0:
            break
        print(fn)
        #print(label)
        #print(label2)
    for i, data in enumerate(loader):
        img, label, label2, fn = data
        if i > 0:
            break
        print(fn)
        
    loader = get_tta_loader(model)
    print(loader.num)
    for i, data in enumerate(loader):
        img, fn = data
        print(fn)
        if i > 0:
            break
