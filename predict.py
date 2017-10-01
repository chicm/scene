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
from utils import save_array, load_array, get_acc_from_w_filename
from utils import create_model
from scenedataset import get_tta_loader
from tqdm import tqdm

data_dir = settings.DATA_DIR
MODEL_DIR = settings.MODEL_DIR

RESULT_DIR = data_dir + '/results'
batch_size = 16

w_file_matcher = ['dense161*pth', 'dense201*pth','dense169*pth','dense121*pth','inceptionv3*pth',
    'res50*pth','res101*pth', 'res152*pth', 'vgg16*pth', 'vgg19*pth']

w_file_matcher = ['res34*pth']

def make_preds(net):
    loader = get_tta_loader(net)
    preds = []
    filenames = []
    m = nn.Softmax()
    net.eval()
    for i, (imgs, fn) in tqdm(enumerate(loader, 0)):
        inputs = Variable(imgs.cuda())
        outputs = net(inputs)
        pred = m(outputs).data.cpu().tolist()
        preds.extend(pred)
        filenames.extend(fn)
    return np.array(preds), filenames

def save_pred_results(filenames, preds, outfile):
    df1 = pd.DataFrame(filenames, columns=['filename'])
    df2 = pd.DataFrame(preds)
    df = df1.join(df2, how='outer')
    df.to_csv(outfile, index=False)

def tta_preds(net, num):
    print('tta {} predict...'.format(num))
    all_preds = [None] * num
    for i in range(num):
        all_preds[i], filenames = make_preds(net)
        #save_pred_results(filenames, all_preds[i], RESULT_DIR+'/test1.csv')
    return np.mean(all_preds, axis=0), filenames

def ensemble():
    for match_str in w_file_matcher:
        os.chdir(MODEL_DIR)
        w_files = glob.glob(match_str)
        #print('cur:' + os.getcwd())
        for w_file in w_files:
            full_w_file = MODEL_DIR + '/' + w_file
            mname = w_file.split('_')[0]
            print(full_w_file)
            model = create_model(mname)
            model.load_state_dict(torch.load(full_w_file))

            pred, filenames = tta_preds(model, 15)
            save_pred_results(filenames, pred, RESULT_DIR + '/' + w_file.split('.')[0] + '.csv')
            del model    

import json

def ensemble_csv(csv_dir, outfile):
    csv_files = glob.glob(csv_dir + '/*.csv')
    
    preds = [None] * len(csv_files)
    filenames = None

    for i, fn in enumerate(csv_files):
        print(fn)
        df = pd.read_csv(fn)
        preds[i] = df.values[:, 1:]
        filenames = df.values[:, 0]

    mean_preds = np.mean(preds, axis=0)
    res = np.argsort(mean_preds)[:, -3:]
    res = np.flip(res, axis=1)
    jsondata = []
    for i, fname in enumerate(filenames):
        jsondata.append({'image_id': fname, 'label_id': res[i].tolist()})

    with open(RESULT_DIR + '/' + outfile, 'w') as f:
        json.dump(jsondata, f, ensure_ascii=False, indent=4)    


parser = argparse.ArgumentParser()
parser.add_argument("--pred", action='store_true', help="ensemble predict")
parser.add_argument("--sub", nargs=1, help="generate submission file")

args = parser.parse_args()
if args.pred:
    ensemble()
    print('done')
if args.sub:
    print('generating submision file...')
    ensemble_csv(RESULT_DIR, args.sub[0])
    print('done')
    print('Please find submisson file at: {}'.format(RESULT_DIR+'/'+args.sub[0]))
