import os  
from pipeline import Pipeline
from config import Config
import glob
import imageio
import cv2
import numpy as np
import argparse
import json
from scipy import misc
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.metrics import accuracy_score
import pandas as pd
from tensorflow.python.client import device_lib
import tensorflow as tf
from keras import backend as K

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def get_parser():
    
    parser = argparse.ArgumentParser('eval')
    parser.add_argument('--inputPath', '-i', required=True, help="Path to the test set")
    parser.add_argument('--configPath', '-c', required=True, help="Path to the json config file")
    parser.add_argument('--dataname', '-f', required=True, help="Path of the CSV where the results will be saved")
    parser.add_argument('--channel', '-ch', type=int, required=True, help = "Channel to optimize either 0 for positve cells to the marker or 1 for positve cells to the marker.")
    parser.add_argument('--minth', '-m', type=int, required=True, help="Min threshold to evaluate")
    parser.add_argument('--maxth', '-x', type=int, required=True, help="Max threshold to evaluate")
    return parser

def read_labels(name,inputShape,imageShape):
    with open(name,'r') as f:
        temp = json.load(f)
        labels=[]
        previous_imshape = '' # Pour éviter d'afficher le format de chaque image
        for d in temp:
            if imageShape != previous_imshape :
                previous_imshape = imageShape
            if imageShape[0] ==255 and imageShape[1]==255:
                x=int(d['x'])
                y=int(d['y'])
            else:
                x=min(max(int(int(d['x'])*(inputShape[0]/imageShape[0])),0),inputShape[0])
                y=min(max(int(int(d['y'])*(inputShape[1]/imageShape[1])),0),inputShape[1])
            c=int(d['label_id'])-1
            labels.append([x,y,c])
        labels=np.array(labels)
    return labels




def metric(pred,label):
    distance_thr=20
    a = np.repeat(pred[...,np.newaxis], label.shape[-1], axis=2)
    b =label.reshape((1,label.shape[0],label.shape[1]))
    b= np.repeat(b,pred.shape[0],axis=0)
    c=np.subtract(a,b)
    d=np.sqrt(c[:,0,:]**2+c[:,1,:]**2)
    d=np.concatenate(((np.ones(label.shape[-1])*distance_thr)[np.newaxis,...],d),axis=0)
    e=np.argmin(d,axis=0)
    TP=np.unique(np.delete(e,np.argwhere(e==0))).shape[0]
    FP=pred.shape[0]-TP
    FN=label.shape[-1]-TP
    return [TP,FP,FN]



def eval(args=None):
    with tf.device('/gpu:0'):
        print(device_lib.list_local_devices())
        pre_ImPosl = []
        rec_ImPosl = []
        F1_ImPosl = []
        pre_ImNegl = []
        rec_ImNegl = []
        F1_ImNegl = []
        thr_l = []
        parser = get_parser()
        args = parser.parse_args(args)
        print(args)
        conf=Config()
        conf.load(args.configPath)
        for th in range(args.minth,args.maxth,10):
            if args.channel == 0:
                conf.update_thresold_0(th)
            else:
                conf.update_thresold_1(th)
            pipeline=Pipeline(conf) 
            data = [args.inputPath+"/"+f for f in os.listdir(args.inputPath) if '.jpg' in f]
            res=np.zeros((len(data),3,3))
            for i,d in enumerate(data):
                img=imageio.imread(d)
                labels=read_labels(d.replace(".jpg",".json"),conf.inputShape,img.shape).reshape((-1,3))
                img=misc.imresize(img,conf.inputShape)
                pred=pipeline.predict(img)
                if len(pred!=0):
                    for j,ch in enumerate(range(3)):
                        label=labels[np.argwhere(labels[:,2]==j)].reshape((-1,3))[:,:2].T
                        res[i,j,:]=metric(pred[np.argwhere(pred[:,2]==j)].reshape((-1,3))[:,:2],label)

            pre=np.sum(res[...,0],axis=0)/(np.sum(res[...,0],axis=0)+np.sum(res[...,1],axis=0)+0.00000001)
            rec=np.sum(res[...,0],axis=0)/(np.sum(res[...,0],axis=0)+np.sum(res[...,2],axis=0)+0.00000001)
            F1=2*(pre*rec)/(pre+rec+0.00000001)
            print(tabulate([['Immunopositive', pre[0],rec[0],F1[0]], ['Immunonegative',pre[1],rec[1],F1[1] ]], headers=['Class', 'Prec.','Rec.','F1']))
            pre_ImPosl.append(pre[0])
            rec_ImPosl.append(rec[0])
            F1_ImPosl.append(F1[0])
            
            pre_ImNegl.append(pre[1])
            rec_ImNegl.append(rec[1])
            F1_ImNegl.append(F1[1])
            thr_l.append(th)
            print(f'End evaluation with threshold 0 = {th}')
        # print('thr_l  ', thr_l , 'pre_ImPosl ', pre_ImPosl, 'Rec_ImPosl ', rec_ImPosl, 'F1_ImPos', F1_ImPosl,
        #    'pre_ImNegl ', pre_ImNegl,  ' rec_ImNegl ', rec_ImNegl, ' F1_ImNegl ', F1_ImNegl )
        df = pd.DataFrame()
        df['Threshold'] = thr_l
        df['Pre_ImPos'] = pre_ImPosl
        df['Rec_ImPos'] = rec_ImPosl
        df['F1_ImPos'] = F1_ImPosl
        df['Pre_ImNeg'] = pre_ImNegl
        df['Rec_ImNeg'] = rec_ImNegl
        df['F1_ImNeg'] = F1_ImNegl
        df.to_csv(args.dataname)

# Function to compute the Root Mean Square Error (RMSE)
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

if __name__ == "__main__":
   eval()



