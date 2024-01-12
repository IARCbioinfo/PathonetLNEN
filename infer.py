from pipeline import Pipeline
from config import Config
import os  
import glob
import imageio
import cv2
import argparse
import numpy as np
import json
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=DeprecationWarning)
def get_parser():
    
    parser = argparse.ArgumentParser('Inference')
    parser.add_argument('--inputPath', '-i', required=True, help="Path to the test set")
    parser.add_argument('--outputPath', '-o', required=True, help="Path to the folder where the results will be saved")
    parser.add_argument('--configPath', '-c', required=True, help="Path to the json config file")
    parser.add_argument('--save_numpy', action='store_true', help="Save the inference results for each tile in npy format")
    parser.add_argument('--visualization', action='store_true', help="Save the tile annotated in jpg format for visualization")
    return parser

def visualizer(img,points):
    r=1
    colors=[
            (255,0,0),
            (0,255,0),
            (0,0,255)
            ]
    image=np.copy(img)
    for p in points:
        x,y,c=p[0],p[1],p[2]
        cv2.circle(image, (int(x), int(y)), int(r), colors[int(c)], 2)
    return image


def infer(args=None):
    parser = get_parser()
    args = parser.parse_args(args)
    conf=Config()
    conf.load(args.configPath)
    pipeline=Pipeline(conf)
 
    if os.path.isdir(args.inputPath):
        #print('args.inputPath  = ', args.inputPath, os.listdir(args.inputPath))
        patient_ids_list = os.listdir(args.inputPath)
        for patient_id in patient_ids_list:
            print('patient_id' , patient_id)
            os.makedirs(os.path.join(args.outputPath, patient_id), exist_ok = True)
            os.makedirs(os.path.join(args.outputPath, patient_id, 'accept'), exist_ok = True)
            data = [ os.path.join(args.inputPath, patient_id, 'accept', f)  for f in os.listdir(os.path.join(args.inputPath, patient_id, 'accept')) if '.jpg' in f]
            for d in data:
                img=imageio.imread(d)
                pred_cell, pred_neg, pred_pos =pipeline.predict(img, exhaustive=True)
                if args.save_numpy:
                    np.save(os.path.join(args.outputPath, patient_id, 'accept',d.split("/")[-1][:-4] + "_pred_neg_mask.npy"), pred_neg)
                    np.save(os.path.join(args.outputPath, patient_id, 'accept',d.split("/")[-1][:-4] + "_pred_pos_mask.npy"), pred_pos)
                if args.visualization:
                    output=visualizer(img,pred_cell)
                    imageio.imwrite(os.path.join(args.outputPath, patient_id, 'accept',d.split("/")[-1][:-4] + "viz.jpg"),output)
                list_cells_for_json = []
                for cell in pred_cell:
                    c_cell= {}
                    c_cell['x'] = cell[0]
                    c_cell['y'] = cell[1]
                    c_cell['label_id'] = int(cell[2]) + 1
                    list_cells_for_json.append(c_cell)
                json_pred_fname = os.path.join(args.outputPath, patient_id, 'accept', d.split("/")[-1][:-3]+'json')
                with open(json_pred_fname, 'w') as f:
                    json.dump(list_cells_for_json, f)

if __name__ == "__main__":
   infer()