import sys
sys.path.append("./detect_face")
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import torchvision.io as io
import torchvision.transforms as F
import os
import json
from tqdm import tqdm
import csv
import timm
import wandb
import numpy as np
from mtcnn import MTCNN
from PIL import Image
import matplotlib.pyplot as plt
import torch
import json
import pandas as pd
import time


from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(prog="Automathon DeepFake - extract faces")
parser.add_argument("--data_dir", type=str)
parser.add_argument("--out_dir", type=str)
parser.add_argument("--data_csv", type=str)

args = parser.parse_args()
data_dir = args.data_dir    
out_dir = args.out_dir
data_csv = args.data_csv

device = torch.device('cuda')

def find_id(filename, pd_dataset):
    row = pd_dataset[pd_dataset['file'] == filename]
    if not row.empty:
        return row['id'].iloc[0]
    else:
        return None



def extract_image(input_dir, output_dir, data_csv):
    
    df = pd.read_csv(data_csv)

    video_files = [f for f in os.listdir(input_dir) if f.endswith('.pt')]

    mtcnn = MTCNN(image_size=256, margin=100, keep_all=True)

    names = []
    id_names = []
    
    for idx, video_file in enumerate(video_files):
        t1 = time.time()
        video_path = os.path.join(input_dir, video_file)
        video = torch.load(video_path)
        frame = video[0]

        # save the frames
        frame = frame.permute(1,2,0)

        outs = mtcnn(frame)

        if outs is None or outs.shape[0] > 1:
            continue

        out = outs[0].permute(1, 2, 0).numpy().astype(np.uint8)
        frame = Image.fromarray(out)
        frame.save(os.path.join(output_dir, f"{video_file[:-3]}.png"))
        names.append(f"{video_file[:-3]}.png")
        id_names.append(find_id(video_file, df))

        t2 = time.time()
        print("Image " , idx, flush=True)
        print((t2-t1), flush=True)

    
    df_ = pd.DataFrame({'names': names, 'id': id_names})
    df_.to_csv("metadata.csv", index=None)
            
extract_image(data_dir, out_dir, data_csv)