import pandas as pd
import os
from dataset import DeepFakeDatasetTest
from torchvision import transforms
import torch
from model import EffNetB0, Resnet50
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(prog="Automathon DeepFake - submit")
parser.add_argument("--data_dir", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--data_csv", type=str)

args = parser.parse_args()
data_dir = args.data_dir    
model_name = args.model_name
data_csv = args.data_csv

def find_id(filename, pd_dataset):
    row = pd_dataset[pd_dataset['file'] == filename]
    if not row.empty:
        return row['id'].iloc[0]
    else:
        return None
    

transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = DeepFakeDatasetTest(data_dir, transform=transform)

device="cuda"
num_classes=1
model = EffNetB0(num_classes).to(device)
model.load_state_dict(torch.load(model_name, map_location=torch.device(device)))


files_name = os.listdir(data_dir)
res = []
ids = []

dataset = pd.read_csv(data_csv)

model.eval()
with torch.no_grad():
    for i, (images, name) in tqdm(enumerate(full_dataset)):
        images = images.to(device)
        outputs = model(images.unsqueeze(0))
        pred = outputs.squeeze()
        #print(outputs.squeeze(0), pred)
        b = (pred.item()>0.5)*1
        res.append(b)
        ids.append(find_id(name, dataset))

df = pd.DataFrame({"id": ids, "label": res})
df.to_csv("submission.csv",index=False)