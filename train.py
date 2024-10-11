from model import EffNetB0, Resnet50
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dataset import DeepFakeDataset
import argparse

parser = argparse.ArgumentParser(prog="Automathon DeepFake - trainer")
parser.add_argument("--data_dir", type=str)
parser.add_argument("--metadata_file", type=str)

args = parser.parse_args()
data_dir = args.data_dir    
metadata_file = args.metadata_file

# Hyper parameters
num_epochs = 200
num_classes = 1
batch_size = 64
learning_rate = 1e-4

def calculate_f1_score(predictions, targets):
    TP = ((predictions == 1) & (targets == 1)).sum().item()
    FP = ((predictions == 1) & (targets == 0)).sum().item()
    FN = ((predictions == 0) & (targets == 1)).sum().item()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1

def train(model, train_loader, test_loader):

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        [
            {"params": model.cnn.parameters(), "lr": learning_rate},
            {"params": model.linear.parameters(), "lr": 10*learning_rate}
        ],
        lr=learning_rate
    )

    wandb.init(
        entity="noerky",
        project="Automathon",
        config={
            "learning_rate": learning_rate,
            "architechture": model.name,
            "epochs": num_epochs
        }
    )

    total_step = len(train_loader)
    for epoch in range(num_epochs):

        totLoss = 0
        totScore = 0

        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to('cuda')
            labels = labels.to('cuda')
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(-1))
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            score = calculate_f1_score(nn.Sigmoid()(outputs)>0.5, labels.unsqueeze(-1))

            totLoss += loss.item()
            totScore += score
            
            if (i+1) % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Score: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item(), score), flush=True)
                
                wandb.log({
                    "loss": totLoss/20,
                    "score": totScore/20,
                })
                totLoss, totScore = 0, 0
                
        model.eval()
        totLoss = 0
        totScore = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images = images.to('cuda')
                labels = labels.to('cuda')

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(-1))

                score = calculate_f1_score(nn.Sigmoid()(outputs)>0.5, labels.unsqueeze(-1))

                totLoss += loss.item()
                totScore += score

                if (i+1) % 1 == 0:
                    print('Epoch Val [{}/{}], Step [{}/{}], Loss: {:.4f}, Score: {:.4f}' 
                        .format(epoch+1, num_epochs, i+1, len(test_loader), loss.item(), score), flush=True)
            

            wandb.log({
                "loss_val": totLoss/len(test_loader),
                "score_val": totScore/len(test_loader),
            })

            torch.save(model.state_dict(), f"{model.name}_{epoch+1}.pth")
        

    wandb.finish()


transform = transforms.Compose([
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    # transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = DeepFakeDataset(data_dir, metadata_file, transform=transform)
torch.manual_seed(521)
train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size


train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

test_dataset.transform = transform_val

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

model_effnet = EffNetB0(num_classes).to('cuda')
model_resnet = Resnet50(num_classes).to('cuda')

train(model_effnet, train_loader, test_loader)
train(model_resnet, train_loader, test_loader)