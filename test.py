import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import timm
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from utils import CameDataset
from cifarnet import ResNept, PADensenet, PAEffnet, PAResnext, PARes2net
import os
import argparse
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
def get_densenet():
    model = timm.create_model("densenet121", pretrained=False)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    return model

def get_resnet():
    model = timm.create_model("resnet34", pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model

def get_res2net50():
    model = timm.create_model("res2net50_14w_8s", pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model
"""
dataset = get_dataset(dataset="camelyon17", download=False)
train_data = dataset.get_subset(
    "train",
    transform=transforms.Compose(
        [transforms.RandomCrop(96),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    ),
)
test_data = dataset.get_subset(
    "test",
    transform=transforms.Compose(
        [transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    ),
)
train_loader = get_train_loader("standard", train_data, batch_size=256)
test_loader = get_eval_loader("standard", test_data, batch_size=256)
"""
data_path = "/data/gpfs/projects/punim1552/resnept/data/camelyon17_v1.0/"
annotations_file=data_path + "metadata.csv"
img_labels = pd.read_csv(annotations_file, index_col=0, dtype={"patient":"str"})
img_labels = img_labels.loc[img_labels["node"]==4]
np.random.seed(10) #若不设置随机种子，则每次抽样的结果都不一样
img_labels = img_labels.sample(frac=1)
train_size = int(len(img_labels) * 0.8)

train_img_labels = img_labels[:train_size]
test_img_labels = img_labels[train_size:]
#train_dataset = CameDataset(annotations_file=data_path + "metadata.csv", img_dir=data_path, is_train=True, transform=transform_train)
#test_dataset = CameDataset(annotations_file=data_path + "metadata.csv", img_dir=data_path, is_train=False, transform=transform_test)
test_dataset = CameDataset(test_img_labels, img_dir=data_path, is_train=True, transform=transform_test)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=4)

model_path1 = "/data/gpfs/projects/punim1552/resnept/model/pares2net/experiment_1_group4_pares2net_18_99.60749681091158.pth"
model_path2 = "/data/gpfs/projects/punim1552/resnept/model/pares2net/experiment_2_group4_pares2net_2_99.63202826022962.pth"
model_path3 = "/data/gpfs/projects/punim1552/resnept/model/pares2net/experiment_3_group4_pares2net_19_99.6222156805024.pth"
model_path4 = "/data/gpfs/projects/punim1552/resnept/model/pares2net/experiment_4_group4_pares2net_1_99.63202826022962.pth"
model_path5 = "/data/gpfs/projects/punim1552/resnept/model/pares2net/experiment_5_group4_pares2net_19_99.55352762241193.pth"
#model_path6 = "/data/gpfs/projects/punim1552/resnept/model/pares2net/cross_res2net_34_86.68022667952124.pth"



def inference(model_path, model_name):
    checkpoint = torch.load(model_path)
    y_true = []
    y_pred = []
    if model_name == "densenet":
        model = get_densenet()
    elif model_name == "padensenet":
        model = PADensenet()
    elif model_name == "resnet":
        model = get_resnet()
    elif model_name == "paresnet":
        model = ResNept()
    elif model_name == "res2net":
        model = get_res2net50()
    elif model_name == "pares2net":
        model = PARes2net()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint["net"])
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader): 
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
            outputs = model(inputs)     
            _, predicted = outputs.max(1)
            y_true.extend(targets.cpu())
            y_pred.extend(predicted.cpu())
  

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")
    print(f"{model_name} inference....")
    print("accuracy", accuracy)
    print("precision", precision)
    print("recall", recall)
    print("f1", f1)

inference(model_path1, "pares2net")
inference(model_path2, "pares2net")
inference(model_path3, "pares2net")
inference(model_path4, "pares2net")
inference(model_path5, "pares2net")
#inference(model_path6, "pares2net")
print("done----------------------------------------------")

