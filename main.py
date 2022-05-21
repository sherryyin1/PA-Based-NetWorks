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
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from utils import CameDataset
from cifarnet import ResNept, PADensenet, PAEffnet, PAResnext, PARes2net
import os
import argparse
import matplotlib.pyplot as plt
from logger import create_logger
import json

from util import progress_bar


parser = argparse.ArgumentParser(description='PyTorch Camelyon17 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--scale', default=0.75, type=float)
parser.add_argument('--reprob', default=0.2, type=float)
parser.add_argument('--ra-m', default=9, type=int)
parser.add_argument('--ra-n', default=2, type=int)
parser.add_argument('--model_name', default="densenet", type=str)
parser.add_argument('--jitter', default=0.2, type=float)
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--hospital_num', default=0, type=int)
parser.add_argument('--experiment_num', default=1, type=int)
parser.add_argument('--output_dir', default="/data/gpfs/projects/punim1552/resnept/model", type=str)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
logger = create_logger(args.output_dir, args.model_name)
logger.info(args)
# Data
logger.info('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(96),
    #transforms.RandomResizedCrop(96, scale=(args.scale, 1.0), ratio=(1.0, 1.0)),

    transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandAugment(num_ops=args.ra_n, magnitude=args.ra_m),
    #transforms.ColorJitter(args.jitter, args.jitter, args.jitter),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #transforms.RandomErasing(p=args.reprob)

])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
data_path = "/data/gpfs/projects/punim1552/resnept/data/camelyon17_v1.0/"
#dataset = get_dataset(dataset="camelyon17", download=False)

annotations_file=data_path + "metadata.csv"
img_labels = pd.read_csv(annotations_file, index_col=0, dtype={"patient":"str"})
img_labels = img_labels.loc[img_labels["node"]==args.hospital_num]
np.random.seed(10) 

img_labels = img_labels.sample(frac=1)
train_size = int(len(img_labels) * 0.8)

train_img_labels = img_labels[:train_size]
test_img_labels = img_labels[train_size:]
#train_dataset = CameDataset(annotations_file=data_path + "metadata.csv", img_dir=data_path, is_train=True, transform=transform_train)
#test_dataset = CameDataset(annotations_file=data_path + "metadata.csv", img_dir=data_path, is_train=False, transform=transform_test)
train_dataset = CameDataset(train_img_labels, img_dir=data_path, is_train=True, transform=transform_train)
test_dataset = CameDataset(test_img_labels, img_dir=data_path, is_train=True, transform=transform_test)

#np.random.seed(10) 
"""
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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=4)

# Model
logger.info('==> Building model..')
def get_densenet():
    model = timm.create_model("densenet121", pretrained=True)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    return model

def get_resnet():
    model = timm.create_model("resnet34", pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model

def get_effnet():
    model = timm.create_model("tf_efficientnet_b4", pretrained=True)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    return model

def get_resnext50():
    model = timm.create_model("resnext50_32x4d", pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model

def get_res2net50():
    model = timm.create_model("res2net50_14w_8s", pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model
net = None
if args.model_name == "densenet":
    net = get_densenet()
elif args.model_name == "padensenet":
    net = PADensenet()
elif args.model_name == "resnet":
    net = get_resnet()
elif args.model_name == "paresnet":
    net = ResNept()
elif args.model_name == "res2net":
    net = get_res2net50()
else:
    net = PARes2net()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)


# Training
def train(epoch):
    logger.info('\nEpoch: %d' % epoch)

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    counter = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        counter += 1
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss / counter, 100.*correct/total


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    counter = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            counter += 1
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    """
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, f'/data/gpfs/projects/punim1552/resnept/model/{args.model_name} \
        /experiment_{args.experiment_num}_group{args.hospital_num}_{args.model_name}_{epoch}_{acc}.pth')
        best_acc = acc
    """
    return test_loss / counter, acc

total_train_acc = []
total_test_acc = []
total_train_loss = []
total_test_loss = []

for i in range(5):
    best_acc = 0
    for epoch in range(start_epoch, start_epoch+20):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        """
        log_stats = {"epoch":epoch,
                 "train_loss":train_loss,
                 "test_loss":test_loss,
                 "train_acc":train_acc,
                 "test_acc":test_acc}
        with open(os.path.join(args.output_dir, f"log_{args.model_name}.txt"), "a") as f:
        f.write(json.dumps(log_stats) + "\n")
        """
        if test_acc > best_acc:
            print('Saving..')

            state = {
            'net': net.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
            }
            torch.save(state, f'/data/gpfs/projects/punim1552/resnept/model/{args.model_name}/experiment_{i+1}_group{args.hospital_num}_{args.model_name}_{epoch}_{test_acc}.pth')
            best_acc = test_acc
        scheduler.step()


