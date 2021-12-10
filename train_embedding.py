import os
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from models.resnet12 import resnet12
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from datasets.dataset import EmbeddingDataset

parser = argparse.ArgumentParser(description='PyTorch Pseduo-loss Confidence Metric for Semi-supervised Few-shot Learning.')
parser.add_argument('--data_path', type=str, default='./data', 
                                    help='Path with datasets.')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save_dir', type=str, default='./ckpt')
parser.add_argument('--dataset', type=str, default='miniimagenet',
                                    help='miniimagenet/tieredimagenet/CIFAR-FS/FC100.')
parser.add_argument('--lr', type=float, default=1e-1,
                        help='learning rate for training the feature extractor.')
parser.add_argument('--img_size', type=int, default=84)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

os.makedirs(args.save_dir, exist_ok=True)

if args.dataset == 'miniimagenet':
    num_classes = 64
    data_root = os.path.join(args.data_path, 'mini-imagenet/')
elif args.dataset == 'tieredimagenet':
    num_classes = 351
    data_root = os.path.join(args.data_path, 'tiered-imagenet/')
elif args.dataset == 'CIFAR-FS':
    num_classes = 64
    data_root = os.path.join(args.data_path, 'CIFAR-FS/')
elif args.dataset == 'FC100':
    num_classes = 60
    data_root = os.path.join(args.data_path, 'FC100/')
else:
    raise NameError


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_embedding():
    setup_seed(2333)
    source_set = EmbeddingDataset(data_root, args.img_size, 'train')
    source_loader = DataLoader(
        source_set, num_workers=4, batch_size=64, shuffle=True)
    test_set = EmbeddingDataset(data_root, args.img_size, 'val')
    test_loader = DataLoader(test_set, num_workers=16, batch_size=32, shuffle=False)

    model = resnet12(num_classes)
    model = model.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in range(120):
        model.train()
        scheduler.step(epoch)
        loss_list = []
        train_acc_list = []
        for images, labels in tqdm(source_loader, ncols=0):
            preds = model(images.cuda())
            loss = criterion(preds, labels.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            train_acc_list.append(preds.max(1)[1].cpu().eq(
                labels).float().mean().item())
        acc = []
        model.eval()
        for images, labels in test_loader:
            preds = model(images.cuda()).detach().cpu()
            preds = torch.argmax(preds, 1).reshape(-1)
            labels = labels.reshape(-1)
            acc += (preds==labels).tolist()
        acc = np.mean(acc)
        print('Epoch:{} Train-loss:{} Train-acc:{} Valid-acc:{}'.format(epoch, str(np.mean(loss_list))[:6], str(
            np.mean(train_acc_list))[:6], str(acc)[:6]))
        if acc > best_acc:
            best_acc = acc
            save_name = args.dataset + '-ResNet12.pth.tar'
            torch.save(model.state_dict(), os.path.join(args.save_dir, save_name))

def main():
    train_embedding()

        
if __name__ == '__main__':
    main()