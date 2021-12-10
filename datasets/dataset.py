import os
import os.path as osp
import pickle
import csv
import collections

import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

class DataSet(Dataset):
    def __init__(self, data_root, setname, img_size, multiT=False):
        self.img_size = img_size
        self.multiT = multiT
        csv_path = osp.join(data_root, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(data_root, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        
        if setname=='test' or setname=='val':
            self.transform = transforms.Compose([
                                               transforms.Resize((img_size, img_size)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ])
        else:
            self.transform = transforms.Compose([
                                            transforms.RandomResizedCrop((img_size, img_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        ])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if i == -1:
            return torch.zeros([3, self.img_size, self.img_size]), 0
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image

class EpisodeSampler():
    def __init__(self, label, n_episodes, n_way, n_shot, n_query, n_unlabel):
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.n_per = n_shot + n_query + n_unlabel

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_episodes
    
    def __iter__(self):
        for i_batch in range(self.n_episodes):
            batch = []
            classes = torch.randperm(len(self.m_ind))
            select_classes = classes[:self.n_way]
            for c in select_classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                cls_batch = l[pos]
                batch.append(cls_batch)
            batch = torch.stack(batch).t().reshape(-1)
            targets = torch.arange(self.n_way).repeat(self.n_per).long()
            yield batch

filenameToPILImage = lambda x: Image.open(x).convert('RGB')

def loadSplit(splitFile):
            dictLabels = {}
            with open(splitFile) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader, None)
                for i,row in enumerate(csvreader):
                    filename = row[0]
                    label = row[1]
                    if label in dictLabels.keys():
                        dictLabels[label].append(filename)
                    else:
                        dictLabels[label] = [filename]
            return dictLabels


class EmbeddingDataset(Dataset):

    def __init__(self, dataroot, img_size, type = 'train'):
        self.img_size = img_size
        # Transformations to the image
        if type=='train':
            self.transform = transforms.Compose([filenameToPILImage,
                                                transforms.Resize((img_size, img_size)),
                                                transforms.RandomCrop(img_size, padding=8),
                                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                                ])
        else:
            self.transform = transforms.Compose([filenameToPILImage,
                                                transforms.Resize((img_size, img_size)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])

        
        self.ImagesDir = os.path.join(dataroot,'images')
        self.data = loadSplit(splitFile = os.path.join(dataroot,'train' + '.csv'))

        self.data = collections.OrderedDict(sorted(self.data.items()))
        keys = list(self.data.keys())
        self.classes_dict = {keys[i]:i  for i in range(len(keys))} # map NLabel to id(0-99)

        self.Files = []
        self.belong = []

        for c in range(len(keys)):
            num = 0
            num_train = int(len(self.data[keys[c]]) * 9 / 10)
            for file in self.data[keys[c]]:
                if type == 'train' and num <= num_train:
                    self.Files.append(file)
                    self.belong.append(c)
                elif type=='val' and num>num_train:
                    self.Files.append(file)
                    self.belong.append(c)
                num = num+1


        self.__size = len(self.Files)

    def __getitem__(self, index):

        c = self.belong[index]
        File = self.Files[index]

        path = os.path.join(self.ImagesDir,str(File))
        try:
            images = self.transform(path)
        except RuntimeError:
            import pdb;pdb.set_trace()
        return images,c

    def __len__(self):
        return self.__size