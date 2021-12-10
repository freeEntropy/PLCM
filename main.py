import os
import math
import torch
import scipy
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import scipy.io as scio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.ssGMM import ss_GaussianMixture
from sklearn.preprocessing import normalize
from models.resnet12 import resnet12
from datasets.dataset import EpisodeSampler, DataSet

parser = argparse.ArgumentParser(description='PyTorch Pseduo-loss Confidence Metric for Semi-supervised Few-shot Learning.')
parser.add_argument('--data_path', type=str, default='./data', 
                                    help='Path with datasets.')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=str, default='miniimagenet',
                                    help='miniimagenet/tieredimagenet/CIFAR-FS/FC100.')
parser.add_argument('--way', type=int, default=5,
                                    help='Number of classes per task, default: 5.')
parser.add_argument('--shot', type=int, default=1,
                                    help='Number of samples per class for support set, default: 1 or 5.')
parser.add_argument('--query', type=int, default=15,
                                    help='Number of samples per class for query set, default: 15.')
parser.add_argument('--unlabel', type=int, default=30,
                                    help='Number of unlabeled samples per class for unlabeled set, default: 30 or 50. 0 means transductive setting.')
parser.add_argument('--network', type=str, default='ResNet12',
                                    help='network for feature extraction.')
parser.add_argument('--resume', type=str, default='./ckpt',
                                    help='path to pre-trained network model.')
parser.add_argument('--train_episodes', type=int, default=600,
                                    help='Number of tasks sampled from training dataset for train, default: 600.')
parser.add_argument('--test_episodes', type=int, default=600,
                                    help='Number of tasks sampled from testing dataset for test, default: 600.')
parser.add_argument('--steps', type=int, default=10,
                                    help='steps of multi-step strategy for instance selection.')
parser.add_argument('--img_size', type=int, default=84)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

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

# laod model from resume
if args.resume is not None:
    resume_dir = args.resume + '/' + args.dataset + '-' + args.network + '.pth.tar'
    model = resnet12(num_classes)
    checkpoint = torch.load(resume_dir)
    
    model.load_state_dict(checkpoint)
    model = model.cuda()
    model.eval()
else:
    model = resnet12(num_classes)
    model = model.cuda()
    model.eval()

# set in semi-supervised few-shot learning
num_support = args.shot * args.way
num_query = args.query * args.way
num_unlabeled = args.unlabel * args.way

if args.unlabel != 0:
  num_select = int(args.unlabel / args.steps)
else:
  num_select = int(args.query * 2 / args.steps)

class Classifier(object):
    def __init__(self):
        self.initial_classifier()
    
    def fit(self, feature, label):
        self.classifier.fit(feature, label)
    
    def predict(self, feature, label=None):
        predicts = self.classifier.predict_proba(feature)
        if label is not None:
            pre_label = np.argmax(predicts, 1).reshape(-1)
            acc = np.mean(pre_label==label)
            return acc
        return predicts
    
    def initial_classifier(self):
        from sklearn.linear_model import LogisticRegression
        self.classifier = LogisticRegression(C=10, multi_class='auto', solver='lbfgs', class_weight='balanced', max_iter=1000)

def get_feature(model, input):
    batch_size = 32
    if input.shape[0] > batch_size:
        embed = []
        i = 0
        while i <= input.shape[0]-1:
            embed.append(
                model(input[i:i+batch_size].cuda(), return_feat=True).detach().cpu())
            i += batch_size
        embed = torch.cat(embed)
    else:
        embed = model(input.cuda(), return_feat=True).detach().cpu()
    assert embed.shape[0] == input.shape[0]
    return embed.numpy()

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, h

def progressiveManager():
    ssGMM_ParameterGroup, min_lossGroup, max_lossGroup = [], [], []
    for i in range(args.steps):
        ss_GMM_parameterGroup, min_lossGroup, max_lossGroup = pseudoLossDistributionLearning(i, ssGMM_ParameterGroup, min_lossGroup, max_lossGroup)
    pseudoLossConfidenceMetric(ss_GMM_parameterGroup, min_lossGroup, max_lossGroup)

def pseudoLossDistributionLearning(i, ssGMM_ParameterGroup, min_lossGroup, max_lossGroup):

    dataset = DataSet(data_root, 'train', args.img_size)
    sampler = EpisodeSampler(dataset.label, args.train_episodes,
                                args.way, args.shot, args.query, args.unlabel)
    trainloader = DataLoader(dataset, batch_sampler=sampler,
                            shuffle=False, num_workers=16, pin_memory=True)
    skc_lr = Classifier()
    print('process {}-step pseudo-loss distribution Learning'.format(i))
    query_select_0, query_select_1, query_unselect_0, query_unselect_1, unlabel_all, loss_assist_ALL = [], [], [], [], [], []
    for data in tqdm(trainloader, ncols=0):
        data = data.cuda()
        targets = torch.arange(args.way).repeat(args.shot+args.query+args.unlabel).long()
    
        support_data = data[:num_support]
        query_data = data[num_support:num_support+num_query]
        unlabel_data = data[num_support+num_query:]

        support_X = normalize(get_feature(model, support_data))
        support_y = targets[:num_support].cpu().numpy()

        query_X = normalize(get_feature(model, query_data))
        query_y = targets[num_support:num_support+num_query].cpu().numpy()

        if args.unlabel == 0:
            assist_X = query_X
            assist_y = query_y
        else:
            unlabel_X = normalize(get_feature(model, unlabel_data))
            assist_X = np.concatenate([unlabel_X, query_X])
            assist_y = np.concatenate([-1 * np.ones(unlabel_X.shape[0]), query_y])

        mix_X, mix_y = support_X, support_y
        select_query_index = np.array([])

        for i in range(len(ssGMM_ParameterGroup)):
            skc_lr.fit(mix_X, mix_y)

            pre_assist = skc_lr.predict(assist_X)
            assist_pseudoLabel = np.argmax(pre_assist, 1).reshape(-1)
            loss_assist = F.nll_loss(torch.log(torch.Tensor(pre_assist)), torch.Tensor(assist_pseudoLabel).long(), reduction='none')
            loss_assist = (loss_assist.numpy() - min_lossGroup[i]) / (max_lossGroup[i] - min_lossGroup[i])

            ssGMM_i = ss_GaussianMixture(ss_GMM_parameter=ssGMM_ParameterGroup[i])
            assist_InstancePredict = ssGMM_i.predict(loss_assist.reshape(-1,1), proba=True)
            assist_InstanceLabel = np.argmax(assist_InstancePredict, 1).reshape(-1)
            assist_InstanceConfidence = np.max(assist_InstancePredict[:,1::2], axis=1)

            select_assist_X, select_assist_y, select_index = [], [], []

            for class_item in range(args.way):

                index_class_i = np.where(assist_pseudoLabel == class_item)
                select_index_classItem = index_class_i[0][assist_InstanceConfidence[index_class_i].argsort()[::-1][:num_select*(i+1)]]
                select_assist_X.extend(assist_X[select_index_classItem])
                select_assist_y.extend(assist_pseudoLabel[select_index_classItem])
                select_index.extend(select_index_classItem)

            select_assist_X = np.array(select_assist_X)
            select_assist_y = np.array(select_assist_y)
            select_index = np.array(select_index)

            select_query_index = select_index[select_index > unlabel_data.size(0)]

            mix_X = np.concatenate([support_X, select_assist_X])
            mix_y = np.concatenate([support_y, select_assist_y])

        unselect_query_index = np.setdiff1d(np.arange(unlabel_data.size(0), len(assist_y)), select_query_index)

        skc_lr.fit(mix_X, mix_y)

        pre_assist = skc_lr.predict(assist_X)
        assist_pseudoLabel = np.argmax(pre_assist, 1).reshape(-1)
        loss_assist = F.nll_loss(torch.log(torch.Tensor(pre_assist)), torch.Tensor(assist_pseudoLabel).long(), reduction='none')

        if len(ssGMM_ParameterGroup) > 0:
            query_select_0.extend(np.array(loss_assist)[select_query_index][assist_pseudoLabel[select_query_index] != assist_y[select_query_index]])
            query_select_1.extend(np.array(loss_assist)[select_query_index][assist_pseudoLabel[select_query_index] == assist_y[select_query_index]])
        
        query_unselect_0.extend(np.array(loss_assist)[unselect_query_index][assist_pseudoLabel[unselect_query_index] != assist_y[unselect_query_index]])
        query_unselect_1.extend(np.array(loss_assist)[unselect_query_index][assist_pseudoLabel[unselect_query_index] == assist_y[unselect_query_index]])

        unlabel_all.extend(np.array(loss_assist)[np.arange(0, unlabel_data.size(0))])

        loss_assist_ALL.extend(np.array(loss_assist))
    
    max_lossItem = max(loss_assist_ALL)
    min_lossItem = min(loss_assist_ALL)

    query_select_0 = (np.array(query_select_0) - min_lossItem) / (max_lossItem - min_lossItem)
    query_select_1 = (np.array(query_select_1) - min_lossItem) / (max_lossItem - min_lossItem)
    query_unselect_0 = (np.array(query_unselect_0) - min_lossItem) / (max_lossItem - min_lossItem)
    query_unselect_1 = (np.array(query_unselect_1) - min_lossItem) / (max_lossItem - min_lossItem)
    unlabel_all = (np.array(unlabel_all) - min_lossItem) / (max_lossItem - min_lossItem)

    x_labeled = np.concatenate([query_select_1, query_select_0, query_unselect_1, query_unselect_0])
    y_labeled = np.concatenate([3*np.ones(len(query_select_1)), 2*np.ones(len(query_select_0)), np.ones(len(query_unselect_1)), np.zeros(len(query_unselect_0))])
    x_unlabeled = unlabel_all

    m_ssGaussianMixture = ss_GaussianMixture()
    ss_GMM_parameter = m_ssGaussianMixture.fit(x_labeled.reshape(-1,1), y_labeled, x_unlabeled.reshape(-1,1), beta = 0.50, tol = 0.1, max_iterations = 20, early_stop = 'True')

    # unpade parameter
    ssGMM_ParameterGroup.append(ss_GMM_parameter)
    min_lossGroup.append(min_lossItem)
    max_lossGroup.append(max_lossItem)

    return ssGMM_ParameterGroup, min_lossGroup, max_lossGroup

def pseudoLossConfidenceMetric(ss_GMM_parameterGroup, min_lossGroup, max_lossGroup):
    all_acc = []
    dataset = DataSet(data_root, 'test', args.img_size)
    sampler = EpisodeSampler(dataset.label, args.test_episodes,
                                args.way, args.shot, args.query, args.unlabel)
    testloader = DataLoader(dataset, batch_sampler=sampler,
                            shuffle=False, num_workers=16, pin_memory=True)
    skc_lr = Classifier()

    for data in tqdm(testloader, ncols=0):
        data = data.cuda()
        targets = torch.arange(args.way).repeat(args.shot+args.query+args.unlabel).long()

        support_data = data[:num_support]
        query_data = data[num_support:num_support+num_query]
        if args.unlabel == 0:
            unlabel_data = query_data
        else:
            unlabel_data = data[num_support+num_query:]

        support_X = normalize(get_feature(model, support_data))
        support_y = targets[:num_support].cpu().numpy()

        query_X = normalize(get_feature(model, query_data))
        query_y = targets[num_support:num_support+num_query].cpu().numpy()

        unlabel_X = normalize(get_feature(model, unlabel_data))

        mix_X, mix_y = support_X, support_y

        for i in range(len(ss_GMM_parameterGroup)):
            skc_lr.fit(mix_X, mix_y)

            pre_unlabel = skc_lr.predict(unlabel_X)
            unlabel_pseudoLabel = np.argmax(pre_unlabel, 1).reshape(-1)
            loss_unlabel = F.nll_loss(torch.log(torch.Tensor(pre_unlabel)), torch.Tensor(unlabel_pseudoLabel).long(), reduction='none')
            loss_unlabel = (loss_unlabel.numpy() - min_lossGroup[i]) / (max_lossGroup[i] - min_lossGroup[i])

            ssGMM_i = ss_GaussianMixture(ss_GMM_parameter=ss_GMM_parameterGroup[i])
            unlabel_InstancePredict = ssGMM_i.predict(loss_unlabel.reshape(-1,1), proba=True)
            unlabel_InstanceLabel = np.argmax(unlabel_InstancePredict, 1).reshape(-1)
            unlabel_InstanceConfidence = np.max(unlabel_InstancePredict[:,1::2], axis=1)

            select_unalebl_X, select_unlabel_y, select_index = [], [], []

            for class_item in range(args.way):
                index_class_i = np.where(unlabel_pseudoLabel == class_item)
                select_index_classItem = index_class_i[0][unlabel_InstanceConfidence[index_class_i].argsort()[::-1][:num_select*(i+1)]]
                select_unalebl_X.extend(unlabel_X[select_index_classItem])
                select_unlabel_y.extend(unlabel_pseudoLabel[select_index_classItem])
                select_index.extend(select_index_classItem)
            
            select_unalebl_X = np.array(select_unalebl_X)
            select_unlabel_y = np.array(select_unlabel_y)
            select_index = np.array(select_index)

            mix_X = np.concatenate([support_X, select_unalebl_X])
            mix_y = np.concatenate([support_y, select_unlabel_y])
        
        skc_lr.fit(mix_X, mix_y)

        query_acc = skc_lr.predict(query_X, query_y)
        all_acc.append(query_acc.tolist())

    m_true, h_true = mean_confidence_interval(all_acc)
    print('Evaluation results: {:4f}, {:4f}'.format(m_true, h_true))
    
def main():
    progressiveManager()


if __name__ == '__main__':
    main()
