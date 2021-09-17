import torch
from torch.autograd import grad
from tqdm import tqdm
import os
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from util.ClassifierDataset import ClassifierDataset
from torch.utils.data import ConcatDataset
import numpy as np
import math


class InF():

    def __init__(self, args, mlp, reps):
        self.args = args
        self.mlp = mlp
        self.reps = reps

        self.criterion = nn.CrossEntropyLoss()
        self.params = list(set(self.mlp.parameters()))        
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')

        if not os.path.isdir('output/weighting/{}/s_result'.format(self.args.dataset)):
            os.mkdir('output/weighting/{}/s_result'.format(self.args.dataset))
        
        if not os.path.isdir('output/weighting/{}/influence_result'.format(self.args.dataset)):
            os.mkdir('output/weighting/{}/influence_result'.format(self.args.dataset))

        self.s_dir = 'output/weighting/{}/s_result/{}'.format(self.args.dataset, self.args.aupr_out)
        self.influence_dir = 'output/weighting/{}/influence_result/{}'.format(self.args.dataset, self.args.aupr_out)
        
        if not os.path.isdir(self.s_dir):
            os.mkdir(self.s_dir)
        
        if not os.path.isdir(self.influence_dir):
            os.mkdir(self.influence_dir)

        self.splits = ['train', 'valid', 'ood']
        self.init_loader()

        test_num = int(len(self.loaders['valid'].dataset.y) / args.split_num)
        self.start_index = args.ith * test_num
        if args.ith == args.split_num - 1:
            self.end_index = len(self.loaders['valid'].dataset.y)
        else:
            self.end_index = (args.ith + 1) * test_num

    def init_loader(self):
        self.loaders = {}
        for split in self.splits:
            if split == 'ood':
                dataset = ClassifierDataset('output/generating/{}/ood_{}.csv'.format(self.args.dataset, self.args.seed))
            else:
                dataset = ClassifierDataset(self.args.train.replace('train', split))
            loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=False)
            self.loaders[split] = loader
        merge_dataset = ConcatDataset([self.loaders['train'].dataset, self.loaders['ood'].dataset])
        self.loaders['merge'] = torch.utils.data.DataLoader(dataset=merge_dataset, batch_size=1, shuffle=False, drop_last=False)

    def cal_s(self):
        if len(os.listdir(self.s_dir)) == len(self.loaders['valid'].dataset.y):
            print('{} has been calculated'.format(self.s_dir))
        else:
            for index, (_, y) in enumerate(self.loaders['valid']):
                if index >= self.start_index and index < self.end_index:
                    print('{}: {}/{}'.format(self.args.ith, index - self.start_index, self.end_index - self.start_index))
                    x = self.reps['valid'][index].unsqueeze(0)
                    x, y = x.to(self.args.device), y.to(self.args.device)
                    s = self.cal_s_single(x, y)
                    torch.save(s, '{}/{}'.format(self.s_dir, index))


    def cal_s_single(self, x_test, y_test):
        ihvp = None
        for i in range(self.args.repeat):
            v = self.grad_z(x_test, y_test)
            h_estimate = v.copy()

            for j, (_, y) in enumerate(self.loaders['merge']):
                x = self.reps['merge'][j].unsqueeze(0).to(self.args.device)
                y = y.to(self.args.device)
                output = self.mlp(x)
                loss = self.loss_with_energy(output, y)
                hv = self.hvp(loss, self.params, h_estimate)
                h_estimate = [_v + (1 - self.args.damp) * _h_e - _hv / self.args.scale for _v, _h_e, _hv in zip(v, h_estimate, hv)]
                h_estimate = [one.detach() for one in h_estimate]
                if j == self.args.recursion - 1:
                    break
                # if j % 50 == 0:
                #     print("Recursion at depth %s: norm is %f" % (j, np.linalg.norm(self.gather_flat_grad(h_estimate).cpu().numpy())))

            if ihvp == None:
                ihvp = [_a / self.args.scale for _a in h_estimate]
            else:
                ihvp = [_a + _b / self.args.scale for _a, _b in zip(ihvp, h_estimate)]
        
        return_ihvp = self.gather_flat_grad(ihvp)
        return_ihvp /= self.args.repeat

        return return_ihvp

    def gather_flat_grad(self, grads):
        views = []
        for p in grads:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)
    
    def loss_with_energy(self, output, y):
        energy = -torch.logsumexp(output, dim=1)
        if y.item() != -1:
            loss = self.criterion(output, y)
            energy_loss = torch.pow(F.relu(energy - self.args.m_in), 2)
            return loss + 0.1 * energy_loss
        else:  # ood
            energy_loss = torch.pow(F.relu(self.args.m_out - energy), 2)
            return energy_loss * 0.1

    def grad_z(self, x, y):
        output = self.mlp(x)
        loss = self.loss_with_energy(output, y)	
        return list(grad(loss, self.params, create_graph=True))

    def hvp(self, loss, model_params, v):
        grad1 = grad(loss, model_params, create_graph=True, retain_graph=True)
        Hv = grad(grad1, model_params, grad_outputs=v)
        return Hv

    def cal_influence(self):
        
        if self.args.dataset == 'clinc150':
            per_intent_num = 20  # valid number
        else:
            per_intent_num = 100  # snips

        if os.path.isdir(self.s_dir) and len(os.listdir(self.s_dir)) == len(self.loaders['valid'].dataset.y):
            print('Start loading s from {}'.format(self.s_dir))
            s_avgs = {}
            for index, (x, y) in enumerate(self.loaders['valid']):
                s_test = torch.load('{}/{}'.format(self.s_dir, index), \
                    map_location='cuda: {}'.format(self.args.gpu) if self.args.gpu != -1 else 'cpu')
                if index % per_intent_num == 0:
                    s_avg = s_test
                elif (index + 1) % per_intent_num == 0:
                    s_avg += s_test
                    s_avg /= per_intent_num
                    s_avgs[y.item()] = s_avg
                else:
                    s_avg += s_test
        else:
            print('Please calculate s first')
            return

        # ood
        data = pd.read_csv('output/generating/{}/ood_{}.csv'.format(self.args.dataset, self.args.seed))
        ind_labels = list(data['ind_index'])

        utts = []
        influences = []
        for index, (utt, y) in enumerate(tqdm(self.loaders['ood'], desc='grad index')):
            s_avg = s_avgs[ind_labels[index]]
            x = self.reps['ood'][index].unsqueeze(0).to(self.args.device)
            y = y.to(self.args.device)
            grad_z_vec = self.grad_z(x, y)
            grad_z_vec = self.gather_flat_grad(grad_z_vec)
            influence = -torch.dot(s_avg, grad_z_vec).item()

            # data['influence'].append(influence)
            # data['utt'].append(utt[0])
            utts.append(utt[0])
            influences.append(influence)
        
        max_influence = np.max(influences)
        min_influence = np.min(influences)
        normalized_influences = [1 / (1 + math.exp(self.args.gamma * influence / (max_influence - min_influence))) for influence in influences]
        
        # utts = ['{}[SPLIT]{}'.format(utt, influence) for utt, influence in zip(utts, normalized_influences)]
        data = pd.DataFrame({'utt': utts, 'influence': influences, \
            'weight': normalized_influences, 'index': [-1] * len(utts)})
        data.to_csv('output/weighting/{}/weight/{}/weight.csv'.format(self.args.dataset, self.args.aupr_out))
    

        # data = pd.read_csv(self.args.train)

        # utts = []
        # influences = []
        
        # for index, (utt, y) in enumerate(tqdm(self.loaders['train'], desc='grad index')):
        #     s_avg = s_avgs[y.item()]
        #     x = self.reps['train'][index].unsqueeze(0).to(self.args.device)
        #     y = y.to(self.args.device)
        #     grad_z_vec = self.grad_z(x, y)
        #     grad_z_vec = self.gather_flat_grad(grad_z_vec)
        #     influence = -torch.dot(s_avg, grad_z_vec).item()

        #     # data['influence'].append(influence)
        #     # data['utt'].append(utt[0])
        #     utts.append(utt[0])
        #     influences.append(influence)
        
        # data = pd.DataFrame({'utt': utts, 'influence': influences, \
        #     'intent': list(data['intent'])})
        # data.to_csv('output/weighting/{}/weight/{}/weight_ind.csv'.format(self.args.dataset, self.args.aupr_out))
    