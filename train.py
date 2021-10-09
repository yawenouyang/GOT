import torch
import time
import numpy as np
import copy
import os
import torch.optim as optim
import torch.nn as nn
from util.ClassifierDataSet import ClassifierDataSet, OODClassifierDataSet
from model.Classifier import Classifier
from sklearn import metrics
import torch.nn.functional as F
from util.utils import load_args, set_seed
from transformers import AdamW


class Train():

    def __init__(self, config):
        
        # get ready
        self.args = load_args(config)

        set_seed(self.args.seed)

        # Load Data files for training
        self.train_loader = self.create_loader(self.args.train, True)
        self.valid_loader = self.create_loader(self.args.valid, False)
        self.test_loader = self.create_loader(self.args.test, False)
        if self.args.al_ratio > 0:
            self.args.ood = 'data/{}/ood.csv'.format(self.args.dataset)
            self.ood_loader = self.create_loader(self.args.ood, True)

        self.mdl = Classifier(self.args).to(self.args.device)
        if self.args.pretrained != '':
            model_param = torch.load(self.args.pretrained, map_location='cuda:{}'.format(self.args.gpu) if self.args.gpu != -1 else 'cpu')
            self.mdl.load_state_dict(model_param)

        self.criterion = nn.CrossEntropyLoss()
        
        bert_param = list(self.mdl.encoder.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_grouped_param = [
                {'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        self.bert_optimizer = AdamW(bert_grouped_param, lr=self.args.bert_lr)
        classifer_params = list(set(self.mdl.parameters()) - set(self.mdl.encoder.parameters()))
        self.classifier_optimizer = optim.Adam(classifer_params, lr=self.args.lr)

        # record best parameter on valid
        self.best_parameter = None
        self.softmax = nn.Softmax(dim=-1)
        
        print(self.args)

    def save_model(self, result):
        self.args.result = result
        args_dict = vars(self.args)
        if result < self.args.save_threshold:
            exit('Exit without saving model parameter.')

        dir_name = 'output/params/{}/{}'.format(self.args.dataset, result)
        is_exists = os.path.exists(dir_name)
        if is_exists:
            print('{} has existed'.format(dir_name))
            dir_name = 'output/params/{}/{}_{}'.format(self.args.dataset, result, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
            print('New dir name: {}'.format(dir_name))
        os.makedirs(dir_name)
        torch.save(self.best_parameter, '{}/params.pkl'.format(dir_name))
        with open('{}/hyper_parameter.txt'.format(dir_name), 'w') as f:
            for key, value in args_dict.items():
                f.write(str(key) + ': ' + str(value) + '\n')

    def create_loader(self, file_name, shuffle):
        batch_size = self.args.batch_size
        if file_name == self.args.ood:
            dataset = OODClassifierDataSet(file_name)
            batch_size = int(len(dataset.x) / len(self.train_loader.dataset.x) * batch_size)
        else:
            dataset = ClassifierDataSet(file_name)
            
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             drop_last=False)
        return loader

    def train_wo_ood(self):
        main_losses = []
        for x, y in self.train_loader:
            self.mdl.zero_grad()
            _, output = self.mdl(x)
            label = y.type(torch.LongTensor).to(self.args.device)
            loss = self.criterion(output, label)
            loss.backward()
            self.classifier_optimizer.step()
            self.bert_optimizer.step()
            main_losses.append(loss.item())
        
        return main_losses

    def train_with_ood(self):
        main_losses = []
        auxiliary_losses = []
        for ind, ood in zip(self.train_loader, self.ood_loader):
            self.mdl.zero_grad()
            ood_x, weights = ood[0], ood[2]
            x = ind[0] + ood_x
            _, output = self.mdl(x)
            ind_len = len(ind[0])

            # ce only for ind
            label = ind[1].type(torch.LongTensor).to(self.args.device)
            loss = self.criterion(output[:ind_len], label)
            main_losses.append(loss.item())

            # energy for both ind and ood
            Ec_out = -torch.logsumexp(output[ind_len:], dim=1)
            Ec_in = -torch.logsumexp(output[:ind_len], dim=1)
            Ec_in_loss = torch.pow(F.relu(Ec_in - self.args.m_in), 2).mean()
            if self.args.weight:
                weights = weights.to(self.args.device)
                Ec_out_loss = torch.pow(F.relu(self.args.m_out - Ec_out), 2)
                Ec_out_loss = torch.sum(Ec_out_loss * weights) / len(ood_x)
            else:
                Ec_out_loss = torch.pow(F.relu(self.args.m_out - Ec_out), 2).mean()
            auxiliary_loss = self.args.al_ratio * (Ec_in_loss + Ec_out_loss)
            
            auxiliary_losses.append(auxiliary_loss.item())
            loss += auxiliary_loss

            loss.backward()
            self.classifier_optimizer.step()
            self.bert_optimizer.step()

        return main_losses, auxiliary_losses


    @staticmethod
    def get_auc(y, pred):
        fpr, tpr, _ = metrics.roc_curve(y, pred, pos_label=1)
        fpr95 = 1  # init
        auroc = metrics.auc(fpr, tpr)
        for i in range(len(tpr)):
            if tpr[i] >= 0.95:
                fpr95 = fpr[i]
                break
        precision, recall, _ = metrics.precision_recall_curve(y, pred, pos_label=1)
        aupr_out = metrics.auc(recall, precision)

        pred = [-1 * one for one in pred]
        precision, recall, _ = metrics.precision_recall_curve(y, pred, pos_label=0)
        aupr_in = metrics.auc(recall, precision)

        return auroc, fpr95, aupr_out, aupr_in

    def msp(self, loader, calc_loss=True):
        all_scores = []
        all_y_index = []
        all_pred_index = []
        all_losses = []
        for x, y in loader:
            _, output = self.mdl(x)
            s_output = self.softmax(output)
            max_pred, predicted = torch.max(s_output.data, 1)
            max_pred = max_pred.tolist()
            pred_index = predicted.tolist()
            all_pred_index += pred_index
            all_scores += max_pred
            y_index = y.tolist()
            all_y_index += y_index
            if calc_loss:
                label = y.type(torch.LongTensor).to(self.args.device)
                loss = self.criterion(output, label)
                all_losses.append(loss.item())
        return all_y_index, all_pred_index, all_scores, all_losses
    
    def energy(self, loader):
        all_y_index = []
        all_pred_index = []
        all_scores = []
        # for x, y in tqdm(loader, desc='energy batch index'):
        for x, y in loader:
            _, output = self.mdl(x)
            output = torch.exp(output / self.args.T)
            energy = - torch.log(torch.sum(output, dim=-1)) * self.args.T
            energy = energy.tolist()
            all_scores += energy
            _, predicted = torch.max(output.data, 1)
            pred_index = predicted.tolist()
            all_pred_index += pred_index
            y_index = y.tolist()  # ood: -1
            all_y_index += y_index
        return all_y_index, all_pred_index, all_scores
    
    @torch.no_grad()
    def evaluate(self, loader):

        if loader == self.valid_loader:
            all_y_index, all_pred_index, _, all_losses = self.msp(loader)
            acc_msp = sum([all_y_index[i] == all_pred_index[i] for i in range(len(all_y_index))]) / len(all_y_index)
            print('Valid acc: {}, loss: {}'.format(acc_msp, np.mean(all_losses)))
            return acc_msp, np.mean(all_losses)
        else:
            all_y_index, all_pred_index, energy = self.energy(loader)
            all_y_is_ood = [1 if index == -1 else 0 for index in all_y_index]
            auroc, fpr95, aupr_out, aupr_in = self.get_auc(all_y_is_ood, energy)
            print('Energy auroc: {}, fpr95: {}, aupr in: {}, aupr out: {}'.format(auroc, fpr95, aupr_in, aupr_out))

            return aupr_out


    def train(self):
        max_acc = 0
        max_acc_epoch = -1
        print("Start training")
        for epoch in range(1, self.args.epoch + 1):
            self.mdl.train()            
            t0 = time.process_time()
            if self.args.al_ratio - 0 < 1e-5:
                main_losses = self.train_wo_ood()
                auxiliary_losses = [0]
            else:
                main_losses, auxiliary_losses = self.train_with_ood()
            t1 = time.process_time()
            print("[Epoch {}] Train Main Loss={} Auxiliary Loss={} T={}s".format(epoch, np.mean(main_losses), np.mean(auxiliary_losses), t1 - t0))
            self.mdl.eval()
            acc, _ = self.evaluate(self.valid_loader)
            if acc > max_acc:
                max_acc = acc
                max_acc_epoch = epoch
                self.best_parameter = copy.deepcopy(self.mdl.state_dict())
            elif self.args.early_stop != -1 and (epoch - max_acc_epoch == self.args.early_stop):
                print("Early Stopping")
                break
        print("Stop training")
        print("Max accuracy on valid is {}".format(max_acc))
    
    def test(self):
        self.mdl.eval()
        self.mdl.load_state_dict(self.best_parameter)
        aupr_out = self.evaluate(self.test_loader)
        self.save_model(aupr_out)
        
    
if __name__ == '__main__':
    exp = Train('configs/train.yaml')
    exp.train()
    exp.test()
