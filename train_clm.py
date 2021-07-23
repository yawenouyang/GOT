from util.utils import load_args, set_seed, init_tokenizer
from transformers import GPT2LMHeadModel
from util.CLMDataset import CLMDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model.ConditionalLM import ConditionalLM
import os
import copy


class Train():

    def __init__(self, config):
        
        self.args = load_args(config)
        set_seed(self.args.seed)
        self.tokenizer = init_tokenizer(self.args.gpt_path)
        self.init_embedding()  # clm use the same word embedding with GPT-2

        # Load Data files for training
        self.train_loader = self.create_loader(self.args.train, True)
        self.valid_loader = self.create_loader(self.args.valid, False)
        
        self.mdl = ConditionalLM(self.args.gpu, self.args.dataset, self.args.label_num).to(self.args.device)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=50256)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.mdl.parameters()), lr=0.001)
        
        self.best_parameter = None
    
    def init_embedding(self):
        if not os.path.exists('output/params/{}/embedding'.format(self.args.dataset)):
            if not os.path.isdir('output/params/{}'.format(self.args.dataset)):
                os.mkdir('output/params/{}'.format(self.args.dataset))
            gpt2 = GPT2LMHeadModel.from_pretrained(self.args.gpt_path, return_dict=True).to(self.args.device)
            embedding = gpt2.get_input_embeddings().weight.data
            torch.save(embedding, 'output/params/{}/embedding'.format(self.args.dataset))

    def create_loader(self, file_name, shuffle):
        dataset = CLMDataset(file_name, self.tokenizer, self.args.device)
        loader = DataLoader(dataset=dataset, batch_size=self.args.batch_size, shuffle=shuffle, drop_last=False)
        return loader

    def train(self):
        J_tr = []
        J_val = []
        best_valid_loss = 0x7fffffff
        for epoch in range(self.args.epoch):
            tr_loss_epoch = []
            for X_in, X_out, _, _, y, _ in self.train_loader:
                score = self.mdl(X_in, y)
                loss = self.criterion(score[:,:-1,:].reshape(-1, self.args.vocab_size), X_out.flatten())
                self.mdl.zero_grad()
                loss.backward()
                self.optimizer.step()
                tr_loss_epoch.append(loss.item())
            J_tr.append(sum(tr_loss_epoch)/len(tr_loss_epoch))

            val_loss_epoch = []
            with torch.no_grad():
                for X_in, X_out, _, _, y, _ in self.valid_loader:
                    score = self.mdl(X_in, y)
                    loss = self.criterion(score[:,:-1,:].reshape(-1, self.args.vocab_size), X_out.flatten())
                    val_loss_epoch.append(loss.item())
                J_val.append(sum(val_loss_epoch) / len(val_loss_epoch))               

                if J_val[-1] < best_valid_loss:
                    best_valid_loss = J_val[-1]
                    self.best_parameter = copy.deepcopy(self.mdl.state_dict())
            print(epoch,"-> train:", J_tr[-1], "valid:", J_val[-1])
        torch.save(self.best_parameter, 'output/params/{}/CondLM.pt'.format(self.args.dataset))
        
    
if __name__ == '__main__':
    exp = Train('configs/clm.yaml')
    exp.train()