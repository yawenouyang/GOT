import torch
from util.ClassifierDataset import ClassifierDataset
import torch.nn as nn
from util.influence_function import InF
import os
from util.utils import load_args
import os

class WeightingModule():

    def __init__(self, config='configs/weighting.yaml'):
        self.args = load_args(config)
        print(self.args)

        if not os.path.isdir('output/weighting/{}'.format(self.args.dataset)):
            os.mkdir('output/weighting/{}'.format(self.args.dataset))
            os.mkdir('output/weighting/{}/weight'.format(self.args.dataset))
            os.mkdir('output/weighting/{}/influence_result'.format(self.args.dataset))
            os.mkdir('output/weighting/{}/s_result'.format(self.args.dataset))
            os.mkdir('output/weighting/{}/rep'.format(self.args.dataset))

            os.mkdir('output/weighting/{}/weight/{}'.format(self.args.dataset, self.args.aupr_out))
            os.mkdir('output/weighting/{}/influence_result/{}'.format(self.args.dataset, self.args.aupr_out))
            os.mkdir('output/weighting/{}/s_result/{}'.format(self.args.dataset, self.args.aupr_out))
            os.mkdir('output/weighting/{}/rep/{}'.format(self.args.dataset, self.args.aupr_out))

        
        self.model_param = torch.load(self.args.param_path, map_location='cuda: {}'.format(self.args.gpu) \
            if self.args.gpu != -1 else 'cpu')
        self.mdl_initialized = False

    @torch.no_grad()
    def get_rep(self):
        splits = ['train', 'valid', 'ood']
        reps = {}
        rep_dir = 'output/weighting/{}/rep/{}'.format(self.args.dataset, self.args.aupr_out)

        rep_list = os.listdir(rep_dir)
        for split in splits:
            if split in rep_list and not self.args.regenerate:
                rep = torch.load('output/weighting/{}/rep/{}/{}'.format(self.args.dataset, self.args.aupr_out, split))
                reps[split] = rep
            else:
                if not self.mdl_initialized:
                    from model.Classifier import Classifier
                    self.mdl = Classifier(self.args).to(self.args.device)
                    self.mdl.load_state_dict(self.model_param)
                    self.mdl_initialized = True

                rep = torch.tensor([])
                if split == 'ood':
                    dataset = ClassifierDataset('output/generating/{}/ood_{}.csv'.format(self.args.dataset, self.args.seed))
                else:
                    dataset = ClassifierDataset(self.args.train.replace('train', split))
                loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
                for x, _ in loader:
                    pooled_output = self.mdl(x)[0]
                    pooled_output = pooled_output.cpu()
                    rep = torch.cat([rep, pooled_output], dim=0)
                reps[split] = rep
                torch.save(rep, 'output/weighting/{}/rep/{}/{}'.format(self.args.dataset, self.args.aupr_out, split))
        reps['merge'] = torch.cat([reps['train'], reps['ood']], dim=0)

        return reps

    def init_MLP(self):
        class MLP(nn.Module):
            def __init__(self, args):
                super(MLP, self).__init__()
                self.args = args
                self.mlp = nn.Linear(768, args.intent_num)
            
            def forward(self, sens):
                logits = self.mlp(sens)
                return logits

        mlp = MLP(self.args).to(self.args.device)

        mlp_param = {}
        for name, param in self.model_param.items():
            if name in ['mlp.weight', 'mlp.bias']:
                mlp_param[name] = param
        mlp.load_state_dict(mlp_param)
        return mlp
    
    @torch.no_grad()
    def test_mlp(self, mlp, reps):
        valid_rep = reps['valid'].to(self.args.device)
        output = mlp(valid_rep)
        y = sum([[i] * 20 for i in range(150)], [])
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.tolist()
        acc = sum([y[i] == predicted[i] for i in range(len(predicted))]) / len(y)
        print(acc)

if __name__ == '__main__':

    w = WeightingModule()

    reps = w.get_rep()
    mlp = w.init_MLP()

    # test mlp and rep
    # w.test_mlp(mlp, reps)
    # test end
    inf = InF(w.args, mlp, reps)

    if w.args.option == 'prepare_s':
        inf.cal_s()
    else:
        inf.cal_influence()

