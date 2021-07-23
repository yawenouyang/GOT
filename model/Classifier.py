import torch.nn as nn
from model.BERT import BERT

class Classifier(nn.Module):
    
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        self.encoder = BERT(args)
        self.mlp = nn.Linear(768, args.intent_num)

    def forward(self, sens):
        pooled_output = self.encoder(sens)
        logits = self.mlp(pooled_output)
        return pooled_output, logits