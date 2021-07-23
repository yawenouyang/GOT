import torch.nn as nn
import torch


class ConditionalLM(nn.Module):
    def __init__(self, gpu, dataset, label_num, fix_word_embedding=False):
        super().__init__()
        pre_embedding = torch.load('output/params/{}/embedding'.format(dataset),\
            map_location='cuda: {}'.format(gpu) if gpu != -1 else 'cpu')
        vocab_size, word_dim = pre_embedding.shape
        self.word_embed = nn.Embedding(vocab_size, word_dim)
        self.word_embed.weight.data = pre_embedding
        self.word_embed.weight.requires_grad = (not fix_word_embedding)

        self.rnn = nn.LSTM(word_dim, word_dim, num_layers=1, bidirectional=False, batch_first=True)

        self.label_embed = nn.Embedding(label_num, word_dim)

        self.fc = nn.Linear(word_dim, vocab_size)
        self.fc.weight.data = pre_embedding.clone()
    
    def forward(self, X, label):  # X (batch_size, seq_len), label (batch_size)
        batch_size, seq_len = X.shape
        X = self.word_embed(X)
        hidden, _ = self.rnn(X)  # hidden (batch_size, seq_len, hidden_dim)
        label = self.label_embed(label).unsqueeze(1).repeat(1, seq_len, 1)
        score = self.fc(hidden+label)
        return score
