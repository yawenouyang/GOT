import pandas as pd
from tqdm import tqdm
import json
import torch
from nltk.corpus import stopwords
from transformers import GPT2LMHeadModel
from model.ConditionalLM import ConditionalLM
from util.utils import load_args, compute_logProb, init_tokenizer
from util.CLMDataset import CLMDataset
from torch.utils.data import DataLoader
from collections import Counter
import json
from tqdm import tqdm
import os

en_stopwords = list(stopwords.words('english'))

class LocatingModule():

    def __init__(self, config='configs/locating.yaml'):
        self.args = load_args(config)
        print(self.args)
        if not os.path.isdir('output/locating/{}'.format(self.args.dataset)):
            os.mkdir('output/locating/{}'.format(self.args.dataset))
            
    def load_utt_intent(self, fname):
        df = pd.read_csv(fname)
        return [(utt, intent) for utt, intent in zip(list(df['utt']), list(df['intent']))]

    def load_semantic(self, fname):
        with open(fname) as f:
            word_score = json.load(f)
        
        intent_related_words = {}
        for intent, scores in word_score.items():
            intent_related_words[intent] = []
            for score in scores:
                if score[1] > self.args.threshold:
                    intent_related_words[intent].append(score[0])
        return intent_related_words

    @torch.no_grad()
    def generate_word_score(self):

        gpt2 = GPT2LMHeadModel.from_pretrained(self.args.gpt_path, return_dict=True).to(self.args.device)
        clm = ConditionalLM(self.args.gpu, self.args.dataset, self.args.label_num).to(self.args.device)
        clm.load_state_dict(torch.load('output/params/{}/{}.pt'.format(self.args.dataset, self.args.cond_name), \
            map_location='cuda: {}'.format(self.args.gpu) if self.args.gpu != -1 else 'cpu'))
        
        gpt2.eval()
        clm.eval()

        self.tokenizer = init_tokenizer(self.args.gpt_path)
        train_loader = self.create_loader(self.args.train, False)

        output = 'output/locating/{}/{}.csv'.format(self.args.dataset, self.args.llr_name)

        with open(output, "w", encoding='utf-8') as f:
            for X_in, X_out, mask, length, y, intent in tqdm(train_loader):
                logProb_Cond = compute_logProb(X_out, clm(X_in, y))
                logProb = compute_logProb(X_out, gpt2(input_ids=X_in, attention_mask=mask).logits)
                llr = logProb_Cond - logProb
                for x, ratio, label in zip(X_out, llr, intent):
                    f.write("{},".format(label))
                    f.write(",".join(self.tokenizer.batch_decode(x.reshape(1,-1))[0].replace("<|endoftext|>","").split()))
                    f.write("\n,"+ ",".join([str(r.item()) for r in ratio])+ "\n")

        summary = {}
        with open(output) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if i % 2 == 0:
                    text = line.split(',')
                else:
                    score = list(map(float, line.split(',')[1:]))
                    if text[0] not in summary:
                        summary[text[0]]={}
                    for j, w in enumerate(text[1:]):
                        if w in en_stopwords:  # not necessary
                            continue
                        if w not in summary[text[0]]:
                            summary[text[0]][w]=score[j]
                        else:
                            summary[text[0]][w]+=score[j]
        old_summary = summary
        summary = {}
        for key in old_summary:
            summary[key] = Counter(old_summary[key]).most_common(10)

        with open('output/locating/{}/{}.json'.format(self.args.dataset, self.args.word_score_name), 'w') as f:
            json.dump(summary, f, indent=4)
        

    def create_loader(self, file_name, shuffle):
        dataset = CLMDataset(file_name, self.tokenizer, self.args.device)
        loader = DataLoader(dataset=dataset, batch_size=self.args.batch_size, shuffle=shuffle, drop_last=False)
        return loader

    def generate_masked_utts(self):
        utt_intents = self.load_utt_intent(self.args.train)
        intent_related_words = self.load_semantic('output/locating/{}/{}.json'.format(self.args.dataset, self.args.word_score_name))
        # print(intent_related_words)

        masked_utts = {'utt':[],'masked_word':[],'intent':[]}
        for utt_intent in tqdm(utt_intents):
            utt, intent = utt_intent
            if intent_related_words.get(intent, None) is None:
                continue
            for word in intent_related_words[intent]:
                if utt.count(word) == 1:
                    masked_utts['utt'].append(utt.replace(word, '[MASK]'))
                    masked_utts['masked_word'].append(word)
                    masked_utts['intent'].append(intent)
                    
        # for intent, num in intent_num.items():
        #     print(intent, num)

        df = pd.DataFrame(masked_utts, columns=list(masked_utts.keys()), index=None)
        df.to_csv('output/locating/{}/{}.csv'.format(self.args.dataset, self.args.masked_utt_name))


if __name__ == '__main__':
    l = LocatingModule()
    l.generate_word_score()
    l.generate_masked_utts()