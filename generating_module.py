from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm
from collections import Counter
from model.ConditionalLM import ConditionalLM
import numpy as np
from util.utils import load_args
import pandas as pd
from collections import Counter, defaultdict
import random
import os
import re, string
from ast import literal_eval

class GeneratingModule():

	def __init__(self, config='configs/generating.yaml'):
		self.args = load_args(config)
		random.seed(self.args.seed)
		print(self.args)

		data = pd.read_csv(self.args.train)
		self.intent2index = {}
		for intent, index in zip(data['intent'], data['index']):
			self.intent2index[intent] = index

		if not os.path.isdir('output/generating/{}'.format(self.args.dataset)):
			os.mkdir('output/generating/{}'.format(self.args.dataset))

		# for p(y)
		data = pd.read_csv(self.args.train)
		indexes = Counter(data['index'])
		py = []
		for index in range(len(indexes)):
			py.append(indexes[index] / len(data['index']))
		self.py = torch.tensor(py).unsqueeze(1).to(self.args.device)

		
	def load_utt_intent(self, fname):
		df = pd.read_csv(fname)
		return [(utt, word, intent) for utt, word, intent \
			in zip(list(df['utt']), list(df['masked_word']), list(df['intent']))]

	@torch.no_grad()
	def evaluate_suitable(self, regenerate=True):  # suitable means fit the context well

		if not regenerate:
			data = pd.read_csv('output/generating/{}/{}.csv'.format(self.args.dataset, self.args.suitable_name), \
				converters={"top_k_tokens_logits": literal_eval, "top_k_tokens": lambda x: x.strip("[]").replace("'","").split(", ")}
				)
			suitable_candidates = [
				(utt, masked_word, top_k_tokens, top_k_tokens_logits, intent) for utt, masked_word, top_k_tokens, top_k_tokens_logits, intent \
				in zip(list(data['utt']), list(data['masked_word']), list(data['top_k_tokens']), list(data['top_k_tokens_logits']), list(data['intent']))
			]
			return suitable_candidates
		
		tokenizer = AutoTokenizer.from_pretrained(self.args.bert_path)
		model = AutoModelForMaskedLM.from_pretrained(self.args.bert_path, return_dict=True).to(self.args.device)
		utt_mword_intents = self.load_utt_intent('output/locating/{}/{}.csv'.format(self.args.dataset, self.args.masked_utt_name))
	
		suitable_candidates = []
		# save suitable_candidates
		data = {'utt': [], 'masked_word': [], 'top_k_tokens':[], 'top_k_tokens_logits': [], 'intent': []}
		for pair in tqdm(utt_mword_intents, desc='suitable: '):
			utt, mword, intent = pair
			add_dot = False
			if utt.endswith('[MASK]'):
				utt += ' .'  # prevent BERT from predicting punctuation
				add_dot = True
			input = tokenizer.encode(utt, return_tensors="pt").to(self.args.device)
			mask_token_index = torch.where(input == tokenizer.mask_token_id)[1].to(self.args.device)
			token_logits = model(input).logits
			mask_token_logits = token_logits[0, mask_token_index.item(), :]
			mask_token_logits = torch.log_softmax(mask_token_logits, dim=0)
			top_k = 10  # to speed up, not necessary
			top_k_tokens_logits = torch.topk(mask_token_logits, top_k, dim=0).values.tolist()
			top_k_tokens = torch.topk(mask_token_logits, top_k, dim=0).indices.tolist()
			top_k_tokens = [tokenizer.decode(token).replace(' ', '') for token in top_k_tokens]
			if add_dot == True:
				utt = utt[:-2]  # remove '.'
			suitable_candidates.append((utt, mword, top_k_tokens, top_k_tokens_logits, intent))

			data['utt'].append(utt)
			data['masked_word'].append(mword)
			data['top_k_tokens'].append(top_k_tokens)
			data['top_k_tokens_logits'].append(top_k_tokens_logits)
			data['intent'].append(intent)

		data = pd.DataFrame(data)
		data.to_csv('output/generating/{}/{}.csv'.format(self.args.dataset, self.args.suitable_name))

		return suitable_candidates


	@torch.no_grad()
	def evaluate_relevance(self, suitable_candidates):
		
		clm = ConditionalLM(self.args.gpu, self.args.dataset, self.args.label_num).to(self.args.device)
		clm.load_state_dict(torch.load('output/params/{}/CondLM.pt'.format(self.args.dataset), \
            map_location='cuda:{}'.format(self.args.gpu) if self.args.gpu != -1 else 'cpu'))
		
		tokenizer = AutoTokenizer.from_pretrained(self.args.gpt_path)
		tokenizer.pad_token = tokenizer.eos_token

		relevance_scores = []
		data = {'utt': [], 'masked_word': [], 'top_k_tokens':[], 'top_k_tokens_relevance': [], 'intent': []}
		for pair in tqdm(suitable_candidates, desc='relevance: '):
			utt, _, candidate_words, _, _ = pair 
			utt = utt[:utt.find(' [MASK]')].replace('[CLS] ', '')  # prepare for GPT-2, drop words after [MASK]
			if utt != '':
				candidate_words = ['\u0120' + word for word in candidate_words]  # in GPT-2，'\u0120' represents space
			utt = [tokenizer.bos_token + utt]
			token_in = tokenizer(utt, return_tensors='pt')['input_ids']
			token_in = token_in.expand(self.args.label_num, token_in.shape[1]).to(self.args.device)
			y = torch.tensor(range(self.args.label_num)).to(self.args.device)
			output = clm(token_in, y)
			output = output[:, -1, :]
			output = torch.softmax(output, dim=1)
			output = torch.sum(output * self.py, dim=0)

			output = torch.log(output)
			word_id = tokenizer.convert_tokens_to_ids(candidate_words)
			scores = output[word_id].tolist()

			data['utt'].append(pair[0])
			data['masked_word'].append(pair[1])
			data['top_k_tokens'].append(pair[2])
			data['top_k_tokens_relevance'].append(scores)
			data['intent'].append(pair[-1])

			relevance_scores.append(scores)

		data = pd.DataFrame(data)
		data.to_csv('output/generating/{}/{}.csv'.format(self.args.dataset, self.args.relevance_name))
		
		return relevance_scores

	def generate_ood(self):
		suitable_candidates = self.evaluate_suitable()
		relevance_scores = self.evaluate_relevance(suitable_candidates)
		oods = {'utt': [], 'masked_word': [], 'generated_word': [], 'q_score': [], 'intent': []}
		for index, (utt, mword, candidate_words, suits, intent) in enumerate(suitable_candidates):
			q_scores = []  # equ. 9
			for suit, rele in zip(suits, relevance_scores[index]):
				q_scores.append(rele - suit)
			sorted_index = np.argsort(q_scores)
			for i in range(self.args.K):
				oods['utt'].append(utt)
				oods['masked_word'].append(mword)
				oods['generated_word'].append(candidate_words[sorted_index[i]])
				oods['q_score'].append(q_scores[sorted_index[i]])
				oods['intent'].append(intent)
		
		df = pd.DataFrame(oods, columns=list(oods.keys()), index=None)
		df.to_csv('output/generating/{}/{}.csv'.format(self.args.dataset, self.args.all_ood_name))

	def sample_ood(self):
		pattern = re.compile("[\d{}]+$".format(re.escape(string.punctuation)))

		df = pd.read_csv('output/generating/{}/{}.csv'.format(self.args.dataset, self.args.all_ood_name))
		
		utts = []
		intent_utt_scores = defaultdict(list)
		for utt, generated_word, intent, score, masked_word in zip(df['utt'], df['generated_word'], df['intent'], df['q_score'], df['masked_word']):
			# remove meaningless generation
			if generated_word != generated_word or generated_word == masked_word or pattern.match(generated_word):
				continue
			replaced_utt = utt.replace('[MASK]', str(generated_word))
			intent_utt_scores[intent].append((replaced_utt, utt, score))
		
		sampled_utts = []
		sampled_intents =[]
		ind_indexes = []
		for intent, utt_scores in intent_utt_scores.items():
			if len(utt_scores) > self.args.sample_num:
				utt_scores = random.sample(utt_scores, self.args.sample_num)

			utts = [utt_score[0] for utt_score in utt_scores]
			sampled_utts += utts
			sampled_intents += len(utts) * [intent]
			ind_indexes += len(utts) * [self.intent2index[intent]]
		
		data = {'utt': sampled_utts, 'intents': sampled_intents, 'index': [-1] * len(sampled_intents), 'ind_index': ind_indexes}
		data = pd.DataFrame(data)
		data.to_csv('output/generating/{}/{}_{}.csv'.format(self.args.dataset, self.args.ood_name, self.args.seed))
	
if __name__ == '__main__':

	g = GeneratingModule()
	g.generate_ood()
	g.sample_ood()