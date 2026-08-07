# src/dataset.py
import pandas as pd
import re
import json
import torch
from torch.utils.data import Dataset


class GrammarErrorCorrectionDataset:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"} ##PAD is for Padding, SOS = Start of Sequence, EOS = End of sentence, UNK = Unkown

    def tokenize(self, sentence): ## used to tokenize a string 
        sentence = str(sentence).lower()
        sentence = re.sub(r"([.!?,'/()])", r" \1 ", sentence)
        return sentence.strip().split()
    
    
    def _add_tokens_to_vocab(self, tokens): # helper function that adds a word to the dictionary if it isnt already there
        for token in tokens:
            if token not in self.word2idx:
                index = len(self.word2idx)
                self.word2idx[token] = index
                self.idx2word[index] = token
    
    def build_vocab(self, min_count=3): #used to create master dictionary/only use on training dataset 
        
        numb_tokens = {}

        for index, row in self.data.iterrows():
            
            input_tokens = self.tokenize(row['sentence'])
            #self._add_tokens_to_vocab(input_tokens) 
            for token in input_tokens:
                if token not in numb_tokens:
                    
                    numb_tokens[token] = 0
                numb_tokens[token] += 1
           
            corrections_list = json.loads(row['corrections'])
            
            for correction in corrections_list:
                
                target_tokens = self.tokenize(correction)
                for token in target_tokens:
                    if token not in numb_tokens:

                        numb_tokens[token] = 0
                    numb_tokens[token] += 1
                # self._add_tokens_to_vocab(target_tokens)
                
        #keep only tokens with >=  min count
        kept_tokens = []
        for token, count in numb_tokens.items():
            if count >= min_count:
                kept_tokens.append(token)
 
        self._add_tokens_to_vocab(kept_tokens)
    

    def encode(self, tokens, max_len):
        ids = [self.word2idx["<SOS>"]]
        
        for token in tokens:
            if token in self.word2idx:
                ids.append(self.word2idx[token])
            else:
                ids.append(self.word2idx["<UNK>"])

        ids.append(self.word2idx["<EOS>"])

        # keep every sentence the same length
        if len(ids) > max_len:
            ids = ids[:max_len]

            ids[-1] = self.word2idx["<EOS>"]  
            
        else:
            while len(ids) < max_len:
                ids.append(self.word2idx["<PAD>"])
 
        return ids

class GECDatasetWrapper(Dataset): #class to wrap GrammarErrorCorrectionDataset 
    def __init__(self, gecdataset, max_len=50, use_first_correction_only=True):
        self.gecdataset = gecdataset
        self.max_len = max_len

        #create pairs of sentence and correction
        self.pairs = []
        for index, row in self.gecdataset.data.iterrows():
            sentence = row['sentence']
            corrections_list = json.loads(row['corrections'])
 
            if use_first_correction_only:
                self.pairs.append((sentence, corrections_list[0]))
            else:
                for correction in corrections_list:
                    self.pairs.append((sentence, correction))
    
    def __len__(self): #get number of rows
        return len(self.pairs)

    def __getitem__(self, idx): #returns one input/target tensor pair for a given index
        sentence, correction = self.pairs[idx] #get the sentence and correction pair
        input_tokens = self.gecdataset.tokenize(sentence)
        target_tokens = self.gecdataset.tokenize(correction)

        input_ids = self.gecdataset.encode(input_tokens, self.max_len)
        target_ids = self.gecdataset.encode(target_tokens, self.max_len)

        #convert to tensors so pytorch can use them
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        target_tensor = torch.tensor(target_ids, dtype=torch.long)
 
        return input_tensor, target_tensor
