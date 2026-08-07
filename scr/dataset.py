# src/dataset.py
import pandas as pd
import re
import json

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
    
    def build_vocab(self): #used to create master dictionary 
        
        for index, row in self.data.iterrows():
            
            input_tokens = self.tokenize(row['sentence'])
            self._add_tokens_to_vocab(input_tokens) 
        
           
            corrections_list = json.loads(row['corrections'])
            
            for correction in corrections_list:
                
                target_tokens = self.tokenize(correction)
                self._add_tokens_to_vocab(target_tokens)
                
        