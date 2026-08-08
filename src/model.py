
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.5):
        super(EncoderLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)##converting tokens to vector using pytorch
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True, 
                            dropout=dropout if num_layers > 1 else 0)#looks at text forward and backward
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size) #compressing hidden states 
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        embedded = self.embedding(x) #turning word into vector embeddings
        encoder_states, (hidden, cell) = self.lstm(embedded) #passing emeddings 
        
        
        hidden = torch.tanh(self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))) #combining forward and backwards so decoder can read them 
        cell = torch.tanh(self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2)))
        
        return encoder_states, hidden, cell

class Attention(nn.Module): 
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_states):
        seq_length = encoder_states.shape[1]
        hidden = hidden.repeat(seq_length, 1, 1).transpose(0, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_states), dim=2))) #calculating energy scores
        attention_scores = self.v(energy).squeeze(2)
        return F.softmax(attention_scores, dim=1) #turn scores into probobilities 

class DecoderLSTM(nn.Module): #Decoder code
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.5):
        super(DecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(embed_size + (hidden_size * 2), hidden_size, num_layers, batch_first=True) #input target word embedding + wegihted context vectore 
        self.fc = nn.Linear(hidden_size, vocab_size) #output layer of probobilities over all of vocab size 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell, encoder_states):
        x = x.unsqueeze(1) #one word at a time 
        embedded = self.dropout(self.embedding(x)) 
        attn_weights = self.attention(hidden, encoder_states).unsqueeze(1) # get attention weights and building context vector
        context = torch.bmm(attn_weights, encoder_states) 
        lstm_input = torch.cat((embedded, context), dim=2) #combine word embedding and attention context 
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1)) #predict the most likley corrected word ID 
        
        return predictions, hidden, cell

class Seq2Seq(nn.Module): #combine encoder and decoder 
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.vocab_size
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device) #matrix to store predictions 
        encoder_states, hidden, cell = self.encoder(source) #pass encoder raw sentence 
        x = target[:, 0] 
        
        for t in range(1, target_len): #prediction loop
            output, hidden, cell = self.decoder(x, hidden, cell, encoder_states)
            outputs[:, t, :] = output
            bestGuess = output.argmax(1) #find the highest prob token
            x = target[:, t] if torch.rand(1).item() < teacher_forcing_ratio else bestGuess #Teacher forcing to speed up training 
            
        return outputs