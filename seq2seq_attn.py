import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import argparse

import random
import math

parser = argparse.ArgumentParser(description='Seq2Seq')

parser.add_argument('--seed', type=int, default=117, help='seed')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
parser.add_argument('--enc_emb', type=int, default=512, help='enc_embed_size')
parser.add_argument('--dec_emb', type=int, default=512, help='dec_embed_size')
parser.add_argument('--hidden_size', type=int, default=1024, help='hidden_size')
parser.add_argument('--n_layer', type=int, default=3, help='n_layers')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--lr', type=float, default=1e-3, help='learning_rate')
parser.add_argument('--clip', type=float, default=1e-2, help='clip')
parser.add_argument('--epoch', type=int, default=30, help='epoch')

args = parser.parse_args()

#folder
if not os.path.exists('./path'):
    os.makedirs('./path')

#seed
torch.manual_seed(args.seed)
random.seed(args.seed)

#GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Let's use", torch.cuda.device_count(), "GPUs!")
print('device:', device)

#French and English Tokenizer  ->  python -m spacy download en && python -m spacy download de
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

#tokenizer function
def tokenizer_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]  #reverse

def tokenizer_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

#Build Field  ->  src : input / trg:target
SRC = Field(tokenize = tokenizer_de, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize = tokenizer_en, init_token='<sos>', eos_token='<eos>', lower=True)

#split data
train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = (SRC, TRG))

#build vocabulary in training set
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

#Build Iterator
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size = args.batch_size, device = device)

#Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout = dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))  #src  :  (src_len, batch_size)
        output, hidden = self.gru(embedded)  #embedded  :  (src_len, batch_size, embed_size)
        output = (output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:])
        hidden = (hidden[:self.n_layers, :, :] + hidden[self.n_layers:, :, :])

        # outputs = [src_len, batch_size, hidden_size * n_directions]
        # hidden = [n_layers * n_directions, batch_size, hidden_size]

        return output, hidden

#Attention
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  #[B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        #[B*T*2H] -> [B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  #[B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  #[B*1*H]
        energy = torch.bmm(v, energy)  #[B*1*T]
        return energy.squeeze(1)  #[B*T]

#Decoder
class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, embed_size)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, last_hidden, encoder_outputs):

        #input = [batch_size]
        #hidden = [n_layers * n_directions, batch_size, hidden_size], where n_layers will always be 1
        #output = [seq_len, batch_size, hidden_size * n_directions]

        input = input.unsqueeze(0)  #input  :  (1, batch_size)
        embedded = self.dropout(self.embedding(input))  #embedded  :  (1, batch_size, embed_size)
        
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  #(B,1,N)
        context = context.transpose(0, 1)  #(1,B,N)
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  #(1,B,N) -> (B,N)
        context = context.squeeze(0)
        prediction = self.out(torch.cat([output, context], 1))  #prediction  :  (batch_size, output_size)

        return prediction, hidden

#Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):

        #src = [src len, batch size]
        #trg = [trg len, batch size]

        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        enc_output, last_hidden, = self.encoder(src)  #encoder
        input = trg[0,:]  #<sos> tokens

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, last_hidden, enc_output)  #decoder
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs

#seq2seq
enc = Encoder(len(SRC.vocab), args.enc_emb, args.hidden_size, args.n_layer, args.dropout)
dec = Decoder(len(TRG.vocab), args.dec_emb, args.hidden_size, args.n_layer, args.dropout)

model = Seq2Seq(enc, dec, device).to(device)

#Initialize model
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)

#optimizer, loss
optimizer = optim.Adam(model.parameters(), lr = args.lr)
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX).to(device)

#train
def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0
    for i, batch in enumerate(iterator):

        src = batch.src.to(device)
        trg = batch.trg.to(device)

        output = model(src, trg)

        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return epoch_loss / len(iterator)

#Evaluate model
def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0
    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src = batch.src.to(device)
            trg = batch.trg.to(device)

            output = model(src, trg, 0)  #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

#main(train, validation)
best_valid_loss = float('inf')  #infinity
for epoch in range(args.epoch):
    train_loss = train(model, train_iterator, optimizer, criterion, args.clip)
    valid_loss = evaluate(model, valid_iterator, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './path/seq2seq.pt')
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

#main(test)
model.load_state_dict(torch.load(f'./path/seq2seq.pt'))
test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')



