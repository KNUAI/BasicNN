# BasicNN
Basic Neural Network
## Installation
```
git clone https://github.com/KNUAI/BasicNN.git
cd BasicNN
pip install -r requirements.txt
```
## Usage
if you want to run MLP or CNN
```
python main.py --epoch 10 --batch_size 256 --lr 1e-4 --dropout 0.2 --model 'CNN'
```
if you want to run RNN(RNN, LSTM, GRU)
```
python main_rnn.py --epoch 10 --batch_size 256 --lr 1e-4 --dropout 0.2 --model 'LSTM' --hidden_size 128 --n_layer 1
```
if you want to run seq2seq
```
python seq2seq.py --epoch 30 --batch_size 256 --lr 1e-3 --clip 1e-2 --dropout 0.5 --hidden_size 1024 --n_layer 3 --enc_emb 512 --dec_emb 512 --bi 2
```
if you want to run attention
```
python seq2seq_attn.py --epoch 30 --batch_size 256 --lr 1e-3 --clip 1e-2 --dropout 0.5 --hidden_size 1024 --n_layer 3 --enc_emb 512 --dec_emb 512
```