import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse

from NN.RNN import RNN, LSTM, GRU

parser = argparse.ArgumentParser(description='Neural Network')

parser.add_argument('--seed', type=int, default=117, help='seed')
parser.add_argument('--epoch', type=int, default=10, help='epoch')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning_rate')
parser.add_argument('--dropout', type=float, default=0.2, help='learning_rate')
parser.add_argument('--model', type=str, default='RNN', help='model')
parser.add_argument('--n_class', type=int, default=10, help='number of class')
parser.add_argument('--input_size', type=int, default=28, help='input_size of rnn model')
parser.add_argument('--sequence_len', type=int, default=28, help='sequence_len of rnn model')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden_size of rnn model')
parser.add_argument('--n_layer', type=int, default=1, help='n_layers of rnn model')

args = parser.parse_args()

#seed
torch.manual_seed(args.seed)

#GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Let's use", torch.cuda.device_count(), "GPUs!")
print('device:', device)

#dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=args.batch_size)

#model
if args.model == 'RNN':
    model = RNN(args.input_size, args.hidden_size, args.n_layer, args.n_class)
elif args.model == 'LSTM':
    model = LSTM(args.input_size, args.hidden_size, args.n_layer, args.n_class)
elif args.model == 'GRU':
    model = GRU(args.input_size, args.hidden_size, args.n_layer, args.n_class)
model.to(device)

#loss, optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

#load_weight
##model.load_state_dict(torch.load(f'./path/{args.model}.pth'))

#train
model.train()
total_loss = 0
total_acc = 0
train_loss = []
train_accuracy = []
i = 1
for epoch in range(args.epoch):
    for data, target in train_loader:
        data = data.reshape(-1, args.sequence_len, args.input_size).to(device)
        target = target.to(device)

        output = model(data)
        loss = criterion(output, target)

        total_loss += loss
        train_loss.append(total_loss/i)

        prediction = output.max(1)[1]  #tensor.max(dim=1)[max, argmax]
        accuracy = prediction.eq(target).sum()/args.batch_size*100

        total_acc += accuracy
        train_accuracy.append(total_acc/i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f'Epoch: {epoch+1}\t Train Step: {i:3d}\t Loss: {loss:.3f}\t Accuracy: {accuracy:.3f}%')
        i += 1
    print(f'Epoch: {epoch+1} finished')

#save_weight
##torch.save(model.state_dict(), f'./path/{args.model}.pth')

#learning_curve
plt.plot(range(len(train_loss)), train_loss)  #marker='o'
plt.xlabel('Train Step')
plt.ylabel('Train Loss')
plt.show()
##plt.savefig(f'./{args.model}_train_loss_result.png')

plt.plot(range(len(train_accuracy)), train_accuracy)  #marker='o'
plt.xlabel('Train Step')
plt.ylabel('Train Accuracy')
plt.show()
##plt.savefig(f'./{args.model}_train_accuracy_result.png')

#validation
with torch.no_grad():
    model.eval()
    test_acc_sum = 0
    for data, target in test_loader:
        data = data.reshape(-1, args.sequence_len, args.input_size).to(device)
        target = target.to(device)

        output = model(data)

        prediction = output.max(1)[1]  #tensor.max(dim=1)[max, argmax]
        test_acc_sum += prediction.eq(target).sum()

print(f'\nTest set: Accuracy: {100 * test_acc_sum / len(test_loader.dataset):.3f}%')