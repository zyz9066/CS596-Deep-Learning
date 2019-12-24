import torch
from torch import FloatTensor, LongTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

cfg = {
    2: [200, 200],
    4: [200,200,200,200],
    6: [200,200,200,200,200,200],
    8: [200, 200, 200, 200, 200, 200, 200, 200],
    10: [200, 200, 200, 200, 200, 200, 200, 200, 200, 200],
    13: [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200],
    16: [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200],
}

class _Net(nn.Module):
    def __init__(self, layer_num, in_dim, out_dim):
        super(_Net, self).__init__()
        self.features = self._set_layers(cfg[layer_num], in_dim)
        self.classifier = nn.Linear(cfg[layer_num][-1], out_dim)
        
    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return F.log_softmax(out, dim=1)
        
    def _set_layers(self, cfg, in_dim):
        layers = []
        for x in cfg:
            layers += [nn.Linear(in_dim, x),
                        nn.ReLU(inplace=True)]
            in_dim = x
        return nn.Sequential(*layers)

class NN():
    def __init__(self, layer_num=8, input_dim=3*32*32, output_dim=10, lr=0.01, epochs=1):
        self.input_dim=input_dim
        self.lr=lr
        self.epochs=epochs
        self.log_interval=2000
        self.model=_Net(layer_num, input_dim, output_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, train_loader):
        self.model.to(self.device)
        # create a stochastic gradient descent optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        # create a loss function
        self.criterion = nn.NLLLoss()
        
        # run the main training loop
        total = 0
        for epoch in range(self.epochs):
            for batch_idx, (data, target) in enumerate(train_loader, 1):
                total+=data.shape[0]
                data, target = FloatTensor(data).requires_grad_(True).to(self.device), LongTensor(target).to(self.device)
                # resize data from (batch_size, 3, 32, 32) to (batch_size, input_dim)
                data = data.view(-1, self.input_dim)
                optimizer.zero_grad()
                net_out = self.model(data)
                loss = self.criterion(net_out, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch+1, batch_idx * len(data), total,
                               100. * batch_idx / len(train_loader), loss.item()))
            
                
    def predict(self, test_loader):
        self.model.to(self.device)
        self.model.eval()
        # run a test loop
        test_loss = 0
        correct = 0
        with torch.no_grad():
            total = 0
            for data, target in test_loader:
                total+=data.shape[0]
                data, target = FloatTensor(data).to(self.device), LongTensor(target).to(self.device)
                data = data.view(-1, self.input_dim)
                net_out = self.model(data)
                # sum up batch loss
                test_loss += self.criterion(net_out, target).item()
                pred = net_out.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data).sum()
        
        test_loss /= total
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, total,
            100. * correct / total))
