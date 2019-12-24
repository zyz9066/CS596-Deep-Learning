'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np
from sklearn.utils import class_weight

import os


cfg = {
    'VGG3D11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG3D13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG3D16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG3D19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class _VGG3D(nn.Module):
    def __init__(self, vgg3d_name):
        super(_VGG3D, self).__init__()
        self.features = self._make_layers(cfg[vgg3d_name])
        self.classifier = nn.Linear(512, 55)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv3d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm3d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool3d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGG3D():
    def __init__(self, vgg3d_name='VGG3D11'):
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._log_interval = 1000
        self._net = _VGG3D(vgg3d_name)
        self._net = self._net.to(self._device)
        if self._device == 'cuda':
            self._net = torch.nn.DataParallel(self._net)
            cudnn.benchmark = True

# Training
    def train(self, X_train, y_train, lr=0.01, epochs=100, batch_size=100):
        N = len(y_train)
        class_weights = torch.FloatTensor(class_weight.compute_class_weight('balanced', np.unique(y_train), y_train))
        X_train = torch.FloatTensor(X_train).to(self._device)
        y_train = torch.LongTensor(y_train).to(self._device)
        self._criterion = nn.CrossEntropyLoss(weight=class_weights.to(self._device))
        optimizer = optim.SGD(self._net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        self._net.train()
        for epoch in range(epochs):
            train_loss = 0
            correct = 0
            total = 0
            perm = torch.randperm(N)
            print('\nEpoch: %d' % epoch)
            for idx in range(0, N, batch_size):
                inputs = X_train[perm[idx:idx+batch_size]]
                targets = y_train[perm[idx:idx+batch_size]]
            
                optimizer.zero_grad()
                outputs = self._net(inputs)
                loss = self._criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                if idx % self._log_interval == 0:
                    print(idx, N, 'Train - Loss: %.3f | Acc: %.3f%% (%d/%d)' 
                        % (train_loss/(idx+1), 100.*correct/total, correct, total))
                        
            print(N, N, 'Train - Loss: %.3f | Acc: %.3f%% (%d/%d)' 
                % (train_loss/N, 100.*correct/total, correct, total))   
                
    def validate(self, X_val, y_val, batch_size=100):
        N = len(y_val)
        X_val = torch.FloatTensor(X_val).to(self._device)
        y_val = torch.LongTensor(y_val).to(self._device)
        self._net.eval()
        valid_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for idx in range(0, len(y_val), batch_size):
                inputs = X_val[idx:idx+batch_size]
                targets = y_val[idx:idx+batch_size]
                
                outputs = self._net(inputs)
                loss = self._criterion(outputs, targets)
    
                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if idx % self._log_interval == 0:
                    print(idx, N, 'Validation - Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (valid_loss/(idx+1), 100.*correct/total, correct, total))
                        
            print(N, N, 'Validation - Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (valid_loss/N, 100.*correct/total, correct, total))
    
    def predict(self, x_test):
        self._net.eval()
        with torch.no_grad():
            inputs = torch.FloatTensor(x_test).to(self._device)
            outputs = self._net(inputs)
    
            _, predicted = outputs.max(1)
            preds = predicted.to(torch.device("cpu")).numpy()
                
        return preds
        
    def save(self):
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(self._net.state_dict(), './checkpoint/ckpt.pth')
        
    def load(self):
        self._net.load_state_dict(torch.load('./checkpoint/ckpt.pth'))