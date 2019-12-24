import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

cfg = {
    6: [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M'],
    8: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    10: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# Define a Convolutional Neural Network, take 3-channel images
class _ConvNet(nn.Module):
    def __init__(self, layer_num, out_dim):
        super(_ConvNet, self).__init__()
        self.features = self._set_layers(cfg[layer_num])
        self.classifier = nn.Linear(cfg[layer_num][-2], out_dim)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
        
    def _set_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers += [nn.Dropout()]
        return nn.Sequential(*layers)

class CNN():
    def __init__(self, layer_num=8, output_dim=10, lr=0.001, epochs=1):
        self.MODEL_STORE_PATH = '.\pytorch_models\\'
        self.lr = lr
        self.epochs = epochs
        self.log_interval = 2000
        self.model = _ConvNet(layer_num, output_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, trainloader):
        self.model.to(self.device)
        # Define a Loss function and optimizer
        # SGD with momentum
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        # use a Classification Cross-Entropy loss
        criterion = nn.CrossEntropyLoss()
        
        # Train the network
        for epoch in range(self.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 1):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
        
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
                # print statistics
                running_loss += loss.item()
                if i % self.log_interval == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i, running_loss / self.log_interval))
                    running_loss = 0.0
        
        print('Finished Training')
        
    def _save(self):
        torch.save(self.model.state_dict(), self.MODEL_STORE_PATH + 'conv_net_model.ckpt')
        
    def _load(self):
        self.model.load_state_dict(torch.load(self.MODEL_STORE_PATH + 'conv_net_model.ckpt'))
                
    # Test the network on the test data
    def predict(self, testloader):
        self.model.to(self.device)
        self.model.eval()
        # run a test loop
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))