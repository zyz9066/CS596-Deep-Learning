import numpy as np
import torch

class _LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs
        
class LogisticRegression():
    def __init__(self, input_dim=3072, output_dim=10, lr=0.5, epochs=100):
        """
        Initialises Logistic Regression classifier with initializing 
        alpha(learning rate), number of epochs.
        """
        self.alpha = lr
        self.epochs = epochs

        self.use_cuda = torch.cuda.is_available()
        self.tensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.model = _LogisticRegressionModel(input_dim, output_dim)
        if self.use_cuda:
            self.model.cuda()
    
    
    def train(self, X_train, y_train):
        """
        Train Logistic Regression classifier using function from Pytorch
        """
        X_train = self.tensor(X_train).requires_grad_(True)
        y_train = self.tensor(y_train).requires_grad_(True).long()
            
        # Computes softmax and then cross entropy
        criterion = torch.nn.MSELoss()
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha)
        
        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = self.model(X_train)
            
            # Compute Loss
            loss = criterion(y_pred, y_train)
            
            # Backward pass
            loss.backward()
            optimizer.step()

    
    def predict(self, X_test):
        """
        Use the trained weights of Logistic Regression classifier to predict labels for
        data points.

        Inputs:
        - X_test: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - pred: Predicted labels for the data in X_test. pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        X_test = self.tensor(X_test).requires_grad_(True)
            
        outputs = self.model(X_test)
        _, pred = torch.max(outputs, 1)
        pred = pred.to(torch.device("cpu")).numpy() if self.use_cuda else pred.numpy()
        return pred 