import numpy as np
import torch

class _LinearRegressionModel(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(_LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim) # One in and one out
        
    def forward(self, x):
        outputs = self.linear(x)
        return outputs
        
class LinearRegression():
    def __init__(self, input_dim=3072, output_dim=10, lr=0.5, epochs=100):
        """
        Initialises Linear Regression classifier with initializing 
        alpha(learning rate), number of epochs.
        """
        self.alpha = lr
        self.epochs = epochs

        # Model
        self.use_cuda = torch.cuda.is_available()
        self.tensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.model = _LinearRegressionModel(input_dim, output_dim)
        if self.use_cuda:
            self.model.cuda()
    
    
    def train(self, X_train, y_train):
        """
        Train Linear Regression classifier using function from Pytorch
        """
        X_train = self.tensor(X_train).requires_grad_(True)
        y_train = self.tensor(y_train).requires_grad_(True)
        
        criterion = torch.nn.MSELoss(reduction='sum')
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha)
        
        self.model.train()
        for epoch in range(self.epochs):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass: compute predicted y by passing x to the model
            y_pred = self.model(X_train)
            _, y_pred = torch.max(y_pred, 1)
            y_pred = y_pred.float()
            
            # Compute loss
            loss = criterion(y_pred, y_train)
            
            # Perform a backward pass
            loss.backward()
            # Update the weights
            optimizer.step()
    
    def predict(self, X_test):
        """
        Use the trained weights of Linear Regression classifier to predict labels for
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