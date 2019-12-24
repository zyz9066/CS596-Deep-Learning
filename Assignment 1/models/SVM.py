import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class _LinearSVM(nn.Module):
    """Support Vector Machine"""
    
    def __init__(self, input_dim, output_dim):
        super(_LinearSVM, self).__init__()
        self.func = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        output = self.func(x)
        return output
        

class SVM():
    def __init__(self, input_dim=3072, output_dim=10, lr=0.01, epochs=100, reg_const=0.05, batch_size=100):
        """
        Initialises Support Vector Machine classifier with initializing
        alpha(learning rate), number of epochs
        and regularization constant.
        """
        self.alpha = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.batch_size = batch_size
        
        self.use_cuda = torch.cuda.is_available()
        self.tensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.model = _LinearSVM(input_dim, output_dim)
        if self.use_cuda:
            self.model.cuda()
            
    '''
    def calc_gradient(self, X_train, y_train):
        """
		It is not mandatory for you to implement this function if you find 
		an equivalent one in Pytorch
		
          Calculate gradient of the svm hinge loss.
          
          Inputs have dimension D, there are C classes, and we operate on minibatches
          of N examples.

          Inputs:
          - X_train: A numpy array of shape (N, D) containing a minibatch of data.
          - y_train: A numpy array of shape (N,) containing training labels; y[i] = c means
            that X[i] has label c, where 0 <= c < C.

          Returns:
          - gradient with respect to weights W; an array of same shape as W
         """
        return grad_w
    '''
    
    def train(self, X_train, y_train):
        """
        Train SVM classifier using stochastic gradient descent.

        Inputs:
        - X_train: A numpy array of shape (N, D) containing training data;
        N examples with D dimensions
        - y_train: A numpy array of shape (N,) containing training labels;
        
        Hint : Operate with Minibatches of the data for SGD
        """
        X_train = self.tensor(X_train)
        y_train = self.tensor(y_train)
        N = len(y_train)
        
        optimizer = optim.SGD(self.model.parameters(), lr=self.alpha)
        
        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(N)
            
            for i in range(0, N, self.batch_size):
                x = X_train[perm[i : i + self.batch_size]]
                y = y_train[perm[i : i + self.batch_size]]
                    
                optimizer.zero_grad()
                output = self.model(x)
                
                loss = torch.mean(torch.clamp(1 - output.t() * y, min=0)) # hinge loss
                loss += self.reg_const * torch.mean(self.model.func.weight ** 2) # l2 penalty
                loss.backward()
                optimizer.step()
        

    def predict(self, X_test):
        """
        Use the trained weights of svm classifier to predict labels for
        data points.

        Inputs:
        - X_test: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - pred: Predicted labels for the data in X_test. pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        X_test = self.tensor(X_test)
            
        outputs = self.model(X_test)
        
        _, pred = torch.max(outputs, 1)
        pred = pred.to(torch.device("cpu")).numpy() if self.use_cuda else pred.numpy()
        return pred