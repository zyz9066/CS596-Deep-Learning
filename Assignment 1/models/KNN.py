import torch
from pykeops.torch import LazyTensor

class KNN():
    def __init__(self, k):
        """
        Initializes the KNN classifier with the k.
        """
        self.k = k
        self.use_cuda = torch.cuda.is_available()
        self.tensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
    
    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just 
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = self.tensor(X.astype('float32'))
        self.y_train = self.tensor(y.astype('int64'))
    
    def find_dist(self, X_test):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train.

        Hint : Use scipy.spatial.distance.cdist

        Returns :
        - dist_ : Distances between each test point and training point
        """
        X_i = LazyTensor(X_test[:, None, :]) # test set
        X_j = LazyTensor(self.X_train[None, :, :]) # train set
        dist_ = ((X_i - X_j) ** 2).sum(-1) # symbolic matrix of squared L2 distances
        return dist_
    
    def predict(self, X_test):
        """
        Predict labels for test data using the computed distances.

        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - pred: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        X_test = self.tensor(X_test.astype('float32'))
        D_ij = self.find_dist(X_test)
        ind_knn = D_ij.argKmin(self.k, dim=1)  # Samples <-> Dataset
        lab_knn = self.y_train[ind_knn]  # array of integers
        pred, _ = lab_knn.mode()   # Compute the most likely label
        pred = pred.to(torch.device("cpu")).numpy() if self.use_cuda else pred.numpy()
        return pred