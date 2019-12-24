import torch

class NearestNeighbor():
    def __init__(self):
        """
        Initializes the KNN classifier with the k.
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def fit(self, X):
        """
        Train the classifier. For nearest neighbors this is just 
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        """
        self._X_train = torch.FloatTensor(X).to(self._device)
    '''
    def _find_dist(self, X_test):
        """
        Compute the distance between each test point in X_test and each training point
        in self._X_train.

        Returns :
        - dist_ : Distances between each test point and training point
        """
        X_i = LazyTensor(X_test) # test set
        X_j = LazyTensor(self._X_train, axis=0) # train set
        dist_ = ((X_i - X_j) ** 2).sum(-1) # symbolic matrix of squared L2 distances
        return dist_
    '''
    def nn_dist(self, X_test):
        """
        Predict labels for test data using the computed distances.

        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - pred: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        X_test = torch.FloatTensor(X_test).to(self._device)
        dists = self._find_dist(X_test)
        dist_nn = dists.min()   # array of distances
        return dist_nn.to(torch.device("cpu"))
    
    def _find_dist(self, X_test):
        r_X_train = torch.sum(self._X_train * self._X_train, dim=1, keepdim=True)  # (B,1)
        r_X_test = torch.sum(X_test * X_test)  # (1)
        mul = torch.matmul(X_test, self._X_train.T)         # (1,B)
        dists_ = r_X_test - 2 * mul + r_X_train.T      # (1,B)
        return dists_