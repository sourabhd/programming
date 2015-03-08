from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
from pylearn2.datasets import DenseDesignMatrix


class LinearSVMCost(DefaultDataSpecsMixin, Cost):
    """Linear SVM cost function"""

    supervised = True

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)

        # One-hot encoding of pylearn2 perhaps consists of 0 and 1
        # Ensure it is  -1 and 1 as required by SVM
        X_s, Z_s = data
        ones_Z_s = T.ones_like(Z_s)
        pos_Z_s = Z_s > 0
        npos_Z_s = Z_s <= 0
        pos_Z_s = pos_Z_s.astype(theano.config.floatX)
        npos_Z_s = npos_Z_s.astype(theano.config.floatX)
        Y_s = 1.0 * ones_Z_s * pos_Z_s - ones_Z_s * npos_Z_s
        # Hinge Loss
        margin_s = Y_s * (theano.dot(X_s, model.W_s) + model.b_s)
        hinge_s = T.ones_like(margin_s) - margin_s
        nneg_s = hinge_s > 0
        nneg_s.astype(theano.config.floatX)
        hinge_s = hinge_s * nneg_s
        loss = T.sum(T.mean(hinge_s, axis=0))
        return loss


class LinearSVM(Model):
    """Linear SVM model"""

    def __init__(self, num_features, num_classes):
        super(LinearSVM, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.dtype = theano.config.floatX

        W0 = np.zeros((self.num_features, self.num_classes), dtype=self.dtype)
        b0 = np.zeros((1, num_classes), dtype=self.dtype)

        self.W_s = sharedX(W0, name="W_s")
        # self.b_s = sharedX(b0, name="b_s")
        self.b_s = theano.shared(b0, name="b_s", broadcastable=[True, False])

        # Parameters
        self._params = [self.W_s, self.b_s]

        # Input - Output
        self.input_space = VectorSpace(dim=self.num_features)
        self.output_space = VectorSpace(dim=self.num_classes)

    def linear_svm(self, X_s):
        return T.argmax(theano.dot(X_s, self.W_s) + self.b_s)


class SyntheticDataset(DenseDesignMatrix):
    """Toy test dataset for classifier"""

    def __init__(self):
        dtype = theano.config.floatX
        num_pos = 1000
        P_sigma = 1
        P_mu = 10
        P = P_sigma * np.random.randn(num_pos, 2) + P_mu
        num_neg = 1000
        N_sigma = 1
        N_mu = 1
        N = N_sigma * np.random.randn(num_neg, 2) + N_mu
        X = np.vstack([P, N]).astype(dtype)
        pos = np.atleast_2d(np.ones(num_pos, dtype=int)).T
        neg = np.atleast_2d(-1 * np.ones(num_neg, dtype=int)).T
        y = np.vstack((pos, neg))
        # Y = np.hstack((-y, y))
        # Y = Y.astype(dtype)
        y_labels = 2

        super(SyntheticDataset, self).__init__(X=X, y=y, y_labels=y_labels)
