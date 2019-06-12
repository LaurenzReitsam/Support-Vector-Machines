
import numpy as np
import pandas as pd
import cvxpy as cp
from sys import exit

class _support_vector_basis_model():
    "Base for the following classifiaction and regression models."
    
    def __init__(self, kernel='linear'):
        self.params = None
        self.support_vectors = None
        self.std = None

        if kernel=='linear'    : self.kernel = self._kernel_linear
        if kernel=='poly'      : self.kernel = self._kernel_poly
        if kernel=='gaussian'  : self.kernel = self._kernel_gaussian
        
        
    def _kernel_linear(self, X1, X2):
        return X1@X2.T
    
    def _kernel_poly(self, X1, X2, p=2.0):
        return (X1@X2.T + 1.0)**p
    
    #def _kernel_gaussian(self, X1, X2, sigma=5.0):
    
    def train(self, X, y_in, c=1):
        pass
    
    def predict(self, X):
        pass
       
    def _normalize(self, X, set_value=False):
        if set_value: self.std = X.std(axis=0)        
        return X/self.std
    
    def get_support_vectors(self):
        return self.support_vectors['X'] * self.std
            


class suppor_vector_classifier(_support_vector_basis_model):
    "Support-Vector classification model"
    
    def train(self, X, y_in, c=1):
        
        self._check_y(y_in)
        
        X = self._normalize(X, set_value=True)
        y = np.copy(y_in)
        
        n_samples, n_features = X.shape   

        y = y.reshape([n_samples,1])   

        K = self.kernel(X, X)   
        P = K* (y@y.T)
        P = cp.Parameter(shape=P.shape, value=P, PSD=True)
        q = np.ones(n_samples)
        A = y.copy().reshape([-1])
        G = np.vstack((np.diag(np.ones(n_samples) * -1), np.identity(n_samples)))
        h = np.hstack((np.zeros(n_samples), np.ones(n_samples) * c))
        
        a = cp.Variable(n_samples)
        
        prob = cp.Problem( cp.Minimize((1/2)*cp.quad_form(a, P) - q.T@a),
                 [G@a <= h, A@a == 0.0])
        
        prob.solve()
        
        lagranges  = np.ravel(a.value)
        
        mask = np.abs(lagranges) > 1e-5
        lagranges[np.invert(mask)] = 0
        
        sv_X = X[mask]
        sv_y = y[mask]
        sv_K = self.kernel(sv_X, sv_X)
        
        lagranges = lagranges.reshape([-1, 1])
        
        intercept = np.mean(sv_y - sv_K.T@(lagranges[mask] * sv_y))
        
        self.params = {'intercept':intercept, 'lagranges' : lagranges[mask]}
        self.support_vectors = {'X': sv_X, 'y' : sv_y}
        
        print(f"Number of support vectors: {mask.sum()}")
        
    def predict(self, X):
        
        X = self._normalize(X)
        
        lagranges = self.params['lagranges']
        intercept = self.params['intercept']
        sv_X = self.support_vectors['X']
        sv_y = self.support_vectors['y'].reshape([-1,1])
        
        K = self.kernel(sv_X, X)
        y = K.T@(lagranges * sv_y) + intercept
        
        return np.sign(y.reshape([-1]))
    
    def _check_y(self, y):
        classes = np.unique(y)
        assert len(classes)==2,      "y must consist of 2 classes"
        assert abs(classes[0]) == 1, "Class labels must be 1 and -1"
        assert abs(classes[1]) == 1, "Class labels must be 1 and -1"

        
        

class suppor_vector_regressor(_support_vector_basis_model):
    "Support-Vector regression model"
    
    def train(self, X, y_in, c=1, eta=0.1):
        
        n_samples, n_features = X.shape
        
        y = np.copy(y_in)
        
        X = self._normalize(X, set_value=True)
        
        y = y.reshape([n_samples])  
        
        K = self.kernel(X, X)   
        P = K
        P = cp.Parameter(shape=P.shape, value=P, PSD=True)
        q = np.ones(n_samples)
        A = y.copy().reshape([-1])
        G = np.vstack((np.diag(np.ones(n_samples) * -1), np.identity(n_samples)))
        h = np.hstack((np.zeros(n_samples), np.ones(n_samples) * c))
        
        a1 = cp.Variable(n_samples)
        a2 = cp.Variable(n_samples)
        delta = a1-a2
        
        
        prob = cp.Problem( cp.Minimize((1/2)*cp.quad_form(delta, P) - y.T@delta + eta*q.T@(a1+a2) ),
                 [G@a1 <= h, G@a2 <= h, q.T@delta == 0.0])
        
        prob.solve()
        
        lagranges  = np.ravel(delta.value)
        
        mask = np.abs(lagranges) > 1e-5
        lagranges[np.invert(mask)] = 0
        
        sv_X = X[mask]
        sv_y = y[mask]
        sv_K = self.kernel(sv_X, sv_X)
        
        lagranges = lagranges.reshape([-1, 1])
        
        intercept = np.mean(sv_y - sv_K.T@(lagranges[mask]))
        
        self.params = {'intercept':intercept, 'lagranges' : lagranges[mask]}
        self.support_vectors = {'X': sv_X, 'y' : sv_y}
        
        print(f"Number of support vectors: {mask.sum()}")
        
        
    def predict(self, X):
        
        X = self._normalize(X)

        lagranges = self.params['lagranges']
        intercept = self.params['intercept']
        sv_X = self.support_vectors['X']

        K = self.kernel(sv_X, X)
        y = K.T@lagranges + intercept

        return y.reshape([-1])
    