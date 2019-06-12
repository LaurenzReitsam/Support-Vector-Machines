
import numpy as np
import pandas as pd
import cvxpy as cp

class support_vector_basis_model():
    
    def __init__(self, kernel='linear'):
        self.params = None
        self.support_vectors = None
        self.std = 1

        
        if kernel=='linear'    : self.kernel = self.kernel_linear
        if kernel=='poly'      : self.kernel = self.kernel_poly
        if kernel=='gaussian'  : self.kernel = self.kernel_gaussian
        
        
    def kernel_linear(self, X1, X2):
        return X1@X2.T
    
    def kernel_poly(self, X1, X2, p=2.0):
        return (X1@X2.T + 1.0)**p
    
    def kernel_gaussian(self, X1, X2, sigma=5.0):
        return #np.exp(-np.linalg.norm(X1-X2)**2 / (2 * (sigma ** 2)))
    
    def train(self, X, y_in, c=1):
        pass
    
    def predict(self, X):
        pass
       
    def normalize(self, X, set_Value=False):
        if set_Value: self.std = X.std()        
        return X/self.std
            


class suppor_vector_classifier(support_vector_basis_model):
    
    def train(self, X, y_in, c=1):
        
        n_samples, n_features = X.shape   
        
        y = np.copy(y_in)
        y[y==0] = -1
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
        lagranges = self.params['lagranges']
        intercept = self.params['intercept']
        sv_X = self.support_vectors['X']
        sv_y = self.support_vectors['y'].reshape([-1,1])
        
        K = self.kernel(sv_X, X)
        y = K.T@(lagranges * sv_y) + intercept
        return np.sign(y.reshape([-1]))
        
        

class suppor_vector_regressor(support_vector_basis_model):
    
    def train(self, X, y_in, c=1, eta = 0.1):
        
        n_samples, n_features = X.shape
        
        y = np.copy(y_in)
        
        X = self.normalize(X, set_Value=True)
        
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
        
        X = self.normalize(X)

        lagranges = self.params['lagranges']
        intercept = self.params['intercept']
        sv_X = self.support_vectors['X']

        K = self.kernel(sv_X, X)
        y = K.T@lagranges + intercept

        return y.reshape([-1])
    