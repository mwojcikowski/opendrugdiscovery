# HACK import BFGS before ffnet, otherwise it will segfault when trying to use BFGS
from scipy.optimize import fmin_l_bfgs_b
fmin_l_bfgs_b
## FIX use ffnet for now, use sklearn in future
from ffnet import ffnet,mlgraph,tmlgraph
import numpy as np
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler

class neuralnetwork:
    def __init__(self, shape = None, full_conn=True, biases=True, random_weights = True, normalize=True):
        """
        shape: shape of a NN given as a tuple
        """
        self.shape = shape
        self.full_conn = full_conn
        self.biases = biases
        self.random_weights = random_weights
        self.normalize = normalize
        if self.normalize:
            self.norm = StandardScaler()
        if shape:
            if full_conn:
                conec = tmlgraph(shape, biases)
            else:
                conec = mlgraph(shapebiases)
            self.model = ffnet(conec)
            if random_weights:
                self.model.randomweights()
    
    def get_params(self, deep=True):
        return {'shape': self.shape, 'full_conn': self.full_conn, 'biases': self.biases, 'random_weights': self.random_weights, 'normalize': self.normalize}
    
    def set_params(self, args):
        self.__init__(**args)
        return self
    
    def fit(self, input_descriptors, target_values, train_alg='tnc', **kwargs):
        if self.normalize:
            descs = self.norm.fit_transform(input_descriptors)
        else:
            descs = input_descriptors
        getattr(self.model, 'train_'+train_alg)(descs, target_values, **kwargs)
        return self
    
    def predict(self, input_descriptors):
        if self.normalize:
            descs = self.norm.fit_transform(input_descriptors)
        else:
            descs = input_descriptors
        return np.array(self.model.call(descs))
    
    def score(self, X, y):
        return linregress(self.predict(X).flatten(), y)[2]**2
        
