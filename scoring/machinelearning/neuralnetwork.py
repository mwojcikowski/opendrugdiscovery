## FIX use ffnet for now, use sklearn in future
from ffnet import ffnet,mlgraph,tmlgraph
import numpy as np
from scipy.stats import linregress

class neuralnetwork:
    def __init__(self, shape, loadnet=None, full_conn=True, biases=False):
        """
        shape: shape of a NN given as a tuple
        """
        if loadnet:
            self.model = ffnet()
            self.model.load(loadnet)
        else:
            if full_conn:
                conec = tmlgraph(shape, biases)
            else:
                conec = mlgraph(shapebiases)
            self.model = ffnet(conec)
    
    def fit(self, input_descriptors, target_values, train_alg='tnc', **kwargs):
        getattr(self.model, 'train_'+train_alg)(input_descriptors, target_values, **kwargs)
    
    def predict(self, input_descriptors):
        return np.array(self.model.call(input_descriptors))
    
    def score(self, X, y):
        return linregress(self.predict(X).flatten(), y)[2]**2
        
