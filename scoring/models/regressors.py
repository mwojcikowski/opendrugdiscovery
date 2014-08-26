from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pls import PLSRegression

from .neuralnetwork import neuralnetwork

__all__ = ['randomforest', 'svm', 'pls', 'neuralnetwork']

class randomforest(RandomForestRegressor):
    pass

class svm(SVR):
    pass

class svm(PLSRegression):
    pass
