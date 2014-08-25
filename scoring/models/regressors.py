from sklearn.ensemble import RandomForestRegressor as randomforest
from sklearn.svm import SVR as svm
from sklearn.pls import PLSRegression as pls

from .neuralnetwork import neuralnetwork

__all__ = ['randomforest', 'svm', 'pls', 'neuralnetwork']
