import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib as pickle

class scorer(object):
    def __init__(self, model_instance, descriptor_generator_instance):
        self.model = model_instance
        self.descriptor_generator = descriptor_generator_instance
        
    def fit(self, ligands, target):
        self.train_descs = self.descriptor_generator.build(ligands)
        self.train_target = target
        return self.model.fit(descs,target)
    
    def predict(self, ligands):
        descs = self.descriptor_generator.build(ligands)
        return self.model.predict(descs)
    
    def score(self, ligands, target):
        descs = self.descriptor_generator.build(ligands)
        return self.model.score(descs,target)
    
    def cross_validate(n = 10, test_set = None, test_target = None):
        if test_set and test_target:
            cv_set = np.vstack((self.train_descs, test_set))
            cv_target = np.vstack((self.train_target, test_target))
        else:
            cv_set = self.train_descs
            cv_target = self.train_target
        return cross_val_score(self.model, cv_set, cv_target, cv = n)
        
    def save(self, filename):
        return pickle.dump(self, filename, compress=1)
    
    @classmethod
    def load(self, filename):
        return pickle.load(filename)
