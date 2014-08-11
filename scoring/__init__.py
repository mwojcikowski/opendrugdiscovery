import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib as pickle

class scorer:
    def __init__(self, model, descriptor_generator, model_opts, desc_opts):
        self.model = model(**model_opts)
        self.descriptor_generator = descriptor_generator(**desc_opts)
        
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
        
    def save(filename):
        f = open(filename,'wb')
        pickle.dump(self, filename)
        f.close()
    
    @classmethod
    def load(filename):
        return pickle.load(open(filename,'rb'))
