import numpy as np
from scipy.stats import linregress
from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib as pickle

### FIX ### If possible make ensemble scorer lazy, for now it consumes all ligands
class scorer(object):
    def __init__(self, model_instances, descriptor_generator_instances, score_title = 'score'):
        self.model = model_instances
        if type(model_instances) is list:
            self.single_model = False
        else:
            self.single_model = True
        
        self.descriptor_generator = descriptor_generator_instances
        if type(descriptor_generator_instances) is list:
            if len(descriptor_generator_instances) == len(model_instances):
                raise ValueError, "Length of models list doesn't equal descriptors list"
            self.single_descriptor = False
        else:
            self.single_descriptor = True
        self.score_title = score_title
        
    def fit(self, ligands, target, *args, **kwargs):
        if self.single_descriptor:
            self.train_descs = self.descriptor_generator.build(ligands)
        else:
            self.train_descs = [desc_gen.build(ligands) for desc_gen in self.descriptor_generator]
        self.train_target = target
        
        if self.single_model and self.single_descriptor:
            return model.fit(self.train_descs,target, *args, **kwargs)
        elif self.single_model and not self.single_descriptor:
            return [model.fit(desc,target, *args, **kwargs) for desc in self.train_descs]
        else:
            return [model.fit(self.train_descs[n],target, *args, **kwargs) for n, model in enumerate(self.model)]
    
    def predict(self, ligands, *args, **kwargs):
        if self.single_model and self.single_descriptor:
            descs = self.descriptor_generator.build(ligands)
            return self.model.predict(descs)
        elif self.single_model and not self.single_descriptor:
            return [self.model.predict(descs, *args, **kwargs) for desc in self.train_descs]
        else:
            descs = [desc_gen.build(ligands) for desc_gen in self.descriptor_generator]
            return [model.predict(descs[n],target, *args, **kwargs) for n, model in enumerate(self.model)]
    
    def score(self, ligands, target, *args, **kwargs):
        if self.single_model and self.single_descriptor:
            descs = self.descriptor_generator.build(ligands)
            return self.model.score(descs, *args, **kwargs)
        elif self.single_model and not self.single_descriptor:
            return [self.model.score(descs, *args, **kwargs) for desc in self.train_descs]
        else:
            descs = [desc_gen.build(ligands) for desc_gen in self.descriptor_generator]
            return [model.score(descs[n],target, *args, **kwargs) for n, model in enumerate(self.model)]
    
    def predict_ligands(self, ligands):
        # make lazy calculation
        for lig in ligands:
            score = self.predict([lig])
            lig.data.update({self.score_title: score[0]})
            yield lig
    
    def set_protein(self, protein):
        self.protein = protein
        if self.single_descriptor:
            self.descriptor_generator.protein = protein
        else:
            for desc in self.descriptor_generator:
                desc.protein = protein
    
    def cross_validate(self, n = 10, test_set = None, test_target = None, n_jobs = 1):
        if test_set and test_target:
            cv_set = np.vstack((self.train_descs, self.test_descs, test_set))
            cv_target = np.hstack((self.train_target.flatten(), self.test_target.flatten(), test_target.flatten()))
        else:
            cv_set = np.vstack((self.train_descs, self.test_descs))
            cv_target = np.hstack((self.train_target.flatten(), self.test_target.flatten()))
        return cross_val_score(self.model, cv_set, cv_target, cv = n, n_jobs = n_jobs)
        
    def save(self, filename):
        self.protein = None
        if self.single_descriptor:
            self.descriptor_generator.protein = None
        else:
            for desc in self.descriptor_generator:
                desc.protein = None
        return pickle.dump(self, filename, compress=9)[0]
    
    @classmethod
    def load(self, filename):
        return pickle.load(filename)
    

class ensemble_model(object):
    def __init__(self, models):
        self._models = models if len(models) else None
    
    def fit(self, X, y, *args, **kwargs):
        for model in self._models:
            model.fit(X, y, *args, **kwargs)
        return self
    
    def predict(self, X, *args, **kwargs):
        return np.array([model.predict(X, *args, **kwargs) for model in self._models]).mean(axis=0)
    
    def score(self, X, y, *args, **kwargs):
        return linregress(self.predict(X, *args, **kwargs).flatten(), y.flatten())[2]**2
