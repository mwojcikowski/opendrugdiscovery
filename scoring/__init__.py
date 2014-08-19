import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib as pickle

#class scorer(object):
#    def __init__(self, model_instance, descriptor_generator_instance):
#        self.model = model_instance
#        self.descriptor_generator = descriptor_generator_instance
#        
#    def fit(self, ligands, target):
#        self.train_descs = self.descriptor_generator.build(ligands)
#        self.train_target = target
#        return self.model.fit(self.train_descs,target)
#    
#    def predict(self, ligands):
#        descs = self.descriptor_generator.build(ligands)
#        return self.model.predict(descs)
#    
#    def score(self, ligands, target):
#        descs = self.descriptor_generator.build(ligands)
#        return self.model.score(descs,target)
#    
#    def cross_validate(n = 10, test_set = None, test_target = None):
#        if test_set and test_target:
#            cv_set = np.vstack((self.train_descs, test_set))
#            cv_target = np.vstack((self.train_target, test_target))
#        else:
#            cv_set = self.train_descs
#            cv_target = self.train_target
#        return cross_val_score(self.model, cv_set, cv_target, cv = n)
#        
#    def save(self, filename):
#        self.protein = None
#        return pickle.dump(self, filename, compress=9)
#    
#    @classmethod
#    def load(self, filename):
#        return pickle.load(filename)


### FIX ### If possible make ensemble scorer lazy, for now it consumes all ligands
class scorer(object):
    def __init__(self, model_instances, descriptor_generator_instances):
        self.model = model_instances
        if type(descriptor_generator_instances) is list:
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
        
    def fit(self, ligands, target):
        if self.single_descriptor:
            self.train_descs = self.descriptor_generator.build(ligands)
        else:
            self.train_descs = [desc_gen.build(ligands) for desc_gen in self.descriptor_generator]
        self.train_target = target
        
        if self.single_model and self.single_descriptor:
            return model.fit(self.train_descs,target)
        elif self.single_model and not self.single_descriptor:
            return [model.fit(desc,target) for desc in self.train_descs]
        else:
            return [model.fit(self.train_descs[n],target) for n, model in enumerate(self.model)]
    
    def predict(self, ligands):
        if self.single_model and self.single_descriptor:
            descs = self.descriptor_generator.build(ligands)
            return self.model.predict(descs)
        elif self.single_model and not self.single_descriptor:
            return [self.model.predict(descs) for desc in self.train_descs]
        else:
            descs = [desc_gen.build(ligands) for desc_gen in self.descriptor_generator]
            return [model.predict(descs[n],target) for n, model in enumerate(self.model)]
    
    def score(self, ligands, target):
        if self.single_model and self.single_descriptor:
            descs = self.descriptor_generator.build(ligands)
            return self.model.score(descs)
        elif self.single_model and not self.single_descriptor:
            return [self.model.score(descs) for desc in self.train_descs]
        else:
            descs = [desc_gen.build(ligands) for desc_gen in self.descriptor_generator]
            return [model.score(descs[n],target) for n, model in enumerate(self.model)]
    
    def set_protein(self, protein):
        self.protein = protein
        if self.single_descriptor:
            self.descriptor_generator.protein = protein
        else:
            for desc in self.descriptor_generator:
                desc.protein = protein
                
    
    ### TODO ### Test
    def cross_validate(n = 10, test_set = None, test_target = None):
        if test_set and test_target:
            cv_set = np.vstack((self.train_descs, test_set))
            cv_target = np.vstack((self.train_target, test_target))
        else:
            cv_set = self.train_descs
            cv_target = self.train_target
        return cross_val_score(self.model, cv_set, cv_target, cv = n)
        
    def save(self, filename):
        self.protein = None
        return pickle.dump(self, filename, compress=9)
    
    @classmethod
    def load(self, filename):
        return pickle.load(filename)
