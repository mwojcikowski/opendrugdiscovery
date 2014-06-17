from sklearn.ensemble import RandomForestRegressor

class randomforest:
    def __init__(self, n_trees = 500, n_jobs = -1):
        self.model = RandomForestRegressor(n_estimators = n_trees, oob_score = True, random_state=1, n_jobs=n_jobs)
    
    def train(self, input_descriptors, target_values):
        self.model.fit(input_descriptors, target_values)
    
    def score(self, input_descriptors):
         self.model.predict(input_descriptors)
         
    def debug():
    
    def performance():
        
    
