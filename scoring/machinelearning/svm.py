from sklearn.svm import SVR

class randomforest:
    def __init__(self, n_trees = 500, n_jobs = -1):
        self.model = SVR(kernel='rbf', C=1e3, gamma=0.1)
        RandomForestRegressor(n_estimators = n_trees, oob_score = True, random_state=1, n_jobs=n_jobs)
    
    def train(self, input_descriptors, target_values):
        self.model.fit(input_descriptors, target_values)
    
    def predict(self, input_descriptors):
         return self.model.predict(input_descriptors)