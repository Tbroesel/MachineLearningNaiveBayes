

import numpy as np


class NaiveBayes:


    def __init__(self):

        self.features = list
        self.likelihoods = {}
        self.class_priors = {}
        self.pred_priors = {}

        self.X_train = np.array
        self.y_train = np.array
        self.train_size = int
        self.num_features = int


    def fit(self, X, y):
        self.features = list(X.columns)
        self.X_train = X
        self.y_train = y
        self.train_size = X.shape[0]
        self.num_features = X.shape[1]

        for feature in self.features:
            self.likelihoods[feature] = {}

            for outcome in np.unique(self.y_train):
                self.likelihoods[feature].update({outcome:{}})
                self.class_priors.update({outcome: 0})

        self._calc_class_prior()
        self._calc_likelihoods()


    def _calc_class_prior(self):
        for outcome in np.unique(self.y_train):
            outcome_count = sum(self.y_train == outcome)
            self.class_priors[outcome] = outcome_count / self.train_size


    def _calc_likelihoods(self):
        for feature in self.features:

            for outcome in np.unique(self.y_train):
                self.likelihoods[feature][outcome]['mean'] = self.X_train[feature][self.y_train[self.y_train == outcome].index.values.tolist()].mean()
                self.likelihoods[feature][outcome]['var'] = self.X_train[feature][self.y_train[self.y_train == outcome].index.values.tolist()].var()


    def _calc_predictor_prior(self):
        for feature in self.features:
            feature_values = self.X_train[feature].value_counts().to_dict()
            for feature_value, count in feature_values.items():
                self.pred_priors[feature][feature_value] = count / self.train_size


    def predict(self, X):
        results = []
        X = np.array(X)

        for query in X:
            probabilities_outcome = {}

            for outcome in np.unique(self.y_train):
                prior = self.class_priors[outcome]
                likelihood = 1
                evidence = 1

                for feat, feature_value in zip(self.features, query):
                    mean = self.likelihoods[feat][outcome]['mean']
                    var = self.likelihoods[feat][outcome]['var']
                    likelihood *= (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(feature_value - mean) ** 2 / (2 * var))

                posterior = (likelihood * prior)
                probabilities_outcome[outcome] = posterior

            result = max(probabilities_outcome, key = lambda x: probabilities_outcome[x])
            results.append(result)

        return np.array(results)


    # calculate accuracy of algorithm compared to known results
    def accuracy(self, y_true, y_pred):
        return float(sum(y_pred == y_true)/(len(y_true)) * 100)


    # calculate precision score
    def precision(self, y_true, y_pred):
        return float(sum(y_pred == y_true) / (sum(y_pred == y_true) + sum(y_pred != y_true)))


    # calculate recall score
    def recall(self, y_true, y_pred):
        return float(sum(y_pred == y_true) / (sum(y_pred == y_true) + sum(y_true != y_pred)))


    # calculate f1 score
    def f1_score(self, y_true, y_pred):
        return float(2 * ((self.precision(y_true, y_pred) * self.recall(y_true, y_pred)) / (self.precision(y_true, y_pred) + self.recall(y_true, y_pred))) * 100)
