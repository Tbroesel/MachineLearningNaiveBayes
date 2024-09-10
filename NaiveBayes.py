import numpy as np


class NaiveBayes:

    def fit(self, X, y):
        sample_count, feature_count = X.shape
        self._classes = np.unique(y)
        classes_count = len(self._classes)

        # calculate mean, variance, and prior for each class
        self._mean = np.zeros((classes_count, feature_count), dtype=np.float64)
        self._var = np.zeros((classes_count, feature_count), dtype=np.float64)
        self._priors = np.zeros(classes_count, dtype=np.float64)

        for i, c in enumerate(self._classes):
            
            X_c = X[y == c]
            self._mean[i, :] = X_c.mean(axis=0)

            self._var[i, :] = X_c.var(axis=0)
            self._priors[i] = X_c.shape[0] / float(sample_count)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for i, c in enumerate(self._classes):
            prior = np.log(self._priors[i])
            posterior = np.sum(np.log(self._pdf(i, x) + 1e-10))
            posterior = posterior + prior
            posteriors.append(posterior)

        # return class with the highest posterior
        return self._classes[np.argmax(posteriors)]

    # calculate probability distribution (using gaussian distribution)
    def _pdf(self, class_index, x):
        epsilon = 1e-10
        mean = self._mean[class_index]
        var = self._var[class_index]
        #print(-((x - mean) ** 2))
        #print(( ((var + epsilon) * 2)))
        numerator = np.exp(-((x - mean) ** 2) / ( ((var + epsilon) * 2)))

        denominator = np.sqrt(2 * np.pi * (var + epsilon))
        return numerator / denominator

    # calculate accuracy of algorithm compared to known results
    def accuracy(self,y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

