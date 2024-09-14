

import pandas as pd
import NaiveBayes as nb
from DataPreProcessing import processData

Naive = nb.NaiveBayes()

print("\n"*3,"~*"*7, "Naive~Bayes~Algorithim", "~*"*7)
fileLocationInput = input(R"Enter File Location of Data: ")

# get raw data from dataset
dataRaw = pd.read_csv(fileLocationInput, header = None)

# process training data
X, y = processData.trainPreProcess(dataRaw)

# fit training data
Naive.fit(X, y)

# process test data
X, y = processData.testPreProcess(dataRaw)

# check accuracy of fit
print("Accuracy Score: {}".format(Naive.accuracy(y, Naive.predict(X))))
print("F1 Score: {}".format(Naive.f1_score(y, Naive.predict(X))))

