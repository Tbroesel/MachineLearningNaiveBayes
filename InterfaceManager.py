

import pandas as pd
import NaiveBayes as nb
from DataPreProcessing import processData
import matplotlib.pyplot as plt
import seaborn as sb

Naive = nb.NaiveBayes()

print("\n"*3,"~*"*7, "Naive~Bayes~Algorithim", "~*"*7)
fileLocationInput = input(R"Enter File Location of Data: ")

# get raw data from dataset
dataRaw = pd.read_csv(fileLocationInput, header = None)

# process training data
X, y = processData.trainPreProcess(dataRaw)

plt.figure()
corr = dataRaw.iloc[:,:-1].corr(method="pearson")
cmap = sb.diverging_palette(250,354,80,60,center='dark',as_cmap=True)
sb.heatmap(corr,vmax=1,vmin=-.5,cmap=cmap,square=True,linewidths=.2)

# fit training data
Naive.fit(X, y)

# process test data
X, y = processData.testPreProcess(dataRaw)

# check accuracy of fit
print("Accuracy Score: {}".format(Naive.accuracy(y, Naive.predict(X))))
print("F1 Score: {}".format(Naive.f1_score(y, Naive.predict(X))))


plt.figure()
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c='blue', marker="*")



plt.show()