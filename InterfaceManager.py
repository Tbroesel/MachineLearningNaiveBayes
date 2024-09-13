

import pandas as pd
import NaiveBayes as nb
from DataPreProcessing import processData


Naive = nb.NaiveBayes()


print("\n"*3,"~*"*7, "Naive~Bayes~Algorithim", "~*"*7)
fileLocationInput = input(R"Enter File Location of Data: ")


# get raw data from dataset
dataRaw = pd.read_csv(fileLocationInput)
#print(dataRaw)

# create training data set with subset of dataRaw


# process training data
X, y = processData.trainPreProcess(dataRaw)


# fit training data
Naive.fit(X, y)

X, y = processData.testPreProcess(dataRaw)


# check accuracy of fit
print("Accuracy Score: {}".format(Naive.accuracy(y, Naive.predict(X))))
print("F1 Score: {}".format(Naive.f1_score(y, Naive.predict(X))))


# test raw data in model








#print("\n"*2,"~~~~~Data Settings~~~~~\n")
#splitPercent = int(input("**What percent of the data do you want to be testing data: "))
#noise = bool(input("\n**Do you want to add noise(Leave empty for false): "))

#X_Train, X_Test, Y_Train, Y_Test = dataProcess.getData(splitPercent,noise)

#Naive.fit(X_Train, Y_Train)

#predictions = Naive.predict(X_Test)

#accuracy = Naive.accuracy(Y_Test,predictions)

#print("Accuracy of Naive Bayes: ", accuracy)