
import NaiveBayes as nb
import DataPreProcessing as dpp


Naive = nb.NaiveBayes()

print("\n"*3,"~*"*7, "Naive~Bayes~Algorithim", "~*"*7)
fileLocationInput = input(R"\n**Enter File Location of Data: ")
dataProcess = dpp.processData(fileLocationInput)

print("\n"*2,"~~~~~Data Settings~~~~~\n")
splitPercent = int(input("**What percent of the data do you want to be testing data: "))
noise = bool(input("\n**Do you want to add noise(Leave empty for false): "))

X_Train, X_Test, Y_Train, Y_Test = dataProcess.getData(splitPercent,noise)

Naive.fit(X_Train, Y_Train)

predictions = Naive.predict(X_Test)

accuracy = Naive.accuracy(Y_Test,predictions)

print("Accuracy of Naive Bayes: ", accuracy)