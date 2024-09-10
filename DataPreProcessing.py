import numpy as np
import pandas as pd
import random

class processData:

    def __init__(self, DataFileLocation):
        self.DataFileLocation = DataFileLocation

        dataRaw = pd.read_csv(DataFileLocation, header=None, delimiter=",")

        self.dataNumpy = dataRaw.to_numpy()
        

         #Shuffle the data
        np.random.shuffle(self.dataNumpy)

        #Y data is one dimensional array that has the 'results' i.e. the thing we are trying to 'guess' [last column]
        self.Y_Data = self.dataNumpy[:,-1]
        #X_Data saves all the data except the 'results' we will use this to see of we can guess the results with this
        self.X_Data = self.dataNumpy[:, :-1]

        numRows, numColumns = self.dataNumpy.shape

        

    def getData(self, split = 30, noise = False):

            if(noise):
                #Get the number of columns to shuffle the .1 is ten percent of the colmns to be shuffled
                numColsShuffled = int((self.X_Data.shape[0]) * .1)
                
                

                colsToBeShuffledCalculated = False

                while colsToBeShuffledCalculated == False:
                    stop = False
                    randCol = np.random.randint(low=1, high=numColsShuffled,size=self.X_Data.shape[1])
        
                #this function checks if the columns have duplicates

                    for i in range(randCol.shape[0]):
                        for j in range(randCol.shape[0]):
                            if((randCol[i] == randCol[j] and  i != j)):
                                stop = True
                            else:
                                pass

                    if(stop):
                        pass
                    else:
                        colsToBeShuffledCalculated = True
    
                for col in range(self.X_Data.shape[1]): # num of cols
                    for colCheck in range(randCol[0]): # if the colum is one of the colums we need to randomize
                        if(colCheck == col):
                            onColumn = True
                        if(onColumn):
                           np.random.shuffle(self.X_Data[:,col]) #this should shuffle to column and not rows


            
            percentSplit = split / 100.0

            #Splits the whole x & y data into parts based on percent giving train majority (percentSplit number is percent of data to be used in testing data)
            X_Train, X_Test = np.split(self.X_Data, [int(len(self.X_Data)*percentSplit)])

            Y_Train, Y_Test = np.split(self.Y_Data, [int(len(self.Y_Data)*percentSplit)])

            return X_Train, X_Test, Y_Train, Y_Test

       

        



