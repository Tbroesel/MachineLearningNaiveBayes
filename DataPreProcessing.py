

import numpy as np
import pandas as pd


class processData:

    def trainPreProcess(dat):

        X = pd.DataFrame(dat)                                       # enforce data type

        X.replace(0.00, 0.01, inplace = True)         # replace 0 values
        X.replace(0, 1, inplace = True)              # replace 0 values
        X.replace("?", 0, inplace = True)           # replace ? values
        X.replace("y", 1, inplace = True)           # replace y values
        X.replace("n", 0, inplace = True)           # replace n values
        X = X.sample(frac = 1)                                      # shuffle the data
        n = len(X) - round(int(len(X) * .20))                       # get size of training dataset and apply training set size parameter
        X.drop(index = X.index[:n], axis = 0, inplace = True)       # cull data from training set
        y = X[X.columns[-1]]                                        # pull labels out into label class
        X.drop([X.columns[-1]], axis = 1, inplace = True)     # remove labels from data class
        X.drop([X.columns[0]], axis = 1, inplace = True)     # remove index from data class

        print(X)
        #print(y)

        return X, y


    def testPreProcess(dat):

        X = pd.DataFrame(dat)
        X.drop([X.columns[-1]], axis = 1, inplace = True)
        y = dat[dat.columns[-1]]

        #print(X)

        return X, y
        
