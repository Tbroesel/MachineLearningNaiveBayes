

import numpy as np
import pandas as pd


class processData:

    def trainPreProcess(dat):

        X = pd.DataFrame(dat)

        X.replace(0.00, 0.01, inplace = True)   # replace 0 values
        X = X.sample(frac = 1)                                  # shuffle the data
        n = round(int(len(X) * .20))                             # get size of training dataset
        X.drop(index = X.index[:n], axis = 0, inplace = True)
        y = X[X.columns[-1]]
        X.drop([X.columns[-1]], axis = 1, inplace = True)   # remove labels


        print(X)

        return X, y


    def testPreProcess(dat):

        X = pd.DataFrame(dat)
        X.drop([X.columns[-1]], axis = 1, inplace = True)
        y = dat[dat.columns[-1]]

        #X.replace(0.00, 0.01, inplace = True)

        print(X)

        return X, y
        
