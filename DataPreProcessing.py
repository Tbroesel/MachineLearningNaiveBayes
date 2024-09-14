

import numpy as np
import pandas as pd


class processData:

    def trainPreProcess(dat):

        X = pd.DataFrame(dat)

        X.replace(0.00, 1, inplace = True)          # replace 0 values
        X.replace(0, 1, inplace = True)             # replace 0 values
        X.replace('?', 1, inplace = True)             # replace 0 values
        X.replace('y', 1, inplace = True)             # replace y values
        X.replace('n', 0, inplace = True)             # replace n values
        X.replace('democrat', 1, inplace = True)    # replace dem values
        X.replace('republican', 0, inplace = True)  # replace rep values
        X = X.sample(frac = 1)                                      # shuffle the data
        n = round(int(len(X) * .50))                                # get size of training dataset
        X.drop(index = X.index[:n], axis = 0, inplace = True)       # cull specified data
        y = X[X.columns[-1]]                                        # move labels to y
        X.drop([X.columns[-1]], axis = 1, inplace = True)     # remove labels from X


        print(X)

        return X, y


    def testPreProcess(dat):

        X = pd.DataFrame(dat)
        X.drop([X.columns[-1]], axis = 1, inplace = True)
        y = dat[dat.columns[-1]]

        #X.replace(0.00, 0.01, inplace = True)

        print(X)

        return X, y
        
