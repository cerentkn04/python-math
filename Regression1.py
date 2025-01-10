import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

data= pd.read_csv("icecraemSellingData.csv")

x=data['Temperature (°C)']
y=data['Ice Cream Sales (units)']


def PolReg(x,y):
    x_vals=np.array(x)
    y_vals=np.array(y)
    n_samp =np.shape(x_vals)
    X_train = np.column_stack((np.ones(n_samp), x_vals)) 
    w = np.zeros(X_train.shape[1])
    print(len(w))
    print(len(X_train))

    solution = np.dot(X_train, w) + 0

    
   
   
    

    return x_vals

solution= PolReg(x,y)
