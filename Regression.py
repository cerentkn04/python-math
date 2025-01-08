import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

advertising = pd.read_csv( "advertising.csv" )

x= advertising['TV']
y= advertising['Sales']
sns.pairplot(advertising, x_vars='TV', y_vars='Sales', height=4, kind='scatter')

X_train, X_test, y_train, y_test = train_test_split( x, y, train_size = 0.7, test_size = 0.3, random_state = 100 )

sol=0
def LinReg( X_train, y_train):

    X_train = np.array( X_train)
    y_train = np.array(y_train)
    n= len(X_train)
 
    sumX = np.sum(X_train)
    sumX2 = np.sum(X_train**2)
    sumY = np.sum(y_train)
    sumXY = np.sum(X_train * y_train)
  
    
    b=(n * sumXY - sumX * sumY)/(n * sumX2 - sumX**2)
    a=(sumY -b*sumX)/n
    sol= a+ b*X_train

    return a,b,sol

def var(array):
    return np.sum((x-np.mean(array))**2 for x in array)/len(array)

def Rsquare(predicted,actual):
    ss_total =np.sum((predicted- np.mean(actual))**2)
    ss_residual = np.sum((actual- predicted)**2)
    return 1-(ss_residual/ss_total)

a,b,sol= LinReg(X_train, y_train)
plt.plot(X_train, sol, color="red", label=f"Regression Line: y = {a:.2f} + {b:.2f}x")
plt.show()
