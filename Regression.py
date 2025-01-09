import numpy as np
import pandas as pd
import seaborn as sns
from sympy import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

advertising = pd.read_csv( "advertising.csv" )

x= advertising['TV']
y= advertising['Sales']
sns.pairplot(advertising, x_vars='TV', y_vars='Sales', height=4, kind='scatter')

X_train, X_test, y_train, y_test = train_test_split( x, y, train_size = 0.7, test_size = 0.3, random_state = 100 )


def LinReg( X_train, y_train):

    X_train = np.array( X_train)
    y_train = np.array(y_train)

   
    a,b = gradientD(X_train,y_train,LL=1e-5)
    print("a=",a,"b=",b)
    sol= a+b*X_train
    print("sol",sol)
    
    sol_original = a + b * ((X_train - np.mean(X_train)) / np.std(X_train))
    return sol
def var(array):
    return np.sum((x-np.mean(array))**2 for x in array)/len(array)

def gradientD(x,actual,LL,max_iters=1000000, tolerance=0.0001):
    intercept=0
    slope = 0

    for _ in range(max_iters):
        difa,difb = numerical_derivative(intercept,slope,x,actual)
        stepb= LL*difb 
        stepa=LL*difa
        intercept -=stepa
        slope-=stepb
        print("-----------------------------------stepa",difa,"stepb",difb)
        if abs(difa) < tolerance or abs(difb) < tolerance:
            break

    return intercept, slope


def numerical_derivative(a,b,x,actual):
    prediction = a + b * x
    error = actual - prediction

    derivative_a = -2 * np.sum(error)/len(actual)
    derivative_b = -2 * np.sum(error * x)/len(actual)
    
    return derivative_a, derivative_b

   

def Rsquare(predicted,actual):
    print(predicted)
    ss_total =np.sum((predicted- np.mean(actual))**2)
    ss_residual = np.sum((actual- predicted)**2)
    return 1-(ss_residual/ss_total)

def ssr(a,b,x,actual):
    return np.sum((actual-(a+b*x))**2)

sol= LinReg(X_train, y_train)
plt.plot(X_train, sol, color="red", )
plt.legend()
plt.show()