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

sol=0
def LinReg( X_train, y_train):

    X_train = np.array( X_train)
    y_train = np.array(y_train)
    n= len(X_train)
    
    b=1
    a=0
    sol= a+ b*X_train
    print("-----------------------------------sol",sol)
    ssr(a,b,X_train,y_train)
    intercept,slope ,sol = gradientD(X_train,y_train,0.1,)

 

    return intercept,slope,sol

def var(array):
    return np.sum((x-np.mean(array))**2 for x in array)/len(array)

def gradientD(x,actual,LL,max_iters=10, tolerance=1e-6):
    intercept=0
    slope = 0.0
    a, b = symbols('a b')
    ssr = sum((actual - (a + b * x))**2 for actual, x in zip(actual, x))
    difa = diff(ssr,a)
    difb = diff(ssr,b)
    for _ in range(max_iters):
        grad_a=difa.subs({a:intercept,b:slope})
        grad_b=difb.subs({a:intercept,b:slope})
        intercept -= LL * grad_a
        slope -= LL * grad_b
        ssr =sum((actual - ( intercept+ slope * x))**2 for actual, x in zip(actual, x))
        
    
    return intercept, slope, ssr


     

def Rsquare(predicted,actual):
    print(predicted)
    ss_total =np.sum((predicted- np.mean(actual))**2)
    ss_residual = np.sum((actual- predicted)**2)
    return 1-(ss_residual/ss_total)

def ssr(a,b,x,actual):
    return np.sum((actual-(a+b*x))**2)
intercept, slope, sol = LinReg(X_train, y_train)

# Evaluating intercept and slope to numeric values (use .evalf())
intercept_value = intercept.evalf()
slope_value = slope.evalf()

# Plotting the regression line
plt.plot(X_train, sol, color="red", label=f"Regression Line: y = {intercept_value:.2f} + {slope_value:.2f}x")
plt.legend()
plt.show()
