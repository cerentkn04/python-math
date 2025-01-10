import numpy as np
import pandas as pd
import seaborn as sns
from sympy import *
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

advertising = pd.read_csv( "advertising.csv" )
studentPerformance=pd.read_csv("Student_Performance.csv")
Xs = studentPerformance[['Hours Studied', 'Previous Scores']]
Y=studentPerformance['Performance Index']

x= advertising['TV']
y= advertising['Sales']

sns.pairplot(studentPerformance, vars=['Hours Studied', 'Previous Scores', 'Performance Index'], height=4, kind='scatter')
sns.pairplot(advertising, x_vars='TV', y_vars='Sales', height=4, kind='scatter')  

X_train, X_test, y_train, y_test = train_test_split( x, y, train_size = 0.7, test_size = 0.3, random_state = 100 )
xs = np.array( Xs)
def LinReg( X_train, y_train):
    X_train = np.array( X_train)
    y_train = np.array(y_train)   
    a,b = gradientD(X_train,y_train,LL=1e-5)
    sol= a+b*X_train
   
    print("Rsquare is: ",Rsquare(sol,y_train))
    
    return sol
#----------------------------------------------------------------
def MultiLinReg( X_train, y_train,max_iters=10000,ll=1e-6,tolerance=0.0000001):
    n_samples, n_features = X_train.shape
    X_train = np.column_stack((np.ones(n_samples), X_train))  
    y_train = np.array(y_train)

    weights = np.zeros(n_features + 1)  

    for _ in range(max_iters):
        predictions = np.dot(X_train, weights)
        errors = y_train - predictions

        gradient = -2 * np.dot(X_train.T, errors) / n_samples
        weights -= ll * gradient

        if np.all(np.abs(gradient) < tolerance):
            break

    bias = weights[0]
    coefficients = weights[1:]
    predicted = np.dot(X_train, weights)
    
    print("Rsquare is: ", Rsquare(predicted, y_train))
    print("Bias and coefficients:", bias, coefficients)
    return bias, coefficients
#----------------------------------------------------------------
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
    ss_total = np.sum((actual - np.mean(actual))**2)
    ss_residual = np.sum((actual- predicted)**2)
    return 1-(ss_residual/ss_total)

def ssr(a,b,x,actual):
    return np.sum((actual-(a+b*x))**2)

#sol= LinReg(X_train, y_train)

#plt.plot(X_train, sol, color="red", )
#plt.legend()
#plt.show()
bias, (coef_hours, coef_scores) = MultiLinReg(Xs, Y)

# Create meshgrid for regression plane
hours_range = np.linspace(Xs['Hours Studied'].min(), Xs['Hours Studied'].max(), 50)
scores_range = np.linspace(Xs['Previous Scores'].min(), Xs['Previous Scores'].max(), 50)
x_surf, y_surf = np.meshgrid(hours_range, scores_range)

# Predict Performance Index using the regression plane
z_surf = bias + coef_hours * x_surf + coef_scores * y_surf

# Plotting the data points and regression plane
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xs['Hours Studied'], Xs['Previous Scores'], Y, color='b', label='Data Points')
ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.5, cmap='viridis', rstride=1, cstride=1, edgecolor='none')

# Labels and title
ax.set_xlabel('Hours Studied')
ax.set_ylabel('Previous Scores')
ax.set_zlabel('Performance Index')
ax.set_title('3D Regression Plane for Student Performance')
plt.legend()
plt.show()