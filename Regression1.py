import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

data= pd.read_csv("icecraemSellingData.csv")

x=data['Temperature (°C)']
y=data['Ice Cream Sales (units)']

sns.pairplot(data, x_vars='Temperature (°C)', y_vars='Ice Cream Sales (units)', height=4, kind='scatter')  
def PolReg(x,y):
    x_vals=np.array(x)
    y_vals=np.array(y)
    n_samp = len(x_vals)
    solution=[]

    Xf=[]
    for i in range(len(x_vals)):
        row=[]
        for j in range(3):
            row.append(x_vals[i]**j)
        Xf.append(row)
    
    w = np.zeros(3)
    bias=0 
    solution = np.dot(Xf,w) + bias  
    w,bias= Gradient(Xf,w,bias,solution,y_vals)
    
   
    yp = np.dot(Xf,w)+bias
    print("bias",bias)
    return yp

def Gradient(xf,w,bias,predicted,actual,lr=0.00001,max_iters=1000):
    Xf = np.array(xf)
    Xf_t = Xf.T
    delta = predicted - actual
    for _ in range(max_iters):
        dw = (-1 / len(xf)) * np.dot(Xf_t, delta)
        w += lr * dw 
        db = (-1 / len(xf)) * np.sum(delta)
        bias += lr * db
    
    return w, bias
def solcompute(Xf,W):
    yp=[]
    for i in range(len(Xf)):
        yp.append(np.dot(Xf[i], W))
    return yp

solution= PolReg(x,y)

print(solution)



plt.scatter(x, y, color="blue", label="Data points")  # Scatter plot of the data
plt.plot(x, solution, color="red", label="Polynomial Regression Line")  # Regression line
plt.xlabel("Temperature (°C)")
plt.ylabel("Ice Cream Sales (units)")
plt.legend()
plt.show()