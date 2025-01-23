import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


data = pd.read_csv("iris.csv")
y = data.iloc[0:99, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
X = data.iloc[0:99, [0, 2]].values

class AdalineGD:
    def __init__(self,Eta=0.01,N_iters=50,random_state=1):
        self.eta =Eta
        self.n_iters = N_iters
        self.random_state = random_state

    def fit(self, X, y):
        rgen= np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0,scale=0.01,size=X.shape[1])
        self.b_ = np.float64(0.)
        self.losses= []

        for i in range(self.n_iters):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            error = y - output
            self.w_ += self.eta * 2.0 * X.T.dot(error) / X.shape[0]
            self.b_ += self.eta*2.0 *error.mean()
            loss= (error**2).mean()
            self.losses.append(loss)
        return self
            

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_      
    def activation(self, x):
        return x
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada1 = AdalineGD(N_iters=15, Eta=0.1).fit(X, y)
ax[0].plot(range(1, len(ada1.losses) + 1),np.log10(ada1.losses), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Mean squared error)')
ax[0].set_title('Adaline - Learning rate 0.1')
ada2 = AdalineGD(N_iters=15, Eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.losses) + 1),ada2.losses, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Mean squared error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()
