import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def loss_1(z):
    return -np.log(sigmoid(z))
def loss_2(z):
    return -np.log(1 - sigmoid(z))
z= np.arange(-10,10,0.1)
sigma_z = sigmoid(z)
c1=[loss_1(x)for x in z]
plt.plot(sigma_z, c1,label='L(w,b)  if y=1')
c0=[loss_2(x)for x in z]
plt.plot(sigma_z, c0,linestyle ='--',label='L(w,b) if y=0')
plt.xlim([0,1])
plt.ylim([0.0,5.1])
plt.xlabel('$\sigma(z)$')
plt.ylabel('L(w, b)')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

