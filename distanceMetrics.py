import numpy as np  

def euclideanDistance(matrix):
    total=0
    for i in range(len(matrix[0])):
        total += (matrix[0][i] - matrix[1][i])**2
    return total**0.5

def ManhattanDistance(matrix):
    return np.einsum('i->',np.abs(matrix[0] - matrix[1]))
def MinkowskiDistance(matrix,p):
  
    return np.einsum('i->', np.abs(matrix[0] - matrix[1]) ** p) ** (1 / p)

x = np.random.randint(1,10,(2,3),dtype=np.int32)
y = np.random.randint(1,5,(3,3),dtype=np.int32)
print("y is:", y)
detofy= np.linalg.det(y)
print("determinant of y is:",detofy)
