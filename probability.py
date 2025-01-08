import numpy as np
def permutation(n):
    if n < 1:
        return 1  
    return np.prod(np.arange(1, n + 1)) 
def permutationsOfSubsets(m,r):
    if m < 1:
        return 1  
    return np.prod(np.arange(1, m + 1)) / np.prod(np.arange(1,m-r))


def Combinations(n,r):
    return permutationsOfSubsets(n,r)*1/permutation(r)


def condP(A,B,W):
    return P(B,A)/P(A,W)
def P(A,W):
    return A/W

    

print(permutation(3))