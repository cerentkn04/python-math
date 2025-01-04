def scalarM(matrix1, num):
    if isinstance(matrix1[0], list):  
        return [[element * num for element in row] for row in matrix1]
    else:  
        return [element * num for element in matrix1]


matrix = [[1, 3], [10, 12]]
matrix = scalarM(matrix, 3)
print(matrix)  