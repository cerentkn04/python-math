def scalarM(matrix1, num):
    if isinstance(matrix1[0], list):  
        return [[element * num for element in row] for row in matrix1]
    else:  
        return [element * num for element in matrix1]
    
def Matrix_Multiplication(matrix1, matrix2):
   
    rows_matrix1 = len(matrix1)
    cols_matrix1 = len(matrix1[0])
    rows_matrix2 = len(matrix2)
    cols_matrix2 = len(matrix2[0])
    
    if cols_matrix1 != rows_matrix2:
        raise ValueError("Number of columns in matrix1 must equal number of rows in matrix2")
    
    result = [[0 for _ in range(cols_matrix2)] for _ in range(rows_matrix1)]

    for i in range(rows_matrix1):
        for j in range(cols_matrix2):
            for k in range(cols_matrix1): 
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    
    return result

def isRowEchelon(matrix):
    leading_one_col = -1 
    for row in matrix:
        
        for col_index, value in enumerate(row):
            if value != 0:

                if col_index <= leading_one_col:
                    return False  
                leading_one_col = col_index
                break 
        else:
  
            if leading_one_col == -1:  
                return False

    return True

def Gaussian_Elimination(matrix):
    columns=len(matrix[0])
    rows = len(matrix)
    
    for r in range(rows):

        pivot_row = r
        for i in range(r+1,rows):
            if abs(matrix[i][r] > abs(matrix[pivot_row][r])):
                pivot_row = i
            
        if pivot_row != r:
            matrix[r], matrix[pivot_row]= matrix[pivot_row],matrix[r]
        
        pivot = matrix[r][r]
        if(pivot !=0):
            matrix[r]= [x/pivot for x in matrix[r]]

        for i in range(r+1,rows):
            
            if matrix[i][r] != 0:  
                factor = matrix[i][r]
                matrix[i] = [matrix[i][j] - factor * matrix[r][j] for j in range(columns)]

    return matrix

matrix = [[1,2,3], [4,5,6]]
matrix1 = [[7,8], [9, 10],[11,12]]
matrix3 = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0]
]  

matrix4 = [
  [2, 3, -1, 8],
    [-1, 2, 1, 3],
    [3, 1, 2, 7],
    [3, 5, -3, 4],

]

result = Gaussian_Elimination(matrix4)
print(result) 