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
    leading_one_col = -1  # Tracks the column index of the first 1 in each row
    for row in matrix:
        # Find the first non-zero element in the row
        for col_index, value in enumerate(row):
            if value != 0:
                # If this is the first non-zero element, check that it follows the leading one rule
                if col_index <= leading_one_col:
                    return False  # Leading 1 in this row must be to the right of the previous row's leading 1
                leading_one_col = col_index
                break  # Stop checking after the first non-zero element
        else:
            # If the row is all zeros, ensure it's after rows with non-zero elements
            if leading_one_col == -1:  # No leading one yet, which is an issue if it's not the first row
                return False

    return True


matrix = [[1,2,3], [4,5,6]]
matrix1 = [[7,8], [9, 10],[11,12]]
matrix3 = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0]
]  # False (Zero row not at the bottom)

matrix4 = [
  [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
]

print(isRowEchelon(matrix3))  # Output: False
print(isRowEchelon(matrix4))