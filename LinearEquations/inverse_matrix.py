from matrix_utility import row_addition_elementary_matrix, scalar_multiplication_elementary_matrix
import numpy as np

"""
Function that finds the inverse of a non-singular matrix
The function performs elementary row operations to transform it into the identity matrix.
The resulting identity matrix will be the inverse of the input matrix if it is non-singular.
If the input matrix is singular (i.e., its diagonal elements become zero during row operations), it raises an error.
"""

def matrix_inverse(matrix):
    print(f"=================== Finding the inverse of a non-singular matrix using elementary row operations ===================\n {matrix}\n")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    n = matrix.shape[0]
    identity = np.identity(n)

    # Initialize variables to store last three elementary matrices
    last_three_elementary_matrices = []

    # Perform row operations to transform the input matrix into the identity matrix
    for i in range(n):
        if matrix[i, i] == 0:
            for k in range(i + 1, n):
                if matrix[k, i] != 0 and matrix[i, k] != 0:
                    # Swap rows i and k
                    matrix[[i, k]] = matrix[[k, i]]
                    identity[[i, k]] = identity[[k, i]]
                    # print(f"Swapped rows {i + 1} and {k + 1}:\n", matrix)
                    break
            else:
                raise ValueError("Matrix is singular, cannot find its inverse.")

        if matrix[i, i] != 1:
            # Scale the current row to make the diagonal element 1
            scalar = 1.0 / matrix[i, i]
            elementary_matrix = scalar_multiplication_elementary_matrix(n, i, scalar)

            # Store the elementary matrix
            last_three_elementary_matrices.append(elementary_matrix)

            print(f"Elementary matrix to make the diagonal element 1: \n {elementary_matrix}")
            matrix = np.dot(elementary_matrix, matrix)
            print(f"The matrix after elementary operation: \n {matrix}")
            print("------------------------------------------------------------------------------------------------------------------")
            identity = np.dot(elementary_matrix, identity)

        # Zero out the elements above and below the diagonal
        for j in range(n):
            if i != j:
                scalar = -matrix[j, i]
                elementary_matrix = row_addition_elementary_matrix(n, j, i, scalar)

                # Store the last three elementary matrices
                if len(last_three_elementary_matrices) < 3:
                    last_three_elementary_matrices.append(elementary_matrix)
                else:
                    last_three_elementary_matrices.pop(0)  # Remove the oldest

                print(f"Elementary matrix for R{j+1} = R{j+1} + ({scalar}R{i+1}):\n {elementary_matrix}")
                matrix = np.dot(elementary_matrix, matrix)
                print(f"The matrix after elementary operation: \n {matrix}")
                print("------------------------------------------------------------------------------------------------------------------")
                identity = np.dot(elementary_matrix, identity)

    print("\nLast three elementary matrices used:")
    for elementary_matrix in last_three_elementary_matrices:
        print(elementary_matrix)

    return identity


if __name__ == '__main__':

    A = np.array([[1, 0.5, 0.15],
                  [0.5, 0.15, 0.20],
                  [0.15, 0.20, 0.25]])

    try:
        A_inverse = matrix_inverse(A)
        print("\nInverse of matrix A: \n", A_inverse)
        print("=====================================================================================================================")

    except ValueError as e:
        print(str(e))
