# Matrix Manupilation
import numpy as np
import matplotlib.pyplot as plt

def input_matrix(m, n):
    A = []
    for i in range(m):
        ac = []
        for j in range(n):
            aij = float(input(f"Give A({i,j}): "))
            ac.append(aij)

        A.append(ac)
    return A

def jacobian_matrix_eq(A, X, B):
    X_new = []
    for i in range(n):
        temp_x = B[i]
        # if i > 0:
        #     for j in range(0, i):
        #         temp_x -= A[i, j] * X[j]
        # if i < n-1:
        for j in range(0, n):
            if j != i:
                temp_x -= A[i, j] * X[j]

        temp_x /= A[i, i]
        X_new.append(float(temp_x))

    return X_new

n = 3

# A = np.random.randint(-10, 10, size = (n, n))
X = np.zeros([n])
# B = np.random.randint(-10, 10, size = (n))

A = np.array([[4, 1, -1], [2, 5, 1], [1, 1, 3]])

B = [3, 9, 7]

n_max = 50

X_arr = [[0, 0, 0]]
for _ in range(n_max):
    Xn = jacobian_matrix_eq(A, X_arr[-1], B)
    X_arr.append(Xn)

print(X_arr)

x1, x2, x3, x4 = [], [], [], []

for x0 in X_arr:
    x1.append(x0[0])
    x2.append(x0[1])
    x3.append(x0[2])


t = range(len(X_arr))

plt.plot(t, x1, label = "x1: ")
plt.plot(t, x2, label = "x2: ")
plt.plot(t, x3, label = "x3: ")
plt.legend()
plt.show()
print(x1[-1], x2[-1], x3[-1])

# Gaussian Elimination Method
import numpy as np

def gaussian_elimination(A, B):
    n = len(B)
    A = A.copy()
    B = B.copy()

    # Forward elimination
    for i in range(n):
        # Partial pivoting
        max_row = i
        for k in range(i+1, n):
            if abs(A[k][i]) > abs(A[max_row][i]):
                max_row = k
        
        # Swap rows properly
        temp_row = A[i].copy()
        A[i] = A[max_row]
        A[max_row] = temp_row
        
        temp_b = B[i]
        B[i] = B[max_row]
        B[max_row] = temp_b
        
        # Eliminate below
        for k in range(i+1, n):
            factor = A[k][i] / A[i][i]
            for j in range(i, n):
                A[k][j] = A[k][j] - factor * A[i][j]
            B[k] = B[k] - factor * B[i]
    
    # Back substitution
    X = [0.0] * n
    
    for i in range(n-1, -1, -1):
        temp_sum = B[i]
        for j in range(i+1, n):
            temp_sum = temp_sum - A[i][j] * X[j]
        X[i] = float(temp_sum / A[i][i])
    
    return X

n = 3

A = np.array([[4, 1, -1],
              [2, 5, 1],
              [1, 1, 3]],
              dtype=float)

B = np.array([3,
              9,
              7],
             dtype=float)

X = gaussian_elimination(A, B)

# print(X)
print("\n",X[0],
      "\n",X[1],
      "\n",X[2])



