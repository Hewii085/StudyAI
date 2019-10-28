import numpy as np


#np.tile을 이용하여 해당 값의 원소를 반복시켜 행렬을 만든다
A = 1
B = np.array([0,1])
C = np.array([[0,1],[2,3]])

print(np.tile(A,3))
print(np.tile(B,3))
print(np.tile(C,3))

print("########################################\n")

print(np.tile(A,(2,3)))
#1번쨰 Dimension 묶음 2번반복, 2번째 DImension에 3번 반복
print(np.tile(B,(2,3))) 
print(np.tile(C,(2,3)))