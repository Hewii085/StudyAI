import numpy as np

print(np.repeat(3,4))
#3을 4번 반복
x=np.array([[1,2],[3,4]])
print(np.repeat(x,2))
#각 원소를 2번씩 반복하고 차원을 낮춤

print(np.repeat(x,2,axis=1))
print(np.repeat(x,2,axis=0))
print(np.repeat(x,[1,2],axis=0))
