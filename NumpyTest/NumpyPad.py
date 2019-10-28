import numpy as np
import matplotlib.pyplot as plt

data = np.array([[1,2,3,4,5],[6,7,8,9,10]])

padding = 2
npad = ((2,2),(0,0))
data_padding = np.pad(data,npad,'constant',constant_values=(0))

data2 = np.array([[[1,1,1,1,1],[1,2,2,2,2]],[[3,3,3,3,3],[4,4,4,4,4]]])
npad2 = ((0,0),(2,2),(2,2))#rank를 맞춰준다?
data_padding2 = np.pad(data2,npad2,'constant',constant_values=(0))

data3 = np.array([[[[1,1],[2,2]],[[1,1],[2,2]]],[[[1,1],[2,2]],[[1,1],[2,2]]]])
npad3 = ((1,1),(1,1),(1,1),(1,1))
data_padding3 = np.pad(data3,npad3,'constant',constant_values=(9))

data4 = np.array([1,2,3,4,5])
npad4 = ((1,1))
data_padding4 = np.pad(data4,npad4,'constant',constant_values=(0))

print(data_padding4)

# plt.imshow(data_padding,cmap='gray')
# plt.show()

#pad_width = 항목의 개수가 변형하고자 하는데이터의 dimension을 넘을 수 없다. 