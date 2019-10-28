import numpy as np 
import tensorflow as tf

def cal_placeholder_test(grid_size):
    y = tf.placeholder(tf.float32, [None]+[grid_size[0],grid_size[1], 8, 5+1])
    npY = np.array([None]+[grid_size[0],grid_size[1],8,6])
    #[None] + [grid_size[0],grid_size[1], 8, 5+1] 앞쪽 차원에 [None] 차원이 추가된다.
    #print(npY.shape)

def cal_grid_size(input_shape):
    grdSize = [x//32 for x in input_shape[:2]]
    
    cal_placeholder_test(grdSize)

    grid_h, grid_w = grdSize
    gridWh=np.reshape([grdSize[0],grdSize[1]],[1,1,1,1,2]).astype(np.float32)
    print("gridWH :")
    print(gridWh)
    rp = np.repeat(np.arange(grid_h),grid_h)
    cxcy = np.transpose([np.tile(np.arange(grid_w),grid_h),np.repeat(np.arange(grid_h),grid_w)])
    cxcy = np.reshape(cxcy,(1,grid_h,grid_w,1,2))
    print("CxCy:")
    print(cxcy)

IM_SIZE = (416,416)
cal_grid_size([IM_SIZE[0],IM_SIZE[1],3])

