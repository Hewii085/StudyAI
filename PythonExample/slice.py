import numpy as np

def ellipsis_case():
    arry = np.arange(1024)
    arry = np.reshape(arry,[2,2,2,128])
    print(arry[...,0:2].shape)

    
def three_dimesion_slice():
    npArry3d = np.zeros([4,4,4],dtype=np.int32)

    idx = 0
    for x in range(0,4):
        for y in range(0,4):
            for j in range(0,4):
                npArry3d[x,y,j] = idx
                idx += 1
    
    #rslt = npArry3d[:,:,1]
    #마지막 Dimension의 1번 원소들만 가져온다
    #rslt = npArry3d[:,1,:]
    #두번째 Dimension의 1번째 원소들만 가져온다
    #rslt = npArry3d[0,:,0]
    #첫번째 Dimension의 0번째 원소중 모든 두번째 Dimension에서 0번째 원소들만 가져온다
    rslt = npArry3d[...,0:1,0]
    print(rslt)


def two_dimension_slice():
    arry = [[1,2,3,4],[5,6,7,8]]
    npArry = np.zeros([2,4],dtype=np.int32)

    idx = 0
    for x in range(0,2):
        for y in range(0,4):
            npArry[x,y] = idx
            idx += 1
    rslt = npArry[0,:] 
    #[:,1]전체 array에서 0번째만 가져온다
    print(rslt)


if __name__ == '__main__':
    case()