import numpy as np

def np_case_six():
    grdH = 13
    grdW = 13 

    grdWH = np.reshape([grdH,grdW],[1,1,1,1,2]).astype(np.float32)
    
    cxcy = np.transpose([np.tile(np.arange(grdW),grdH),np.repeat(np.arange(grdH),grdW)])
    print(cxcy)

def np_case_five():
    x = np.arange(13*13*1024)
    x = np.reshape(x,[-1,13,13,1024])
    x = np.reshape(x,(-1,13,13,8,6))
    #원소의 개수가 일치하지 않아서 reshape 불가능
    print(x)
    

def np_case_four():
    x = np.arange(8112)
    rslt = np.reshape(x,(-1,13,13,8,6))
    txty = rslt[...,0:2]
    print(rslt[0])

def np_case_three():
    x = np.arange(12).reshape(1,1,1,12)
    print(x)
    print(x.reshape(-1))

def np_case_two():
    # x = np.arange(15).reshape(3,5)
    x = np.arange(24).reshape(2,3,4)
    print("Array\n")
    print(x)
    x = np.transpose(x)
    #Array 재배열
    #https://rfriend.tistory.com/289
    print("TransPose\n")
    print(x)
    x=np.transpose(x,(2,1,0))
    print("TransPose_2\n")
    print(x)

def np_case_one():
    arry = np.arange(2)
    print("Array : \n")
    print(arry)

    print("Reshape : \n")
    print(np.reshape(arry,[1,2]))
    #[원소 개수, 원소 개수 .....]
        

np_case_six()