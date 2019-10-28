import numpy as np

def show_arry(data):
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            print(data[0,i,j,:],end='')
        print('')

def pad(data):
    rslt = np.pad(data,((0,0),(2,2),(2,2),(0,0)),'constant')
    return rslt



x = [[[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]],[[[7,7],[8,8],[9,9]],[[10,10],[10,10],[10,10]]]]

m = [ 
        [ 
            [ [1],[1],[1] ],
            [ [1],[1],[1] ],
            [ [1],[1],[1] ] 
        ],
        [ 
            [ [2],[2],[2] ] ,
            [ [2],[2],[2] ] ,
            [ [2],[2],[2] ] 
        ],
        [ 
            [ [3],[3],[3] ] ,
            [ [3],[3],[3] ] ,
            [ [3],[3],[3] ] 
        ] 
    ]
#mnist data strcture, not use reshape
mPad = pad(m)

m = np.array(m)
mDim = m.ndim
shape = m.shape

show_arry(mPad)    


