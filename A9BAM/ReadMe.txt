$ python3 BAM.py

---------------Bidirectional---Associative---Memory-----------------

set A :
 [[ 1 -1  1  1]
 [ 1 -1 -1  1]
 [ 1 -1 -1 -1]
 [ 1 -1  1 -1]
 [ 1 -1  1 -1]
 [ 1 -1  1 -1]]

set B :
 [[ 1 -1 -1  1]
 [ 1 -1  1 -1]
 [ 1 -1  1  1]]

weight :
 [[2 2 4]
 [4 0 2]
 [2 2 0]
 [0 4 2]
 [0 4 2]
 [0 4 2]]

---------------------Testing--with--examples----------------------

x = trans( [[-1  1  1]] ), y_pred = trans( [[ 1 -1 -1  1  1  1]] )
x = trans( [[-1 -1 -1 -1 -1 -1]] ), y_pred = trans( [[-1 -1 -1]] )
x = trans( [[ 1 -1  1]] ), y_pred = trans( [[ 1  1 -1 -1 -1 -1]] )
x = trans( [[ 1 -1 -1  1  1  1]] ), y_pred = trans( [[-1  1  1]] )
x = trans( [[1 1 1]] ), y_pred = trans( [[1 1 1 1 1 1]] )
