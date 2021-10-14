import numpy as np

def makeArray(arr, dim1, dim2):
    vec = np.array(arr).reshape(dim1, dim2)
    return vec

def makeSet(v1, v2, v3, v4, ax):
    comb = np.concatenate((v1, v2, v3, v4), axis = ax)
    return comb

def computeWeight(setA, setB):
    x = setA.copy()
    y = setB.copy()
    weight = np.dot(x, y)
    return weight

def testing(u, weight, flag):
    v = None
    if flag == 0:
        v = np.dot(weight.T, u)
        v[v >= 0] = 1
        v[v < 0] = -1
    else:
        v = np.dot(weight, u)
        v[v > 0] = 1
        v[v <= 0] = -1
    
    return np.array(v)
    
inp1, out1 = makeArray([1,1,1,1,1,1], 6, 1), makeArray([1,1,1], 3, 1)
inp2, out2 = makeArray([-1,-1,-1,-1,-1,-1], 6, 1), makeArray([-1,-1,-1], 3, 1)
inp3, out3 = makeArray([1,-1,-1,1,1,1], 6, 1), makeArray([-1,1,1], 3, 1)
inp4, out4 = makeArray([1,1,-1,-1,-1,-1], 6, 1), makeArray([1,-1,1], 3, 1)

setA = makeSet(inp1, inp2, inp3, inp4, 1)
setB = makeSet(out1.T, out2.T, out3.T, out4.T, 0)

print("\n---------------Bidirectional---Associative---Memory-----------------\n")
print("set A :\n", setA)
print("\nset B :\n", setB.T)
W = computeWeight(setA, setB)

print("\nweight :\n", W)

print("\n---------------------Testing--with--examples----------------------\n")
print("x = trans(", out3.T, "), y_pred = trans(", testing(out3, W, 1).T, ")")
print("x = trans(", inp2.T, "), y_pred = trans(", testing(inp2, W, 0).T, ")")
print("x = trans(", out4.T, "), y_pred = trans(", testing(out4, W, 1).T, ")")
print("x = trans(", inp3.T, "), y_pred = trans(", testing(inp3, W, 0).T, ")")
print("x = trans(", out1.T, "), y_pred = trans(", testing(out1, W, 1).T, ")")
