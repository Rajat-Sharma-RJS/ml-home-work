import numpy as np
import matplotlib.pyplot as plt

def initialization(row, col):
  matrix = []
  for i in range(row):
    arr = []
    for j in range(col):
      w1 = np.random.randint(-100, 100)/100
      w2 = np.random.randint(-100, 100)/100

      arr.append([w1, w2])

    matrix.append(arr)

  matrix = np.array(matrix)
  return matrix

def createTrainingData(N):
  vec = []
  for i in range(N):
    x1 = np.random.randint(-100, 100)/100
    x2 = np.random.randint(-100, 100)/100

    vec.append([x1, x2])
  
  vec = np.array(vec)
  return vec

def safe(x, y, M, N):
  if (x >= 0 and x < M and y >= 0 and y < N):
    return True
  return False

def neighbourhood(i, j, M, N, radius):
  neigh = []
  for r in range(radius):
    rd = r+1
    for v in range(rd+1):
      if safe((i+v), (j+(rd-v)), M, N) and ([i+v, j+(rd-v)] not in neigh):
        neigh.append([i+v, j+(rd-v)])
      if safe((i-v), (j+(rd-v)), M, N) and ([i-v, j+(rd-v)] not in neigh):
        neigh.append([i-v, j+(rd-v)])
      if safe((i+v), (j-(rd-v)), M, N) and ([i+v, j-(rd-v)] not in neigh):
        neigh.append([i+v, j-(rd-v)])
      if safe((i-v), (j-(rd-v)), M, N) and ([i-v, j-(rd-v)] not in neigh):
        neigh.append([i-v, j-(rd-v)])
  
  neigh.append([i, j])
  neigh = np.array(neigh)
  return neigh

def trainingNeurons(neurons, train, alpha):
  initial = neurons.copy()
  M = len(initial)
  N = len(initial[0])

  radius = None
  if M % 2 == 1:
    radius = int(M/2)
  else:
    radius = int((M-1)/2)
  
  matrix100 = None
  matrix500 = None
  converge = False
  itr = 0
  e = 1
  n = 1e3
  while not converge:
    for x in train:
      x1, x2 = x[0], x[1]
      r, c = 0, 0
      dist = 100
      for i in range(M):
        for j in range(N):
          w1, w2 = initial[i, j][0], initial[i, j][1]
          val = (w1 - x1)**2 + (w2 - x2)**2
          if(val < dist):
            dist = val  ## min value
            r, c = i, j
      
      neigh = neighbourhood(r, c, M, N, radius)
      for ent in neigh:
        p1, p2 = ent[0], ent[1]
        ## change matrix at (p1,p2)
        initial[p1, p2] = initial[p1, p2] + alpha*(x - initial[p1, p2])
    
    e += 1
    
    ## reduce radius
    if (itr == 50 or itr == 250 or itr == 500):
      print("iteration %d : changing radius from %d to %d"%(itr, radius, radius-1))
      if radius > 1:
        radius = radius - 1
      ## else don't change radius
    
    if (itr == 100):
      matrix100 = initial.copy()
    if (itr == 500):
      matrix500 = initial.copy()
    
    itr += 1
    if (e > n):
      converge = True
      print("Converged : %d iteration"%itr)
  return (initial, matrix100, matrix500)

def distribute(matrix):
  vec1, vec2 = [], []
  for i in range(len(matrix)):
    for j in range(len(matrix[0])):
      vec1.append(matrix[i, j][0])
      vec2.append(matrix[i, j][1])
  
  vec1 = np.array(vec1)
  vec2 = np.array(vec2)
  return vec1, vec2

def testing(final, test):
  M = len(final)
  N = len(final[0])
  output = []
  for x in test:
    x1, x2 = x[0], x[1]
    dist = 100
    r, c = 0, 0
    for i in range(M):
      for j in range(N):
        w1, w2 = final[i, j][0], final[i, j][1]
        val = (w1 - x1)**2 + (w2 - x2)**2
        if (val < dist):
          dist = val
          r, c = i, j
    vec = []
    vec.append(N*r+c+1)
    vec.append(round(final[r, c][0],2))
    vec.append(round(final[r, c][1],2))
    output.append(vec)
  
  output = np.array(output)
  return output

neurons = initialization(10, 10)
train = createTrainingData(1500)
alpha = 0.1
print("\nTraining neurons\n")
final, mid100, mid500 = trainingNeurons(neurons, train, alpha)
vec1, vec2 = distribute(final)
arr1, arr2 = distribute(neurons)
arr3, arr4 = distribute(mid100)
arr5, arr6 = distribute(mid500)

## subplots
fig, axis = plt.subplots(1, 4, figsize=(20, 5))
axis[0].plot(arr1, arr2, 'r-')
axis[0].set_title("Initially")
axis[1].plot(arr3, arr4, 'r-')
axis[1].set_title("After 100 iterations")
axis[2].plot(arr5, arr6, 'r-')
axis[2].set_title("After 500 iterations")
axis[3].plot(vec1, vec2, 'r-')
axis[3].set_title("After 1000 iterations")

## testing performance with input vectors
print("\nTesting neurons\n")
X1 = [0.1, 0.8]
X2 = [0.5, -0.2]
X3 = [-0.8, -0.9]
X4 = [-0.6, 0.9]

print("X1 = [0.1, 0.8].T")
print("X2 = [0.5, -0.2].T")
print("X3 = [-0.8, -0.9].T")
print("X4 = [-0.6, 0.9].T")

test = [X1, X2, X3, X4]
test = np.array(test)
output = testing(final, test)
print("\nAfter Testing the neurons with input data :\n")
print("Neuron %d responds to the input vector X1 i.e., n1 = [%0.2f, %0.2f]"%(output[0,0], output[0, 1], output[0, 2]))
print("Neuron %d responds to the input vector X2 i.e., n2 = [%0.2f, %0.2f]"%(output[1,0], output[1, 1], output[1, 2]))
print("Neuron %d responds to the input vector X3 i.e., n3 = [%0.2f, %0.2f]"%(output[2,0], output[2, 1], output[2, 2]))
print("Neuron %d responds to the input vector X4 i.e., n4 = [%0.2f, %0.2f]"%(output[3,0], output[3, 1], output[3, 2]))

## plotting graph
plt.show()
