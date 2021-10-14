import pandas as pd
import numpy as np

def gini(val, train):
  small = train[train['age'] < val]
  other = train[train['age'] >= val]

  small_yes = len(small[small['num'] == 1])
  small_no = len(small[small['num'] == 0])

  other_yes = len(other[other['num'] == 1])
  other_no = len(other[other['num'] == 0])

  g1 = 1 - (small_yes/len(small))**2 - (small_no/len(small))**2
  g2 = 1 - (other_yes/len(other))**2 - (other_no/len(other))**2

  ans = (len(small)/len(train))*g1 + (len(other)/len(train))*g2

  return ans
  
def decidingFactor(factor, df):
  min = 1
  x = 0
  for val in factor:
    gi = gini(val, df)
    if gi < min:
      min = gi
      x = val
  return x

def decision(x, df):
  left = df[df['age'] < x]
  right = df[df['age'] >= x]

  y1 = len(left[left['num'] == 1])
  n1 = len(left) - y1
  y2 = len(right[right['num'] == 1])
  n2 = len(right) - y2

  return (y1/len(left), n1/len(left), y2/len(right), n2/len(right))

def decide(py1, pn1, py2, pn2):
  if py1 > pn1 and pn2 > py2:
    pleft = 1
    pright = 0
  else:
    pleft = 0
    pright = 1
  
  return (pleft, pright)

def predict(x, pleft, pright, val):
  ans = None
  if val < x:
    ans = pleft
  else:
    ans = pright
  return ans
  
def prediction(x, pleft, pright, df):
  output = []
  for val in df['age']:
    expect = predict(x, pleft, pright, val)
    output.append(expect)
  output = np.array(output)
  return output

def confusion(label, output):
  tp, fp, fn, tn = 0, 0, 0, 0
  for i in range(len(label)):
    if label[i] == 1:
      if output[i] == 1:
        tp = tp+1
      else:
        fn = fn+1
    else:
      if output[i] == 1:
        fp = fp+1
      else:
        tn = tn+1
  
  return (tp, fp, fn, tn)

def factors(array):
  factor = []
  for i in range(len(array)):
    if i == len(array)-1:
      continue
    factor.append(((array[i]+array[i+1])/2))
  return np.array(factor)

df = pd.read_csv('Cleveland_heart_disease_data.csv')
df = df[['age', 'num']]
training, test = df[:round(len(df)*0.70)], df[round(len(df)*0.70):]

print("\n------------Training--data-------------\n")
#sorting
train = training.copy()
train.sort_values('age', axis=0, ascending=True, inplace=True)
ages = train['age'].to_numpy()
heart = train['num'].to_numpy()

factor = factors(ages)
x = decidingFactor(factor, train)
print("Calculating deciding factor with least gini impurity :")
print("x =", round(x+0.5), ";  gini impurity =", round(gini(x, train),2))

print("\nCalculating probabilities on both sides of the decision node :")
py1, pn1, py2, pn2 = decision(x, train)
print('probability of having heart disease with age <', round(x+0.5), 'is :', round(py1,2))
print('probability of NOT having heart disease with age <', round(x+0.5), 'is :', (1-round(py1,2)))
print('probability of having heart disease with age >=', round(x+0.5), 'is :', round(py2,2))
print('probability of NOT having heart disease with age >=', round(x+0.5), 'is :', (1-round(py2,2)))

print("\nMarking the sides of the decision node as true/false :")
pleft, pright = decide(py1, pn1, py2, pn2)
print('left :',bool(pleft))
print('right :', bool(pright))

print("\n-------------Testing--data--------------\n")
output = prediction(x, pleft, pright, test)
label = test['num'].to_numpy()

print("         -------------")
print("         |  AGE < %d |"%round(x+0.5))
print("         -------------")
print("           /       \ ")
print("          /         \ ")
print("      ------      -------")
print("      | NO |      | YES |")
print("      ------      -------")
print("\nConfusion Matrix :\n")
tp, fp, fn, tn = confusion(label, output)
print('true positive :', tp, '  false positive :', fp)
print('false negative :', fn, ' true negative :', tn)

acc = (tp+tn)/len(output)
print('\nAccuracy :', round(acc, 2)*100, '%')

