import numpy as np
import pandas as pd
from math import log2

def entropy(n1, n2):
  t = n1+n2
  e1 = -(n1/t)*log2((n1/t))
  e2 = -(n2/t)*log2((n2/t))
  ans = e1+e2
  return ans

def entropyTarget(train):
  yes = len(train[train['num'] == 1])
  no = len(train[train['num'] == 0])

  ent = entropy(yes, no)
  return ent

def entropyCond(df):
  y1 = len(df[df['num'] == 1])
  n1 = len(df)-y1

  e1 = entropy(y1, n1)
  return e1

def entropyWAvg(df1, df2, df3, ent1, ent2, ent3):
  total = len(df1)+len(df2)+len(df3)
  w1 = (len(df1)/total)*ent1

  w2 = (len(df2)/total)*ent2
  w3 = (len(df3)/total)*ent3

  wavg = w1+w2+w3
  return wavg

def probab(df):
  y = len(df[df['num'] == 1])
  n = len(df) - y

  pby = y/len(df)
  pbn = n/len(df)
  
  if pby > pbn:
    dec = True
  else:
    dec = False
  
  return (pby, pbn, dec)

def prediction(test, dec1, dec2, dec3):
  output = []
  for val in test['slope']:
    if val == 1:
      output.append(int(dec1))
    elif val == 2:
      output.append(int(dec2))
    else:
      output.append(int(dec3))
  
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

df = pd.read_csv('Cleveland_heart_disease_data.csv')
train, test = df[:round(len(df)*0.70)], df[round(len(df)*0.70):]

print("\n------------Training--data-------------\n")
print("Calculating entropy of target variable :")
entH = entropyTarget(train)
print("E(Heart disease) =", round(entH,2))

slope1, slope2, slope3 = train[train['slope'] == 1], train[train['slope'] == 2], train[train['slope'] == 3]
ent1 = entropyCond(slope1)
ent2 = entropyCond(slope2)
ent3 = entropyCond(slope3)

print("\nCalculating entropy for each value of slope i.e. 1, 2 and 3 :")
print("E(H | S = 1) =", round(ent1,3))
print("E(H | S = 2) =", round(ent2,3))
print("E(H | S = 3) =", round(ent3,3))

print("\nCalculating Weighted Average of all three :")
entWavg = entropyWAvg(slope1, slope2, slope3, ent1, ent2, ent3)
print("E(H | S) =", round(entWavg,2))

print("\nCalculating Information Gain :")
infoGain = entH - entWavg
print("IG(H, S) =", round(infoGain,2))

print("\nCalculating probabilities :")
pby1, pbn1, dec1 = probab(slope1)
pby2, pbn2, dec2 = probab(slope2)
pby3, pbn3, dec3 = probab(slope3)

print("for slope 1 P(yes) =", round(pby1,2), ", P(No) =", round(pbn1,2), " and decision =",dec1)
print("for slope 2 P(yes) =", round(pby2,2), ", P(No) =", round(pbn2,2), " and decision =",dec2)
print("for slope 3 P(yes) =", round(pby3,2), ", P(No) =", round(pbn3,2), " and decision =",dec3)

print("\n-------------Testing--data--------------\n")
output = prediction(test, dec1, dec2, dec3)
label = test['num'].to_numpy()

tp, fp, fn, tn = confusion(label, output)
print('true positive :', tp, '  false positive :', fp)
print('false negative :', fn, ' true negative :', tn)

acc = (tp+tn)/len(output)
print('\nAccuracy :', round(acc, 2)*100, '%')


