import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def numpyData(df, features, label):
    features = df[features]
    features = features.to_numpy()
    
    output = df[label].apply(lambda x: 1 if x == 1 else -1)
    output = output.to_numpy()
    return (features, output)

def scaling(features):
    m = features.shape[0]
    for i in range(features.shape[1]):
        feature = features[:, i]
        avg = feature.mean()
        vari = np.sqrt((1/m)*((feature - avg)**2).sum())
        feature = (feature - avg)/vari
        #mx = abs(max(feature)) if abs(max(feature)) > abs(min(feature)) else abs(min(feature))
        #feature = feature/(mx + 0.1)
        features[:, i] = np.round(feature, 6)

def takeStep(i1, i2, seq):
    alpha1 = seq.alphas[i1]
    y1 = seq.y[i1]
    if alpha1 > 0 and alpha1 < seq.C:
        E1 = seq.e_cache[i1]
    else:
        E1 = seq.X[i1] * seq.w + seq.b - seq.y[i1]
    alpha2 = seq.alphas[i2]
    y2 = seq.y[i2]
    E2 = seq.e_cache[i2]
    s = y1 * y2
    if y1 == y2:
        L = max(0, alpha1+alpha2-seq.C)
        H = min(seq.C, alpha1+alpha2)
    else:
        L = max(0, alpha2-alpha1)
        H = min(seq.C, seq.C+alpha2-alpha1)
    if L == H:
        return 0
    eta = seq.K[i1, i1] + seq.K[i2, i2] - 2*seq.K[i1, i2]
    if eta > 0:
        a2 = alpha2 + y2*(E1-E2)/eta
        if a2 < L:
            a2 = L
        elif a2 > H:
            a2 = H
    else:
        c1 = eta / 2.0
        c2 = y2 * (E1 - E2) - eta * alpha2
        Lobj = c1 * L * L + c2 * L
        Hobj = c1 * H * H + c2 * H
        if Lobj > Hobj + seq.tol:
            a2 = L
        elif Lobj < Hobj - seq.tol:
            a2 = H
        else:
            a2 = alpha2
    if abs(a2 - alpha2) < seq.tol:
        return 0
    a1 = alpha1 - s*(a2 - alpha2)
    if a1 > 0 and a1 < seq.C:
        bnew = seq.b - E1 - y1 * (a1 - alpha1) * seq.K[i1, i1] - y2 * (a2 - alpha2) * seq.K[i1, i2]
    elif a2 > 0 and a2 < seq.C:
        bnew = seq.b - E2 - y1 * (a1 - alpha1) * seq.K[i1, i2] - y2 * (a2 - alpha2) * seq.K[i2, i2]
    else:
        b1 = seq.b - E1 - y1 * (a1 - alpha1) * seq.K[i1, i1] - y2 * (a2 - alpha2) * seq.K[i1, i2]
        b2 = seq.b - E2 - y1 * (a1 - alpha1) * seq.K[i1, i2] - y2 * (a2 - alpha2) * seq.K[i2, i2]
        bnew = (b1 + b2) / 2.0
    seq.b = bnew
    seq.alphas[i1] = a1
    seq.alphas[i2] = a2
    seq.w = seq.X.T * np.multiply(seq.alphas, seq.y)
    for i in range(seq.m):
        if (seq.alphas[i] > 0) and (seq.alphas[i] < seq.C):
            seq.e_cache[i] = seq.X[i] * seq.w + seq.b - seq.y[i]
    return 1

def examEx(i2, seq):
    y2 = seq.y[i2]
    alpha2 = seq.alphas[i2]
    if alpha2 > 0 and alpha2 <seq.C:
        E2 = seq.e_cache[i2]
    else:
        E2 = seq.X[i2] * seq.w + seq.b - seq.y[i2]
        seq.e_cache[i2] = E2
    r2 = E2 * y2
    if((r2 < -seq.tol) and (seq.alphas[i2] < seq.C)) or ((r2 > seq.tol) and (seq.alphas[i2] > 0)):
        
        max_delta_E = 0
        i1 = -1
        for i in range(seq.m):
            if seq.alphas[i] > 0 and seq.alphas[i] < seq.C:
                if i == i2:
                    continue
                E1 = seq.e_cache[i]
                delta_E = abs(E1 - E2)
                if delta_E > max_delta_E:
                    max_delta_E = delta_E
                    i1 = i
        if i1 >= 0:
            if takeStep(i1, i2, seq):
                return 1
        
        random_index = np.random.permutation(seq.m)
        for i in random_index:
            if seq.alphas[i] > 0 and seq.alphas[i] < seq.C:
                if i == i2:
                    continue
                if takeStep(i, i2, seq):
                    return 1
        
        random_index = np.random.permutation(seq.m)
        for i in random_index:
            if i == i2:
                continue
            if takeStep(i1, i2, seq):
                return 1
    return 0

class SeqMinOpt:
    
    def __init__(self, data_X, data_y, C, toler, kernel_tup):
        self.X = np.mat(data_X)
        self.y = np.mat(data_y).T
        self.C = C
        self.tol = toler
        self.kernel_tup = kernel_tup
        self.m = np.shape(data_X)[0]
        self.n = np.shape(data_X)[1]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.w = np.mat(np.zeros(self.n)).T
        self.e_cache = np.mat(np.zeros(self.m)).T
        
        if kernel_tup[0] == 'lin':
            self.K = self.X * self.X.T
        elif kernel_tup[0] == 'rbf':
            self.K = np.mat(np.zeros((self.m, self.m)))
            for i in range(self.m):
                for j in range(self.m):
                    self.K[i, j] = (self.X[i] - self.X[j]) * (self.X[i] - self.X[j]).T
                    self.K[i, j] = np.exp(self.K[i, j]/(-1 * self.kernel_tup[1]**2))
        else:
            pass


def mainRoutine(seq, max_iter=5):
    num_changed = 0
    examine_all = 1
    passes = 0
    while(passes <= max_iter):
        num_changed = 0
        if (examine_all == 1):
            for i2 in range(seq.m):
                num_changed += examEx(i2, seq)
        else:
            for i2 in range(seq.m):
                if (seq.alphas[i2] > 0) and (seq.alphas[i2] < seq.C):
                    num_changed += examEx(i2, seq)
        if (num_changed == 0):
            passes += 1
        if (examine_all == 1):
            examine_all = 0
        elif (num_changed == 0):
            examine_all = 1


df = pd.read_csv("Cleveland_heart_disease_data.csv")

train, test = df[:round(len(df)*0.7)], df[round(len(df)*0.7):]

features, label = numpyData(train, ['age', 'trestbps'], 'num')
testing, output = numpyData(train, ['age', 'trestbps'], 'num')

scaling(features)
scaling(testing)

label = label.astype("float64")
output = output.astype("float64")

seq = SeqMinOpt(features, label, 100, 0.001, ('lin', 0.1))
print("-----------------Training data-------------------\n")
mainRoutine(seq)

print("weights :\n", seq.w)
print("b :", seq.b)

print("\n-----------------Testing data-------------------\n")
seqTest = SeqMinOpt(testing, output, 100, 0.001, ('lin', 0.1))
error = 0.0
for i in range(seqTest.m):
    if seqTest.y[i] * (seqTest.X[i] * seq.w + seq.b) < 0:
        error += 1.0

error = error/seqTest.m
accuracy = (1 - error) * 100
print("accuracy :", round(accuracy), "%")

plt.plot(seq.e_cache, 'r.')
plt.title("Error")
plt.show()

