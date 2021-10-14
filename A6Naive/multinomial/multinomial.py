import numpy as np
import pandas as pd
import string

def remove_punctuation(x):
    new_x = x.lower()
    inp_x = new_x.translate(str.maketrans("", "", string.punctuation))
    return inp_x.split()

def phi_j(word, y, dictonary, label, alpha, total):
    count = 0
    val = 0
    for i in range(len(label)):
        if label[i] == y:
            val += total[i]
            if word in dictonary[i]:
                count += dictonary[i][word]
    count += alpha
    val += 6500*alpha
    return (count/val)

def comp_y(y, label):
    count = 0
    for i in range(len(label)):
        if label[i] == y:
            count += 1
    return (count/len(label))

def out_y(dictonary, label, test_consider, total):
    exp = []
    proby0 = comp_y(0, label)
    proby1 = comp_y(1, label)
    
    for vec in test_consider:
        prob0 = 1
        prob1 = 1
        for word in vec:
            prob0 *= phi_j(word, 0, dictonary, label, 1, total)
            prob1 *= phi_j(word, 1, dictonary, label, 1, total)
        
        prob0 *= proby0
        prob1 *= proby1
        
        if prob0 > prob1:
            exp.append(0)
            #print("0")
        else:
            exp.append(1)
            #print("1")
    return exp

def accuracy(exp, label):
    count = 0
    j = 0
    for i in label:
        if exp[j] == 1 and i == 1:
            count += 1
        elif exp[j] == 0 and i == 0:
            count += 1
        j += 1
    return (count/len(label))

def frequency(x):
    freq = {}
    for word in x:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1
    return freq

def total(x):
    count = 0
    for i in x.values():
        count += i
    return count


df = pd.read_table('SMSSpamCollection', sep="\t", names=['label', 'sms_message'])
df['label'] = df.label.map({'ham':0, 'spam':1})

df['consider'] = df['sms_message'].apply(remove_punctuation)

size = round(len(df)*0.7)
train, test = df[:size], df[size:]

train['dictonary'] = train['consider'].apply(frequency)
train['total'] = train['dictonary'].apply(total)

exp = out_y(train['dictonary'], train['label'], test['consider'], train['total'])
acc = accuracy(exp, test['label'])

print("Multinomial Model")
print("Accuracy :", round(acc, 2))
