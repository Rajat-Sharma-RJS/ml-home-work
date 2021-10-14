$ python3 problem2.py 

------------Training--data-------------

Calculating entropy of target variable :
E(Heart disease) = 0.99

Calculating entropy for each value of slope i.e. 1, 2 and 3 :
E(H | S = 1) = 0.764
E(H | S = 2) = 0.908
E(H | S = 3) = 0.998

Calculating Weighted Average of all three :
E(H | S) = 0.85

Calculating Information Gain :
IG(H, S) = 0.15

Calculating probabilities :
for slope 1 P(yes) = 0.22 , P(No) = 0.78  and decision = False
for slope 2 P(yes) = 0.68 , P(No) = 0.32  and decision = True
for slope 3 P(yes) = 0.53 , P(No) = 0.47  and decision = True

-------------Testing--data--------------

true positive : 29   false positive : 19
false negative : 14  true negative : 29

Accuracy : 64.0 %
