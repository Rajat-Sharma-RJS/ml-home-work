$ python3 problem1.py 

------------Training--data-------------

Calculating deciding factor with least gini impurity :
x = 55.0 ;  gini impurity = 0.45

Calculating probabilities on both sides of the decision node :
probability of having heart disease with age < 55.0 is : 0.29
probability of NOT having heart disease with age < 55.0 is : 0.71
probability of having heart disease with age >= 55.0 is : 0.61
probability of NOT having heart disease with age >= 55.0 is : 0.39

Marking the sides of the decision node as true/false :
left : False
right : True

-------------Testing--data--------------

         -------------
         |  AGE < 55 |
         -------------
           /       \ 
          /         \ 
      ------      -------
      | NO |      | YES |
      ------      -------

Confusion Matrix :

true positive : 29   false positive : 22
false negative : 14  true negative : 26

Accuracy : 60.0 %
