import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# loading datasets
iris=datasets.load_iris()

#   assigning feature to x and target or label to y
x=iris.data
y=iris.target

# from sklearn importing classifier
from sklearn.neighbors import KNeighborsClassifier

#  FROM SKLEARN  IMPORTING train_test_split to split the data into training and testing
from sklearn.model_selection import train_test_split

x_train , x_test , y_train, y_test =train_test_split(x,y,test_size=.3,random_state=21,stratify=y)
knn=KNeighborsClassifier()

# fitting the model or trrain the model
knn.fit(x_train,y_train)

# predicting the test set and assigning into y_pred

y_pred=knn.predict(x_test)
print("test set prediction : \n {}".format(y_pred))
# checking the accuracy of our model

print(knn.score(x_test,y_test))
# 0.9555555555555556
# it is 95% accurate which is very good
plt.xlabel("feature")
plt.ylabel("label")
plt.plot(x_test,y_test)

plt.show()

