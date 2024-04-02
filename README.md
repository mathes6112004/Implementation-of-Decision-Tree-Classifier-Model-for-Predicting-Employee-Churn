# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas module and import the required data set.
2.Find the null values and count them.
3.Count number of left values.
4.From sklearn import LabelEncoder to convert string values to numerical values.
5.From sklearn.model_selection import train_test_split.
6.Assign the train dataset and test dataset.
7.From sklearn.tree import DecisionTreeClassifier.
8.Use criteria as entropy.
9.From sklearn import metrics.
10.Find the accuracy of our model and predict the require values.

## Program:
```
import pandas as pd
data=pd.read_csv('/content/Employee.csv')
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
x

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
![image](https://github.com/mathes6112004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477782/1ca427f4-71db-4cea-bf28-6294ca92b7e7)
###
![image](https://github.com/mathes6112004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477782/7963233b-2cfb-4600-bf42-e3169227b083)
###
![image](https://github.com/mathes6112004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477782/223f7d0c-6dbd-46fb-ab0e-fb13bd78b102)
###
![image](https://github.com/mathes6112004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477782/c0478d18-4b69-407c-a536-11875baeadc7)
###
![image](https://github.com/mathes6112004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477782/ef1caf85-0cb1-4c8f-b57a-0bd5d3232022)
###
![image](https://github.com/mathes6112004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477782/d11d1764-f78d-4b3a-96a3-ec5fb2e9462f)
###
![image](https://github.com/mathes6112004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477782/350ef7e2-43f9-4f80-93b3-3f8eefe0304e)
###
![image](https://github.com/mathes6112004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477782/45167227-cdf6-4553-adaa-06f13ab3dc04)
###
![image](https://github.com/mathes6112004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477782/43007f89-2807-423a-b9e5-6c2fdbf4d2c2)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
