# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
 # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])

data

![image](https://github.com/user-attachments/assets/1cea1cb3-3cb0-4b52-a1d6-67388570a388)

data.isnull().sum()

![image](https://github.com/user-attachments/assets/9150f037-2879-43ce-911f-7a238f1c82e8)

missing=data[data.isnull().any(axis=1)]
missing

![image](https://github.com/user-attachments/assets/343d2a3c-2781-4c49-a654-75ae7a38de69)

data2=data.dropna(axis=0)

data2

![image](https://github.com/user-attachments/assets/78a17b9a-5b92-422b-ad0e-9fd018351017)

sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})

print(data2['SalStat'])

![image](https://github.com/user-attachments/assets/17f6e175-defb-4f31-8318-1238dc071b91)

sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)

dfs

![image](https://github.com/user-attachments/assets/03596fdd-b0bf-4995-b59b-c8bb4bda52c0)

data2

![image](https://github.com/user-attachments/assets/31e5115e-cb22-4fdf-8cc6-c02c2f55eec2)

new_data=pd.get_dummies(data2, drop_first=True)

new_data

![image](https://github.com/user-attachments/assets/e2c52313-7517-44f4-9a9f-8071f845269f)

columns_list=list(new_data.columns)

print(columns_list)

![image](https://github.com/user-attachments/assets/9de2e67c-b209-4ad7-b19f-5593fca316b4)


features=list(set(columns_list)-set(['SalStat']))

print(features)

![image](https://github.com/user-attachments/assets/b06e1475-de46-4b81-b7ed-e26865beb289)

y=new_data['SalStat'].values

print(y)

![image](https://github.com/user-attachments/assets/9e7fb01b-3b8a-4829-a1b2-a146c48c1e94)

x=new_data[features].values

print(x)

![image](https://github.com/user-attachments/assets/c02b1ca8-e98f-41bd-9093-b31bce5638a6)

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)

![image](https://github.com/user-attachments/assets/39c189d8-a5eb-44dd-a5a6-26b73247a5b6)

prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)

print(confusionMatrix)

![image](https://github.com/user-attachments/assets/5e2058d0-34b0-4d59-8004-9e9fd3718250)

accuracy_score=accuracy_score(test_y,prediction)

print(accuracy_score)

![image](https://github.com/user-attachments/assets/cdd21e71-54c2-4952-900d-19c496b16e01)

print("Misclassified Samples : %d" % (test_y !=prediction).sum())

![image](https://github.com/user-attachments/assets/b648ce09-0101-498f-9ebf-e3b3fb199c64)

data.shape

![image](https://github.com/user-attachments/assets/5ed725c2-7d32-4e69-86c0-67bd59bdad75)

import pandas as pd

from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)

x=df[['Feature1','Feature3']]

y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)

x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]

print("Selected Features:")

print(selected_features)

![image](https://github.com/user-attachments/assets/4ed63b81-6909-4bcb-8e90-5fdaf7080c46)

import pandas as pd

import numpy as np

from scipy.stats import chi2_contingency

import seaborn as sns

tips=sns.load_dataset('tips')

tips.head()

![image](https://github.com/user-attachments/assets/a2c76027-8d35-4b91-950c-dc4d0ee946b8)

tips.time.unique()

![image](https://github.com/user-attachments/assets/763fbca0-b79f-4d92-81db-461b1a7179d1)

contingency_table=pd.crosstab(tips['sex'],tips['time'])

print(contingency_table)

![image](https://github.com/user-attachments/assets/aae0feab-7445-495c-8062-796cf9b0ab52)

chi2,p,_,_=chi2_contingency(contingency_table)

print(f"Chi-Square Statistics: {chi2}")

print(f"P-Value: {p}")

![image](https://github.com/user-attachments/assets/b2c4ae8e-0721-48d6-806d-87089fc8b824)





# RESULT:
       # INCLUDE YOUR RESULT HERE
       
       Thus, Feature selection and Feature scaling has been used on thegiven dataset.
