# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
# Data.csv:
```python
import pandas as pd
df=pd.read_csv("data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
```
# Encoding.csv:
```python
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
oe=OrdinalEncoder()

df1["nom_0"] = oe.fit_transform(df1[["nom_0"]])
temp=['Cold','Warm','Hot']
oe2=OrdinalEncoder(categories=[temp])
df1['ord_2'] = oe2.fit_transform(df1[['ord_2']])

df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df0=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df2=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df3=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df4=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df4
```

# Titanic.csv:
```python
import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

#removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

#data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

#feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
```

# OUPUT:
# Data.csv:
### Initial dataset:


![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/a5cecad1-b301-4569-9b97-c2922bdf68bd)


### Binary Encoding:


![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/eb7ef7e2-56ed-4557-a064-2719ac20d75c)![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/ae96cd61-9bc6-49e4-897f-171fb96d1d39)


### Encoded Dataset:


![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/fa0b81ae-f0a1-4cc6-8577-7c3a138453ee)



### Data Scaling using MinMaxScaler:



![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/c28b17fc-9c23-44de-a331-51091759dda7)



### Data Scaling using StandardScaler:



![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/83f3cdd7-bfd8-49c1-819d-68d88523d4a7)



### Data Scaling using MaxAbsScaler:


![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/23a00f4a-6877-4983-8947-a245af1a49f1)



### Data Scaling using RobustScaler:



![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/0d949f51-a3a1-413e-96ff-7885f9bb3dea)



# Encoding.csv :
### Initial Dataset:

![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/d3f4977a-1585-47ca-b579-416761c11493)


### Binary Encoding:


![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/1c4fbfa1-aaeb-4398-9891-2457202dc6a9)![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/075b3055-779b-4da8-a437-9bc6bdb98233)



### Encoded Dataset:


![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/a9cc2ae5-20b9-405a-9b92-ffdab5473ab9)



### Data Scaling using MinMaxScaler:


![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/290c2a6b-af99-44ae-a30a-fed3724b6c84)



###  Data Scaling using StandardScaler:


![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/a5ae91d7-6385-4819-9d6c-759f06729e65)


### Data Scaling using MaxAbsScaler:



![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/315a7371-0a0b-42fb-a34d-e131d8b1b410)



### Data Scaling using RobustScaler:


![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/76ced006-47a4-49e1-a521-a1bf24518f97)



# Titanic.csv :
### Initial Dataset:


![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/2a919d35-c4fa-45df-8a0f-24bbbe92fcf7)



### Data cleaning before encoding:


![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/34d1f588-0d83-49ab-ba6e-c9fe21d69047)



### Cleaned Dataset:


![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/533aeb8f-6711-4400-9806-d3dff776df8d)



### Binary Encoding:



![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/fb7c3035-dc9a-4c32-bdb7-46617e9202cf)



### Encoded Dataset:


![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/010e885c-eb69-4603-9f25-ff86d4dbed56)



### Data Scaling using MinMaxScaler:



![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/ce3101e9-733f-44a7-8f47-6039b51c69d4)



### Data Scaling using StandardScaler:


![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/cff6eecf-d37b-4382-a14e-65b3e0d5a3ba)



### Data Scaling using MaxAbsScaler:



![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/5b90c049-e5ca-4845-803b-66d14c028d89)



### Data Scaling using RobustScaler:



![image](https://github.com/vinushcv/EX-05-Feature-Generation/assets/113975318/a42f9a1a-3c06-4de8-9d8d-9df68e4a8e5c)



# RESULT:
Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.

