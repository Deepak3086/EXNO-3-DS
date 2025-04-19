## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

### REG NO:212224220019
### NAME: DEEPAK JG
```
import pandas as pd
df=pd.read_csv('/content/Encoding Data (2).csv')
df
```
![image](https://github.com/user-attachments/assets/65471992-b410-4132-9fce-0f1b3e2c53e4)
```
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/f9996b65-3f6a-413e-8fb2-2a3e46e14668)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/20be73c0-cf49-4720-bd6c-b9087d2b3425)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/ec5656a1-93d2-4dbe-bd24-77175abb61c2)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=dfc.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/cd8c751c-ff0b-4820-89dc-acebf4cd425b)
```
pd.get_dummies(df,columns=['nom_0'])
```
![image](https://github.com/user-attachments/assets/23c47c0c-e8fb-47aa-905a-140456d5ed85)
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/1125ab05-b17e-495f-9475-58da9fabcca0)
```
from category_encoders import BinaryEncoder
df=pd.read_csv('/content/data (2).csv')
df
```
![image](https://github.com/user-attachments/assets/ef01cd01-1cfc-4da1-8098-3b9824c367e8)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/97e5757e-02ee-4644-b797-e52635f09c09)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/85c7c22a-15fa-484e-8c6e-aa1836de31ff)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv('/content/Data_to_Transform (1).csv')
df
```
![image](https://github.com/user-attachments/assets/91f2f1f6-77be-4b53-9652-4587d9b73e50)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/19b7c580-830f-4736-8280-d81564bf2678)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/be074947-4b3a-4512-846d-7d7d21731bef)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/6319eb5d-4def-4781-a672-62275d4372cc)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/8bcb5960-a074-4833-8709-b480acddb7f3)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/d96d69fc-ef50-4f75-8064-616921d08db4)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/351eb798-41f3-4ca0-b5bf-665cb496f86e)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/bf4bc326-f508-4173-8a5e-fddd06e83256)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/6258706f-a345-4e2d-bc8b-9c9edad5a315)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative_Skew1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/36441a91-fe70-46d5-a0f6-581df4ae5912)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/3314f2bf-7f7e-43b4-bc6e-14b8ed317d02)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/751f0964-9f73-4988-baeb-01948ab0366a)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative_Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative_Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/8bc9b1a6-7940-4a58-b0a9-a5cfb6a68fc6)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

```
![image](https://github.com/user-attachments/assets/66ff31b1-4a7a-427e-a2df-741923851ec0)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/71e3d9e9-3c66-46bd-b416-512ebbb4b600)
```
dt = pd.read_csv("/content/titanic_dataset (2).csv")

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal', n_quantiles=891)

dt["Age_1"] = qt.fit_transform(dt[["Age"]])

sm.qqplot(dt['Age'], line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/9528c035-0e80-48f6-9547-c54dffc6a6f1)
```
sm.qqplot(dt['Age_1'], line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/6dd15815-8621-449e-b1f2-c16f6ca38cf2)


# RESULT:
       # INCLUDE YOUR RESULT HERE

       
