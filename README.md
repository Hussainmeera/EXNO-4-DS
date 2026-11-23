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
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv('/content/bmi.csv')
df.head()
```

<img width="428" height="207" alt="image" src="https://github.com/user-attachments/assets/a6911a54-d1bd-40dc-beed-5d22addeaa4f" />




```
df.dropna()
```

<img width="538" height="362" alt="image" src="https://github.com/user-attachments/assets/1d3837dd-31fa-4f35-88ae-6eca54552e7b" />



```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```

<img width="392" height="127" alt="image" src="https://github.com/user-attachments/assets/02316a16-5c05-4b13-abf5-1ad049717b91" />


```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("bmi.csv")
df1.head()
```


<img width="422" height="205" alt="image" src="https://github.com/user-attachments/assets/349e0192-2416-486a-afd4-f04edb521241" />



```
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

df1[['Height', 'Weight']] = sc.fit_transform(df1[['Height', 'Weight']])

df1.head(10)
```


<img width="605" height="348" alt="image" src="https://github.com/user-attachments/assets/aba8229b-13f3-4086-a9cc-67b39d45dfce" />



```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```


<img width="595" height="347" alt="image" src="https://github.com/user-attachments/assets/e6f626fb-50b1-449c-add9-f4436b09b5d7" />



```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```


<img width="562" height="392" alt="image" src="https://github.com/user-attachments/assets/8dc7c9b6-544d-49ab-ac86-69223df50f8e" />



```
df3=pd.read_csv("bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3[['Height', 'Weight']] = scaler.fit_transform(df3[['Height', 'Weight']])
df3
```


<img width="623" height="393" alt="image" src="https://github.com/user-attachments/assets/44f01df8-a4b0-4d7a-828a-e9282ffee168" />



```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df4[['Height', 'Weight']] = scaler.fit_transform(df4[['Height', 'Weight']])
df4.head()
```


<img width="567" height="210" alt="image" src="https://github.com/user-attachments/assets/d2a7dbfa-a3d0-41f2-8dca-8d3f0e9910e6" />



```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```


<img width="832" height="217" alt="image" src="https://github.com/user-attachments/assets/4ca744ac-6e52-4293-9337-a9d2071e5c90" />



```
contingency_table = pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

<img width="527" height="68" alt="image" src="https://github.com/user-attachments/assets/85d4155d-58a3-43fb-a434-8f92d05d1497" />



```
chi2, p,_,_  = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```


<img width="873" height="38" alt="image" src="https://github.com/user-attachments/assets/193ee683-8cef-4d6b-a00c-16e04501d6b5" />




```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data = {
'Feature1': [1, 2, 3, 4, 5],
'Feature2': ['A', 'B', 'C', 'A', 'B'],
'Feature3': [0, 1, 1, 0, 1],
'Target': [0, 1, 1, 0, 1]
}
df = pd.DataFrame (data)
X = df [['Feature1', 'Feature3']]
y = df ['Target']
selector=SelectKBest(score_func=mutual_info_classif, k = 1)
X_new = selector.fit_transform(X, y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="557" height="41" alt="image" src="https://github.com/user-attachments/assets/167cf345-ae9e-4a49-8f91-6fd2c648d6b3" />

# RESULT:
       Feature Scaling and Feature Selection process was executed successfully.
