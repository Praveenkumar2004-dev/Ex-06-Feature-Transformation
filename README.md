# Ex-06-Feature-Transformation

## Aim:
To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a
mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform 
the values which are useful for our further analysis.

## ALGORITHM:
# STEP 1:
Read the given Data

# STEP 2:
Clean the Data Set using Data Cleaning Process

# STEP 3:
Apply Feature Transformation techniques to all the features of the data set

# STEP 4:
Print the transformed features

## PROGRAM:
```
NAME:PRAVEEN KUMAR S
REG.NO: 212222230108
```


```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```
## Output:

![i1](https://user-images.githubusercontent.com/119559827/234005991-7f2acfd3-785a-47a1-a8f9-1345a434b5ed.png)

![i2](https://user-images.githubusercontent.com/119559827/234006073-ef58fab0-81a5-4a26-9a1b-c6a756b03ace.png)

![i3](https://user-images.githubusercontent.com/119559827/234006172-0be9f420-0ce2-4e04-8b02-ab5ea119c382.png)

![i4](https://user-images.githubusercontent.com/119559827/234006227-44587bed-337a-4a41-9ceb-bf9946d2efc9.png)

![i5](https://user-images.githubusercontent.com/119559827/234006266-8ff432c0-eb5c-4203-bb56-211bd4fad333.png)

![i6](https://user-images.githubusercontent.com/119559827/234006311-7a51c750-9823-43dc-b176-19026d63439d.png)

![i7](https://user-images.githubusercontent.com/119559827/234006343-978ea45a-2ccf-47c6-8dc6-09b89148a601.png)

![i8](https://user-images.githubusercontent.com/119559827/234006382-746200ae-dc53-4a35-9464-d6080cd13245.png)

![i9](https://user-images.githubusercontent.com/119559827/234006434-f5c74020-b55d-4624-a135-13c02a5009d8.png)

![i10](https://user-images.githubusercontent.com/119559827/234006488-65afb3aa-38a8-465f-9eeb-63adc8ffb1fa.png)

![i11](https://user-images.githubusercontent.com/119559827/234006533-ed101834-22aa-4847-a5d7-40a52bd69d51.png)

## RESULT:
Thus feature transformation is done for the given dataset.

