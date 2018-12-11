
# coding: utf-8

# In[1]:


#
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
os.chdir("E/project3")

# Any results you write to the current directory are saved as output.


# In[7]:


#Load libraries
import os
import pandas as pd
from fancyimpute import KNN   
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
from random import randrange, uniform
import xlrd


# In[12]:



#Loading the Data
data = pd.read_excel('../input/data.xls', error_bad_lines=False)


# In[13]:


data


# In[16]:


## Missing Value Analysis
#Create dataframe with missing percentage
missing_val = pd.DataFrame(data.isnull().sum())


# In[17]:


#Reset index
missing_val = missing_val.reset_index()



# In[22]:


#Rename variable
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})

#Calculate percentage
missing_val['Missing_percentage'] = (missing_val['Missing_percentage']/len(data))*100

#descending order
missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)




# In[23]:


missing_val


# In[24]:


#Apply KNN imputation algorithm
data = pd.DataFrame(KNN(k = 9).complete(data), columns = data.columns)


# In[28]:


data.head(10)


# In[26]:


## Missing Value Analysis
#Create dataframe with missing percentage
missing_val = pd.DataFrame(data.isnull().sum())


# In[27]:


missing_val


# In[29]:


## Outlier Analysis
 
df = data.copy()


# In[31]:


# #Plot boxplot to visualize Outliers
get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(data['Height'])


# In[43]:


#Detect and replace with NA
# #Extract quartiles
for i in cnames:
    q75, q25 = np.percentile(data.iloc[:,i], [75 ,25])

# #Calculate IQR
     iqr = q75 - q25

# #Calculate inner and outer fence
    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)

# #Replace with NA
    data[data.iloc[[:,i] < minimum] = np.nan
    data[data.loc[:,i] > maximum] = np.nan

# #Calculate missing value
    missing_val = pd.DataFrame(data.isnull().sum())
         
# #Impute with KNN
    data = pd.DataFrame(KNN(k = 3).complete(data), columns = data.columns)


# In[47]:


##Correlation analysis
#Correlation plot
df_corr = data


# In[48]:


#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(7, 5))

#Generate correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[49]:


#Load Libraries

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[50]:


#Divide data into train and test
train, test = train_test_split(data, test_size=0.2)


# In[51]:


#Decision tree for regression
fit_DT = DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,0:20], train.iloc[:,20])

#Apply model on test data
predictions_DT = fit_DT.predict(test.iloc[:,0:20])


# In[52]:


#Calculate MAPE
def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape



# In[53]:


MAPE(test.iloc[:,20], predictions_DT)


# In[54]:


#Accuracy = 89.67


# In[55]:


#Import libraries for Linear Regression
import statsmodels.api as sm


# In[56]:


#Train the model using training sets
model = sm.OLS(train.iloc[:,20],train.iloc[:,0:20]).fit()


# In[57]:


#print out statistics
model.summary()


# In[58]:


#make the prediction
predictions_LR = model.predict(test.iloc[:,0:20])


# In[67]:


#Calculate MAPE
MAPE(test.iloc[:,20],predictions_LR)


# In[60]:


#Accuracy = 91.2%


# In[61]:


#RandomForest Methods
#Divide data into train and test
X = data.values[:, 0:20]
Y = data.values[:,20]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2)


# In[72]:


#Random Forest
from sklearn.ensemble import RandomForestRegressor

RF_model = RandomForestRegressor(n_estimators = 20).fit(X_train, y_train)


# In[73]:


RF_Predictions = RF_model.predict(X_test)


# In[74]:


#Calculate MAPE
MAPE(test.iloc[:,20],RF_Predictions)


# In[ ]:


#Accuracy = 93.4%


# In[75]:


#XGboost Model
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score


# In[76]:


# fit model no training data
model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
model.fit(X_train, y_train)


# In[77]:


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# In[78]:


#Calculate MAPE
MAPE(y_test,RF_Predictions)


# In[ ]:


#Accuracy = 95.2%

