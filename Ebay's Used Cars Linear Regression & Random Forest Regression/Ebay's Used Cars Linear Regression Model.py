
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Import Data

# In[2]:


df = pd.read_csv('autos.csv',sep=',',header=0,encoding = 'latin1')


# #### Explore the data and look at the first few records

# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.head()


# #### Identify missing records 

# In[6]:


df.isnull().sum()


# Lets explore the missing values to see whether we can replace them with an appropriate value or decide to drop the values.

# In[7]:


df['notRepairedDamage'].value_counts()


# 87.7% of 'notRepairedDamage' feature is 'nein'. Would be okay to replace the missing entries with the most common 
# response.

# In[8]:


df['notRepairedDamage'].fillna('nein',inplace=True)


# Let's explore some of the other features.

# In[9]:


df['fuelType'].value_counts()


# 66% of 'fuelType' feature is 'benzin'. The vast majority is not 'benzin' so replacing the missing values with 'benzin' might
# not be the best approach. Furthermore, the accuracy of our model drops 1% if you were to replace the missing value with 
# 'benzin'.

# In[10]:


#df['fuelType'].fillna('benzin',inplace=True)


# Replace missing values with the most common entry for 'gearbox'

# In[11]:


df['gearbox'].value_counts()


# In[12]:


df['gearbox'].fillna('manuell',inplace=True)


# #### Drop missing records. 

# DataSet is large enough that we can affort to drop a couple of the missing records without compromising our model.

# In[13]:


df.dropna(inplace=True)


# Look at heatmap of null values to make sure all are gone.

# In[14]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# #### Remove unnecessary columns 

# Some of these columns do not provide any significant information. For example, 'abtest' only has 1 unique value, 'name' is
# a bit redundant since the information is also stored in 'brand' and 'vehicle type' but more organized, 'lastSeen' seems to 
# be irrelevant information. Explored through the kernels to see what the community opinion is on what they
# think is relevant or not.

# In[15]:


df.drop(['name','dateCrawled','seller','offerType','abtest','nrOfPictures',
         'lastSeen','postalCode','dateCreated','monthOfRegistration'],axis=1,inplace=True)


# #### Remove outliers from the data.

# Since we are doing a linear regression, it is important to address any potential outliers since they can greatly influence
# the model.

# In[16]:


print('Min Value: {} Max Value: {}'.format(df['price'].min(),df['price'].max()))


# In[17]:


df['price'].describe()


# Choosing to select price values above 200 and uder 40000. 

# In[19]:


df = df[(df['price']>200) & (df['price']<40000)]


# #### Let's evaluate some of the other features.

# By removing potential outliers, I've been able to increase my models accuracy from 58% up to 80%. 

# In[20]:


sns.distplot(df['yearOfRegistration'])


# In[21]:


df = df[(df['yearOfRegistration']>1995) & (df['yearOfRegistration']<2016)]


# In[22]:


df['powerPS'].describe()


# In[23]:


df = df[(df['powerPS']>50)& (df['powerPS']<200)]


# In[24]:


df.head()


# #### Converting categorical data into dummy variables.

# In[25]:


df=pd.get_dummies(data=df,columns=['notRepairedDamage','vehicleType','model','brand','gearbox','fuelType'],drop_first=True)


# In[26]:


df.head()


# #### Determine the indepedent and the dependent (target) variable

# In[27]:


X = df.drop('price',axis=1)

y = df['price']


# #### Split your data into a training set and testing set

# In[28]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)


# ## Linear Regression

# Instantiate Linear Regression model  

# In[29]:


from sklearn.linear_model import LinearRegression

LR=LinearRegression()


# Fit Model to training data

# In[30]:


LR.fit(X_train,y_train)


# Predict using testing data

# In[31]:


y_pred = LR.predict(X_test)


# Evaluate your model and Accuracy

# In[32]:


from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test, y_pred))


# In[33]:


print(LR.score(X_test, y_test)*100,'% Prediction Accuracy')


# ## Random Forest Regression

# In[34]:


from sklearn.ensemble import RandomForestRegressor


# Instantiate Random Forest Regressor

# In[35]:


regr = RandomForestRegressor()


# Fit model to training data

# In[36]:


regr.fit(X_train, y_train)


# Evaluate your model and Accuracy

# In[38]:


print(regr.score(X_test, y_test)*100,'% Prediction Accuracy')


# ### Identifying important features

# In[50]:


feat_imp = regr.feature_importances_
x = np.argsort(feat_imp)[::1]

for element in range(X.shape[1]):
    print('%d. feature %d {%f}' % (element+1,x[element],feat_imp[x[element]]))


# Feature 0,1, and 2 is collectively 86.8% important when explaining the valuation of a vehicle. 
# 
# Feature 0 ='yearOfRegistration'
# 
# Feature 1 ='powerPS'
# 
# Feature 2 ='kilometer'
# 
