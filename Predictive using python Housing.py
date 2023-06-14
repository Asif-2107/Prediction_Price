#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[5]:


data = pd.read_csv("The dataset")


# In[6]:


data.head()


# In[7]:


data.tail()


# In[8]:


data.columns


# In[9]:


data.shape


# In[10]:


data.describe() #data exploration


# In[11]:


#we would replace the null values with the mean but since we do not have null values here which means its a very clean dataset, we do not do anything and move on with visuaization


# In[12]:


data.isnull()  #null values???


# In[13]:


data.isnull().sum()    #now what is the sum of the null values 


# In[15]:


#visualisation = aim is to predict the price of the house


# In[16]:


sns.relplot(x = 'price', y = 'bedrooms', data = data)


# In[17]:


sns.relplot(x = 'price', y = 'bathrooms', data = data)


# In[18]:


sns.relplot(x = 'price', y = 'parking', data = data)


# In[19]:


sns.relplot(x = 'price', y = 'area', data = data)


# In[23]:


sns.relplot(x = 'price', y = 'area', hue = 'basement', data = data)


# In[24]:


#In your example code, you are using sns.relplot() 
#to create a scatter plot of the "price" and "area" columns from your dataset,
#with the "basement" column as the hue parameter. 
#This means that the data points in the scatter plot will be colored according to whether the property has a basemen


# In[28]:


#modelling
data.head()


# In[26]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Training data is a subset of the data used to train a machine learning model.
#The purpose of training data is to allow the machine learning algorithm to learn the relationships between the input features (independent variables) and the target variable (dependent variable) in order to make accurate predictions on new, unseen data. In supervised learning, the target variable is known and is used to train the algorithm to make accurate predictions.

#Test data is a subset of the data that is used to evaluate the performance of the machine learning model. 
#The purpose of test data is to simulate the performance of the machine learning model on new, unseen data. Typically, the test data is not used during the training phase, and the model's performance on the test data is used to measure the model's generalization performance.

#To split the data into training and test sets, a common practice is to randomly select a percentage of the data (e.g., 70% or 80%) for training, and the remaining data for testing. The training data is used to fit the machine learning model, and the test data is used to evaluate the model's performance. It is important to ensure that the test data is representative of the data that the model will be applied to in practice.


# In[65]:


train = data.drop(['price', 'furnishingstatus'], axis=1)  #1 = colums dropped, 0 = rows dropped
test = data['price'] # dependent variable


# In[66]:


# assume 'train' is a DataFrame containing the input features and target variable
from sklearn.preprocessing import OrdinalEncoder

# create an ordinal encoder object
oe = OrdinalEncoder()

# list of categorical column names
cat_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# encode the categorical columns using the ordinal encoder
train[cat_cols] = oe.fit_transform(train[cat_cols])





# In[67]:


#segregate data
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size = 0.3, random_state = 2) # input features of training set, testing set, target variables for training set, testing set, then split between training and testing with a random percentage to testing set being 30% data, random number generator being 2 


# In[68]:


regr = LinearRegression() #make the model


# In[69]:


regr.fit(X_train, y_train) #fitting the training data


# In[70]:


pred = regr.predict(X_test)  

#regr.predict(X_test) is a method call that generates predictions for the target variable using the input features in the X_test DataFrame, based on the linear regression model regr that was fit on the training data.


# In[71]:


pred


# In[74]:


regr.score(X_test, y_test)



# In[ ]:


#If you get an accuracy of 0.65, it means that your linear regression model is able to predict the target variable correctly for 65% of the test instances.

