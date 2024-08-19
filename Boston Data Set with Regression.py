#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


boston_dataset = pd.read_csv('boston.csv')
boston_dataset.keys()


# In[17]:


boston_dataset.head(5)


# In[19]:


boston_dataset.info()


# In[31]:


boston_dataset.isnull().sum()


# In[34]:


correlation_matrix = boston_dataset.corr().round(2)


# In[35]:


sns.heatmap(data=correlation_matrix, annot=True)


# In[39]:


plt.figure(figsize=(20, 5))
features = ['LSTAT', 'RM']
target = boston_dataset['Price']
for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston_dataset[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('Price')


# In[46]:


X = pd.DataFrame(np.c_[boston_dataset['LSTAT'], boston_dataset['RM']], columns=['LSTAT', 'RM'])
Y = boston_dataset['Price']


# In[49]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[50]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


# In[51]:


y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)
print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

y_test_predict = lin_model.predict(X_test)
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
# r-squared score of the model
r2 = r2_score(Y_test, y_test_predict)
print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[52]:


plt.scatter(Y_test, y_test_predict)
plt.show()


# In[53]:


import numpy as np
import matplotlib.pyplot as plt


# In[55]:


get_ipython().run_line_magic('matplotlib', 'inline')
def gradient_descent(x,y):
    m = b = 1
    rate = 0.01
    n = len(x)
    plt.scatter(x,y)
    for i in range(100):
        y_predicted = m * x + b
        plt.plot(x,y_predicted,color='green')
        md = -(2/n)*sum(x*(y-y_predicted))
        yd = -(2/n)*sum(y-y_predicted)
        m = m - rate * md
        b = b - rate * yd


# In[56]:


x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])


# In[57]:


gradient_descent(x,y)


# In[58]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
datas = pd.read_csv('boston.csv')
datas


# In[59]:


X = datas.iloc[:, 1:2].values
y = datas.iloc[:, 2].values


# In[60]:


from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X, y)


# In[62]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 4)
X_poly = poly.fit_transform(X)
poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)


# In[63]:


plt.scatter(X, y, color = 'blue')
plt.plot(X, lin.predict(X), color = 'red')
plt.title('Linear Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()


# In[65]:


plt.scatter(X, y, color = 'blue')
plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red')
plt.title('Polynomial Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()


# # Advertising CSV regression

# In[67]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
df = pd.read_csv('Advertising.csv')
df


# In[68]:


df.dropna(inplace=True,axis=0)
df


# In[70]:


y = df['Sales']
X = df.drop('Sales',axis=1)


# In[71]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# In[72]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[73]:


lr = LinearRegression()
model = lr.fit(X_train,y_train)


# In[74]:


y_pred = model.predict(X_test)
ydf = pd.DataFrame({'y_test':y_test,'y_pred':y_pred})
rslt_df = ydf.sort_values(by = 'y_test')


# In[75]:


print(mean_squared_error(y_test,y_pred)) 


# In[76]:


print(r2_score(y_test, y_pred))


# In[77]:


import matplotlib.pyplot as plt
plt.scatter(ydf['y_test'],ydf['y_pred'])


# In[78]:


model.coef_


# In[79]:


model.intercept_


# In[ ]:




