#!/usr/bin/env python
# coding: utf-8

# ### `MODEL BUILDING -- LINEAR REGRESSION`

# `SPLIT THE TRAIN AND TEST DATASET`

# In[10]:


# !pip install pandas
import pandas as pd
# !pip install numpy
import numpy as np
# !pip install seaborn
import seaborn as sns
# !pip install matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# !pip install ipywidgets
import ipywidgets as widgets


# `READ THE CLEANED DATASET`

# In[11]:


df = pd.read_csv('Cleaned_df.csv')
df = df.drop(['Unnamed: 0'],axis = 1)


# `SPLIT THE DATASET INTO X AND Y`

# In[12]:


x = df.iloc[:,2:]
y = df.iloc[:,1]


# `SPLIT THE DATASET INTO TRAIN AND TEST OF X AND Y`

# In[13]:


# !pip install sklearn
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, random_state = 42)


# `CHECK FOR SHAPE OF TRAIN AND TEST SETS`

# In[14]:


print('Shape of x_train: ',x_train.shape)
print('Shape of x_test: ',x_test.shape)
print('Shape of y_train: ',y_train.shape)
print('Shape of y_test: ',y_test.shape)


# #### `FIT THE MODELS`

# In[15]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr = lr.fit(x_train,y_train)


# `PREDICT THE MODEL`

# In[16]:


y_pred = lr.predict(x_test)


# #### `MODEL-1`

# In[17]:


# !pip install statmodels
import statsmodels.api as sm
x_train_sm = x_train

x_train_sm = sm.add_constant(x_train_sm)

mlm = sm.OLS(y_train,x_train_sm).fit()

mlm.params
print(mlm.summary())


# In[18]:


#Actual vs Predicted
c = [i for i in range(1,49041,1)]
plt.plot(c,y_test, color="blue", linewidth=0.1, linestyle="-")
plt.plot(c,y_pred, color="red",  linewidth=0.1, linestyle="-")
plt.title('Actual_Power Vs Predicted_Power', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('Active_Power', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[19]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x_train.values, i) 
                          for i in range(len(x_train.columns))] 
  
print(vif_data)


# In[20]:


c = [i for i in range(1,49041,1)]

plt.plot(c,y_test-y_pred, color="blue", linewidth=0.1, linestyle="-")
plt.title('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[21]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :', round(r_squared,3),"% Variance of the Active Power is Explained by the Wind Speed, Pitch Angle and Rotor Speed")


# #### `MODEL-2`

# In[22]:


x1_train = x_train[['Avg_Ambient_Temp', 'Avg_Generator_Speed', 'Avg_Pitch_Angle', 'Avg_Rotor_Speed',
                    'Avg_Wind_Speed', 'Bearing_DE_Temp', 'Bearing_NDE_Temp', 'Gearbox_bearing_Temp', 'Gearbox_oil_Temp',
                    'Generator_wind_Temp_1', 'Generator_wind_Temp_2', 'Generator_wind_Temp_3', 'Generators_sliprings_Temp',
                    'Hidraulic_group_pressure', 'Nacelle_Misalignment_Avg_Wind_Dir', 'Trafo_1_wind_Temp', 'Trafo_2_wind_Temp',
                    'Trafo_3_wind_Temp']]
x1_test = x_test[['Avg_Ambient_Temp', 'Avg_Generator_Speed', 'Avg_Pitch_Angle', 'Avg_Rotor_Speed',
                    'Avg_Wind_Speed', 'Bearing_DE_Temp', 'Bearing_NDE_Temp', 'Gearbox_bearing_Temp', 'Gearbox_oil_Temp',
                    'Generator_wind_Temp_1', 'Generator_wind_Temp_2', 'Generator_wind_Temp_3', 'Generators_sliprings_Temp',
                    'Hidraulic_group_pressure', 'Nacelle_Misalignment_Avg_Wind_Dir', 'Trafo_1_wind_Temp', 'Trafo_2_wind_Temp',
                    'Trafo_3_wind_Temp']]


# Avg_Nacelle_Pos -- varible is removed because of its significant value is more than 0.05.

# In[23]:


lr1 = lr.fit(x1_train,y_train)


# In[24]:


y_pred1 = lr1.predict(x1_test)


# In[25]:


import statsmodels.api as sm
x1_train_sm = x1_train

x1_train_sm = sm.add_constant(x1_train_sm)

mlm1 = sm.OLS(y_train,x1_train_sm).fit()


mlm1.params
print(mlm1.summary())


# In[26]:


#Actual vs Predicted
c = [i for i in range(1,49041,1)]
plt.plot(c,y_test, color="blue", linewidth=0.1, linestyle="-")
plt.plot(c,y_pred1, color="red",  linewidth=0.1, linestyle="-")
plt.title('Actual_Power Vs Predicted_Power', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('Active_Power', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[27]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x1_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x1_train.values, i) 
                          for i in range(len(x1_train.columns))] 
  
print(vif_data)


# In[28]:


c = [i for i in range(1,49041,1)]

plt.plot(c,y_test-y_pred, color="blue", linewidth=0.1, linestyle="-")
plt.title('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[29]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred1)
r_squared = r2_score(y_test, y_pred1)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :', round(r_squared,3),"% Variance of the Active Power is Explained by the Wind Speed, Pitch Angle and Rotor Speed")


# #### `MODEL-3`

# In[30]:


x2_train = x_train[['Avg_Ambient_Temp', 'Avg_Generator_Speed', 'Avg_Pitch_Angle', 'Avg_Rotor_Speed',
                    'Avg_Wind_Speed', 'Bearing_DE_Temp', 'Bearing_NDE_Temp', 'Gearbox_bearing_Temp', 'Gearbox_oil_Temp',
                    'Generator_wind_Temp_1', 'Generator_wind_Temp_2', 'Generator_wind_Temp_3', 'Generators_sliprings_Temp',
                    'Hidraulic_group_pressure', 'Trafo_1_wind_Temp', 'Trafo_2_wind_Temp',
                    'Trafo_3_wind_Temp']]
x2_test = x_test[['Avg_Ambient_Temp', 'Avg_Generator_Speed', 'Avg_Pitch_Angle', 'Avg_Rotor_Speed',
                    'Avg_Wind_Speed', 'Bearing_DE_Temp', 'Bearing_NDE_Temp', 'Gearbox_bearing_Temp', 'Gearbox_oil_Temp',
                    'Generator_wind_Temp_1', 'Generator_wind_Temp_2', 'Generator_wind_Temp_3', 'Generators_sliprings_Temp',
                    'Hidraulic_group_pressure', 'Trafo_1_wind_Temp', 'Trafo_2_wind_Temp',
                    'Trafo_3_wind_Temp']]


# Nacelle_Misalignment_Avg_Wind_Dir varible is removed because of its significant value is more than 0.05.

# In[31]:


lr2 = lr.fit(x2_train,y_train)


# In[32]:


y_pred2 = lr2.predict(x2_test)


# In[33]:


import statsmodels.api as sm
x2_train_sm = x2_train

x2_train_sm = sm.add_constant(x2_train_sm)

mlm2 = sm.OLS(y_train,x2_train_sm).fit()

mlm2.params
print(mlm2.summary())


# In[34]:


#Actual vs Predicted
c = [i for i in range(1,49041,1)]
plt.plot(c,y_test, color="blue", linewidth=0.1, linestyle="-")
plt.plot(c,y_pred2, color="red",  linewidth=0.1, linestyle="-")
plt.title('Actual_Power Vs Predicted_Power', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('Active_Power', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[35]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x2_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x2_train.values, i) 
                          for i in range(len(x2_train.columns))] 
  
print(vif_data)


# In[36]:


c = [i for i in range(1,49041,1)]

plt.plot(c,y_test-y_pred2, color="blue", linewidth=0.1, linestyle="-")
plt.title('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[37]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred2)
r_squared = r2_score(y_test, y_pred2)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :', round(r_squared,3),"% Variance of the Active Power is Explained by the Wind Speed, Pitch Angle and Rotor Speed")


# #### `MODEL-4`

# In[38]:


x3_train = x_train[['Avg_Ambient_Temp', 'Avg_Pitch_Angle', 'Avg_Rotor_Speed',
                    'Avg_Wind_Speed', 'Bearing_DE_Temp', 'Bearing_NDE_Temp', 'Gearbox_bearing_Temp', 'Gearbox_oil_Temp',
                    'Generator_wind_Temp_1', 'Generator_wind_Temp_2', 'Generator_wind_Temp_3', 'Generators_sliprings_Temp',
                    'Hidraulic_group_pressure', 'Trafo_1_wind_Temp', 'Trafo_2_wind_Temp',
                    'Trafo_3_wind_Temp']]
x3_test = x_test[['Avg_Ambient_Temp', 'Avg_Pitch_Angle', 'Avg_Rotor_Speed',
                    'Avg_Wind_Speed', 'Bearing_DE_Temp', 'Bearing_NDE_Temp', 'Gearbox_bearing_Temp', 'Gearbox_oil_Temp',
                    'Generator_wind_Temp_1', 'Generator_wind_Temp_2', 'Generator_wind_Temp_3', 'Generators_sliprings_Temp',
                    'Hidraulic_group_pressure', 'Trafo_1_wind_Temp', 'Trafo_2_wind_Temp',
                    'Trafo_3_wind_Temp']]


# 'Avg_Generator_Speed' varible is removed because of its significant value is more than 0.05.

# In[39]:


lr3 = lr.fit(x3_train,y_train)


# In[40]:


y_pred3 = lr3.predict(x3_test)


# In[41]:


import statsmodels.api as sm
x3_train_sm = x3_train

x3_train_sm = sm.add_constant(x3_train_sm)

mlm3 = sm.OLS(y_train,x3_train_sm).fit()

mlm3.params
print(mlm3.summary())


# In[42]:


#Actual vs Predicted
c = [i for i in range(1,49041,1)]
plt.plot(c,y_test, color="blue", linewidth=0.1, linestyle="-")
plt.plot(c,y_pred3, color="red",  linewidth=0.1, linestyle="-")
plt.title('Actual_Power Vs Predicted_Power', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('Active_Power', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[43]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x3_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x3_train.values, i) 
                          for i in range(len(x3_train.columns))] 
  
print(vif_data)


# In[44]:


c = [i for i in range(1,49041,1)]

plt.plot(c,y_test-y_pred3, color="blue", linewidth=0.1, linestyle="-")
plt.title('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[45]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred3)
r_squared = r2_score(y_test, y_pred3)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :', round(r_squared,3),"% Variance of the Active Power is Explained by the Wind Speed, Pitch Angle and Rotor Speed")


# #### `MODEL-5`

# In[46]:


x4_train = x_train[['Avg_Ambient_Temp', 'Avg_Pitch_Angle', 'Avg_Rotor_Speed', 'Avg_Wind_Speed', 'Bearing_DE_Temp',
                    'Gearbox_bearing_Temp', 'Gearbox_oil_Temp', 'Generator_wind_Temp_1',
                    'Generator_wind_Temp_2', 'Generator_wind_Temp_3', 'Generators_sliprings_Temp',
                    'Hidraulic_group_pressure', 'Trafo_1_wind_Temp', 'Trafo_2_wind_Temp',
                    'Trafo_3_wind_Temp']]
x4_test = x_test[['Avg_Ambient_Temp', 'Avg_Pitch_Angle', 'Avg_Rotor_Speed', 'Avg_Wind_Speed', 'Bearing_DE_Temp',
                    'Gearbox_bearing_Temp', 'Gearbox_oil_Temp', 'Generator_wind_Temp_1',
                    'Generator_wind_Temp_2', 'Generator_wind_Temp_3', 'Generators_sliprings_Temp',
                    'Hidraulic_group_pressure', 'Trafo_1_wind_Temp', 'Trafo_2_wind_Temp',
                    'Trafo_3_wind_Temp']]


# 'Bearing_NDE_Temp' & 'Bearing_DE_Temp' These varibles are Auto Correlated each other. So, i removed one variable ('Bearing_NDE_Temp')

# In[47]:


lr4 = lr.fit(x4_train,y_train)


# In[48]:


y_pred4 = lr4.predict(x4_test)


# In[49]:


import statsmodels.api as sm
x4_train_sm = x4_train

x4_train_sm = sm.add_constant(x4_train_sm)

mlm4 = sm.OLS(y_train,x4_train_sm).fit()

mlm4.params
print(mlm4.summary())


# In[50]:


#Actual vs Predicted
c = [i for i in range(1,49041,1)]
plt.plot(c,y_test, color="blue", linewidth=0.1, linestyle="-")
plt.plot(c,y_pred4, color="red",  linewidth=0.1, linestyle="-")
plt.title('Actual_Power Vs Predicted_Power', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('Active_Power', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[51]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x4_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x4_train.values, i) 
                          for i in range(len(x4_train.columns))] 
  
print(vif_data)


# In[52]:


c = [i for i in range(1,49041,1)]

plt.plot(c,y_test-y_pred4, color="blue", linewidth=0.1, linestyle="-")
plt.title('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[53]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred4)
r_squared = r2_score(y_test, y_pred4)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :', round(r_squared,3),"% Variance of the Active Power is Explained by the Wind Speed, Pitch Angle and Rotor Speed")


# #### `MODEL-6`

# In[54]:


x5_train = x_train[['Avg_Ambient_Temp', 'Avg_Pitch_Angle', 'Avg_Rotor_Speed', 'Avg_Wind_Speed', 'Bearing_DE_Temp',
                    'Gearbox_bearing_Temp', 'Gearbox_oil_Temp', 'Generator_wind_Temp_1',
                    'Generator_wind_Temp_3', 'Generators_sliprings_Temp', 'Hidraulic_group_pressure', 'Trafo_1_wind_Temp',
                    'Trafo_2_wind_Temp', 'Trafo_3_wind_Temp']]
x5_test = x_test[['Avg_Ambient_Temp', 'Avg_Pitch_Angle', 'Avg_Rotor_Speed', 'Avg_Wind_Speed', 'Bearing_DE_Temp',
                    'Gearbox_bearing_Temp', 'Gearbox_oil_Temp', 'Generator_wind_Temp_1',
                    'Generator_wind_Temp_3', 'Generators_sliprings_Temp', 'Hidraulic_group_pressure', 'Trafo_1_wind_Temp',
                    'Trafo_2_wind_Temp', 'Trafo_3_wind_Temp']]


# 'Generator_wind_Temp_1', 'Generator_wind_Temp_2' These varibles are Auto Correlated each other. So, i removed one variable ('Generator_wind_Temp_2')

# In[55]:


lr5 = lr.fit(x5_train,y_train)


# In[56]:


y_pred5 = lr5.predict(x5_test)


# In[57]:


import statsmodels.api as sm
x5_train_sm = x5_train

x5_train_sm = sm.add_constant(x5_train_sm)

mlm5 = sm.OLS(y_train,x5_train_sm).fit()

mlm5.params
print(mlm5.summary())


# In[58]:


#Actual vs Predicted
c = [i for i in range(1,49041,1)]
plt.plot(c,y_test, color="blue", linewidth=0.1, linestyle="-")
plt.plot(c,y_pred5, color="red",  linewidth=0.1, linestyle="-")
plt.title('Actual_Power Vs Predicted_Power', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('Active_Power', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[59]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x5_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x5_train.values, i) 
                          for i in range(len(x5_train.columns))] 
  
print(vif_data)


# In[60]:


c = [i for i in range(1,49041,1)]

plt.plot(c,y_test-y_pred5, color="blue", linewidth=0.1, linestyle="-")
plt.title('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[61]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred5)
r_squared = r2_score(y_test, y_pred5)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :', round(r_squared,3),"% Variance of the Active Power is Explained by the Wind Speed, Pitch Angle and Rotor Speed")


# #### `MODEL-7`

# In[62]:


x6_train = x_train[['Avg_Ambient_Temp', 'Avg_Pitch_Angle', 'Avg_Rotor_Speed', 'Avg_Wind_Speed', 'Bearing_DE_Temp',
                    'Gearbox_bearing_Temp', 'Gearbox_oil_Temp', 'Generator_wind_Temp_1', 'Generators_sliprings_Temp',
                    'Hidraulic_group_pressure', 'Trafo_1_wind_Temp', 'Trafo_2_wind_Temp', 'Trafo_3_wind_Temp']]
x6_test = x_test[['Avg_Ambient_Temp', 'Avg_Pitch_Angle', 'Avg_Rotor_Speed', 'Avg_Wind_Speed', 'Bearing_DE_Temp',
                    'Gearbox_bearing_Temp', 'Gearbox_oil_Temp', 'Generator_wind_Temp_1', 'Generators_sliprings_Temp',
                    'Hidraulic_group_pressure', 'Trafo_1_wind_Temp', 'Trafo_2_wind_Temp', 'Trafo_3_wind_Temp']]


# 'Generator_wind_Temp_1', 'Generator_wind_Temp_3' These varibles are Auto Correlated each other. So, i removed one variable ('Generator_wind_Temp_3')

# In[63]:


lr6 = lr.fit(x6_train,y_train)


# In[64]:


y_pred6 = lr6.predict(x6_test)


# In[65]:


import statsmodels.api as sm
x6_train_sm = x6_train

x6_train_sm = sm.add_constant(x6_train_sm)

mlm6 = sm.OLS(y_train,x6_train_sm).fit()

mlm6.params
print(mlm6.summary())


# In[66]:


#Actual vs Predicted
c = [i for i in range(1,49041,1)]
plt.plot(c,y_test, color="blue", linewidth=0.1, linestyle="-")
plt.plot(c,y_pred6, color="red",  linewidth=0.1, linestyle="-")
plt.title('Actual_Power Vs Predicted_Power', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('Active_Power', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[67]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x6_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x6_train.values, i) 
                          for i in range(len(x6_train.columns))] 
  
print(vif_data)


# In[68]:


c = [i for i in range(1,49041,1)]

plt.plot(c,y_test-y_pred6, color="blue", linewidth=0.1, linestyle="-")
plt.title('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[69]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred6)
r_squared = r2_score(y_test, y_pred6)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :', round(r_squared,3),"% Variance of the Active Power is Explained by the Wind Speed, Pitch Angle and Rotor Speed")


# #### `MODEL-8`

# In[70]:


x7_train = x_train[['Avg_Ambient_Temp', 'Avg_Pitch_Angle', 'Avg_Rotor_Speed', 'Avg_Wind_Speed', 'Bearing_DE_Temp',
                    'Gearbox_bearing_Temp', 'Gearbox_oil_Temp', 'Generator_wind_Temp_1', 'Generators_sliprings_Temp',
                    'Hidraulic_group_pressure', 'Trafo_2_wind_Temp', 'Trafo_3_wind_Temp']]
x7_test = x_test[['Avg_Ambient_Temp', 'Avg_Pitch_Angle', 'Avg_Rotor_Speed', 'Avg_Wind_Speed', 'Bearing_DE_Temp',
                    'Gearbox_bearing_Temp', 'Gearbox_oil_Temp', 'Generator_wind_Temp_1', 'Generators_sliprings_Temp',
                    'Hidraulic_group_pressure', 'Trafo_2_wind_Temp', 'Trafo_3_wind_Temp']]


# 'Trafo_1_wind_Temp', 'Trafo_2_wind_Temp' These varibles are Auto Correlated each other. So, i removed one variable ('Trafo_1_wind_Temp')

# In[71]:


lr7 = lr.fit(x7_train,y_train)


# In[72]:


y_pred7 = lr7.predict(x7_test)


# In[73]:


import statsmodels.api as sm
x7_train_sm = x7_train

x7_train_sm = sm.add_constant(x7_train_sm)

mlm7 = sm.OLS(y_train,x7_train_sm).fit()

mlm7.params
print(mlm7.summary())


# In[74]:


#Actual vs Predicted
c = [i for i in range(1,49041,1)]
plt.plot(c,y_test, color="blue", linewidth=0.1, linestyle="-")
plt.plot(c,y_pred7, color="red",  linewidth=0.1, linestyle="-")
plt.title('Actual_Power Vs Predicted_Power', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('Active_Power', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[75]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x7_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x7_train.values, i) 
                          for i in range(len(x7_train.columns))] 
  
print(vif_data)


# In[76]:


c = [i for i in range(1,49041,1)]

plt.plot(c,y_test-y_pred7, color="blue", linewidth=0.1, linestyle="-")
plt.title('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[77]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred7)
r_squared = r2_score(y_test, y_pred7)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :', round(r_squared,3),"% Variance of the Active Power is Explained by the Wind Speed, Pitch Angle and Rotor Speed")


# #### `MODEL-9`

# In[78]:


x8_train = x_train[['Avg_Ambient_Temp', 'Avg_Pitch_Angle', 'Avg_Rotor_Speed', 'Avg_Wind_Speed', 'Bearing_DE_Temp',
                    'Gearbox_oil_Temp', 'Generator_wind_Temp_1', 'Generators_sliprings_Temp',
                    'Hidraulic_group_pressure', 'Trafo_2_wind_Temp', 'Trafo_3_wind_Temp']]
x8_test = x_test[['Avg_Ambient_Temp', 'Avg_Pitch_Angle', 'Avg_Rotor_Speed', 'Avg_Wind_Speed', 'Bearing_DE_Temp',
                    'Gearbox_oil_Temp', 'Generator_wind_Temp_1', 'Generators_sliprings_Temp',
                    'Hidraulic_group_pressure', 'Trafo_2_wind_Temp', 'Trafo_3_wind_Temp']]


# 'Gearbox_bearing_Temp', 'Gearbox_oil_Temp' These varibles are Auto Correlated each other. So, i removed one variable ('Gearbox_bearing_Temp')

# In[79]:


lr8 = lr.fit(x8_train,y_train)


# In[80]:


y_pred8 = lr8.predict(x8_test)


# In[81]:


import statsmodels.api as sm
x8_train_sm = x8_train

x8_train_sm = sm.add_constant(x8_train_sm)

mlm8 = sm.OLS(y_train,x8_train_sm).fit()

mlm8.params
print(mlm8.summary())


# In[82]:


#Actual vs Predicted
c = [i for i in range(1,49041,1)]
plt.plot(c,y_test, color="blue", linewidth=0.1, linestyle="-")
plt.plot(c,y_pred8, color="red",  linewidth=0.1, linestyle="-")
plt.title('Actual_Power Vs Predicted_Power', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('Active_Power', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[83]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x8_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x8_train.values, i) 
                          for i in range(len(x8_train.columns))] 
  
print(vif_data)


# In[84]:


c = [i for i in range(1,49041,1)]

plt.plot(c,y_test-y_pred8, color="blue", linewidth=0.1, linestyle="-")
plt.title('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[85]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred8)
r_squared = r2_score(y_test, y_pred8)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :', round(r_squared,3),"% Variance of the Active Power is Explained by the Wind Speed, Pitch Angle and Rotor Speed")


# #### `MODEL-10`

# In[86]:


x9_train = x_train[['Avg_Ambient_Temp', 'Avg_Pitch_Angle', 'Avg_Rotor_Speed', 'Avg_Wind_Speed', 'Bearing_DE_Temp',
                    'Generator_wind_Temp_1', 'Generators_sliprings_Temp', 'Gearbox_oil_Temp']]
x9_test = x_test[['Avg_Ambient_Temp', 'Avg_Pitch_Angle', 'Avg_Rotor_Speed', 'Avg_Wind_Speed', 'Bearing_DE_Temp',
                  'Generator_wind_Temp_1', 'Generators_sliprings_Temp', 'Gearbox_oil_Temp']]


# 'Hidraulic_group_pressure', 'Trafo_2_wind_Temp', 'Trafo_3_wind_Temp' These varibles are Auto Correlated. So, i removed the variable

# In[87]:


lr9 = lr.fit(x9_train,y_train)


# In[88]:


y_pred9 = lr9.predict(x9_test)


# In[89]:


import statsmodels.api as sm
x9_train_sm = x9_train

x9_train_sm = sm.add_constant(x9_train_sm)

mlm9 = sm.OLS(y_train,x9_train_sm).fit()

mlm9.params
print(mlm9.summary())


# In[90]:


#Actual vs Predicted
c = [i for i in range(1,49041,1)]
plt.plot(c,y_test, color="blue", linewidth=0.1, linestyle="-")
plt.plot(c,y_pred9, color="red",  linewidth=0.1, linestyle="-")
plt.title('Actual_Power Vs Predicted_Power', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('Active_Power', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[91]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x9_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x9_train.values, i) 
                          for i in range(len(x9_train.columns))] 
  
print(vif_data)


# In[92]:


c = [i for i in range(1,49041,1)]

plt.plot(c,y_test-y_pred9, color="blue", linewidth=0.1, linestyle="-")
plt.title('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[93]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred9)
r_squared = r2_score(y_test, y_pred9)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :', round(r_squared,3),"% Variance of the Active Power is Explained by the Wind Speed, Pitch Angle and Rotor Speed")


# #### `MODEL-10`

# In[94]:


x10_train = x_train[['Avg_Pitch_Angle', 'Avg_Rotor_Speed', 'Avg_Wind_Speed']]
x10_test = x_test[['Avg_Pitch_Angle', 'Avg_Rotor_Speed', 'Avg_Wind_Speed']]


# Removed all temperature variables.

# In[95]:


lr10 = lr.fit(x10_train,y_train)


# In[96]:


y_pred10 = lr10.predict(x10_test)


# In[97]:


import statsmodels.api as sm
x10_train_sm = x10_train

x10_train_sm = sm.add_constant(x10_train_sm)

mlm10 = sm.OLS(y_train,x10_train_sm).fit()

mlm10.params
print(mlm10.summary())


# In[98]:


#Actual vs Predicted
c = [i for i in range(1,49041,1)]
plt.plot(c,y_test, color="blue", linewidth=0.1, linestyle="-")
plt.plot(c,y_pred10, color="red",  linewidth=0.1, linestyle="-")
plt.title('Actual_Power Vs Predicted_Power', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('Active_Power', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[99]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x10_train.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x10_train.values, i) 
                          for i in range(len(x10_train.columns))] 
  
print(vif_data)


# In[100]:


c = [i for i in range(1,49041,1)]

plt.plot(c,y_test-y_pred10, color="blue", linewidth=0.1, linestyle="-")
plt.title('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.5, top=1.6)


# In[101]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred10)
r_squared = r2_score(y_test, y_pred10)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :', round(r_squared,3),"% Variance of the Active Power is Explained by the Wind Speed, Pitch Angle and Rotor Speed")


# `REGRESSION EQUATION`

# In[102]:


@widgets.interact(Wind_Speed = range(1, 25, 1), 
                  Pitch_Angle = range(1, 95, 1), Rotor_Speed = range(1, 20, 1))
def Regression_Equation(Wind_Speed, Pitch_Angle, Rotor_Speed):
    Intercept = mlm10.params[0]
    Coefficient = mlm10.params[1]*Pitch_Angle + mlm10.params[2]*Rotor_Speed + mlm10.params[3]*Wind_Speed
    
    y_Predicted = Intercept + Coefficient  # y = a + bx  Equation.
    if y_Predicted < 0:
        print("Please Choose Other Values: Active Power is '0' at these values")
    else:
        print('Active Power Predicted: ',y_Predicted)  # Basic Starting Values are 3, 77, 1

Thank You...