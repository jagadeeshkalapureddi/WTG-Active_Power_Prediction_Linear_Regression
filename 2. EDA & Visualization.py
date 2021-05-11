#!/usr/bin/env python
# coding: utf-8

# ### `IMPORT REQUIRED PACKAGES(LIBRARIES)`

# In[1]:


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
from matplotlib.cm import get_cmap


# ### `IMPORT DATASET`

# #### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# **Import dataset from google drive and assigned the dataset as "df"** ----`It will take few minutes based on internet because its file size is large.`
# #### --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# In[2]:


URL = 'https://drive.google.com/file/d/1haTtjbR90YnbAGls8QVdREmHNmx1a5mQ/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+URL.split('/')[-2]


# `Set up display area to show dataframe in jupyter console`

# In[3]:


get_ipython().run_cell_magic('time', '', "print('*' * 127)\nprint('WIND TURBINE GENERATOR PARAMETER DATASET')\nMaster = pd.read_csv(path,low_memory = False, encoding = 'cp1252', parse_dates=['Date'], skipinitialspace=True)\nprint('*' * 127)")


# In[4]:


df = Master


# ### `EXPLORATORY DATA ANALYSIS`

# #### `CONFIRM THE DATA TRANSFER FROM GOOGLE DRIVE TO NOTEBOOK`

# In[5]:


print('First Five Records of the Dataset')
df.head()  # As default shows the top 5 Rows.


# In[6]:


print('Last Five Records of the Dataset')
df.tail()  # As default shows the Last 5 Rows.


# `Check for the Shape`

# In[7]:


print("The Data Frame having the Rows of '{}' and Columns of '{}'".format (df.shape[0],df.shape[1]))


# `Check for the Detailed Information of the Dataset`

# In[8]:


print('Total_Columns: ', len(df.columns),'\n')
print(df.columns,'\n')
print('Shape :',df.shape)


# In[9]:


df.rename({"Generator�s_sliprings_Temp":"Generators_sliprings_Temp", "Generator’s_sliprings_Temp":"Generators_sliprings_Temp"}, axis = 1, inplace = True)


# In[10]:


df.info()


# `CHANGE THE DATA TYPE`

# In[11]:


df['Date'] = pd.to_datetime(df['Date'], errors='coerce')


# `Displays memory consumed by each column`

# In[12]:


print(df.memory_usage(),'\n')
print('Dataset uses {0} MB'.format(df.memory_usage().sum()/1024**2))


# In[13]:


df.info()


# `CHECK FOR NULL VALUES`

# In[14]:


df.isnull().sum()


# `NULL VALUE COUNTS PLOT BEFORE TREATMENT`

# In[15]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
x = df.columns
y = df.isnull().sum()
plt.bar(x,y,color = colors, align = 'center')
plt.xticks(rotation=90)
plt.title('Barplot_Null Value_Count', fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.3, top=1.3)
for i in range(len(x)):
    pos = y[i]
    string = '{:}'.format(pos)
    plt.text(i,pos,string,ha='left',color='black',rotation = 'vertical', va = 'top')


# `NULL VALUE TREATMENT`

#         Every parameter of the Wind Turbine is a Numerical Variable. So, to avoid any Outliers effect on variables. I am doing Null values treatment with its Median.

# In[16]:


constant_values = {'Avg_Active_Power': df.iloc[:,1].median(), 'Avg_Ambient_Temp': df.iloc[:,2].median(),
       'Avg_Generator_Speed': df.iloc[:,3].median(), 'Avg_Nacelle_Pos': df.iloc[:,4].median(), 'Avg_Pitch_Angle': df.iloc[:,5].median(),
       'Avg_Rotor_Speed': df.iloc[:,6].median(), 'Avg_Wind_Speed': df.iloc[:,7].median(), 'Bearing_DE_Temp': df.iloc[:,8].median(),
       'Bearing_NDE_Temp': df.iloc[:,9].median(), 'Gearbox_bearing_Temp': df.iloc[:,10].median(), 'Gearbox_oil_Temp': df.iloc[:,11].median(),
       'Generator_wind_Temp_1': df.iloc[:,12].median(), 'Generator_wind_Temp_2': df.iloc[:,13].median(),
       'Generator_wind_Temp_3': df.iloc[:,14].median(), "Generators_sliprings_Temp": df.iloc[:,15].median(),
       'Hidraulic_group_pressure': df.iloc[:,16].median(), 'Nacelle_Misalignment_Avg_Wind_Dir': df.iloc[:,17].median(),
       'Trafo_1_wind_Temp': df.iloc[:,18].median(), 'Trafo_2_wind_Temp': df.iloc[:,19].median(), 'Trafo_3_wind_Temp': df.iloc[:,20].median()}
df[['Avg_Active_Power', 'Avg_Ambient_Temp', 'Avg_Generator_Speed', 'Avg_Nacelle_Pos', 'Avg_Pitch_Angle', 'Avg_Rotor_Speed', 'Avg_Wind_Speed', 'Bearing_DE_Temp',
    'Bearing_NDE_Temp', 'Gearbox_bearing_Temp', 'Gearbox_oil_Temp', 'Generator_wind_Temp_1', 'Generator_wind_Temp_2', 'Generator_wind_Temp_3', "Generators_sliprings_Temp",
    'Hidraulic_group_pressure', 'Nacelle_Misalignment_Avg_Wind_Dir', 
    'Trafo_1_wind_Temp', 'Trafo_2_wind_Temp', 'Trafo_3_wind_Temp']] = df[['Avg_Active_Power', 'Avg_Ambient_Temp', 'Avg_Generator_Speed', 'Avg_Nacelle_Pos', 'Avg_Pitch_Angle',
                                                                          'Avg_Rotor_Speed', 'Avg_Wind_Speed', 'Bearing_DE_Temp', 'Bearing_NDE_Temp', 'Gearbox_bearing_Temp',
                                                                          'Gearbox_oil_Temp', 'Generator_wind_Temp_1', 'Generator_wind_Temp_2', 'Generator_wind_Temp_3', 
                                                                          "Generators_sliprings_Temp", 'Hidraulic_group_pressure', 'Nacelle_Misalignment_Avg_Wind_Dir',
                                                                          'Trafo_1_wind_Temp', 'Trafo_2_wind_Temp', 'Trafo_3_wind_Temp']].fillna(value = constant_values)


# `NULL VALUE COUNTS PLOT AFTER TREATMENT`

# In[17]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
x = df.columns
y = df.isnull().sum()
plt.bar(x,y,color = colors, align = 'center')
plt.xticks(rotation=90)
plt.title('Barplot_Null Value_Count', fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.3, top=1.3)
for i in range(len(x)):
    pos = y[i]
    string = '{:}'.format(pos)
    plt.text(i,pos,string,ha='left',color='black',rotation = 'vertical', va = 'top')


# `Statistical Information :`

# In[18]:


df.describe()


# `CHECK FOR OUTLIER VALUES USING BOX PLOTS`

# In[19]:


plt.subplot(8,2,1)
sns.boxplot(df.iloc[:,1], color = 'green')
plt.xlabel("")
plt.title('Boxplot_ '+ df.iloc[:,1].name +'_(No Outliers)_Skew_' + str(round(df.iloc[:,1].skew(),2)), fontsize = 10)
plt.subplot(8,2,2)
sns.boxplot(df.iloc[:,2], color = 'green')
plt.xlabel("")
plt.title('Boxplot_ '+ df.iloc[:,2].name +'_(No Outliers)_Skew_' + str(round(df.iloc[:,2].skew(),2)), fontsize = 10)
plt.subplot(8,2,3)
sns.boxplot(df.iloc[:,3], color = 'green')
plt.xlabel("")
plt.title('Boxplot_ '+ df.iloc[:,3].name +'_(No Outliers)_Skew_' + str(round(df.iloc[:,3].skew(),2)), fontsize = 10)
plt.subplot(8,2,4)
sns.boxplot(df.iloc[:,4], color = 'green')
plt.xlabel("")
plt.title('Boxplot_ '+ df.iloc[:,4].name +'_(No Outliers)_Skew_' + str(round(df.iloc[:,4].skew(),2)), fontsize = 10)
plt.subplot(8,2,5)
sns.boxplot(df.iloc[:,5], color = 'green')
plt.xlabel("")
plt.title('Boxplot_ '+ df.iloc[:,5].name +'_(No Outliers)_Skew_' + str(round(df.iloc[:,5].skew(),2)), fontsize = 10)
plt.subplot(8,2,6)
sns.boxplot(df.iloc[:,6], color = 'green')
plt.xlabel("")
plt.title('Boxplot_ '+ df.iloc[:,6].name +'_(No Outliers)_Skew_' + str(round(df.iloc[:,6].skew(),2)), fontsize = 10)
plt.subplot(8,2,7)
sns.boxplot(df.iloc[:,7], color = 'green')
plt.xlabel("")
plt.title('Boxplot_ '+ df.iloc[:,7].name +'_(No Outliers)_Skew_' + str(round(df.iloc[:,7].skew(),2)), fontsize = 10)
plt.subplot(8,2,8)
sns.boxplot(df.iloc[:,8], color = 'green')
plt.xlabel("")
plt.title('Boxplot_ '+ df.iloc[:,8].name +'_(No Outliers)_Skew_' + str(round(df.iloc[:,8].skew(),2)), fontsize = 10)
plt.subplot(8,2,9)
sns.boxplot(df.iloc[:,9], color = 'green')
plt.xlabel("")
plt.title('Boxplot_ '+ df.iloc[:,9].name +'_(No Outliers)_Skew_' + str(round(df.iloc[:,9].skew(),2)), fontsize = 10)
plt.subplot(8,2,10)
sns.boxplot(df.iloc[:,10], color = 'green')
plt.xlabel("")
plt.title('Boxplot_ '+ df.iloc[:,10].name +'_(No Outliers)_Skew_' + str(round(df.iloc[:,10].skew(),2)), fontsize = 10)

plt.subplots_adjust(left=0.45, bottom=0, right=2.5, top=4.9)


# In[20]:


plt.subplot(8,2,1)
sns.boxplot(df.iloc[:,11], color = 'green')
plt.xlabel("")
plt.title('Boxplot_ '+ df.iloc[:,11].name +'_(No Outliers)_Skew_' + str(round(df.iloc[:,11].skew(),2)), fontsize = 10)
plt.subplot(8,2,2)
sns.boxplot(df.iloc[:,12], color = 'green')
plt.xlabel("")
plt.title('Boxplot_ '+ df.iloc[:,12].name +'_(No Outliers)_Skew_' + str(round(df.iloc[:,12].skew(),2)), fontsize = 10)
plt.subplot(8,2,3)
sns.boxplot(df.iloc[:,13], color = 'green')
plt.xlabel("")
plt.title('Boxplot_ '+ df.iloc[:,13].name +'_(No Outliers)_Skew_' + str(round(df.iloc[:,13].skew(),2)), fontsize = 10)
plt.subplot(8,2,4)
sns.boxplot(df.iloc[:,14], color = 'green')
plt.xlabel("")
plt.title('Boxplot_ '+ df.iloc[:,14].name +'_(Found Outliers)_Skew_' + str(round(df.iloc[:,14].skew(),2)), fontsize = 10)
plt.subplot(8,2,5)
sns.boxplot(df.iloc[:,15], color = 'green')
plt.xlabel("")
plt.title('Boxplot_ '+ df.iloc[:,15].name +'_(No Outliers)_Skew_' + str(round(df.iloc[:,15].skew(),2)), fontsize = 10)
plt.subplot(8,2,6)
sns.boxplot(df.iloc[:,16], color = 'green')
plt.xlabel("")
plt.title('Boxplot_ '+ df.iloc[:,16].name +'_(No Outliers)_Skew_' + str(round(df.iloc[:,16].skew(),2)), fontsize = 10)
plt.subplot(8,2,7)
sns.boxplot(df.iloc[:,17], color = 'green')
plt.xlabel("")
plt.title('Boxplot_ '+ df.iloc[:,17].name +'_(No Outliers)_Skew_' + str(round(df.iloc[:,17].skew(),2)), fontsize = 10)
plt.subplot(8,2,8)
sns.boxplot(df.iloc[:,18], color = 'green')
plt.xlabel("")
plt.title('Boxplot_ '+ df.iloc[:,18].name +'_(No Outliers)_Skew_' + str(round(df.iloc[:,18].skew(),2)), fontsize = 10)
plt.subplot(8,2,9)
sns.boxplot(df.iloc[:,19], color = 'green')
plt.xlabel("")
plt.title('Boxplot_ '+ df.iloc[:,19].name +'_(No Outliers)_Skew_' + str(round(df.iloc[:,19].skew(),2)), fontsize = 10)
plt.subplot(8,2,10)
sns.boxplot(df.iloc[:,20], color = 'green')
plt.xlabel("")
plt.title('Boxplot_ '+ df.iloc[:,20].name +'_(No Outliers)_Skew_' + str(round(df.iloc[:,20].skew(),2)), fontsize = 10)

plt.subplots_adjust(left=0.45, bottom=0, right=2.5, top=4.9)


# #### -----------------------------------------------------------------------------------------------------
# 
#     From above box plots i found that the data is almost not having the outliers except one variable that is Generator Winding Temperature-3. So, we need Outliers Treatment as using Quantiles.
#     
# #### ----------------------------------------------

# `Check for the Quartile Ranges`

# In[21]:


print('Lower Limit - 5% :', df.iloc[:,14].quantile(0.05), '\n Upper Limit - 95% :', df.iloc[:,14].quantile(0.95))


# `Replace the Outliers with its Quartile ranges`

# In[22]:


df.iloc[:,14] = np.where(df.iloc[:,14] < df.iloc[:,14].quantile(0.05), df.iloc[:,14].quantile(0.05), df.iloc[:,14])
df.iloc[:,14].describe()


# `Box Plot and Histogram plot for re-checking the Outliers`

# In[23]:


plt.subplot(1,2,1)
sns.boxplot(df.iloc[:,14], color = 'green')
plt.xlabel(df.iloc[:,14].name, fontsize = 10)
plt.title('Boxplot_ '+ df.iloc[:,14].name +'_(No Outliers)_Skew_' + str(round(df.iloc[:,14].skew(),2)), fontsize = 10)
plt.subplot(1,2,2)
plt.hist(df.iloc[:,14], color = 'orange')
plt.xlabel(df.iloc[:,14].name, fontsize = 10)
plt.title('Histogram_ '+ df.iloc[:,14].name, fontsize = 10)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# In[24]:


df.to_csv('Cleaned_df.csv')


# `VISUALIZATION`

# In[25]:


def Scatter_Plot(X,Y,Size,Color):
        fig = plt.figure(figsize=(8,4))

        plt.scatter(x = X,
                    y = Y, s = float(Size), color = Color)

        plt.xlabel(X.name.capitalize())
        plt.ylabel(Y.name.capitalize())

        plt.title("%s vs %s"%(X.name.capitalize(), Y.name.capitalize()))


# `POWER CURVE - Average Wind Speed Vs Average Actvie Power`

# In[26]:


Scatter_Plot(df.iloc[:,7],df.iloc[:,1], 8, 'green')


# `Average Generator Speed Vs Average Actvie Power`

# In[27]:


Scatter_Plot(df.iloc[:,3],df.iloc[:,1], 8, 'green')


# `Average Generator Speed Vs Average Wind Speed`

# In[28]:


Scatter_Plot(df.iloc[:,7],df.iloc[:,3], 8, 'green')


# `Average Pitch Angle Vs Average Generator Speed`

# In[29]:


Scatter_Plot(df.iloc[:,5],df.iloc[:,3], 8, 'green')


# `Average Pitch Angle Vs Average Wind Speed`

# In[30]:


Scatter_Plot(df.iloc[:,7],df.iloc[:,5], 8, 'green')


# `Average Wind Speed Vs Average Rotor Speed`

# In[31]:


Scatter_Plot(df.iloc[:,7],df.iloc[:,6], 8, 'green')


# `All Temperature Parameters Vs Active Power`

# In[32]:


# line 1 points

x1 = df.iloc[:,1]
y1 = df.iloc[:,8]

z1 = np.polyfit(x1, y1, 1)
p1 = np.poly1d(z1)
plt.plot(x1,p1(x1),"r-.", label = df.iloc[:,8].name.upper())

# line 2 points
x2 = df.iloc[:,1]
y2 = df.iloc[:,9]

z2 = np.polyfit(x2, y2, 1)
p2 = np.poly1d(z2)
plt.plot(x2,p2(x2),"b-.",label = df.iloc[:,9].name.upper())

# line 3 points
x3 = df.iloc[:,1]
y3 = df.iloc[:,2]

z3 = np.polyfit(x3, y3, 1)
p3 = np.poly1d(z3)
plt.plot(x3,p3(x3),"g--", label = df.iloc[:,2].name.upper())

# line 4 points
x4 = df.iloc[:,1]
y4 = df.iloc[:,10]

z4 = np.polyfit(x4, y4, 1)
p4 = np.poly1d(z4)
plt.plot(x4,p4(x4),"c--",label = df.iloc[:,10].name.upper())

# line 5 points
x5 = df.iloc[:,1]
y5 = df.iloc[:,11]

z5 = np.polyfit(x5, y5, 1)
p5 = np.poly1d(z5)
plt.plot(x5,p5(x5),"m--", label = df.iloc[:,11].name.upper())

# line 6 points
x6 = df.iloc[:,1]
y6 = df.iloc[:,12]

z6 = np.polyfit(x6, y6, 1)
p6 = np.poly1d(z6)
plt.plot(x6,p6(x6),"r_",label = df.iloc[:,12].name.upper())

# line 7 points
x7 = df.iloc[:,1]
y7 = df.iloc[:,13]

z7 = np.polyfit(x7, y7, 1)
p7 = np.poly1d(z7)
plt.plot(x7,p7(x7),"y_",label = df.iloc[:,13].name.upper())

# line 8 points
x8 = df.iloc[:,1]
y8 = df.iloc[:,14]

z8 = np.polyfit(x8, y8, 1)
p8 = np.poly1d(z8)
plt.plot(x8,p8(x8),"b_",label = df.iloc[:,14].name.upper())

# line 9 points
x9 = df.iloc[:,1]
y9 = df.iloc[:,15]

z9 = np.polyfit(x9, y9, 1)
p9 = np.poly1d(z9)
plt.plot(x9,p9(x9),"g:",label = df.iloc[:,15].name.upper())

# line 10 points
x10 = df.iloc[:,1]
y10 = df.iloc[:,18]

z10 = np.polyfit(x10, y10, 1)
p10 = np.poly1d(z10)
plt.plot(x10,p10(x10),"r:",label = df.iloc[:,18].name.upper())

# line 11 points
x11 = df.iloc[:,1]
y11 = df.iloc[:,19]

z11 = np.polyfit(x11, y11, 1)
p11 = np.poly1d(z11)
plt.plot(x11,p11(x11),"y:",label = df.iloc[:,19].name.upper())

# line 12 points
x12 = df.iloc[:,1]
y12 = df.iloc[:,20]

z12 = np.polyfit(x12, y12, 1)
p12 = np.poly1d(z12)
plt.plot(x12,p12(x12),"b:",label = df.iloc[:,20].name.upper())


# Set the y & x axis label of the current axis.
plt.ylabel('Wind Turbine all Temperatures', fontsize = 15)
plt.xlabel('Average Active Power', fontsize = 15)
# Set a title of the current axes.
plt.title('Temperatures with respect to Active Power', fontsize = 20)
# show a legend on the plot
plt.legend()
# Display a figure.
plt.subplots_adjust(left=0.5, bottom=0, right=2.4, top=1.5)
plt.show()


# `ACTIVE POWER TREND DURING A TIME PERIOD`

# In[33]:


pf1 = df[['Date','Avg_Active_Power']]
pf1 = pf1.set_index('Date')
y1 = pf1['Avg_Active_Power'].resample('D').mean()
y1 = y1.fillna(y1.mean())
plt.figure(figsize=(15,6))
plt.plot(y1)
plt.title('Time Variant Plot for Avg_Active Power')
plt.xlabel("Time in Days")
plt.ylabel("Avg_Active_Power")
plt.show()


# `WIND SPEED TREND OVER A TIME PERIOD`

# In[34]:


pf2 = df[['Date','Avg_Wind_Speed']]
pf2 = pf2.set_index('Date')
y2 = pf2['Avg_Wind_Speed'].resample('D').mean()
y2 = y2.fillna(y2.mean())
plt.figure(figsize=(15,6))
plt.plot(y2)
plt.title('Time Variant Plot for Avg_Wind_Speed')
plt.xlabel("Time in Days")
plt.ylabel("Avg_Wind_Speed")
plt.show()


# `WIND ROSE PLOTS`

# In[35]:


pf3 = df[['Avg_Wind_Speed', 'Avg_Nacelle_Pos','Nacelle_Misalignment_Avg_Wind_Dir', 'Avg_Active_Power']]
pf3


# In[36]:


pf3['Direction'] = np.where((pf3['Avg_Nacelle_Pos']>337.5) & (pf3['Avg_Nacelle_Pos']<=22.5),"0N","0N")

pf3['Direction'] = np.where((pf3['Avg_Nacelle_Pos']>22.5) & (pf3['Avg_Nacelle_Pos']<=67.5),"1NE",pf3['Direction'])

pf3['Direction'] = np.where((pf3['Avg_Nacelle_Pos']>67.5) & (pf3['Avg_Nacelle_Pos']<=112.5),"2E",pf3['Direction'])
                            
pf3['Direction'] = np.where((pf3['Avg_Nacelle_Pos']>112.5) & (pf3['Avg_Nacelle_Pos']<=157.7),"3SE",pf3['Direction'])

pf3['Direction'] = np.where((pf3['Avg_Nacelle_Pos']>157.7) & (pf3['Avg_Nacelle_Pos']<=202.5),"4S",pf3['Direction'])

pf3['Direction'] = np.where((pf3['Avg_Nacelle_Pos']>202.5) & (pf3['Avg_Nacelle_Pos']<=247.5),"5SW",pf3['Direction'])

pf3['Direction'] = np.where((pf3['Avg_Nacelle_Pos']>247.5) & (pf3['Avg_Nacelle_Pos']<=292.5),"6W",pf3['Direction'])
                            
pf3['Direction'] = np.where((pf3['Avg_Nacelle_Pos']>292.5) & (pf3['Avg_Nacelle_Pos']<=337.5),"7NW",pf3['Direction'])


# In[37]:


pf4 = pf3.groupby('Direction').mean()
pf4


# `NACELLE POSITION AS PER WIND DIRECTION`

# In[38]:


# !pip install plotly
import plotly.express as px
fig = px.bar_polar(pf4, r="Avg_Nacelle_Pos", theta=pf4.index.values,
                   color=pf4.index.values, template="plotly_dark",
                   color_discrete_sequence= px.colors.sequential.Plasma_r, title = 'WIND ROSE PLOT ON NACELLE POSITION AS PER DIRECTION')
fig.show()


# `AVERAGE WIND DIRECTION`

# In[39]:


fig = px.bar_polar(pf4, r="Avg_Wind_Speed", theta=pf4.index.values,
                   color=pf4.index.values, template="plotly_dark",
                   color_discrete_sequence= px.colors.sequential.Plasma_r, title = 'WIND ROSE PLOT ON AVERAGE WIND DIRECTION')
fig.show()


# `AVERAGE NACELLE MISALIGNMENT WITH RESPECT TO WIND DIRECTION`

# In[40]:


fig = px.bar_polar(pf4, r="Nacelle_Misalignment_Avg_Wind_Dir", theta=pf4.index.values,
                   color=pf4.index.values, template="plotly_dark",
                   color_discrete_sequence= px.colors.sequential.Plasma_r, title = 'AVERAGE NACELLE MISALIGNMENT WITH RESPECT TO WIND DIRECTION')
fig.show()


# `CORRELATION PLOT`

# In[41]:


sns.heatmap(df.corr(), annot = True)
plt.subplots_adjust(left=0.8, bottom=0, right=2.8, top=1.8)
plt.show()

Thank You.....