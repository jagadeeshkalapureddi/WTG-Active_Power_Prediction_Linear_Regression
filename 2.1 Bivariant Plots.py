#!/usr/bin/env python
# coding: utf-8

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


# In[2]:


df = pd.read_csv('Cleaned_df.csv')


# In[3]:


pf = df[(df['Avg_Wind_Speed'] > 3) & (df['Avg_Active_Power'] > 10)]


# In[4]:


import ipywidgets as widgets

@widgets.interact(Feature1=pf.columns, Feature2=pf.columns)
def create_scatter(Feature1, Feature2):
    with plt.style.context("ggplot"):
        fig = plt.figure(figsize=(8,4))

        plt.scatter(x = pf[Feature1],
                    y = pf[Feature2], s = 8, color = 'green')

        plt.xlabel(Feature1.capitalize())
        plt.ylabel(Feature2.capitalize())

        plt.title("%s vs %s"%(Feature1.capitalize(), Feature2.capitalize()))


# In[ ]:




