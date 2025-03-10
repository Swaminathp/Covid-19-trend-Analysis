#!/usr/bin/env python
# coding: utf-8

# # Project : Analyzing the trends of COVID-19 with Python

# In[1]:


# Import important libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


# # Loading the dataset

# In[2]:


df = pd.read_csv('Covid_19_Clean_Complete (1) (1).csv')


# In[3]:


df.head()


# In[4]:


df = df.rename(columns={'Province/State':'state','Country/Region':'country','Lat':'lat','Long':'long','Date':'date','Confirmed':'confirmed','Deaths':'deaths','Recovered':'recovered', 'Active':'active'})


# In[5]:


df


# # Finding active cases

# In[6]:


df['active'] = df['confirmed'] - df['deaths'] - df['recovered']


# In[7]:


df['active']


# # To find the top cases in the last value in dates

# In[8]:


top = df[df['date']==df['date'].max()]
top


# # Grouping the data by country

# In[9]:


w = top.groupby('country')[['confirmed', 'active', 'deaths']].sum().reset_index()


# In[10]:


w


# # Plotting the data on world map

# In[11]:


fig = px.choropleth(w,locations='country',locationmode='country names',color ='active', hover_name='country',
                    range_color=[1,1500], color_continuous_scale='Peach',title='Active Cases Countries')
fig.show()


# # Visualizing confirmed cases

# In[12]:


plt.figure(figsize = (15,10))

t_cases= df.groupby('date')['confirmed'].sum().reset_index()
t_cases['date'] = pd.to_datetime(t_cases['date'])

a = sns.pointplot(x= t_cases.date.dt.date,
                 y= t_cases.confirmed,
                 color = 'r')
a.set(xlabel = 'Date', ylabel = 'Total Cases')

plt.xticks(rotation = 90, fontsize = 7)
plt.yticks(fontsize = 15)
a


# In[13]:


t_cases


# #  Top 20 countries with the most number of active cases

# In[14]:


top_active = top.groupby(by = 'country')['active'].sum().sort_values(ascending= False).head(20).reset_index()
top_active


# # Visualization

# In[15]:


plt.figure(figsize= (20,15))
plt.title('Top 20 countries with the most number of active cases', fontsize = 25)
a = sns.barplot(x= top_active.active,
               y= top_active.country)
a.set(xlabel = 'Active Cases', ylabel = 'Countries')
for i,(value, name) in enumerate(zip(top_active.active, top_active.country)):
    a.text(value, i-0.5,f'{value:,.0f}', size=10, va='top')

plt.xticks(rotation = 90, fontsize = 7)
plt.yticks(fontsize = 15)
a


# #  Top 20 countries with the most number of deaths

# In[16]:


top_death = top.groupby(by = 'country')['deaths'].sum().sort_values(ascending= False).head(20).reset_index()
top_death


# # Visualization

# In[17]:


plt.figure(figsize= (15,15))
plt.title('Top 20 countries with the most number of death cases', fontsize = 25)
a = sns.barplot(x= top_death.deaths,
               y= top_active.country)
a.set(xlabel = 'Number of deaths', ylabel = 'Countries')
for i,(value, name) in enumerate(zip(top_death.deaths, top_active.country)):
    a.text(value, i-0.5,f'{value:,.0f}', size=10, va='top')

plt.xticks(rotation = 90, fontsize = 7)
plt.yticks(fontsize = 15)
a


# # Recovery

# In[18]:


top_recovery = top.groupby(by = 'country')['recovered'].sum().sort_values(ascending= False).head(20).reset_index()
top_recovery


# # Visualizing recovery

# In[19]:


plt.figure(figsize= (20,15))
plt.title('Top 20 countries with the most number of recovery', fontsize = 25)
a = sns.barplot(x= top_recovery.recovered,
               y= top_recovery.country)
a.set(xlabel = 'Number of recovery', ylabel = 'Countries')
for i,(value, name) in enumerate(zip(top_recovery.recovered, top_recovery.country)):
    a.text(value, i-0.5,f'{value:,.0f}', size=10, va='top')

plt.xticks(rotation = 90, fontsize = 7)
plt.yticks(fontsize = 15)
a


# # Country wise analysis

# # Brazil

# In[20]:


Brazil=df[df.country =='Brazil']
Brazil = Brazil.groupby(by='date')[['confirmed', 'active','recovered','deaths']].sum().reset_index()
Brazil.tail(20)


# # USA

# In[21]:


US=df[df.country =='US']
US = US.groupby(by='date')[['confirmed', 'active','recovered','deaths']].sum().reset_index()
US.tail(20)


# # India

# In[22]:


India=df[df.country =='India']
India = India.groupby(by='date')[['confirmed', 'active','recovered','deaths']].sum().reset_index()
India.tail(20)


# # Russia

# In[23]:


Russia=df[df.country =='Russia']
Russia = Russia.groupby(by='date')[['confirmed', 'active','recovered','deaths']].sum().reset_index()
Russia.tail(20)


# # Visualizing comparison of confimed cases  

# In[24]:


sns.pointplot(data=Brazil, x = Brazil.index, y= 'confirmed', color = 'green', label = 'Brazil')
sns.pointplot(data=India, x = India.index, y= 'confirmed', color = 'yellow', label = 'India')
sns.pointplot(data=US, x = US.index, y= 'confirmed', color = 'red', label = 'USA')
sns.pointplot(data=Russia, x = Russia.index, y= 'confirmed', color = 'blue', label = 'Russia')


plt.xlabel('No. of Days', fontsize = 5)
plt.ylabel('Confirmed cases', fontsize = 5)
plt.title('Confirmed cases over the period of time', fontsize=5)

plt.legend()
plt.show


# # Visualizing comparison of active cases   

# In[25]:


sns.pointplot(data=Brazil, x = Brazil.index, y= 'active', color = 'green', label = 'Brazil')
sns.pointplot(data=India, x = India.index, y= 'active', color = 'yellow', label = 'India')
sns.pointplot(data=US, x = US.index, y= 'active', color = 'red', label = 'USA')
sns.pointplot(data=Russia, x = Russia.index, y= 'active', color = 'blue', label = 'Russia')


plt.xlabel('No. of Days', fontsize = 5)
plt.ylabel('Active cases', fontsize = 5)
plt.title('Active cases over the period of time', fontsize=5)

plt.legend()
plt.show


# # Visualizing comparison of death cases  

# In[26]:


sns.pointplot(data=Brazil, x = Brazil.index, y= 'deaths', color = 'green', label = 'Brazil')
sns.pointplot(data=India, x = India.index, y= 'deaths', color = 'yellow', label = 'India')
sns.pointplot(data=US, x = US.index, y= 'deaths', color = 'red', label = 'USA')
sns.pointplot(data=Russia, x = Russia.index, y= 'deaths', color = 'blue', label = 'Russia')


plt.xlabel('No. of Days', fontsize = 5)
plt.ylabel('Death cases', fontsize = 5)
plt.title('Death cases over the period of time', fontsize=5)

plt.legend()
plt.show


# # Visualizing comparison of recovered cases  

# In[27]:


sns.pointplot(data=Brazil, x = Brazil.index, y= 'recovered', color = 'green', label = 'Brazil')
sns.pointplot(data=India, x = India.index, y= 'recovered', color = 'yellow', label = 'India')
sns.pointplot(data=US, x = US.index, y= 'recovered', color = 'red', label = 'USA')
sns.pointplot(data=Russia, x = Russia.index, y= 'recovered', color = 'blue', label = 'Russia')


plt.xlabel('No. of Days', fontsize = 5)
plt.ylabel('Recovered cases', fontsize = 5)
plt.title('Recovered cases over the period of time', fontsize=5)

plt.legend()
plt.show


# # Forecasting using prophet from Facebook

# In[28]:


pip install prophet


# In[29]:


from prophet import Prophet


# In[30]:


df


# In[31]:


df.groupby(by='date').head()


# In[32]:


total_active=df['active'].sum()
print('Total number active case around the world is',total_active)


# In[33]:


confirmed = df.groupby('date').sum()['confirmed'].reset_index()
death = df.groupby('date').sum()['deaths'].reset_index()
recovered = df.groupby('date').sum()['recovered'].reset_index()


# In[34]:


confirmed


# # Forecasting confirmed cases

# In[35]:


confirmed.columns = ['ds', 'y']
confirmed['ds'] = pd.to_datetime(confirmed['ds'])
confirmed.tail()


# In[36]:


m = Prophet(interval_width = 0.95)
m.fit(confirmed)


# In[37]:


future = m.make_future_dataframe(periods = 7, freq = 'D')
future


# In[38]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(10)


# In[39]:


confirmed_forecast_plot = m.plot(forecast)


# In[40]:


confirmed_forecast_plot = m.plot_components(forecast)


# # Forecasting death cases

# In[41]:


death.columns = ['ds', 'y']
death['ds'] = pd.to_datetime(death['ds'])
m = Prophet()
m.fit(death)
future = m.make_future_dataframe(periods= 7, freq = "D")
forecast = m.predict(future)
death_forecast_plot = m.plot(forecast)


# # Recovery forecasting

# In[42]:


recovered.columns = ['ds', 'y']
recovered['ds'] = pd.to_datetime(recovered['ds'])
m = Prophet()
m.fit(recovered)
future = m.make_future_dataframe(periods= 7, freq = "D")
forecast = m.predict(future)
recovery_forecast_plot = m.plot(forecast)


# # Active cases forecasting

# In[43]:


active = df.groupby('date').sum()['active'].reset_index()
active


# In[44]:


active.columns = ['ds', 'y']
active['ds'] = pd.to_datetime(active['ds'])
m = Prophet()
m.fit(recovered)
future = m.make_future_dataframe(periods= 7, freq = "D")
forecast = m.predict(future)
active_forecast_plot = m.plot(forecast)


# In[ ]:




