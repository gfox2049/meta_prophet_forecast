#!/usr/bin/env python
# coding: utf-8

# In[1]:


customer_id = ''
output_bucket = 's3://'
input_file = 's3://whatever.csv'
output_file = output_bucket + 'dynamic_projection_monthly_linear_w_season_' + customer_id + '.csv'
#
# Note: linear non-seasonality is appropriate for customers in non-seasonal industries, 
# that are expecting to experience linear growth in the future.
# Example use cases - customer is moving from on-prem resources to cloud without much duplication / parallel infrastructure.
# Ideal for non-seasonal industries - healthcare, utilities, etc.


# In[2]:


# required input fields: key_field,month_start,sum_recs
# month_start should be YYYY-MM-DD
# output format:     
#     ds
#    ,key_field
#    ,yhat (projection)
#    ,conf_score
#    ,updated_on
# note:  conf_score is a composite of mean abs percentage and ratio of observations to full year


# In[3]:


## environment requires intall of following packages ##
#!pip install Prophet --quiet
#!pip install SQLAlchemy==1.4.46 --quiet
#!pip install pandasql --quiet


# In[4]:


import logging
import warnings 
logging.getLogger('prophet').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


# In[5]:


from datetime import datetime as dt


# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')


# In[7]:


from pandasql import sqldf


# In[8]:


from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation


# In[9]:


req_obs = 0
cap_growth = 1.05
fperiods = 14
cap_pctile = 1


# In[10]:


def replace_negatives(x):
    if x < 0:
        return 0
    else:
        return x


# In[11]:


pd.set_option('display.max_columns', None)


# In[12]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[13]:


pdfin = pd.read_csv(input_file)


# In[14]:


pdfin.count()


# In[15]:


pdfin.head()


# In[16]:


# estimates full last month based on daily average
pdfin2 = sqldf('''
        select
             key_field
            ,month_start
            ,case when month_start = DATE('now', 'start of month')
                  then (sum_recs / round(julianday('now') - julianday(DATE('now', 'start of month'))))*30
                  else sum_recs
             end sum_recs
        from
            pdfin
        where
            month_start != '2023-01-01'
            and month_start != '2023-02-01'
        order by
            key_field desc
            ,month_start asc
        ''')
pdfin2.head()


# In[17]:


pdz = sqldf('''
select
    month_start
    ,sum(sum_recs) sum_recs
from
    pdfin2
group by
    month_start
''')
pdz


# In[18]:


# forecast aggregate for cv scores

pdz['ds'] = pdz['month_start']
pdz['y'] = pdz['sum_recs']
pdm = pdz[['ds','y']]

pdm = pdm.sort_values(by=['ds'], ascending=True)

pdm['cap'] = pdm['y'].quantile(cap_pctile) + 1 # plus 1 prevents floor issue

m = Prophet(growth = 'linear', yearly_seasonality=True)
m.fit(pdm)
future = m.make_future_dataframe(periods=18, freq='MS')
future['cap'] = pdm['y'].quantile(cap_pctile) + 1 # plus 1 prevents floor issue
forecast = m.predict(future)
fout = pd.DataFrame(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

df_cv = cross_validation(m, initial='90 days', period='60 days', horizon = '60 days')
df_p = performance_metrics(df_cv)

pcalc = 1 - (df_p['mape'].mean())
mcalc = pdm['ds'].count() / 12 
if ( mcalc ) >= 1:
    mscore = 1
else:
    mscore = mcalc
fcalc = (pcalc + mscore) / 2


# In[19]:


fig1 = m.plot(forecast)


# In[20]:


fcalc


# In[21]:


# forecast on raw data 

pdd = pdfin2.groupby(['key_field','month_start'])['sum_recs'].sum().reset_index()
dfin = pd.DataFrame(columns = ['ds','yhat','yhat_lower','yhat_upper','key_field','conf'])

dfin.to_csv('dfin_temp_out_hi_' + customer_id + '.csv', index=False)

for y in pdd['key_field'].unique():

    print('forecasting ' + str(y))

    pdz = pdd[(pdd['key_field'] == y)]
    pdz['ds'] = pdz['month_start']
    pdz['y'] = pdz['sum_recs']
    pdm = pdz[['ds','y']]

    pdm = pdm.sort_values(by=['ds'], ascending=True)

    pdm['cap'] = pdm['y'].quantile(cap_pctile) + 1 # plus 1 prevents floor issue

    if len(pdm) >= req_obs:

        m = Prophet(growth = 'linear', yearly_seasonality=True)
        try:
            m.fit(pdm)
            future = m.make_future_dataframe(periods=18, freq='MS')
            future['cap'] = pdm['y'].quantile(cap_pctile) + 1 # plus 1 prevents floor issue
            forecast = m.predict(future)
            fout = pd.DataFrame(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
            fout['key_field'] = y
            mcalc = pdm['ds'].count() / 12 
            if ( mcalc ) >= 1:
                mscore = 1
            else:
                mscore = mcalc
            zcalc = (fcalc + mscore) / 2
            fout['conf'] = zcalc

            fout['yhat'] = fout['yhat'].apply(lambda x: replace_negatives(x))        
            fout.to_csv('dfin_temp_out_hi_' + customer_id + '.csv', mode='a', index=False, header=False)
        except:
            print('exception ' + str(y))


# In[22]:


dfin = pd.read_csv('dfin_temp_out_hi_' + customer_id + '.csv')


# In[23]:


draw = sqldf('''
select
    ds
    ,key_field
    ,yhat
    ,conf conf_score
    ,current_timestamp updated_on
from
    dfin
where
    ds >= DATE('now', 'start of month')
''')
draw


# In[24]:


draw.count()


# In[25]:


mean_conf = draw['conf_score'].mean()
min_conf = draw['conf_score'].min()
max_conf = draw['conf_score'].max()
print('mean_conf = ' + str(mean_conf))
print('min_conf = ' + str(min_conf))
print('max_conf = ' + str(max_conf))


# In[26]:


draw.head()


# In[27]:


# check actuals of real history
hist = sqldf('''
select 
    month_start
    ,sum(sum_recs) mon_sum_recs
from
    pdfin
group by
    month_start
order by month_start asc
''')
hist


# In[28]:


hist.index = hist['month_start']
hist.plot(figsize=(20, 10))


# In[29]:


# check forecast based on raw - lowest level top
fore = sqldf('''
select 
     date(ds, 'start of month') month_start
     ,sum(yhat) fore_mon_sum_recs
from
    draw
group by
    date(ds, 'start of month')
''')
fore


# In[30]:


fore.index = fore['month_start']


# In[31]:


fore.plot(figsize=(15, 10))


# In[32]:


# combine history and forecast to visualize and summarize
forehist = sqldf('''
select 
     date(ds, 'start of month') month_start
     ,sum(yhat) fore_mon_sum_recs
from
    draw
group by
    date(ds, 'start of month')
union
select 
    month_start
    ,sum(sum_recs) mon_sum_recs
from
    pdfin
group by
    month_start
order by month_start asc
''')
forehist


# In[33]:


forehist.index = forehist['month_start']


# In[34]:


forehist.plot(figsize=(15, 10))


# In[35]:


draw.to_csv(output_file, index=False, header=True)

