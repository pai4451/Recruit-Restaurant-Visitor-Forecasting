#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Data Description
# 
# In this competition, you are provided a time-series forecasting problem centered around restaurant visitors. The data comes from two separate sites:
# 
# * Hot Pepper Gourmet (hpg): similar to Yelp, here users can search restaurants and also make a reservation online
# * AirREGI / Restaurant Board (air): similar to Square, a reservation control and cash register system
# 
# You must use the reservations, visits, and other information from these sites to forecast future restaurant visitor totals on a given date. <span style="color:red">The training data covers the dates from 2016 until April 2017</span>. <span style="color:red">The test set covers the last week of April and May of 2017</span>. The test set is split based on time (the public fold coming first, the private fold following the public) and covers a chosen subset of the air restaurants. Note that the test set intentionally spans a holiday week in Japan called the "Golden Week."
# 
# There are days in the test set where the restaurant were closed and had no visitors. These are ignored in scoring. The training set omits days where the restaurants were closed.
# ### File Descriptions
# 
# This is a relational dataset from two systems. Each file is prefaced with the source (either air_ or hpg_) to indicate its origin. Each restaurant has a unique air_store_id and hpg_store_id. Note that not all restaurants are covered by both systems, and that you have been provided data beyond the restaurants for which you must forecast. Latitudes and Longitudes are not exact to discourage de-identification of restaurants.

# ### air_reserve.csv
# 
# This file contains reservations made in the air system. Note that the reserve_datetime indicates the time when the reservation was created, whereas the visit_datetime is the time in the future where the visit will occur.
# 
# * `air_store_id` - the restaurant's id in the air system
# * `visit_datetime` - the time of the reservation
# * `reserve_datetime` - the time the reservation was made
# * `reserve_visitors` - the number of visitors for that reservation

# In[20]:


air_reserve = pd.read_csv('./data/recruit-restaurant-visitor-forecasting/air_reserve.csv') 
air_reserve.head()


# ### hpg_reserve.csv
# 
# This file contains reservations made in the hpg system.
# 
# * `hpg_store_id` - the restaurant's id in the hpg system
# * `visit_datetime` - the time of the reservation
# * `reserve_datetime` - the time the reservation was made
# * `reserve_visitors` - the number of visitors for that reservation

# In[21]:


hpg_reserve = pd.read_csv('./data/recruit-restaurant-visitor-forecasting/hpg_reserve.csv') 
hpg_reserve.head()


# ### air_store_info.csv
# 
# This file contains information about select air restaurants. Column names and contents are self-explanatory.
# 
# * `air_store_id`
# * `air_genre_name`
# * `air_area_name`
# * `latitude`
# * `longitude`
# Note: latitude and longitude are the latitude and longitude of the area to which the store belongs

# In[22]:


air_store_info = pd.read_csv('./data/recruit-restaurant-visitor-forecasting/air_store_info.csv') 
air_store_info.head()


# ### hpg_store_info.csv
# 
# This file contains information about select hpg restaurants. Column names and contents are self-explanatory.
# 
# * `hpg_store_id`
# * `hpg_genre_name`
# * `hpg_area_name`
# * `latitude`
# * `longitude`
# Note: latitude and longitude are the latitude and longitude of the area to which the store belongs

# In[23]:


hpg_store_info = pd.read_csv('./data/recruit-restaurant-visitor-forecasting/hpg_store_info.csv') 
hpg_store_info.head()


# ### store_id_relation.csv
# 
# This file allows you to join select restaurants that have both the air and hpg system.
# 
# * `hpg_store_id`
# * `air_store_id`

# In[24]:


store_id_relation = pd.read_csv('./data/recruit-restaurant-visitor-forecasting/store_id_relation.csv') 
store_id_relation.head()


# ### air_visit_data.csv
# 
# This file contains historical visit data for the air restaurants.
# 
# * `air_store_id`
# * `visit_date - the date`
# * `visitors - the number of visitors to the restaurant on the date`

# In[25]:


air_visit_data = pd.read_csv('./data/recruit-restaurant-visitor-forecasting/air_visit_data.csv') 
air_visit_data.head()


# ### date_info.csv
# 
# This file gives basic information about the calendar dates in the dataset.
# 
# * `calendar_date`
# * `day_of_week`
# * `holiday_flg - is the day a holiday in Japan`

# In[26]:


date_info = pd.read_csv('./data/recruit-restaurant-visitor-forecasting/date_info.csv') 
date_info.head()


# ### sample_submission.csv
# 
# This file shows a submission in the correct format, including the days for which you must forecast.
# 
# * `id` - the id is formed by concatenating the air_store_id and visit_date with an underscore
# * `visitors`- the number of visitors forecasted for the store and date combination

# In[27]:


sample_submission = pd.read_csv('./data/recruit-restaurant-visitor-forecasting/sample_submission.csv') 
sample_submission.head()


# In[ ]:




