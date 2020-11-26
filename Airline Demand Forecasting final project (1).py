#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import os
os.chdir("C:\\Users\\swath\\Downloads\\")

#trainingdataset#
training = pd.read_csv("airline_data_training.csv", sep=',', header=0)
training['departure_date'] = pd.to_datetime(training['departure_date'])
training['booking_date'] = pd.to_datetime(training['booking_date'])
training['days_prior'] = (training['departure_date'] - training['booking_date']).dt.days
training['day'] = training['departure_date'].dt.day_name()
training['demand'] = training['cum_bookings'].groupby(training['departure_date']).transform(max)
training['Forecast_remaining_demand'] = training['demand'] - training['cum_bookings']
training['Historical_Booking_Rate']=training['cum_bookings']/training['demand']
training['avg_Forecast_remaining_demand'] = training.groupby(['days_prior','day'])['Forecast_remaining_demand'].transform(np.mean)
training['avg_Historical_Booking_rate'] = training["Historical_Booking_Rate"].groupby(training['days_prior']).transform(np.mean)

#validation data set#
validation=pd.read_csv("airline_data_validation.csv",sep=',',header=0)
validation['departure_date'] = pd.to_datetime(validation['departure_date'])
validation['booking_date'] = pd.to_datetime(validation['booking_date'])
validation['days_prior'] = (validation['departure_date'] - validation['booking_date']).dt.days
validation['day'] = validation['departure_date'].dt.day_name()
validation['demand'] = validation['cum_bookings'].groupby(validation['departure_date']).transform(max)
validation = validation.loc[validation['days_prior'] > 0]
validation['error'] = abs(validation['demand']-validation['naive_fcst'])
Total_error=validation['error'].sum()


#Multiplicative model:
Dataframe = training.drop_duplicates(['days_prior'])
Dataframe1= Dataframe[['days_prior','avg_Historical_Booking_rate']]
Dataframe2= validation.merge(Dataframe1,how="left",on=['days_prior'])
Dataframe2["estimate"]=Dataframe2['cum_bookings']/Dataframe2['avg_Historical_Booking_rate']
Dataframe2['Error_multi'] = abs(Dataframe2['demand']-Dataframe2['estimate'])
Error_multi=Dataframe2["Error_multi"].sum()

#Calculate MASE : (Total_error_multi)/(Total_error)
MASE_multi=Error_multi/Total_error
print("MASE(Multiplicative model):",MASE_multi)

#Additive model:
Df= training.drop_duplicates(['days_prior', 'day'])
Df1 = Df[['days_prior','day','avg_Forecast_remaining_demand']]
Df2 = validation.merge(Df1, how = 'left', on = ['days_prior', 'day'])
Df2['estimate1'] = Df2['cum_bookings'] + Df2['avg_Forecast_remaining_demand']
Df2['Error_add'] = abs(Df2['demand']-Df2['estimate1'])
Error_add=Df2["Error_add"].sum()

#Percentage
P1=round((Error_add/Total_error),4)
improve_a=round((1-P1)*100,2)



#calculate MASE:(Total_error_add)/(Total_error)
MASE_add=Error_add/Total_error
print("MASE(Additive model):",MASE_add)
print("MASE additive model improvement to the basemodel is :",improve_a,"%")
