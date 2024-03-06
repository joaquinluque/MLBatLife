"""
ML Battery Life Predictor (MLBatLife)

It uses Machine Learning techniques to predict the SOH evolution of a battery
in a context of Self-Consumption Increase applications.

The input to the predictor is a CSV file containing the input power profile
(electricity demand minus photovoltaic generation) of a certain user.
This CSV file should contain two columns. 
The first row is used for a header: 
    "Time" (1st column) and "Power" (2nd column).
The information on this file has a resolution of one minute per row.
The second row in the "Time" column contains the initial time of the prediction
expressed in minutes from the beginning of the year. The following rows in
this column adds 1 minute to the previous one.
The (2nd column) contains the input power profile expressed in watts.

The predictor also requires as inputs:
    - The operational strategy. 0: Greedy; 1: FeedInDamp
    - The nominal capacity of the battery (in kWh)

The output of the predictor is a vector containing the SOH 
(capacity over initial capacity) estimated at the end
of each day.

The FeedInDamp strategy has been trained with the solar radiation in Munich

The methods used in this predictor are explained in the following paper:
Luque, J., Tepe, B., Carrasco, A., Heidarabadi, H., León, C., & Hesse, H. 
Machine Learning Estimation of Battery State of Health in Residential Photovoltaic Systems.
Journal of Energy Storage.

"""

import numpy as np
import pandas as pd
import pickle


#Function to read the CSV file
def read_csv(fname):
    """
    Inputs:
       fname: Name of the CSV file containing the input power profile
    Returns:
       t: time column (in minutes)
       input_prof: input power profile (in watts)
    """
    df = pd.read_csv(fname)
    t = df['Time'].to_numpy()
    input_prof = df['Power'].to_numpy()
    return t,input_prof


#Function to extract the features characterizing the input power profile
#during a day
def extract_features(t,input_prof):
    """
    Inputs:
       t: time column (in minutes)
       input_prof: input power profile (in watts)
    Returns:
       bf: basic features. A matrix containing one row per day and two columns
           with the mean value (T1) and mean absolute value (T4) of the 
           input power profile
       day: A vector containing one value per day indicating the day from
           the beginning of the first year of prediction
    """

    ls = 24*60 #Length of a segment (in minutes). It corresponds to one day
    n = len(input_prof) #Length of the input power profile (in minutes)
    nseg = int(n/ls) #Number of segments
    bf = np.zeros((nseg,2)) #Basic features (T1 & T4)
    day = np.zeros(nseg) #Day
    for iseg in range(nseg):
        i1 = iseg*ls
        i2 = i1+ls
        v = input_prof[i1:i2]
        bf[iseg,0] = np.mean(v) #T1: mean
        bf[iseg,1] = np.mean( np.abs(v) )  #T4: Absolute mean value
        day[iseg] = t[i1] / (24*60)
    return bf,day


#Function to estimate the SOH
def estimate_soh(bf,day,strategy,Qnom):
    """
    Inputs:
       bf: basic features. A matrix containing one row per day and two columns
           with the mean value (T1) and mean absolute value (T4) of the 
           input power profile
       day: A vector containing one value per day indicating the day from
           the beginning of the first year of prediction
       strategy: Operational strategy 0: Greedy; 1: FeedInDamp
       Qnom: Nominal capacity of the battery (kWh)
    Returns:
       Qhat: A vector containing one value per day indicating the 
           SOH (capacity) estimated ath the end of the day
    """

    n = bf.shape[0]
    d = 5 #Number of features: ['SOH0','T1','Strategy','T4','Day']

    #Machine Learing (ML) Model
    #   model: A trained Random Forest model
    #   mu: mean value of the features in the training dataset
    #   sigma: standard deviation of the features in the training dataset
    #   Qtr: Nominal capacity of the battery in the training dataset (in kWh)
    PickleFileName = 'MLBatLife_Model.pkl'  #File containing the ML model
    with open(PickleFileName, 'rb') as f:
        model,mu,sigma,Qtr = pickle.load(f)
    
    X = np.zeros((n,d)) #Matrix design 
    X[:,1] = bf[:,0] * (Qtr/Qnom) #T1
    X[:,2] = strategy #Operational strategy
    X[:,3] = bf[:,1] * (Qtr/Qnom)  #T4
    X[:,4] = day #Day

    Q0 = 1 #Capacity at the begining of the first day
    Qhat = np.zeros(n)
    for i in range(n):
        Xday = X[i,:] #Matrix design for one day
        Xday[0] = Q0
        Qhat[i] = Q0
        Xnorm = (Xday-mu) / sigma #Normalize matrix design
        Xnorm = Xnorm.reshape((1,-1))
        Qlosshat = model.predict(Xnorm) #ML Estimation of Q lost in one day
        Q0 = Q0 - Qlosshat #Capacity at the begining of the next day
    
    return Qhat


