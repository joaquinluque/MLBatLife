"""
ML Battery Life Predictor. Example of use.

It requires:
    - The MLBatLife_Pred library
    - A CSV file containing the input power profile (an example is provided)
    - The trained Machine Learning model used for estimating SOH
        This model is provided in the file "MLBatLife_Model.pkl"

"""

import MLBatLife_Pred as blp

fname = 'input_profile_sample.csv'
t,input_prof= blp.read_csv(fname) #Reads the CSV file
bf,day = blp.extract_features(t,input_prof) #Obtains the features of each day
strategy = 0 #Greedy
Qnom = 5 #Nominal capacity of the battery in kWh
SOH_hat = blp.estimate_soh(bf,day,strategy,Qnom) #Estimates SOH for each day

import matplotlib.pyplot as plt
plt.plot(day,SOH_hat)
plt.xlabel('Day')
plt.ylabel('SOH')
plt.tight_layout()

